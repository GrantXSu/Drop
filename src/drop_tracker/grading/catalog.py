"""Local Pokémon card catalog and visual-reference matching."""

from __future__ import annotations

import argparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import httpx
import numpy as np

from .features import CardAnalysis, analyze_image


API_BASE = "https://api.tcgdex.net/v2/en"
DEFAULT_CATALOG_PATH = Path("data/grading/card_catalog.sqlite")
DEFAULT_SERIES = ("swsh", "sv", "me")
USER_AGENT = "DropCardGrader/0.1 (catalog sync; TCGdex)"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS series (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            release_date TEXT,
            synced_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sets (
            id TEXT PRIMARY KEY,
            series_id TEXT NOT NULL,
            name TEXT NOT NULL,
            release_date TEXT,
            official_count INTEGER,
            total_count INTEGER,
            logo_url TEXT,
            symbol_url TEXT,
            FOREIGN KEY(series_id) REFERENCES series(id)
        );
        CREATE TABLE IF NOT EXISTS cards (
            id TEXT PRIMARY KEY,
            series_id TEXT NOT NULL,
            set_id TEXT NOT NULL,
            set_name TEXT NOT NULL,
            local_id TEXT NOT NULL,
            name TEXT NOT NULL,
            image_url TEXT,
            perceptual_hash TEXT,
            color_signature TEXT,
            reference_features TEXT,
            synced_at TEXT NOT NULL,
            FOREIGN KEY(set_id) REFERENCES sets(id)
        );
        CREATE INDEX IF NOT EXISTS cards_name_idx ON cards(name);
        CREATE INDEX IF NOT EXISTS cards_set_idx ON cards(set_id);
        """
    )
    return connection


def _json_get(client: httpx.Client, path: str) -> Dict[str, object]:
    response = client.get(f"{API_BASE}/{path}")
    response.raise_for_status()
    return response.json()


def _perceptual_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8]
    median = float(np.median(coefficients[1:]))
    bits = (coefficients > median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _color_signature(image: np.ndarray) -> List[float]:
    resized = cv2.resize(image, (128, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
    return [round(float(value), 6) for value in histogram.reshape(-1)]


def _reference_profile(data: bytes) -> Tuple[str, str, str]:
    analysis = analyze_image(data, side="front")
    features = {
        name: analysis.features[name]
        for name in (
            "edge_pale",
            "corner_pale_mean",
            "corner_pale_max",
            "surface_glare",
            "surface_dark",
        )
    }
    return (
        _perceptual_hash(analysis.image),
        json.dumps(_color_signature(analysis.image), separators=(",", ":")),
        json.dumps(features, separators=(",", ":")),
    )


def _download_profile(card: Dict[str, str]) -> Tuple[str, Optional[Tuple[str, str, str]], Optional[str]]:
    image_base = card.get("image")
    if not image_base:
        return card["id"], None, "no reference image"
    try:
        response = httpx.get(
            f"{image_base}/low.webp",
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        return card["id"], _reference_profile(response.content), None
    except (httpx.HTTPError, ValueError) as error:
        return card["id"], None, str(error)


def sync_catalog(
    path: Path = DEFAULT_CATALOG_PATH,
    series_ids: Sequence[str] = DEFAULT_SERIES,
    with_images: bool = True,
    workers: int = 8,
) -> Dict[str, object]:
    """Sync series, sets, card summaries, and optional visual fingerprints."""
    connection = _connect(path)
    now = datetime.now(timezone.utc).isoformat()
    cards_to_profile: List[Dict[str, str]] = []
    set_count = 0

    with httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
        for series_id in series_ids:
            series = _json_get(client, f"series/{series_id}")
            connection.execute(
                """
                INSERT INTO series(id, name, release_date, synced_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    release_date=excluded.release_date,
                    synced_at=excluded.synced_at
                """,
                (
                    series["id"],
                    series["name"],
                    series.get("releaseDate"),
                    now,
                ),
            )
            for set_summary in series.get("sets", []):
                set_data = _json_get(client, f"sets/{set_summary['id']}")
                counts = set_data.get("cardCount", {})
                connection.execute(
                    """
                    INSERT INTO sets(
                        id, series_id, name, release_date, official_count,
                        total_count, logo_url, symbol_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        series_id=excluded.series_id,
                        name=excluded.name,
                        release_date=excluded.release_date,
                        official_count=excluded.official_count,
                        total_count=excluded.total_count,
                        logo_url=excluded.logo_url,
                        symbol_url=excluded.symbol_url
                    """,
                    (
                        set_data["id"],
                        series_id,
                        set_data["name"],
                        set_data.get("releaseDate"),
                        counts.get("official"),
                        counts.get("total"),
                        set_data.get("logo"),
                        set_data.get("symbol"),
                    ),
                )
                set_count += 1
                for card in set_data.get("cards", []):
                    connection.execute(
                        """
                        INSERT INTO cards(
                            id, series_id, set_id, set_name, local_id, name,
                            image_url, synced_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            series_id=excluded.series_id,
                            set_id=excluded.set_id,
                            set_name=excluded.set_name,
                            local_id=excluded.local_id,
                            name=excluded.name,
                            image_url=excluded.image_url,
                            synced_at=excluded.synced_at
                        """,
                        (
                            card["id"],
                            series_id,
                            set_data["id"],
                            set_data["name"],
                            card["localId"],
                            card["name"],
                            card.get("image"),
                            now,
                        ),
                    )
                    if with_images and card.get("image"):
                        existing = connection.execute(
                            "SELECT perceptual_hash FROM cards WHERE id = ?", (card["id"],)
                        ).fetchone()
                        if not existing or not existing["perceptual_hash"]:
                            cards_to_profile.append(card)
                connection.commit()

    failures = []
    completed = 0
    if cards_to_profile:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(_download_profile, card): card["id"]
                for card in cards_to_profile
            }
            for future in as_completed(futures):
                card_id, profile, error = future.result()
                if profile:
                    connection.execute(
                        """
                        UPDATE cards SET
                            perceptual_hash = ?,
                            color_signature = ?,
                            reference_features = ?
                        WHERE id = ?
                        """,
                        (*profile, card_id),
                    )
                    completed += 1
                    if completed % 100 == 0:
                        connection.commit()
                elif error:
                    failures.append({"card_id": card_id, "error": error})
        connection.commit()

    card_count = connection.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    profile_count = connection.execute(
        "SELECT COUNT(*) FROM cards WHERE perceptual_hash IS NOT NULL"
    ).fetchone()[0]
    connection.close()
    return {
        "database": str(path),
        "series": list(series_ids),
        "sets": set_count,
        "cards": card_count,
        "visual_profiles": profile_count,
        "new_profiles": completed,
        "failures": failures[:20],
        "failure_count": len(failures),
        "source": "TCGdex",
        "source_url": "https://tcgdex.dev/",
    }


def _hamming(first: str, second: str) -> int:
    return bin(int(first, 16) ^ int(second, 16)).count("1")


def _histogram_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(np.abs(np.asarray(first) - np.asarray(second)).sum())


def identify_card(
    image: np.ndarray,
    path: Path = DEFAULT_CATALOG_PATH,
    limit: int = 3,
) -> List[Dict[str, object]]:
    """Return the closest visual catalog matches for a normalized card front."""
    if not path.exists():
        return []
    hashes = [
        _perceptual_hash(image),
        _perceptual_hash(cv2.rotate(image, cv2.ROTATE_180)),
    ]
    colors = _color_signature(image)
    connection = _connect(path)
    rows = connection.execute(
        """
        SELECT id, series_id, set_id, set_name, local_id, name, image_url,
               perceptual_hash, color_signature, reference_features
        FROM cards
        WHERE perceptual_hash IS NOT NULL AND color_signature IS NOT NULL
        """
    ).fetchall()
    scored = []
    for row in rows:
        hamming = min(_hamming(value, row["perceptual_hash"]) for value in hashes)
        color_distance = _histogram_distance(colors, json.loads(row["color_signature"]))
        score = hamming + color_distance * 8.0
        scored.append((score, hamming, color_distance, row))
    connection.close()
    scored.sort(key=lambda item: item[0])

    results = []
    for score, hamming, color_distance, row in scored[:limit]:
        confidence = max(0.0, min(1.0, 1.0 - score / 38.0))
        results.append(
            {
                "id": row["id"],
                "name": row["name"],
                "series_id": row["series_id"],
                "set_id": row["set_id"],
                "set_name": row["set_name"],
                "number": row["local_id"],
                "image_url": (
                    f"{row['image_url']}/high.webp" if row["image_url"] else None
                ),
                "confidence": round(confidence, 3),
                "hash_distance": hamming,
                "color_distance": round(color_distance, 3),
                "reference_features": (
                    json.loads(row["reference_features"])
                    if row["reference_features"]
                    else None
                ),
            }
        )
    return results


def apply_reference_baseline(
    analysis: CardAnalysis, match: Dict[str, object]
) -> None:
    """Subtract legitimate printed pale areas from physical-wear proxies."""
    if float(match.get("confidence", 0.0)) < 0.55:
        return
    reference = match.get("reference_features")
    if not reference:
        return
    for name in ("edge_pale", "corner_pale_mean", "corner_pale_max"):
        analysis.features[name] = max(
            0.0, analysis.features[name] - float(reference.get(name, 0.0))
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the local Pokémon card catalog.")
    parser.add_argument("--database", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--series", nargs="+", default=list(DEFAULT_SERIES))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    try:
        report = sync_catalog(
            args.database,
            args.series,
            with_images=not args.metadata_only,
            workers=args.workers,
        )
    except (httpx.HTTPError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
