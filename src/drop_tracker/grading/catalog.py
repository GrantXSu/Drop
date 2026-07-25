"""Local Pokémon card catalog and visual-reference matching."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote

import cv2
import httpx
import numpy as np

from .features import CardAnalysis, analyze_image
from .embeddings import cosine_similarity, visual_embedding


API_BASE = "https://api.tcgdex.net/v2/en"
DEFAULT_CATALOG_PATH = Path("data/grading/card_catalog.sqlite")
DEFAULT_SERIES = ("all",)
USER_AGENT = "DropCardGrader/0.1 (catalog sync; TCGdex)"
FALLBACK_IMAGE_BASE = "https://images.pokemontcg.io"


def _fallback_set_id(set_id: str) -> str:
    if set_id.endswith(".5tg"):
        return set_id.replace(".5tg", "tg")
    return set_id


def _fallback_image_url(
    set_id: str, local_id: str, high_resolution: bool = False
) -> str:
    suffix = "_hires.png" if high_resolution else ".png"
    return (
        f"{FALLBACK_IMAGE_BASE}/{quote(_fallback_set_id(set_id))}/"
        f"{quote(local_id)}{suffix}"
    )


def _image_asset_url(
    image_url: Optional[str],
    set_id: str,
    local_id: str,
    high_resolution: bool = False,
) -> str:
    if not image_url:
        return _fallback_image_url(set_id, local_id, high_resolution)
    if image_url.endswith((".png", ".jpg", ".jpeg", ".webp")):
        if high_resolution and image_url.endswith(".png"):
            return image_url.removesuffix(".png") + "_hires.png"
        return image_url
    size = "high" if high_resolution else "low"
    return f"{image_url}/{size}.webp"


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
            visual_embedding TEXT,
            synced_at TEXT NOT NULL,
            FOREIGN KEY(set_id) REFERENCES sets(id)
        );
        CREATE INDEX IF NOT EXISTS cards_name_idx ON cards(name);
        CREATE INDEX IF NOT EXISTS cards_set_idx ON cards(set_id);
        """
    )
    card_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(cards)")
    }
    if "visual_embedding" not in card_columns:
        connection.execute("ALTER TABLE cards ADD COLUMN visual_embedding TEXT")
    return connection


def _json_get(client: httpx.Client, path: str) -> object:
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


def _profile_from_analysis(analysis: CardAnalysis) -> Tuple[str, str, str]:
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
    features["centering_distances"] = analysis.diagnostics["centering"]["distances"]
    features["card_dimensions"] = analysis.diagnostics["centering"]["card_dimensions"]
    thumbnail = cv2.resize(analysis.image, (256, 358), interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(
        ".jpg", thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 72]
    )
    if success:
        features["reference_thumbnail"] = base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")
    return (
        _perceptual_hash(analysis.image),
        json.dumps(_color_signature(analysis.image), separators=(",", ":")),
        json.dumps(features, separators=(",", ":")),
    )


def _reference_profile(data: bytes) -> Tuple[str, str, str]:
    return _profile_from_analysis(analyze_image(data, side="front"))


def _download_profile(
    card: Dict[str, str],
) -> Tuple[str, Optional[Tuple[str, str, str, Optional[str]]], Optional[str]]:
    image_base = card.get("image")
    image_url = _image_asset_url(
        image_base,
        card["set_id"],
        card["localId"],
    )
    try:
        response = httpx.get(
            image_url,
            timeout=20.0,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        analysis = analyze_image(response.content, side="front")
        embedding = visual_embedding(analysis.image, allow_download=True)
        embedding_json = (
            json.dumps(
                [round(float(value), 7) for value in embedding],
                separators=(",", ":"),
            )
            if embedding is not None
            else None
        )
        return card["id"], (*_profile_from_analysis(analysis), embedding_json), None
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
        resolved_series = list(series_ids)
        if "all" in resolved_series:
            series_index = _json_get(client, "series")
            if not isinstance(series_index, list):
                raise ValueError("TCGdex returned an invalid English series index.")
            resolved_series = [
                str(item["id"])
                for item in series_index
                if isinstance(item, dict) and item.get("id")
            ]
        for series_id in resolved_series:
            series = _json_get(client, f"series/{series_id}")
            if not isinstance(series, dict):
                raise ValueError(f"TCGdex returned invalid series data for {series_id}.")
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
                    image_url = card.get("image") or _fallback_image_url(
                        set_data["id"], card["localId"]
                    )
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
                            image_url,
                            now,
                        ),
                    )
                    if with_images:
                        existing = connection.execute(
                            """
                            SELECT perceptual_hash, reference_features,
                                   visual_embedding
                            FROM cards WHERE id = ?
                            """,
                            (card["id"],),
                        ).fetchone()
                        reference = (
                            json.loads(existing["reference_features"])
                            if existing and existing["reference_features"]
                            else {}
                        )
                        if (
                            not existing
                            or not existing["perceptual_hash"]
                            or "centering_distances" not in reference
                            or "reference_thumbnail" not in reference
                            or not existing["visual_embedding"]
                        ):
                            cards_to_profile.append(
                                {
                                    **card,
                                    "image": image_url,
                                    "set_id": set_data["id"],
                                }
                            )
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
                            reference_features = ?,
                            visual_embedding = ?
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
    embedding_count = connection.execute(
        "SELECT COUNT(*) FROM cards WHERE visual_embedding IS NOT NULL"
    ).fetchone()[0]
    connection.close()
    return {
        "database": str(path),
        "series": resolved_series,
        "sets": set_count,
        "cards": card_count,
        "visual_profiles": profile_count,
        "ml_embeddings": embedding_count,
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


def _artwork_keypoint_similarity(
    image: np.ndarray, reference: Optional[Dict[str, object]]
) -> float:
    if not reference or not reference.get("reference_thumbnail"):
        return 0.0
    try:
        encoded = base64.b64decode(
            str(reference["reference_thumbnail"]), validate=True
        )
    except (binascii.Error, ValueError, TypeError):
        return 0.0
    thumbnail = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if thumbnail is None:
        return 0.0
    observed = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        (thumbnail.shape[1], thumbnail.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    detector = cv2.ORB_create(nfeatures=900, fastThreshold=12)
    _, observed_descriptors = detector.detectAndCompute(observed, None)
    _, reference_descriptors = detector.detectAndCompute(thumbnail, None)
    if observed_descriptors is None or reference_descriptors is None:
        return 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        observed_descriptors, reference_descriptors, k=2
    )
    good = sum(
        1
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    )
    expected = max(
        15.0,
        min(len(observed_descriptors), len(reference_descriptors)) * 0.20,
    )
    return float(np.clip(good / expected, 0.0, 1.0))


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
               perceptual_hash, color_signature, reference_features,
               visual_embedding
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
    has_embeddings = any(row["visual_embedding"] for row in rows)
    query_embedding = (
        visual_embedding(image, allow_download=True) if has_embeddings else None
    )
    reranked = []
    for score, hamming, color_distance, row in scored[: max(400, limit)]:
        reference = (
            json.loads(row["reference_features"])
            if row["reference_features"]
            else None
        )
        keypoint_similarity = _artwork_keypoint_similarity(image, reference)
        embedding_similarity = (
            cosine_similarity(
                query_embedding,
                json.loads(row["visual_embedding"]),
            )
            if query_embedding is not None and row["visual_embedding"]
            else 0.0
        )
        embedding_evidence = float(
            np.clip((embedding_similarity - 0.55) / 0.45, 0.0, 1.0)
        )
        reranked.append(
            (
                score - keypoint_similarity * 18.0 - embedding_evidence * 24.0,
                score,
                hamming,
                color_distance,
                keypoint_similarity,
                embedding_similarity,
                embedding_evidence,
                reference,
                row,
            )
        )
    reranked.sort(key=lambda item: item[0])

    results = []
    for (
        adjusted_score,
        score,
        hamming,
        color_distance,
        keypoint_similarity,
        embedding_similarity,
        embedding_evidence,
        reference,
        row,
    ) in reranked[:limit]:
        confidence = max(0.0, min(1.0, 1.0 - max(0.0, adjusted_score) / 38.0))
        results.append(
            {
                "id": row["id"],
                "name": row["name"],
                "series_id": row["series_id"],
                "set_id": row["set_id"],
                "set_name": row["set_name"],
                "number": row["local_id"],
                "image_url": _image_asset_url(
                    row["image_url"],
                    row["set_id"],
                    row["local_id"],
                    high_resolution=True,
                ),
                "confidence": round(confidence, 3),
                "match_method": (
                    "ml_visual" if embedding_evidence >= 0.50 else "visual"
                ),
                "hash_distance": hamming,
                "color_distance": round(color_distance, 3),
                "keypoint_similarity": round(keypoint_similarity, 3),
                "embedding_similarity": round(embedding_similarity, 3),
                "reference_features": reference,
            }
        )
    return results


def search_cards(
    query: str,
    path: Path = DEFAULT_CATALOG_PATH,
    limit: int = 20,
) -> List[Dict[str, object]]:
    """Search English catalog metadata by name, number, set, or card id."""
    tokens = [token.strip() for token in query.split() if token.strip()]
    if not path.exists() or not tokens:
        return []
    clauses = []
    parameters: List[object] = []
    for token in tokens:
        clauses.append(
            "(name LIKE ? OR local_id LIKE ? OR set_name LIKE ? OR id LIKE ?)"
        )
        value = f"%{token}%"
        parameters.extend((value, value, value, value))
    connection = _connect(path)
    rows = connection.execute(
        f"""
        SELECT id, series_id, set_id, set_name, local_id, name, image_url
        FROM cards
        WHERE {" AND ".join(clauses)}
        ORDER BY
            CASE WHEN lower(name) = lower(?) THEN 0 ELSE 1 END,
            name, set_name, local_id
        LIMIT ?
        """,
        (*parameters, query.strip(), max(1, min(limit, 50))),
    ).fetchall()
    connection.close()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "series_id": row["series_id"],
            "set_id": row["set_id"],
            "set_name": row["set_name"],
            "number": row["local_id"],
            "image_url": _image_asset_url(
                row["image_url"],
                row["set_id"],
                row["local_id"],
                high_resolution=True,
            ),
        }
        for row in rows
    ]


def get_card(
    card_id: str, path: Path = DEFAULT_CATALOG_PATH
) -> Optional[Dict[str, object]]:
    """Return one catalog card with its clean-reference profile."""
    if not path.exists():
        return None
    connection = _connect(path)
    row = connection.execute(
        """
        SELECT id, series_id, set_id, set_name, local_id, name, image_url,
               reference_features
        FROM cards WHERE id = ?
        """,
        (card_id,),
    ).fetchone()
    connection.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "series_id": row["series_id"],
        "set_id": row["set_id"],
        "set_name": row["set_name"],
        "number": row["local_id"],
        "image_url": _image_asset_url(
            row["image_url"],
            row["set_id"],
            row["local_id"],
            high_resolution=True,
        ),
        "confidence": 1.0,
        "match_method": "manual",
        "reference_features": (
            json.loads(row["reference_features"])
            if row["reference_features"]
            else None
        ),
    }


def apply_reference_baseline(
    analysis: CardAnalysis, match: Dict[str, object]
) -> None:
    """Calibrate print placement against the matched clean card layout."""
    if float(match.get("confidence", 0.0)) < 0.55:
        return
    reference = match.get("reference_features")
    if not reference:
        return
    _apply_surface_reference(analysis, reference)
    expected = reference.get("centering_distances")
    if not expected:
        return
    centering = analysis.diagnostics["centering"]
    observed = centering["distances"]

    def calibrated_axis(
        observed_first: float,
        observed_second: float,
        expected_first: float,
        expected_second: float,
    ) -> Tuple[float, float, float]:
        observed_total = max(2.0, observed_first + observed_second)
        expected_total = max(2.0, expected_first + expected_second)
        expected_scaled = observed_total * expected_first / expected_total
        displacement = observed_first - expected_scaled
        first = float(np.clip(observed_total / 2.0 + displacement, 1.0, observed_total - 1.0))
        second = observed_total - first
        balance = min(first, second) / max(first, second)
        return first, second, balance

    left, right, horizontal = calibrated_axis(
        observed["left"],
        observed["right"],
        expected["left"],
        expected["right"],
    )
    top, bottom, vertical = calibrated_axis(
        observed["top"],
        observed["bottom"],
        expected["top"],
        expected["bottom"],
    )
    centering["raw_percent"] = {
        "left": centering["left_percent"],
        "right": centering["right_percent"],
        "top": centering["top_percent"],
        "bottom": centering["bottom_percent"],
    }
    centering.update(
        {
            "left_percent": round(left / (left + right) * 100),
            "right_percent": round(right / (left + right) * 100),
            "top_percent": round(top / (top + bottom) * 100),
            "bottom_percent": round(bottom / (top + bottom) * 100),
            "balance_x": horizontal,
            "balance_y": vertical,
            "reference_calibrated": True,
            "reference_card_id": match.get("id"),
        }
    )
    analysis.features["centering_x"] = horizontal
    analysis.features["centering_y"] = vertical


def _apply_surface_reference(
    analysis: CardAnalysis, reference: Dict[str, object]
) -> None:
    """Locate thin observed edges absent from aligned clean artwork."""
    encoded_thumbnail = reference.get("reference_thumbnail")
    if not encoded_thumbnail:
        return
    try:
        reference_bytes = base64.b64decode(str(encoded_thumbnail), validate=True)
    except (binascii.Error, ValueError, TypeError):
        return
    reference_image = cv2.imdecode(
        np.frombuffer(reference_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if reference_image is None:
        return

    observed = cv2.resize(
        analysis.image,
        (reference_image.shape[1], reference_image.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    observed_gray = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=1200, fastThreshold=10)
    reference_points, reference_descriptors = detector.detectAndCompute(
        reference_gray, None
    )
    observed_points, observed_descriptors = detector.detectAndCompute(
        observed_gray, None
    )
    if reference_descriptors is None or observed_descriptors is None:
        return
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        reference_descriptors, observed_descriptors, k=2
    )
    reliable = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            reliable.append(pair[0])
    if len(reliable) < 12:
        return
    source = np.float32(
        [reference_points[item.queryIdx].pt for item in reliable]
    ).reshape(-1, 1, 2)
    destination = np.float32(
        [observed_points[item.trainIdx].pt for item in reliable]
    ).reshape(-1, 1, 2)
    transform, inliers = cv2.findHomography(
        source, destination, cv2.RANSAC, 3.0
    )
    if (
        transform is None
        or inliers is None
        or float(inliers.mean()) < 0.45
    ):
        return
    aligned_reference = cv2.warpPerspective(
        reference_image,
        transform,
        (observed.shape[1], observed.shape[0]),
    )
    aligned_gray = cv2.cvtColor(aligned_reference, cv2.COLOR_BGR2GRAY)
    observed_edges = cv2.Canny(
        cv2.GaussianBlur(observed_gray, (3, 3), 0), 45, 125
    )
    reference_edges = cv2.Canny(
        cv2.GaussianBlur(aligned_gray, (3, 3), 0), 45, 125
    )
    known_edges = cv2.dilate(reference_edges, np.ones((5, 5), np.uint8))
    extra_edges = cv2.bitwise_and(observed_edges, cv2.bitwise_not(known_edges))

    hsv = cv2.cvtColor(observed, cv2.COLOR_BGR2HSV)
    glare_candidates = (
        ((hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 235)).astype(np.uint8) * 255
    )
    glare_count, glare_labels, glare_stats, _ = cv2.connectedComponentsWithStats(
        glare_candidates
    )
    broad_glare = np.zeros_like(glare_candidates)
    for index in range(1, glare_count):
        _, _, glare_width, glare_height, glare_area = glare_stats[index]
        glare_length = max(glare_width, glare_height)
        if glare_area >= 30 and glare_area / max(1, glare_length) > 7:
            broad_glare[glare_labels == index] = 255
    broad_glare = cv2.dilate(broad_glare, np.ones((9, 9), np.uint8))
    extra_edges[broad_glare > 0] = 0
    margin_x = round(observed.shape[1] * 0.06)
    margin_y = round(observed.shape[0] * 0.06)
    interior = np.zeros_like(extra_edges)
    interior[
        margin_y : observed.shape[0] - margin_y,
        margin_x : observed.shape[1] - margin_x,
    ] = 255
    extra_edges = cv2.bitwise_and(extra_edges, interior)
    extra_edges = cv2.morphologyEx(
        extra_edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        extra_edges
    )
    anomalies = []
    for index in range(1, component_count):
        x, y, width, height, area = stats[index]
        short, long = sorted((width, height))
        if area < 7 or long < 8 or short <= 0:
            continue
        coordinates = np.column_stack(np.where(component_labels == index))
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = float(
            np.sqrt(max(eigenvalues) / max(0.5, min(eigenvalues)))
        )
        diagonal = float(np.hypot(width, height))
        if elongation < 2.2 or area / max(1.0, diagonal) > 6.0:
            continue
        if width > observed.shape[1] * 0.85 or height > observed.shape[0] * 0.85:
            continue
        anomalies.append((int(area), int(x), int(y), int(width), int(height)))

    scale_x = analysis.image.shape[1] / observed.shape[1]
    scale_y = analysis.image.shape[0] / observed.shape[0]
    def surface_severity(area: int) -> str:
        if area >= 80:
            return "high"
        if area >= 20:
            return "medium"
        return "small"

    for area, x, y, width, height in sorted(anomalies, reverse=True)[:20]:
        analysis.diagnostics["defects"].append(
            {
                "type": "Surface scratch/crease candidate",
                "location": "front surface",
                "severity": surface_severity(area),
                "evidence": f"{area} reference-unmatched edge pixels",
                "bbox": (
                    round(x * scale_x),
                    round(y * scale_y),
                    round((x + width) * scale_x),
                    round((y + height) * scale_y),
                ),
            }
        )
    surface_weight = sum(
        {"small": 0.10, "medium": 0.50, "high": 2.0}[
            surface_severity(item[0])
        ]
        for item in anomalies
    )
    damage = float(np.clip(surface_weight / 30.0, 0, 1))
    analysis.features["surface_damage"] = damage
    analysis.features["surface_assessed"] = 1.0
    surface_signals = analysis.diagnostics["condition_signals"]["surface"]
    surface_signals["reference_compared"] = True
    surface_signals["anomaly_count"] = len(anomalies)
    surface_signals["severity_weight"] = round(surface_weight, 2)


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
