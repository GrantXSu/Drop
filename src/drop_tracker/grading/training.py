"""Train a grading model from a provenance-aware CSV manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from .features import CardImageError, analyze_image
from .model import DEFAULT_MODEL_PATH, FEATURE_NAMES, feature_vector


MAX_IMAGE_BYTES = 15 * 1024 * 1024
REQUIRED_COLUMNS = {
    "grading_company",
    "overall_grade",
    "front",
    "source_url",
    "usage_rights",
}
TARGET_COLUMNS = {
    "PSA": {"psa_overall": "overall_grade"},
    "BGS": {
        "bgs_overall": "overall_grade",
        "bgs_corners": "bgs_corners",
        "bgs_edges": "bgs_edges",
    },
}


def _read_image(reference: str, manifest_directory: Path) -> bytes:
    parsed = urlparse(reference)
    if parsed.scheme in {"http", "https"}:
        with httpx.stream(
            "GET",
            reference,
            follow_redirects=True,
            timeout=20.0,
            headers={"User-Agent": "DropCardGrader/0.1 (dataset import)"},
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if not content_type.lower().startswith("image/"):
                raise ValueError(f"Expected an image from {reference}, got {content_type!r}")
            chunks = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_IMAGE_BYTES:
                    raise ValueError(f"Image exceeds 15 MB: {reference}")
                chunks.append(chunk)
            return b"".join(chunks)

    path = (manifest_directory / reference).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds 15 MB: {path}")
    return path.read_bytes()


def _validate_row(row: Dict[str, str], row_number: int) -> Dict[str, float]:
    company = row.get("grading_company", "").strip().upper()
    if company not in TARGET_COLUMNS:
        raise ValueError(f"Row {row_number}: grading_company must be PSA or BGS.")
    targets: Dict[str, float] = {}
    for target, column in TARGET_COLUMNS[company].items():
        raw_value = row.get(column, "").strip()
        if not raw_value:
            if target.endswith("overall"):
                raise ValueError(f"Row {row_number}: overall_grade is required.")
            continue
        try:
            value = float(raw_value)
        except ValueError as error:
            raise ValueError(f"Row {row_number}: {column} must be a number.") from error
        if not 1.0 <= value <= 10.0:
            raise ValueError(f"Row {row_number}: {column} must be between 1 and 10.")
        if company == "BGS" and abs(value * 2 - round(value * 2)) > 1e-6:
            raise ValueError(
                f"Row {row_number}: BGS labels must use 0.5-point increments."
            )
        targets[target] = value
    if not row.get("front", "").strip():
        raise ValueError(f"Row {row_number}: front is required.")
    if not row.get("source_url", "").startswith(("http://", "https://")):
        raise ValueError(f"Row {row_number}: source_url must document the public source.")
    if not row.get("usage_rights", "").strip():
        raise ValueError(f"Row {row_number}: usage_rights is required.")
    if not row.get("certification_number", "").strip():
        raise ValueError(f"Row {row_number}: certification_number is required.")
    return targets


def _quality_issue(features: Dict[str, float], side: str) -> Optional[str]:
    """Reject photos whose capture quality could be learned as card damage."""
    if features["sharpness"] < 0.08:
        return f"{side} image is too blurry"
    if features["surface_glare"] > 0.08:
        return f"{side} image has excessive glare"
    if not 0.20 <= features["exposure"] <= 0.85:
        return f"{side} image exposure is unsuitable"
    return None


def _fit_target(
    records: List[Dict[str, object]], target: str, random_seed: int
) -> Optional[Dict[str, object]]:
    labeled = [record for record in records if target in record["targets"]]
    if len(labeled) < 100:
        return None
    groups = [str(record["group"]) for record in labeled]
    targets = np.asarray(
        [float(record["targets"][target]) for record in labeled],
        dtype=np.float64,
    )
    if len(set(groups)) < 80:
        return None
    if len(set(np.rint(targets).astype(int))) < 4:
        return None
    features = np.vstack([record["vector"] for record in labeled])
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    train_indices, test_indices = next(split.split(features, targets, groups))
    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=random_seed,
    )
    model.fit(features[train_indices], targets[train_indices])
    predictions = model.predict(features[test_indices])
    validation_mae = float(mean_absolute_error(targets[test_indices], predictions))
    return {
        "model": model,
        "samples": len(labeled),
        "validation_mae": validation_mae,
        "validation_within_one": float(
            np.mean(np.abs(predictions - targets[test_indices]) <= 1.0)
        ),
        "grade_distribution": {
            str(grade): int(np.sum(np.rint(targets) == grade))
            for grade in sorted(set(np.rint(targets).astype(int)))
        },
    }


def train(
    manifest_path: Path,
    output_path: Path = DEFAULT_MODEL_PATH,
    random_seed: int = 42,
) -> Dict[str, object]:
    """Extract visual features, evaluate a holdout, and save a model artifact."""
    manifest_path = manifest_path.resolve()
    records: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            try:
                targets = _validate_row(row, row_number)
                front = analyze_image(
                    _read_image(row["front"].strip(), manifest_path.parent),
                    side="front",
                ).features
                issue = _quality_issue(front, "front")
                if issue:
                    raise ValueError(issue)
                back_reference = row.get("back", "").strip()
                back = (
                    analyze_image(
                        _read_image(back_reference, manifest_path.parent), side="back"
                    ).features
                    if back_reference
                    else None
                )
                if back:
                    issue = _quality_issue(back, "back")
                    if issue:
                        raise ValueError(issue)
                records.append(
                    {
                        "vector": feature_vector(front, back),
                        "targets": targets,
                        "group": row["source_url"].strip(),
                    }
                )
            except (CardImageError, FileNotFoundError, httpx.HTTPError, ValueError) as error:
                rejected.append({"row": row_number, "reason": str(error)})

    if len(records) < 100:
        raise ValueError(
            f"Only {len(records)} valid samples were found; at least 100 are required."
        )
    trained = {
        target: result
        for target in (
            "psa_overall",
            "bgs_overall",
            "bgs_corners",
            "bgs_edges",
        )
        if (result := _fit_target(records, target, random_seed)) is not None
    }
    if not trained:
        raise ValueError(
            "No target has 100 labels from 80 distinct cards across four grade bands."
        )
    artifact = {
        "models": {target: result["model"] for target, result in trained.items()},
        "feature_names": FEATURE_NAMES,
        "sample_count": len(records),
        "targets": {
            target: {
                key: value
                for key, value in result.items()
                if key != "model"
            }
            for target, result in trained.items()
        },
        "training_policy": (
            "PSA overall and BGS category labels; verified certification; "
            "capture-quality gated; grouped holdout"
        ),
        "manifest": str(manifest_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return {
        "model_path": str(output_path),
        "samples": len(records),
        "rejected": rejected,
        "targets": artifact["targets"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Pokémon card grade estimator.")
    parser.add_argument("manifest", type=Path, help="CSV sample manifest")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    try:
        report = train(args.manifest, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
