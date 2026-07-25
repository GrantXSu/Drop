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
REQUIRED_COLUMNS = {"grade", "front", "source_url", "usage_rights"}


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


def _validate_row(row: Dict[str, str], row_number: int) -> float:
    try:
        grade = float(row["grade"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Row {row_number}: grade must be a number.") from error
    if not 1.0 <= grade <= 10.0:
        raise ValueError(f"Row {row_number}: grade must be between 1 and 10.")
    if not row.get("front", "").strip():
        raise ValueError(f"Row {row_number}: front is required.")
    if not row.get("source_url", "").startswith(("http://", "https://")):
        raise ValueError(f"Row {row_number}: source_url must document the public source.")
    if not row.get("usage_rights", "").strip():
        raise ValueError(f"Row {row_number}: usage_rights is required.")
    return grade


def _quality_issue(features: Dict[str, float], side: str) -> Optional[str]:
    """Reject photos whose capture quality could be learned as card damage."""
    if features["sharpness"] < 0.08:
        return f"{side} image is too blurry"
    if features["surface_glare"] > 0.08:
        return f"{side} image has excessive glare"
    if not 0.20 <= features["exposure"] <= 0.85:
        return f"{side} image exposure is unsuitable"
    return None


def train(
    manifest_path: Path,
    output_path: Path = DEFAULT_MODEL_PATH,
    random_seed: int = 42,
) -> Dict[str, object]:
    """Extract visual features, evaluate a holdout, and save a model artifact."""
    manifest_path = manifest_path.resolve()
    vectors: List[np.ndarray] = []
    grades: List[float] = []
    groups: List[str] = []
    rejected: List[Dict[str, object]] = []

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            try:
                grade = _validate_row(row, row_number)
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
                vectors.append(feature_vector(front, back))
                grades.append(grade)
                groups.append(row["source_url"].strip())
            except (CardImageError, FileNotFoundError, httpx.HTTPError, ValueError) as error:
                rejected.append({"row": row_number, "reason": str(error)})

    if len(vectors) < 100:
        raise ValueError(
            f"Only {len(vectors)} valid samples were found; at least 100 are required."
        )

    features = np.vstack(vectors)
    targets = np.asarray(grades)
    if len(set(groups)) < 80:
        raise ValueError("At least 80 distinct source cards are required for a holdout.")
    if len(set(round(grade) for grade in grades)) < 4:
        raise ValueError("Samples must cover at least 4 distinct whole-grade bands.")
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    train_indices, test_indices = next(split.split(features, targets, groups))
    x_train, x_test = features[train_indices], features[test_indices]
    y_train, y_test = targets[train_indices], targets[test_indices]
    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=random_seed,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    validation_mae = float(mean_absolute_error(y_test, predictions))
    within_one = float(np.mean(np.abs(predictions - y_test) <= 1.0))
    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "sample_count": len(vectors),
        "validation_mae": validation_mae,
        "validation_within_one": within_one,
        "grade_distribution": {
            str(grade): int(np.sum(np.rint(targets) == grade))
            for grade in sorted(set(np.rint(targets).astype(int)))
        },
        "training_policy": "verified labels; capture-quality gated; grouped holdout",
        "manifest": str(manifest_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return {
        "model_path": str(output_path),
        "samples": len(vectors),
        "rejected": rejected,
        "validation_mae": round(validation_mae, 3),
        "validation_within_one": round(within_one, 3),
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
