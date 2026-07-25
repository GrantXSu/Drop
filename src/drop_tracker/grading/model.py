"""Grade prediction and model artifact handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np


BASE_FEATURES = (
    "centering_x",
    "centering_y",
    "edge_pale",
    "corner_pale_mean",
    "corner_pale_max",
    "edge_defect_load",
    "corner_defect_load",
    "surface_glare",
    "surface_dark",
    "surface_damage",
    "surface_assessed",
    "sharpness",
    "exposure",
    "contrast",
)
MODEL_FEATURES = (
    "centering_x",
    "centering_y",
    "edge_pale",
    "corner_pale_mean",
    "corner_pale_max",
    "edge_defect_load",
    "corner_defect_load",
)
SIDES = ("front", "back")
FEATURE_NAMES = tuple(
    f"{side}_{feature}" for side in SIDES for feature in MODEL_FEATURES
) + ("has_back",)
DEFAULT_MODEL_PATH = Path("models/card_grader.joblib")


@dataclass(frozen=True)
class GradePrediction:
    grade: float
    low: float
    high: float
    confidence: str
    method: str
    model_samples: int
    caveat: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def feature_vector(
    front: Dict[str, float], back: Optional[Dict[str, float]]
) -> np.ndarray:
    values = [front[name] for name in MODEL_FEATURES]
    if back is None:
        values.extend([0.0] * len(MODEL_FEATURES))
        values.append(0.0)
    else:
        values.extend(back[name] for name in MODEL_FEATURES)
        values.append(1.0)
    return np.asarray(values, dtype=np.float64)


def _heuristic_grade(
    front: Dict[str, float], back: Optional[Dict[str, float]]
) -> GradePrediction:
    subgrades = category_subgrades(front, back)
    by_name = {
        item["key"]: float(item["score"])
        for item in subgrades
        if item["score"] is not None
    }
    weights = {"centering": 0.10, "corners": 0.30, "edges": 0.30, "surface": 0.30}
    available_weight = sum(weights[name] for name in by_name)
    grade = sum(by_name[name] * weights[name] for name in by_name) / available_weight
    # Severe physical damage caps the estimate; centering influences the
    # weighted score but does not impose the same damage ceiling.
    physical_scores = [
        by_name[name] for name in ("corners", "edges", "surface") if name in by_name
    ]
    grade = min(grade, min(physical_scores) + 1.5)
    grade = float(np.clip(grade, 1.0, 10.0))
    uncertainty = 1.75 if back is None else 1.25
    return GradePrediction(
        grade=round(grade, 1),
        low=round(max(1.0, grade - uncertainty), 1),
        high=round(min(10.0, grade + uncertainty), 1),
        confidence="low",
        method="untrained visual heuristic",
        model_samples=0,
        caveat=(
            "Untrained visual estimate only. Train the model on verified, licensed PSA "
            "examples before using it to make buying or submission decisions."
        ),
    )


def _load_artifact(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    artifact = joblib.load(path)
    if artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("The saved grading model uses an incompatible feature set.")
    return artifact


def predict_grade(
    front: Dict[str, float],
    back: Optional[Dict[str, float]] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> GradePrediction:
    artifact = _load_artifact(model_path)
    if artifact is None:
        return _heuristic_grade(front, back)

    models = artifact.get("models", {})
    model = models.get("psa_overall") if isinstance(models, dict) else None
    target_metrics = (
        artifact.get("targets", {}).get("psa_overall", {})
        if isinstance(artifact.get("targets", {}), dict)
        else {}
    )
    if model is None:
        model = artifact.get("model")
        target_metrics = artifact
    if model is None:
        return _heuristic_grade(front, back)
    vector = feature_vector(front, back).reshape(1, -1)
    grade = float(np.clip(model.predict(vector)[0], 1.0, 10.0))
    validation_mae = float(target_metrics.get("validation_mae", 1.5))
    sample_count = int(target_metrics.get("samples", artifact.get("sample_count", 0)))
    uncertainty = max(0.6, min(2.5, validation_mae * 1.65))
    confidence = "high" if sample_count >= 1000 and validation_mae <= 0.6 else "medium"
    if sample_count < 250 or validation_mae > 1.0:
        confidence = "low"

    return GradePrediction(
        grade=round(grade, 1),
        low=round(max(1.0, grade - uncertainty), 1),
        high=round(min(10.0, grade + uncertainty), 1),
        confidence=confidence,
        method="trained verified-sample model",
        model_samples=sample_count,
        caveat=(
            "Unofficial estimate, not a PSA grade. Confidence reflects held-out "
            "sample error and cannot account for defects hidden by the photos."
        ),
    )


def defect_summary(
    front: Dict[str, float],
    back: Optional[Dict[str, float]],
    model_path: Optional[Path] = None,
) -> Tuple[Dict[str, object], ...]:
    return category_subgrades(front, back, model_path)


def _condition(score: float) -> str:
    if score >= 9.5:
        return "Pristine"
    if score >= 9.0:
        return "Gem Mint"
    if score >= 8.0:
        return "Near Mint–Mint"
    if score >= 7.0:
        return "Near Mint"
    if score >= 6.0:
        return "Excellent–Mint"
    if score >= 4.0:
        return "Very Good"
    return "Poor–Good"


def grade_label(score: float) -> str:
    """Return a plain-language condition band for a decimal estimate."""
    return _condition(score)


def _larger_border_share(balance: float) -> float:
    """Convert smaller/larger border balance to the conventional 50–100 share."""
    return float(100.0 / (1.0 + np.clip(balance, 0.0, 1.0)))


def _curve(value: float, points: Tuple[Tuple[float, float], ...]) -> float:
    x_values, y_values = zip(*points)
    return float(np.interp(value, x_values, y_values))


def centering_standards(
    front: Dict[str, float], back: Optional[Dict[str, float]]
) -> Dict[str, object]:
    """Calculate PSA-style and Beckett-style centering subgrades."""
    front_axes = (
        _larger_border_share(front["centering_x"]),
        _larger_border_share(front["centering_y"]),
    )
    front_worst = max(front_axes)
    psa_front = _curve(
        front_worst,
        (
            (50.0, 10.0),
            (55.0, 10.0),
            (60.0, 9.0),
            (65.0, 8.0),
            (70.0, 7.0),
            (75.0, 6.0),
            (80.0, 5.0),
            (85.0, 4.0),
            (90.0, 3.0),
            (100.0, 1.0),
        ),
    )
    bgs_front = _curve(
        front_worst,
        (
            (50.0, 10.0),
            (55.0, 9.0),
            (60.0, 8.0),
            (65.0, 7.0),
            (70.0, 6.0),
            (75.0, 5.0),
            (80.0, 4.0),
            (85.0, 3.0),
            (90.0, 2.0),
            (100.0, 1.0),
        ),
    )
    if min(front_axes) <= 50.5 and 50.5 < front_worst <= 55.0:
        bgs_front = max(bgs_front, 9.5)

    back_axes = None
    psa_back = bgs_back = 9.0
    if back is not None:
        back_axes = (
            _larger_border_share(back["centering_x"]),
            _larger_border_share(back["centering_y"]),
        )
        back_worst = max(back_axes)
        psa_back = _curve(
            back_worst,
            ((50.0, 10.0), (75.0, 10.0), (90.0, 9.0), (95.0, 7.0), (100.0, 1.0)),
        )
        bgs_back = _curve(
            back_worst,
            (
                (50.0, 10.0),
                (60.0, 10.0),
                (70.0, 9.0),
                (80.0, 8.0),
                (90.0, 7.0),
                (95.0, 6.0),
                (100.0, 1.0),
            ),
        )

    front_deviation = max(front_axes) - 50.0
    back_deviation = max(back_axes) - 50.0 if back_axes else 25.0
    precision_score = max(
        1.0, 10.0 - 0.03 * front_deviation - 0.015 * back_deviation
    )
    psa_score = min(psa_front, psa_back, precision_score)
    psa_10_eligible = bool(
        back_axes
        and max(front_axes) < 55.0
        and max(back_axes) < 75.0
        and psa_score >= 9.95
    )

    return {
        "psa": round(psa_score, 1),
        "bgs": round(min(bgs_front, bgs_back), 1),
        "psa_10_eligible": psa_10_eligible,
        "front_axes": tuple(round(value, 1) for value in front_axes),
        "back_axes": (
            tuple(round(value, 1) for value in back_axes) if back_axes else None
        ),
    }


def _trained_bgs_subgrades(
    front: Dict[str, float],
    back: Optional[Dict[str, float]],
    model_path: Optional[Path],
) -> Dict[str, Dict[str, object]]:
    if model_path is None:
        return {}
    artifact = _load_artifact(model_path)
    if artifact is None or not isinstance(artifact.get("models"), dict):
        return {}
    vector = feature_vector(front, back).reshape(1, -1)
    predictions = {}
    for category, target in (("corners", "bgs_corners"), ("edges", "bgs_edges")):
        model = artifact["models"].get(target)
        metrics = artifact.get("targets", {}).get(target, {})
        if (
            model is None
            or int(metrics.get("samples", 0)) < 100
            or float(metrics.get("validation_mae", 99.0)) > 1.0
        ):
            continue
        predictions[category] = {
            "score": float(np.clip(model.predict(vector)[0], 1.0, 10.0)),
            "samples": int(metrics["samples"]),
            "validation_mae": float(metrics["validation_mae"]),
        }
    return predictions


def category_subgrades(
    front: Dict[str, float],
    back: Optional[Dict[str, float]],
    model_path: Optional[Path] = None,
) -> Tuple[Dict[str, object], ...]:
    """Produce DGC-style decimal subgrades from measurable visual signals."""
    sides = (front,) if back is None else (front, back)
    standards = centering_standards(front, back)
    edge = max(side["edge_pale"] for side in sides)
    corner = max(side["corner_pale_max"] for side in sides)
    edge_load = max(side["edge_defect_load"] for side in sides)
    corner_load = max(side["corner_defect_load"] for side in sides)
    surface_assessed = any(side["surface_assessed"] >= 0.5 for side in sides)
    surface_damage = max(
        (
            side["surface_damage"]
            for side in sides
            if side["surface_assessed"] >= 0.5
        ),
        default=0.0,
    )
    scores = {
        "centering": float(standards["psa"]),
        "corners": float(
            np.clip(10.0 - 12.0 * corner - 5.0 * corner_load, 1.0, 10.0)
        ),
        "edges": float(
            np.clip(10.0 - 15.0 * edge - 6.0 * edge_load, 1.0, 10.0)
        ),
        "surface": (
            float(np.clip(10.0 - 30.0 * surface_damage, 1.0, 10.0))
            if surface_assessed
            else None
        ),
    }
    trained = _trained_bgs_subgrades(front, back, model_path)
    for category in ("corners", "edges"):
        if category in trained:
            scores[category] = trained[category]["score"]
    scores = {
        name: round(value, 1) if value is not None else None
        for name, value in scores.items()
    }
    return (
        {
            "key": "centering",
            "name": "Centering",
            "score": scores["centering"],
            "condition": f"PSA {standards['psa']:.1f} · BGS {standards['bgs']:.1f}",
            "detail": (
                f"Front larger-border shares: {standards['front_axes']}. "
                f"Back: {standards['back_axes'] or 'not supplied'}."
            ),
            "standards": standards,
        },
        {
            "key": "corners",
            "name": "Corners",
            "score": scores["corners"],
            "condition": _condition(scores["corners"]),
            "detail": (
                (
                    f"BGS subgrade model: {trained['corners']['samples']} samples, "
                    f"holdout MAE {trained['corners']['validation_mae']:.2f}."
                )
                if "corners" in trained
                else "Uses localized corner damage area and repeated finding count "
                "shown in the visual report."
            ),
            "method": "trained BGS subgrade" if "corners" in trained else "visual heuristic",
        },
        {
            "key": "edges",
            "name": "Edges",
            "score": scores["edges"],
            "condition": _condition(scores["edges"]),
            "detail": (
                (
                    f"BGS subgrade model: {trained['edges']['samples']} samples, "
                    f"holdout MAE {trained['edges']['validation_mae']:.2f}."
                )
                if "edges" in trained
                else "Uses localized edge damage area and repeated finding count "
                "shown in the visual report."
            ),
            "method": "trained BGS subgrade" if "edges" in trained else "visual heuristic",
        },
        {
            "key": "surface",
            "name": "Surface",
            "score": scores["surface"],
            "condition": (
                _condition(scores["surface"])
                if scores["surface"] is not None
                else "Not assessed"
            ),
            "detail": (
                "Compared with aligned clean reference artwork for localized "
                "scratches and creases."
                if surface_assessed
                else "Requires a confident catalog match and usable clean reference."
            ),
        },
    )
