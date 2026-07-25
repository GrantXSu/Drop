"""FastAPI web scanner for unofficial Pokémon card grade estimates."""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .catalog import (
    DEFAULT_CATALOG_PATH,
    apply_reference_baseline,
    identify_card,
)
from .features import (
    CardAnalysis,
    CardImageError,
    analyze_image,
    annotated_image,
    source_boundary_image,
)
from .model import DEFAULT_MODEL_PATH, defect_summary, grade_label, predict_grade


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
STATIC_DIRECTORY = Path(__file__).with_name("static")
app = FastAPI(title="Pokémon Card Grade Scanner", version="0.1.0")


def _visual_report(side: str, analysis: CardAnalysis) -> dict:
    def encode(image) -> str:
        success, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not success:
            raise ValueError("Could not render the annotated card image.")
        return "data:image/jpeg;base64," + base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")

    centering = analysis.diagnostics["centering"]
    return {
        "side": side,
        "source_image": encode(source_boundary_image(analysis)),
        "image": encode(annotated_image(analysis)),
        "centering": {
            "horizontal": f"{centering['left_percent']}/{centering['right_percent']}",
            "vertical": f"{centering['top_percent']}/{centering['bottom_percent']}",
            "left": centering["left_percent"],
            "right": centering["right_percent"],
            "top": centering["top_percent"],
            "bottom": centering["bottom_percent"],
            "distances": centering["distances"],
            "distance_mm": centering["distance_mm"],
            "card_dimensions": centering["card_dimensions"],
            "offset": centering["offset"],
        },
        "findings": analysis.diagnostics["defects"],
        "condition_signals": analysis.diagnostics["condition_signals"],
    }


async def _read_upload(upload: UploadFile) -> bytes:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 15 MB or smaller.")
    return data


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIRECTORY / "index.html")


@app.get("/api/status")
def status() -> dict:
    model_path = Path(os.getenv("CARD_GRADER_MODEL", str(DEFAULT_MODEL_PATH)))
    return {
        "ready": True,
        "trained_model": model_path.exists(),
        "model_path": str(model_path),
    }


@app.post("/api/grade")
async def grade_card(
    front: UploadFile = File(...),
    back: Optional[UploadFile] = File(default=None),
) -> dict:
    try:
        front_analysis = analyze_image(await _read_upload(front), side="front")
        back_analysis = (
            analyze_image(await _read_upload(back), side="back") if back else None
        )
        catalog_path = Path(
            os.getenv("CARD_CATALOG", str(DEFAULT_CATALOG_PATH))
        )
        matches = identify_card(front_analysis.image, catalog_path)
        identified_card = (
            matches[0] if matches and float(matches[0]["confidence"]) >= 0.55 else None
        )
        if identified_card:
            apply_reference_baseline(front_analysis, identified_card)
        model_path = Path(os.getenv("CARD_GRADER_MODEL", str(DEFAULT_MODEL_PATH)))
        prediction = predict_grade(
            front_analysis.features,
            back_analysis.features if back_analysis else None,
            model_path,
        )
    except CardImageError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail=f"Model error: {error}") from error

    warnings = [f"Front: {warning}" for warning in front_analysis.warnings]
    if back_analysis:
        warnings.extend(f"Back: {warning}" for warning in back_analysis.warnings)
    else:
        warnings.append("Add a back photo for a more complete estimate.")

    prediction_payload = prediction.to_dict()
    if warnings:
        prediction_payload["confidence"] = "low"
        prediction_payload["caveat"] = (
            f"{prediction_payload['caveat']} Retake the flagged photos before "
            "relying on this estimate."
        )
    prediction_payload["label"] = grade_label(prediction.grade)

    visual_reports = [_visual_report("Front", front_analysis)]
    if back_analysis:
        visual_reports.append(_visual_report("Back", back_analysis))

    return {
        "prediction": prediction_payload,
        "categories": defect_summary(
            front_analysis.features,
            back_analysis.features if back_analysis else None,
        ),
        "warnings": warnings,
        "visual_reports": visual_reports,
        "identification": {
            "match": identified_card,
            "candidates": matches,
            "catalog_ready": catalog_path.exists(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Pokémon card grading scanner.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "drop_tracker.grading.web:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
