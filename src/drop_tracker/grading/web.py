"""FastAPI web scanner for unofficial Pokémon card grade estimates."""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import threading
import uuid
import webbrowser
from pathlib import Path
from typing import Literal, Optional

import cv2
import stripe
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .billing import (
    COOKIE_NAME,
    clear_grade_history,
    consume_scan,
    customer_for_device,
    delete_back_reference,
    device_token,
    get_back_reference,
    grade_history,
    record_grade,
    resolve_device,
    save_back_reference,
    scan_already_counted,
    set_subscription,
    usage_status,
)
from .catalog import (
    DEFAULT_CATALOG_PATH,
    apply_back_reference,
    apply_reference_baseline,
    get_card,
    identify_card,
    search_cards,
)
from .features import (
    CardAnalysis,
    CardImageError,
    analyze_image,
    annotated_image,
    apply_manual_centering,
    source_boundary_image,
)
from .model import DEFAULT_MODEL_PATH, defect_summary, grade_label, predict_grade


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
STATIC_DIRECTORY = Path(__file__).with_name("static")
app = FastAPI(title="Pokémon Card Grade Scanner", version="0.1.0")


class CheckoutRequest(BaseModel):
    plan: Literal["monthly", "annual"]


class DeveloperUnlockRequest(BaseModel):
    password: str


class CardSearchRequest(BaseModel):
    query: str


@app.middleware("http")
async def ensure_device_cookie(request: Request, call_next):
    device_id, token = resolve_device(request.cookies.get(COOKIE_NAME))
    request.state.device_id = device_id
    response = await call_next(request)
    if request.cookies.get(COOKIE_NAME) != token:
        response.set_cookie(
            COOKIE_NAME,
            device_token(device_id),
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="lax",
        )
    return response


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
        "raw_source_image": encode(analysis.source_image),
        "source_boundary": analysis.source_boundary.tolist(),
        "source_dimensions": {
            "width": analysis.source_image.shape[1],
            "height": analysis.source_image.shape[0],
        },
        "image": encode(annotated_image(analysis)),
        "card_image": encode(analysis.image),
        "centering": {
            "horizontal": f"{centering['left_percent']}/{centering['right_percent']}",
            "vertical": f"{centering['top_percent']}/{centering['bottom_percent']}",
            "left": centering["left_percent"],
            "right": centering["right_percent"],
            "top": centering["top_percent"],
            "bottom": centering["bottom_percent"],
            "distances": centering["distances"],
            "guides": centering["guides"],
            "distance_mm": centering["distance_mm"],
            "card_dimensions": centering["card_dimensions"],
            "offset": centering["offset"],
            "reference_calibrated": centering.get("reference_calibrated", False),
            "raw_percent": centering.get("raw_percent"),
            "measurement_limit_mm": centering.get("measurement_limit_mm"),
            "retest_recommended": centering.get("retest_recommended", False),
            "manual_override": centering.get("manual_override", False),
        },
        "findings": analysis.diagnostics["defects"],
        "condition_signals": analysis.diagnostics["condition_signals"],
    }


def _confident_catalog_match(matches: list) -> Optional[dict]:
    if not matches:
        return None
    first = matches[0]
    confidence = float(first.get("confidence", 0.0))
    if confidence < 0.75:
        return None
    if len(matches) > 1:
        margin = confidence - float(matches[1].get("confidence", 0.0))
        if margin < 0.08 and confidence < 0.92:
            return None
    return first


def _parse_manual_boundary(value: Optional[str]):
    if value is None:
        return None
    try:
        boundary = json.loads(value)
    except json.JSONDecodeError as error:
        raise CardImageError("Manual card boundary is invalid JSON.") from error
    if (
        not isinstance(boundary, list)
        or len(boundary) != 4
        or any(not isinstance(point, list) or len(point) != 2 for point in boundary)
    ):
        raise CardImageError("Manual card boundary must contain four [x, y] points.")
    return boundary


async def _read_upload(upload: UploadFile) -> bytes:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 15 MB or smaller.")
    return data


@app.get("/settings", include_in_schema=False)
@app.get("/cards", include_in_schema=False)
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIRECTORY / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/api/status")
def status(request: Request) -> dict:
    model_path = Path(os.getenv("CARD_GRADER_MODEL", str(DEFAULT_MODEL_PATH)))
    return {
        "ready": True,
        "trained_model": model_path.exists(),
        "model_path": str(model_path),
        "billing": usage_status(request.state.device_id),
    }


@app.post("/api/billing/checkout")
def create_checkout(payload: CheckoutRequest, request: Request) -> dict:
    secret_key = os.getenv("STRIPE_SECRET_KEY")
    price_id = os.getenv(
        "STRIPE_PRO_MONTHLY_PRICE_ID"
        if payload.plan == "monthly"
        else "STRIPE_PRO_ANNUAL_PRICE_ID"
    )
    if not secret_key or not price_id or not os.getenv("CARDLENS_COOKIE_SECRET"):
        raise HTTPException(
            status_code=503,
            detail="Payments are not configured. Add the Stripe price and secret keys.",
        )
    stripe.api_key = secret_key
    base_url = str(request.base_url).rstrip("/")
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        client_reference_id=request.state.device_id,
        metadata={"device_id": request.state.device_id, "plan": payload.plan},
        success_url=f"{base_url}/?checkout=success",
        cancel_url=f"{base_url}/?checkout=cancelled",
        allow_promotion_codes=True,
    )
    return {"url": session.url}


@app.post("/api/billing/portal")
def create_billing_portal(request: Request) -> dict:
    secret_key = os.getenv("STRIPE_SECRET_KEY")
    customer_id = customer_for_device(request.state.device_id)
    if not secret_key or not customer_id:
        raise HTTPException(status_code=404, detail="No Pro subscription was found.")
    stripe.api_key = secret_key
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=str(request.base_url),
    )
    return {"url": session.url}


@app.post("/api/billing/webhook")
async def stripe_webhook(request: Request) -> dict:
    secret_key = os.getenv("STRIPE_SECRET_KEY")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    if not secret_key or not webhook_secret:
        raise HTTPException(status_code=503, detail="Stripe webhook is not configured.")
    stripe.api_key = secret_key
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, signature, webhook_secret)
    except (ValueError, stripe.error.SignatureVerificationError) as error:
        raise HTTPException(status_code=400, detail="Invalid Stripe webhook.") from error
    event_type = event["type"]
    item = event["data"]["object"]
    if event_type == "checkout.session.completed":
        metadata = item.get("metadata") or {}
        checkout_status = (
            "active"
            if item.get("payment_status") in {"paid", "no_payment_required"}
            else "incomplete"
        )
        set_subscription(
            device_id=metadata.get("device_id") or item.get("client_reference_id"),
            customer_id=item.get("customer"),
            subscription_id=item.get("subscription"),
            status=checkout_status,
        )
    elif event_type in {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    }:
        set_subscription(
            customer_id=item.get("customer"),
            subscription_id=item.get("id"),
            status=item.get("status") or "inactive",
        )
    return {"received": True}


@app.post("/api/developer/unlock")
def developer_unlock(payload: DeveloperUnlockRequest, request: Request) -> dict:
    expected = os.getenv("CARDLENS_DEVELOPER_PASSWORD")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Developer access is not configured on this server.",
        )
    if not hmac.compare_digest(payload.password.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Incorrect developer password.")
    set_subscription(device_id=request.state.device_id, status="developer")
    return {"billing": usage_status(request.state.device_id)}


@app.post("/api/developer/lock")
def developer_lock(request: Request) -> dict:
    set_subscription(device_id=request.state.device_id, status="inactive")
    return {"billing": usage_status(request.state.device_id)}


@app.get("/api/cards")
def card_search(q: str) -> dict:
    catalog_path = Path(os.getenv("CARD_CATALOG", str(DEFAULT_CATALOG_PATH)))
    return {"results": search_cards(q, catalog_path), "catalog_ready": catalog_path.exists()}


@app.post("/api/cards/search")
def secure_card_search(payload: CardSearchRequest, request: Request) -> dict:
    query = payload.query.strip()
    expected = os.getenv("CARDLENS_DEVELOPER_PASSWORD")
    if expected and hmac.compare_digest(query.encode(), expected.encode()):
        set_subscription(device_id=request.state.device_id, status="developer")
        return {
            "developer_unlocked": True,
            "billing": usage_status(request.state.device_id),
            "results": [],
        }
    catalog_path = Path(os.getenv("CARD_CATALOG", str(DEFAULT_CATALOG_PATH)))
    return {
        "developer_unlocked": False,
        "results": search_cards(query, catalog_path),
        "catalog_ready": catalog_path.exists(),
    }


@app.get("/api/history")
def history(request: Request) -> dict:
    return {"cards": grade_history(request.state.device_id)}


@app.delete("/api/history")
def delete_history(request: Request) -> dict:
    clear_grade_history(request.state.device_id)
    return {"cleared": True}


@app.post("/api/reference/back")
async def upload_back_reference(
    request: Request, image: UploadFile = File(...)
) -> dict:
    analysis = analyze_image(await _read_upload(image), side="back")
    if min(analysis.source_image.shape[:2]) < 1000:
        raise HTTPException(
            status_code=422,
            detail="Clean back reference must be at least 1000 pixels on the short edge.",
        )
    if analysis.features["sharpness"] < 0.12 or analysis.features["surface_glare"] > 0.08:
        raise HTTPException(
            status_code=422,
            detail="Clean back reference must be sharp and free of strong glare.",
        )
    success, encoded = cv2.imencode(
        ".jpg", analysis.image, [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    if not success:
        raise HTTPException(status_code=500, detail="Could not save clean back reference.")
    save_back_reference(request.state.device_id, encoded.tobytes())
    return {"saved": True, "billing": usage_status(request.state.device_id)}


@app.delete("/api/reference/back")
def clear_back_reference(request: Request) -> dict:
    delete_back_reference(request.state.device_id)
    return {"cleared": True, "billing": usage_status(request.state.device_id)}


@app.post("/api/grade")
async def grade_card(
    request: Request,
    front: UploadFile = File(...),
    back: Optional[UploadFile] = File(default=None),
    scan_id: Optional[str] = Form(default=None),
    card_id: Optional[str] = Form(default=None),
    front_boundary: Optional[str] = Form(default=None),
    back_boundary: Optional[str] = Form(default=None),
    front_left_mm: Optional[float] = Form(default=None),
    front_right_mm: Optional[float] = Form(default=None),
    front_top_mm: Optional[float] = Form(default=None),
    front_bottom_mm: Optional[float] = Form(default=None),
    back_left_mm: Optional[float] = Form(default=None),
    back_right_mm: Optional[float] = Form(default=None),
    back_top_mm: Optional[float] = Form(default=None),
    back_bottom_mm: Optional[float] = Form(default=None),
) -> dict:
    billing_before = usage_status(request.state.device_id)
    if not billing_before["can_scan"] and not scan_already_counted(
        request.state.device_id, scan_id
    ):
        raise HTTPException(
            status_code=402,
            detail=(
                "Free plan limit reached: 3 card analyses per UTC day. "
                "Upgrade to CardLens Pro for unlimited analyses."
            ),
        )
    try:
        front_analysis = analyze_image(
            await _read_upload(front),
            side="front",
            manual_boundary=_parse_manual_boundary(front_boundary),
        )
        back_analysis = (
            analyze_image(
                await _read_upload(back),
                side="back",
                manual_boundary=_parse_manual_boundary(back_boundary),
            )
            if back
            else None
        )
        back_reference_applied = False
        if back_analysis:
            clean_back = get_back_reference(request.state.device_id)
            if clean_back:
                back_reference_applied = apply_back_reference(
                    back_analysis, clean_back
                )
        catalog_path = Path(
            os.getenv("CARD_CATALOG", str(DEFAULT_CATALOG_PATH))
        )
        matches = identify_card(front_analysis.image, catalog_path)
        if card_id:
            identified_card = get_card(card_id, catalog_path)
            if identified_card is None:
                raise CardImageError("The selected catalog card no longer exists.")
        else:
            identified_card = _confident_catalog_match(matches)
        if identified_card:
            apply_reference_baseline(front_analysis, identified_card)
        manual_front = {
            "left": front_left_mm,
            "right": front_right_mm,
            "top": front_top_mm,
            "bottom": front_bottom_mm,
        }
        if any(value is not None for value in manual_front.values()):
            if any(value is None for value in manual_front.values()):
                raise CardImageError(
                    "Provide all four front measurements for manual centering."
                )
            apply_manual_centering(
                front_analysis,
                {name: float(value) for name, value in manual_front.items()},
            )
        manual_back = {
            "left": back_left_mm,
            "right": back_right_mm,
            "top": back_top_mm,
            "bottom": back_bottom_mm,
        }
        if any(value is not None for value in manual_back.values()):
            if back_analysis is None:
                raise CardImageError("Upload a back image before adjusting its guides.")
            if any(value is None for value in manual_back.values()):
                raise CardImageError(
                    "Provide all four back measurements for manual centering."
                )
            apply_manual_centering(
                back_analysis,
                {name: float(value) for name, value in manual_back.items()},
            )
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

    effective_scan_id = scan_id or uuid.uuid4().hex
    categories = defect_summary(
        front_analysis.features,
        back_analysis.features if back_analysis else None,
        model_path,
    )
    billing_after = consume_scan(
        request.state.device_id, scan_id=effective_scan_id
    )
    record_grade(
        device_id=request.state.device_id,
        scan_id=effective_scan_id,
        grade=prediction.grade,
        grade_label=prediction_payload["label"],
        categories=categories,
        card=identified_card,
    )
    return {
        "scan_id": effective_scan_id,
        "prediction": prediction_payload,
        "categories": categories,
        "warnings": warnings,
        "visual_reports": visual_reports,
        "identification": {
            "match": identified_card,
            "candidates": matches,
            "catalog_ready": catalog_path.exists(),
        },
        "billing": billing_after,
        "back_reference_applied": back_reference_applied,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Pokémon card grading scanner.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--open", action="store_true", help="Open CardLens in a browser.")
    args = parser.parse_args()
    if args.open:
        browser_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://{browser_host}:{args.port}")
        ).start()
    uvicorn.run(
        "drop_tracker.grading.web:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
