from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from drop_tracker.grading.catalog import (
    _connect,
    _reference_profile,
    apply_reference_baseline,
    identify_card,
)
from drop_tracker.grading.features import (
    CARD_HEIGHT,
    CARD_WIDTH,
    CardImageError,
    _border_measurements,
    _region_stats,
    analyze_image,
    annotated_image,
    source_boundary_image,
)
from drop_tracker.grading.model import (
    BASE_FEATURES,
    category_subgrades,
    centering_standards,
    feature_vector,
    predict_grade,
)
from drop_tracker.grading.web import app


def card_image_bytes() -> bytes:
    image = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (35, 75, 175), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (CARD_WIDTH - 21, CARD_HEIGHT - 21), (30, 200, 235), 24)
    cv2.rectangle(image, (85, 110), (CARD_WIDTH - 86, CARD_HEIGHT - 150), (85, 120, 170), 8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def back_photo_bytes() -> bytes:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (145, 70, 12), dtype=np.uint8)
    cv2.rectangle(card, (35, 40), (CARD_WIDTH - 36, CARD_HEIGHT - 41), (220, 145, 40), -1)
    cv2.circle(card, (CARD_WIDTH // 2, CARD_HEIGHT // 2), 210, (230, 230, 230), -1)
    background = np.full((1250, 1250, 3), (115, 165, 195), dtype=np.uint8)
    source = np.float32(
        [[0, 0], [CARD_WIDTH - 1, 0], [CARD_WIDTH - 1, CARD_HEIGHT - 1], [0, CARD_HEIGHT - 1]]
    )
    destination = np.float32([[260, 90], [860, 160], [810, 1130], [210, 1040]])
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(card, transform, (1250, 1250))
    mask = cv2.warpPerspective(
        np.full((CARD_HEIGHT, CARD_WIDTH), 255, dtype=np.uint8),
        transform,
        (1250, 1250),
    )
    background[mask > 0] = warped[mask > 0]
    success, encoded = cv2.imencode(".jpg", background)
    assert success
    return encoded.tobytes()


def front_photo_bytes() -> bytes:
    card = cv2.imdecode(np.frombuffer(card_image_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    rng = np.random.default_rng(7)
    background = rng.integers(12, 42, size=(1250, 1250, 1), dtype=np.uint8)
    background = np.repeat(background, 3, axis=2)
    source = np.float32(
        [[0, 0], [CARD_WIDTH - 1, 0], [CARD_WIDTH - 1, CARD_HEIGHT - 1], [0, CARD_HEIGHT - 1]]
    )
    destination = np.float32([[250, 110], [850, 135], [820, 1135], [225, 1090]])
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(card, transform, (1250, 1250))
    mask = cv2.warpPerspective(
        np.full((CARD_HEIGHT, CARD_WIDTH), 255, dtype=np.uint8),
        transform,
        (1250, 1250),
    )
    background[mask > 0] = warped[mask > 0]
    success, encoded = cv2.imencode(".jpg", background)
    assert success
    return encoded.tobytes()


def test_analyze_image_extracts_normalized_features() -> None:
    analysis = analyze_image(card_image_bytes(), side="front")

    assert analysis.image.shape == (CARD_HEIGHT, CARD_WIDTH, 3)
    assert set(analysis.features) == set(BASE_FEATURES)
    assert all(np.isfinite(value) for value in analysis.features.values())
    assert all(0.0 <= value <= 1.0 for value in analysis.features.values())
    distances = analysis.diagnostics["centering"]["distances"]
    side_anchor = (distances["left"] + distances["right"]) / 2
    assert distances["top"] <= side_anchor * 1.8
    assert distances["bottom"] <= side_anchor * 1.8
    assert not {"Corner color change", "Surface glare"} & {
        finding["type"] for finding in analysis.diagnostics["defects"]
    }


def test_front_analysis_removes_contrasting_scanner_mat() -> None:
    analysis = analyze_image(front_photo_bytes(), side="front")
    boundary = analysis.source_boundary

    assert "inferred from the image edges" not in " ".join(analysis.warnings)
    assert boundary[:, 0].min() > 150
    assert boundary[:, 0].max() < 950
    assert boundary[:, 1].min() > 60
    assert boundary[:, 1].max() < 1200


def test_front_analysis_isolates_modern_silver_border() -> None:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (185, 185, 185), dtype=np.uint8)
    cv2.rectangle(card, (28, 30), (CARD_WIDTH - 29, CARD_HEIGHT - 31), (120, 180, 65), -1)
    background = np.full((1250, 1250, 3), (24, 24, 24), dtype=np.uint8)
    source = np.float32(
        [[0, 0], [CARD_WIDTH - 1, 0], [CARD_WIDTH - 1, CARD_HEIGHT - 1], [0, CARD_HEIGHT - 1]]
    )
    destination = np.float32([[245, 100], [850, 125], [825, 1135], [220, 1095]])
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(card, transform, (1250, 1250))
    mask = cv2.warpPerspective(
        np.full((CARD_HEIGHT, CARD_WIDTH), 255, dtype=np.uint8),
        transform,
        (1250, 1250),
    )
    background[mask > 0] = warped[mask > 0]
    success, encoded = cv2.imencode(".jpg", background)
    assert success

    analysis = analyze_image(encoded.tobytes(), side="front")

    assert analysis.source_boundary[:, 0].min() > 150
    assert analysis.source_boundary[:, 0].max() < 950


def test_front_bottom_centering_ignores_copyright_text() -> None:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (190, 190, 190), dtype=np.uint8)
    cv2.rectangle(card, (30, 32), (CARD_WIDTH - 31, CARD_HEIGHT - 41), (80, 145, 190), -1)
    for x in range(140, CARD_WIDTH - 120, 28):
        cv2.rectangle(card, (x, CARD_HEIGHT - 25), (x + 17, CARD_HEIGHT - 20), (25, 25, 25), -1)

    centering = _border_measurements(card, side="front")

    assert 35 <= centering["distances"]["bottom"] <= 45
    assert centering["layout_adjustment"] is None


def test_photo_quality_does_not_create_hidden_damage_penalties() -> None:
    clean = {name: 0.0 for name in BASE_FEATURES}
    clean.update(
        {
            "centering_x": 1.0,
            "centering_y": 1.0,
            "surface_glare": 0.30,
            "surface_dark": 0.25,
            "sharpness": 0.01,
        }
    )

    categories = {
        category["key"]: category
        for category in category_subgrades(clean, clean)
    }

    assert categories["corners"]["score"] == 10.0
    assert categories["edges"]["score"] == 10.0
    assert categories["surface"]["score"] == 10.0
    assert "confidence" in categories["surface"]["detail"]
    pristine_photo = dict(clean)
    pristine_photo.update(
        {"surface_glare": 0.0, "surface_dark": 0.0, "sharpness": 1.0}
    )
    assert np.array_equal(
        feature_vector(clean, clean),
        feature_vector(pristine_photo, pristine_photo),
    )


def test_back_analysis_isolates_card_before_measuring_centering() -> None:
    analysis = analyze_image(back_photo_bytes(), side="back")
    centering = analysis.diagnostics["centering"]

    assert "inferred from the image edges" not in " ".join(analysis.warnings)
    assert centering["balance_x"] > 0.80
    assert centering["balance_y"] > 0.80
    assert centering["guides"]["left"] < 100
    assert centering["guides"]["top"] < 120
    assert centering["distances"]["left"] == centering["guides"]["left"]
    assert centering["distances"]["right"] == CARD_WIDTH - 1 - centering["guides"]["right"]
    assert centering["distance_mm"]["left"] > 0
    assert centering["card_dimensions"] == {"width": CARD_WIDTH, "height": CARD_HEIGHT}
    assert analysis.diagnostics["defects"] == []
    assert analysis.features["edge_pale"] == 0.0
    assert analysis.features["corner_pale_max"] == 0.0
    assert annotated_image(analysis).shape[:2] == (CARD_HEIGHT + 180, CARD_WIDTH + 180)
    assert source_boundary_image(analysis).shape[0] > analysis.source_image.shape[0]


def test_back_analysis_marks_non_blue_corner_whitening() -> None:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (145, 70, 12), dtype=np.uint8)
    cv2.rectangle(card, (38, 42), (CARD_WIDTH - 39, CARD_HEIGHT - 43), (215, 145, 45), -1)
    cv2.rectangle(card, (0, 0), (55, 55), (235, 235, 235), -1)
    success, encoded = cv2.imencode(".jpg", card)
    assert success

    analysis = analyze_image(encoded.tobytes(), side="back")

    assert any(
        finding["type"] == "Localized whitening"
        for finding in analysis.diagnostics["defects"]
    )


def test_back_whitening_ignores_smooth_glare_but_finds_small_chip() -> None:
    clean = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (145, 70, 12), dtype=np.uint8)
    y, x = np.mgrid[:CARD_HEIGHT, :CARD_WIDTH]
    glare_alpha = 0.48 * np.exp(
        -(((x - 8) / 75.0) ** 2 + ((y - CARD_HEIGHT / 2) / 310.0) ** 2)
    )
    glare = (
        clean.astype(np.float32) * (1.0 - glare_alpha[:, :, None])
        + 255.0 * glare_alpha[:, :, None]
    ).astype(np.uint8)

    _, glare_diagnostics = _region_stats(glare, side="back")

    assert glare_diagnostics["defects"] == []

    chipped = clean.copy()
    cv2.rectangle(
        chipped,
        (CARD_WIDTH - 24, 0),
        (CARD_WIDTH - 12, 12),
        (235, 235, 235),
        -1,
    )
    _, chip_diagnostics = _region_stats(chipped, side="back")

    assert any(
        finding["type"] == "Localized whitening"
        and finding["bbox"][0] > CARD_WIDTH - 50
        for finding in chip_diagnostics["defects"]
    )


def test_back_whitening_finds_wear_on_outermost_corner_pixels() -> None:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (145, 70, 12), dtype=np.uint8)
    cv2.rectangle(card, (10, 0), (20, 4), (225, 225, 225), -1)
    cv2.rectangle(card, (0, 12), (4, 24), (225, 225, 225), -1)
    cv2.rectangle(card, (25, 0), (31, 3), (190, 200, 205), -1)

    features, diagnostics = _region_stats(card, side="back")

    corner_findings = [
        finding
        for finding in diagnostics["defects"]
        if finding["bbox"][0] < 50 and finding["bbox"][1] < 50
    ]
    assert len(corner_findings) >= 2
    assert diagnostics["condition_signals"]["corners"]["top-left"] > 0
    assert features["corner_pale_max"] > 0


def test_analyze_image_rejects_non_image() -> None:
    try:
        analyze_image(b"not an image")
    except CardImageError as error:
        assert "supported image" in str(error)
    else:
        raise AssertionError("Expected an invalid image to be rejected")


def test_prediction_uses_disclosed_fallback_without_model(tmp_path: Path) -> None:
    features = analyze_image(card_image_bytes()).features

    prediction = predict_grade(features, features, tmp_path / "missing.joblib")

    assert 1.0 <= prediction.grade <= 10.0
    assert prediction.low <= prediction.grade <= prediction.high
    assert prediction.method == "untrained visual heuristic"
    assert prediction.confidence == "low"


def test_centering_uses_psa_and_beckett_thresholds() -> None:
    perfect = {name: 0.0 for name in BASE_FEATURES}
    perfect.update({"centering_x": 1.0, "centering_y": 1.0})
    psa_limit_front = dict(perfect)
    psa_limit_front.update(
        {"centering_x": 45.0 / 55.0, "centering_y": 45.0 / 55.0}
    )
    psa_limit_back = dict(perfect)
    psa_limit_back.update(
        {"centering_x": 25.0 / 75.0, "centering_y": 25.0 / 75.0}
    )

    assert centering_standards(perfect, perfect)["psa"] == 10.0
    assert centering_standards(perfect, perfect)["bgs"] == 10.0
    assert centering_standards(perfect, None)["psa"] < 10.0
    assert centering_standards(psa_limit_front, psa_limit_back)["psa"] < 10.0
    assert not centering_standards(psa_limit_front, psa_limit_back)[
        "psa_10_eligible"
    ]
    assert centering_standards(psa_limit_front, psa_limit_back)["bgs"] < 10.0


def test_catalog_identifies_matching_reference(tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite"
    connection = _connect(database)
    perceptual_hash, color_signature, reference_features = _reference_profile(
        card_image_bytes()
    )
    connection.execute(
        "INSERT INTO series VALUES (?, ?, ?, ?)",
        ("sv", "Scarlet & Violet", "2023-03-31", "now"),
    )
    connection.execute(
        "INSERT INTO sets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("sv-test", "sv", "Test Set", "2024-01-01", 1, 1, None, None),
    )
    connection.execute(
        """
        INSERT INTO cards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sv-test-1",
            "sv",
            "sv-test",
            "Test Set",
            "1",
            "Pikachu",
            "https://example.com/card",
            perceptual_hash,
            color_signature,
            reference_features,
            "now",
        ),
    )
    connection.commit()
    connection.close()

    analysis = analyze_image(card_image_bytes(), side="front")
    matches = identify_card(analysis.image, database)

    assert matches[0]["id"] == "sv-test-1"
    assert matches[0]["confidence"] == 1.0


def test_catalog_reference_calibrates_layout_without_moving_guides() -> None:
    analysis = analyze_image(card_image_bytes(), side="front")
    original_guides = dict(analysis.diagnostics["centering"]["guides"])
    expected = dict(analysis.diagnostics["centering"]["distances"])

    apply_reference_baseline(
        analysis,
        {
            "id": "reference-card",
            "confidence": 1.0,
            "reference_features": {"centering_distances": expected},
        },
    )

    centering = analysis.diagnostics["centering"]
    assert centering["guides"] == original_guides
    assert centering["left_percent"] == 50
    assert centering["right_percent"] == 50
    assert centering["top_percent"] == 50
    assert centering["bottom_percent"] == 50
    assert centering["reference_calibrated"]


def test_grade_api_returns_breakdown(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CARD_GRADER_MODEL", str(tmp_path / "missing.joblib"))
    client = TestClient(app)

    response = client.post(
        "/api/grade",
        files={"front": ("front.jpg", card_image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert 1.0 <= payload["prediction"]["grade"] <= 10.0
    assert [category["name"] for category in payload["categories"]] == [
        "Centering",
        "Corners",
        "Edges",
        "Surface",
    ]
    assert all(1.0 <= category["score"] <= 10.0 for category in payload["categories"])
    assert payload["prediction"]["label"]
    assert "Add a back photo" in payload["warnings"][-1]
    assert payload["visual_reports"][0]["side"] == "Front"
    assert payload["visual_reports"][0]["image"].startswith("data:image/jpeg;base64,")
    assert payload["visual_reports"][0]["source_image"].startswith("data:image/jpeg;base64,")
    assert "/" in payload["visual_reports"][0]["centering"]["horizontal"]
    assert set(payload["visual_reports"][0]["condition_signals"]) == {
        "corners",
        "edges",
        "surface",
    }
