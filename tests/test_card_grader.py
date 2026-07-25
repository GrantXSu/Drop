import json
from pathlib import Path

import joblib
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from drop_tracker.grading.billing import consume_scan, set_subscription, usage_status
import drop_tracker.grading.catalog as catalog_module
import drop_tracker.grading.web as web_module
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
    FEATURE_NAMES,
    category_subgrades,
    centering_standards,
    feature_vector,
    predict_grade,
)
from drop_tracker.grading.training import _validate_row
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


def test_front_centering_rejects_content_bars_beyond_physical_limit() -> None:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (35, 205, 235), dtype=np.uint8)
    cv2.rectangle(
        card,
        (35, 38),
        (CARD_WIDTH - 36, CARD_HEIGHT - 42),
        (55, 180, 220),
        3,
    )
    cv2.rectangle(
        card,
        (45, CARD_HEIGHT - 96),
        (CARD_WIDTH - 46, CARD_HEIGHT - 84),
        (235, 235, 235),
        -1,
    )

    centering = _border_measurements(card, side="front")

    assert centering["distance_mm"]["bottom"] <= 5.0
    assert centering["distances"]["bottom"] < 70


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
    assert categories["surface"]["score"] is None
    assert categories["surface"]["condition"] == "Not assessed"
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


def test_inner_blue_border_streak_is_surface_not_edge_whitening() -> None:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (145, 70, 12), dtype=np.uint8)
    cv2.line(card, (CARD_WIDTH // 2, 13), (CARD_WIDTH // 2, 31), (235, 235, 235), 3)

    features, diagnostics = _region_stats(card, side="back")
    finding_types = [finding["type"] for finding in diagnostics["defects"]]

    assert "Surface scratch/print-line candidate" in finding_types
    assert "Localized whitening" not in finding_types
    assert features["surface_assessed"] == 1.0
    assert features["surface_damage"] > 0
    assert diagnostics["condition_signals"]["surface"][
        "localized_border_inspection"
    ]


def test_tiny_inner_border_streak_is_labeled_small() -> None:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (145, 70, 12), dtype=np.uint8)
    cv2.line(card, (CARD_WIDTH // 2, 13), (CARD_WIDTH // 2, 21), (235, 235, 235), 1)

    _, diagnostics = _region_stats(card, side="back")
    surface_findings = [
        finding
        for finding in diagnostics["defects"]
        if finding["type"] == "Surface scratch/print-line candidate"
    ]

    assert surface_findings
    assert surface_findings[0]["severity"] == "small"


def test_smooth_top_border_glare_is_not_whitening() -> None:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (145, 70, 12), dtype=np.uint8)
    y, x = np.mgrid[:CARD_HEIGHT, :CARD_WIDTH]
    alpha = 0.40 * np.exp(
        -(((x - CARD_WIDTH / 2) / 230.0) ** 2 + ((y - 5) / 28.0) ** 2)
    )
    glare = (
        card.astype(np.float32) * (1.0 - alpha[:, :, None])
        + 255.0 * alpha[:, :, None]
    ).astype(np.uint8)

    _, diagnostics = _region_stats(glare, side="back")

    assert not any(
        finding["type"] == "Localized whitening"
        for finding in diagnostics["defects"]
    )


def test_shadowed_left_edge_whitening_is_detected() -> None:
    card = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (145, 70, 12), dtype=np.uint8)
    card[:, :85] = (72, 35, 6)
    cv2.rectangle(card, (0, 420), (8, 442), (98, 98, 98), -1)

    _, diagnostics = _region_stats(card, side="back")

    assert any(
        finding["type"] == "Localized whitening" and finding["bbox"][0] < 15
        for finding in diagnostics["defects"]
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
    assert "Prototype" not in prediction.caveat


def test_training_labels_separate_psa_overall_from_bgs_subgrades() -> None:
    common = {
        "front": "front.jpg",
        "back": "back.jpg",
        "source_url": "https://example.com/cert",
        "usage_rights": "owner permission",
        "certification_number": "12345",
    }
    psa = _validate_row(
        {
            **common,
            "grading_company": "PSA",
            "overall_grade": "9",
            "bgs_corners": "",
            "bgs_edges": "",
        },
        2,
    )
    bgs = _validate_row(
        {
            **common,
            "grading_company": "BGS",
            "overall_grade": "9.5",
            "bgs_corners": "9.5",
            "bgs_edges": "9",
        },
        3,
    )

    assert psa == {"psa_overall": 9.0}
    assert bgs == {
        "bgs_overall": 9.5,
        "bgs_corners": 9.5,
        "bgs_edges": 9.0,
    }


def test_validated_bgs_models_supply_corner_and_edge_subgrades(
    tmp_path: Path,
) -> None:
    features = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float64)
    corners = DummyRegressor(strategy="constant", constant=8.5).fit(
        features, np.array([8.5, 8.5])
    )
    edges = DummyRegressor(strategy="constant", constant=7.0).fit(
        features, np.array([7.0, 7.0])
    )
    artifact = {
        "feature_names": FEATURE_NAMES,
        "models": {"bgs_corners": corners, "bgs_edges": edges},
        "targets": {
            "bgs_corners": {"samples": 150, "validation_mae": 0.5},
            "bgs_edges": {"samples": 150, "validation_mae": 0.6},
        },
    }
    model_path = tmp_path / "categories.joblib"
    joblib.dump(artifact, model_path)
    clean = {name: 0.0 for name in BASE_FEATURES}
    clean.update({"centering_x": 1.0, "centering_y": 1.0})

    categories = {
        item["key"]: item
        for item in category_subgrades(clean, clean, model_path)
    }

    assert categories["corners"]["score"] == 8.5
    assert categories["edges"]["score"] == 7.0
    assert categories["corners"]["method"] == "trained BGS subgrade"
    assert categories["edges"]["method"] == "trained BGS subgrade"


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
    assert matches[0]["keypoint_similarity"] >= 0.0


def test_catalog_all_mode_discovers_every_english_series(
    monkeypatch, tmp_path: Path
) -> None:
    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    responses = {
        "series": [{"id": "base"}, {"id": "sv"}, {"id": "me"}],
        "series/base": {"id": "base", "name": "Base", "sets": []},
        "series/sv": {"id": "sv", "name": "Scarlet & Violet", "sets": []},
        "series/me": {"id": "me", "name": "Mega Evolution", "sets": []},
    }
    monkeypatch.setattr(catalog_module.httpx, "Client", FakeClient)
    monkeypatch.setattr(
        catalog_module,
        "_json_get",
        lambda client, path: responses[path],
    )

    report = catalog_module.sync_catalog(
        tmp_path / "all-english.sqlite",
        series_ids=("all",),
        with_images=False,
    )

    assert report["series"] == ["base", "sv", "me"]


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


def test_catalog_reference_grades_visible_surface_scratch() -> None:
    image = np.full((CARD_HEIGHT, CARD_WIDTH, 3), (75, 115, 165), dtype=np.uint8)
    rng = np.random.default_rng(19)
    for _ in range(80):
        center = (
            int(rng.integers(45, CARD_WIDTH - 45)),
            int(rng.integers(55, CARD_HEIGHT - 55)),
        )
        color = tuple(int(value) for value in rng.integers(30, 230, size=3))
        cv2.circle(image, center, int(rng.integers(4, 18)), color, -1)
    cv2.rectangle(image, (28, 30), (CARD_WIDTH - 29, CARD_HEIGHT - 31), (190, 190, 190), 8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    _, _, reference_json = _reference_profile(encoded.tobytes())
    reference_features = json.loads(reference_json)
    clean_analysis = analyze_image(encoded.tobytes(), side="front")
    apply_reference_baseline(
        clean_analysis,
        {
            "id": "surface-reference",
            "confidence": 1.0,
            "reference_features": reference_features,
        },
    )
    assert clean_analysis.features["surface_assessed"] == 1.0
    assert clean_analysis.features["surface_damage"] == 0.0

    analysis = analyze_image(encoded.tobytes(), side="front")
    cv2.line(analysis.image, (130, 170), (620, 830), (245, 245, 245), 5)

    apply_reference_baseline(
        analysis,
        {
            "id": "surface-reference",
            "confidence": 1.0,
            "reference_features": reference_features,
        },
    )

    assert analysis.features["surface_assessed"] == 1.0
    assert analysis.features["surface_damage"] > 0
    assert any(
        finding["type"] == "Surface scratch/crease candidate"
        for finding in analysis.diagnostics["defects"]
    )
    surface = category_subgrades(analysis.features, None)[-1]
    assert surface["score"] is not None
    assert surface["score"] < 10.0


def test_grade_api_returns_breakdown(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CARD_GRADER_MODEL", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("CARDLENS_BILLING_DB", str(tmp_path / "billing.sqlite"))
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
    assert all(
        category["score"] is None or 1.0 <= category["score"] <= 10.0
        for category in payload["categories"]
    )
    assert payload["categories"][-1]["condition"] == "Not assessed"
    assert payload["prediction"]["label"]
    assert payload["billing"]["remaining_today"] == 2
    assert "Add a back photo" in payload["warnings"][-1]
    assert payload["visual_reports"][0]["side"] == "Front"
    assert payload["visual_reports"][0]["image"].startswith("data:image/jpeg;base64,")
    assert payload["visual_reports"][0]["card_image"].startswith(
        "data:image/jpeg;base64,"
    )
    assert payload["visual_reports"][0]["source_image"].startswith("data:image/jpeg;base64,")
    assert "/" in payload["visual_reports"][0]["centering"]["horizontal"]
    assert set(payload["visual_reports"][0]["condition_signals"]) == {
        "corners",
        "edges",
        "surface",
        "defect_counts",
    }


def test_ui_collapses_detected_findings() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"].startswith("no-store")
    assert 'class="findings-dropdown"' in response.text
    assert "Detected findings (" in response.text
    assert "Enable line adjustment" in response.text
    assert 'class="drag-guide' in response.text
    assert "Accuracy checklist before scanning" in response.text
    assert "viewport-fit=cover" in response.text
    assert "Prototype" not in response.text
    assert 'id="camera-mode"' in response.text
    assert 'id="camera-video"' in response.text
    assert 'id="camera-front-slot"' in response.text
    assert 'id="camera-back-slot"' in response.text
    assert "Capture front" in response.text
    assert "Analyze captured card" in response.text
    assert "3 card analyses per UTC day" in response.text
    assert "$9.99" in response.text
    assert "$59.99" in response.text
    assert 'data-pro-ad' in response.text
    assert "Keep grading after your three free cards." in response.text
    assert "Build your collection without a daily cap." in response.text
    assert 'data-nav="analyze"' in response.text
    assert 'data-nav="cards"' in response.text
    assert 'data-nav="settings"' in response.text
    assert 'id="prescan-card-query"' in response.text
    assert 'id="developer-password"' not in response.text

    assert TestClient(app).get("/cards").status_code == 200
    assert TestClient(app).get("/settings").status_code == 200


def test_grade_api_applies_manual_centering_guides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CARD_GRADER_MODEL", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("CARDLENS_BILLING_DB", str(tmp_path / "billing.sqlite"))
    response = TestClient(app).post(
        "/api/grade",
        files={"front": ("front.jpg", card_image_bytes(), "image/jpeg")},
        data={
            "front_left_mm": "1.0",
            "front_right_mm": "2.0",
            "front_top_mm": "2.5",
            "front_bottom_mm": "2.5",
        },
    )

    assert response.status_code == 200
    centering = response.json()["visual_reports"][0]["centering"]
    assert centering["manual_override"]
    assert centering["distance_mm"] == {
        "left": 1.0,
        "right": 2.0,
        "top": 2.5,
        "bottom": 2.5,
    }
    assert centering["horizontal"] == "33/67"
    assert centering["vertical"] == "50/50"


def test_repeated_whitening_outweighs_centering_in_fallback_grade(
    tmp_path: Path,
) -> None:
    clean = {name: 0.0 for name in BASE_FEATURES}
    clean.update({"centering_x": 1.0, "centering_y": 1.0})
    whitened = dict(clean)
    whitened.update({"edge_pale": 0.03, "edge_defect_load": 1.0})
    poor_centering = dict(clean)
    poor_centering.update({"centering_x": 0.25, "centering_y": 0.25})

    whitening_grade = predict_grade(
        whitened, whitened, tmp_path / "missing.joblib"
    ).grade
    centering_grade = predict_grade(
        poor_centering, poor_centering, tmp_path / "missing.joblib"
    ).grade

    assert whitening_grade < 7.0
    assert centering_grade > whitening_grade


def test_medium_whitening_findings_do_not_score_like_high_damage() -> None:
    features = {name: 0.0 for name in BASE_FEATURES}
    features.update(
        {
            "centering_x": 1.0,
            "centering_y": 1.0,
            "edge_pale": 0.03,
            "edge_defect_load": 19 * 0.25 / 12.0,
        }
    )

    edges = category_subgrades(features, features)[2]

    assert 6.5 <= edges["score"] <= 8.0


def test_many_small_surface_marks_do_not_collapse_surface_grade() -> None:
    features = {name: 0.0 for name in BASE_FEATURES}
    features.update(
        {
            "centering_x": 1.0,
            "centering_y": 1.0,
            "surface_assessed": 1.0,
            "surface_damage": 19 * 0.10 / 30.0,
        }
    )

    surface = category_subgrades(features, features)[-1]

    assert surface["score"] >= 8.0


def test_manual_catalog_search_and_confirmation(monkeypatch, tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite"
    connection = _connect(database)
    connection.execute(
        "INSERT INTO series VALUES (?, ?, ?, ?)",
        ("swsh", "Sword & Shield", "2020-02-07", "now"),
    )
    connection.execute(
        "INSERT INTO sets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("swsh1", "swsh", "Sword & Shield", "2020-02-07", 202, 202, None, None),
    )
    connection.execute(
        "INSERT INTO cards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "swsh1-65",
            "swsh",
            "swsh1",
            "Sword & Shield",
            "065",
            "Pikachu",
            "https://example.com/pikachu",
            None,
            None,
            None,
            "now",
        ),
    )
    connection.commit()
    connection.close()
    monkeypatch.setenv("CARD_CATALOG", str(database))
    monkeypatch.setenv("CARD_GRADER_MODEL", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("CARDLENS_BILLING_DB", str(tmp_path / "billing.sqlite"))
    client = TestClient(app)

    search = client.get("/api/cards", params={"q": "Pikachu 065"})
    assert search.status_code == 200
    assert search.json()["results"][0]["id"] == "swsh1-65"

    grade = client.post(
        "/api/grade",
        files={"front": ("front.jpg", card_image_bytes(), "image/jpeg")},
        data={"card_id": "swsh1-65"},
    )
    assert grade.status_code == 200
    match = grade.json()["identification"]["match"]
    assert match["id"] == "swsh1-65"
    assert match["match_method"] == "manual"


def test_free_quota_counts_unique_cards_and_pro_is_unlimited(tmp_path: Path) -> None:
    database = tmp_path / "billing.sqlite"
    device = "test-device"

    assert usage_status(device, database)["remaining_today"] == 3
    assert consume_scan(device, "card-1", database)["remaining_today"] == 2
    assert consume_scan(device, "card-1", database)["remaining_today"] == 2
    assert consume_scan(device, "card-2", database)["remaining_today"] == 1
    assert consume_scan(device, "card-3", database)["remaining_today"] == 0
    assert not usage_status(device, database)["can_scan"]

    set_subscription(device_id=device, status="active", path=database)
    status = usage_status(device, database)
    assert status["is_pro"]
    assert status["daily_limit"] is None
    assert status["can_scan"]


def test_grade_api_enforces_three_unique_cards_per_day(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CARD_GRADER_MODEL", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("CARDLENS_BILLING_DB", str(tmp_path / "billing.sqlite"))
    client = TestClient(app)

    for index in range(3):
        response = client.post(
            "/api/grade",
            files={"front": ("front.jpg", card_image_bytes(), "image/jpeg")},
            data={"scan_id": f"card-{index}"},
        )
        assert response.status_code == 200

    blocked = client.post(
        "/api/grade",
        files={"front": ("front.jpg", card_image_bytes(), "image/jpeg")},
        data={"scan_id": "card-4"},
    )
    assert blocked.status_code == 402
    assert "3 card analyses" in blocked.json()["detail"]
    recalculated = client.post(
        "/api/grade",
        files={"front": ("front.jpg", card_image_bytes(), "image/jpeg")},
        data={
            "scan_id": "card-2",
            "front_left_mm": "2.0",
            "front_right_mm": "2.0",
            "front_top_mm": "2.0",
            "front_bottom_mm": "2.0",
        },
    )
    assert recalculated.status_code == 200
    assert recalculated.json()["billing"]["used_today"] == 3
    history = client.get("/api/history")
    assert history.status_code == 200
    assert len(history.json()["cards"]) == 3

    cleared = client.delete("/api/history")
    assert cleared.status_code == 200
    assert client.get("/api/history").json()["cards"] == []


def test_pro_checkout_uses_configured_stripe_price(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CARDLENS_BILLING_DB", str(tmp_path / "billing.sqlite"))
    monkeypatch.setenv("CARDLENS_COOKIE_SECRET", "test-cookie-secret")
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_example")
    monkeypatch.setenv("STRIPE_PRO_MONTHLY_PRICE_ID", "price_monthly")
    monkeypatch.setenv("STRIPE_PRO_ANNUAL_PRICE_ID", "price_annual")
    captured = {}

    class CheckoutSession:
        url = "https://checkout.stripe.test/session"

    def fake_create(**kwargs):
        captured.update(kwargs)
        return CheckoutSession()

    monkeypatch.setattr(web_module.stripe.checkout.Session, "create", fake_create)
    client = TestClient(app)
    client.get("/api/status")

    response = client.post("/api/billing/checkout", json={"plan": "annual"})

    assert response.status_code == 200
    assert response.json()["url"] == CheckoutSession.url
    assert captured["mode"] == "subscription"
    assert captured["line_items"] == [{"price": "price_annual", "quantity": 1}]
    assert captured["metadata"]["plan"] == "annual"


def test_developer_password_grants_unlimited_access(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CARDLENS_BILLING_DB", str(tmp_path / "billing.sqlite"))
    monkeypatch.setenv("CARDLENS_COOKIE_SECRET", "test-cookie-secret")
    monkeypatch.setenv("CARDLENS_DEVELOPER_PASSWORD", "correct horse battery staple")
    client = TestClient(app)
    client.get("/api/status")

    denied = client.post("/api/cards/search", json={"query": "incorrect"})
    assert denied.status_code == 200
    assert not denied.json()["developer_unlocked"]

    unlocked = client.post(
        "/api/cards/search",
        json={"query": "correct horse battery staple"},
    )
    assert unlocked.status_code == 200
    billing = unlocked.json()["billing"]
    assert billing["plan"] == "developer"
    assert billing["is_developer"]
    assert billing["is_unlimited"]
    assert billing["daily_limit"] is None

    locked = client.post("/api/developer/lock")
    assert locked.status_code == 200
    assert locked.json()["billing"]["plan"] == "free"
