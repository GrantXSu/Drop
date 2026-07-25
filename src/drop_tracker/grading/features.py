"""Computer-vision measurements used by the card grade estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


CARD_WIDTH = 750
CARD_HEIGHT = 1050
CARD_WIDTH_MM = 63.5
CARD_HEIGHT_MM = 88.9
MAX_BORDER_MM = 5.0


class CardImageError(ValueError):
    """Raised when an uploaded image cannot be treated as a card photo."""


@dataclass(frozen=True)
class CardAnalysis:
    image: np.ndarray
    features: Dict[str, float]
    warnings: Tuple[str, ...]
    diagnostics: Dict[str, object]
    source_image: np.ndarray
    source_boundary: np.ndarray


def decode_image(data: bytes) -> np.ndarray:
    """Decode an uploaded image into OpenCV BGR format."""
    if not data:
        raise CardImageError("The image is empty.")
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise CardImageError("The file is not a supported image.")
    if min(image.shape[:2]) < 240:
        raise CardImageError("Use an image at least 240 pixels on its shortest side.")
    return image


def _order_points(points: np.ndarray) -> np.ndarray:
    points = points.reshape(4, 2).astype(np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ],
        dtype=np.float32,
    )


def _card_contour(image: np.ndarray) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    image_area = resized.shape[0] * resized.shape[1]
    candidates = []
    for lower, upper in ((25, 85), (45, 140), (70, 210)):
        edges = cv2.Canny(gray, lower, upper)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not image_area * 0.08 <= area <= image_area * 0.96:
                continue
            rectangle = cv2.minAreaRect(contour)
            short, long = sorted(rectangle[1])
            if short <= 0 or not 0.60 <= short / long <= 0.82:
                continue
            rectangle_area = short * long
            rectangularity = area / rectangle_area if rectangle_area else 0.0
            if rectangularity < 0.62:
                continue
            box = cv2.boxPoints(rectangle)
            candidates.append((area * rectangularity, box))
    if not candidates:
        return None
    _, best_box = max(candidates, key=lambda candidate: candidate[0])
    return best_box.astype(np.float32) / scale


def _foreground_card_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Separate a front card from a contrasting tabletop or scanner mat."""
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    frame = max(4, round(min(resized.shape[:2]) * 0.035))
    background_pixels = np.concatenate(
        (
            lab[:frame].reshape(-1, 3),
            lab[-frame:].reshape(-1, 3),
            lab[:, :frame].reshape(-1, 3),
            lab[:, -frame:].reshape(-1, 3),
        )
    )
    background_color = np.median(background_pixels, axis=0)
    distance = np.linalg.norm(lab - background_color, axis=2)
    normalized = np.clip(distance, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(18.0, float(otsu_threshold) * 0.72)
    mask = np.where(distance >= threshold, 255, 0).astype(np.uint8)
    kernel_size = max(9, int(min(resized.shape[:2]) * 0.025))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = resized.shape[0] * resized.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not image_area * 0.12 <= area <= image_area * 0.96:
            continue
        rectangle = cv2.minAreaRect(contour)
        short, long = sorted(rectangle[1])
        if short <= 0 or not 0.52 <= short / long <= 0.86:
            continue
        rectangle_area = short * long
        if rectangle_area <= 0 or area / rectangle_area < 0.55:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, 0.025 * perimeter, True)
        boundary = polygon.reshape(4, 2) if len(polygon) == 4 else cv2.boxPoints(rectangle)
        candidates.append((area, boundary))
    if not candidates:
        return None
    _, boundary = max(candidates, key=lambda candidate: candidate[0])
    return boundary.astype(np.float32) / scale


def _pokemon_front_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Locate the connected yellow/gold outer border used by Pokémon fronts."""
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (8, 20, 60), (55, 255, 255))
    kernel_size = max(7, int(min(resized.shape[:2]) * 0.018))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(yellow, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = resized.shape[0] * resized.shape[1]
    candidates = []
    for contour in contours:
        rectangle = cv2.minAreaRect(contour)
        short, long = sorted(rectangle[1])
        rectangle_area = short * long
        if short <= 0 or not 0.52 <= short / long <= 0.86:
            continue
        if not image_area * 0.10 <= rectangle_area <= image_area * 0.92:
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if rectangle_area <= 0 or hull_area / rectangle_area < 0.20:
            continue
        perimeter = cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, 0.025 * perimeter, True)
        boundary = polygon.reshape(4, 2) if len(polygon) == 4 else cv2.boxPoints(rectangle)
        candidates.append((rectangle_area, boundary))
    if not candidates:
        return None
    _, boundary = max(candidates, key=lambda candidate: candidate[0])
    return boundary.astype(np.float32) / scale


def _pokemon_silver_front_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Locate modern silver/gray Pokémon borders on a darker background."""
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    silver = cv2.inRange(hsv, (0, 0, 85), (179, 105, 255))
    kernel_size = max(7, int(min(resized.shape[:2]) * 0.016))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    silver = cv2.morphologyEx(silver, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(silver, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = resized.shape[0] * resized.shape[1]
    candidates = []
    for contour in contours:
        rectangle = cv2.minAreaRect(contour)
        short, long = sorted(rectangle[1])
        rectangle_area = short * long
        if short <= 0 or not 0.52 <= short / long <= 0.86:
            continue
        if not image_area * 0.12 <= rectangle_area <= image_area * 0.92:
            continue
        hull = cv2.convexHull(contour)
        if rectangle_area <= 0 or cv2.contourArea(hull) / rectangle_area < 0.20:
            continue
        perimeter = cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        boundary = polygon.reshape(4, 2) if len(polygon) == 4 else cv2.boxPoints(rectangle)
        candidates.append((rectangle_area, boundary))
    if not candidates:
        return None
    _, boundary = max(candidates, key=lambda candidate: candidate[0])
    return boundary.astype(np.float32) / scale


def _refine_card_edges(
    image: np.ndarray, boundary: np.ndarray, color_mask: np.ndarray
) -> np.ndarray:
    """Fit four independent diagonal lines to the outer blue border."""
    ordered = _order_points(boundary)
    minimum = ordered.min(axis=0)
    maximum = ordered.max(axis=0)
    inset_x = max(8, round((maximum[0] - minimum[0]) * 0.08))
    inset_y = max(8, round((maximum[1] - minimum[1]) * 0.08))

    def robust_fit(independent: list, dependent: list) -> Optional[np.ndarray]:
        if len(independent) < 60:
            return None
        x = np.asarray(independent, dtype=np.float32)
        y = np.asarray(dependent, dtype=np.float32)
        coefficients = np.polyfit(x, y, 1)
        residuals = y - np.polyval(coefficients, x)
        median = np.median(residuals)
        deviation = np.median(np.abs(residuals - median))
        keep = np.abs(residuals - median) <= max(2.5, deviation * 3.0)
        if keep.sum() < 40:
            return None
        return np.polyfit(x[keep], y[keep], 1)

    rows = range(
        max(0, round(minimum[1]) + inset_y),
        min(image.shape[0], round(maximum[1]) - inset_y),
    )
    row_values, left_values, right_values = [], [], []
    x_min = max(0, round(minimum[0]) - inset_x)
    x_max = min(image.shape[1], round(maximum[0]) + inset_x)
    for row in rows:
        colored = np.flatnonzero(color_mask[row, x_min:x_max]) + x_min
        if colored.size:
            row_values.append(row)
            left_values.append(int(colored.min()))
            right_values.append(int(colored.max()))

    columns = range(
        max(0, round(minimum[0]) + inset_x),
        min(image.shape[1], round(maximum[0]) - inset_x),
    )
    column_values, top_values, bottom_values = [], [], []
    y_min = max(0, round(minimum[1]) - inset_y)
    y_max = min(image.shape[0], round(maximum[1]) + inset_y)
    for column in columns:
        colored = np.flatnonzero(color_mask[y_min:y_max, column]) + y_min
        if colored.size:
            column_values.append(column)
            top_values.append(int(colored.min()))
            bottom_values.append(int(colored.max()))

    left = robust_fit(row_values, left_values)
    right = robust_fit(row_values, right_values)
    top = robust_fit(column_values, top_values)
    bottom = robust_fit(column_values, bottom_values)
    if any(line is None for line in (left, right, top, bottom)):
        center = ordered.mean(axis=0)
        return center + (ordered - center) * 1.025

    # Convert x=a*y+b and y=a*x+b to normal-form lines. Two pixels of
    # outward allowance preserve rounded and lightly whitened cut edges.
    fitted_lines = [
        (np.array([-top[0], 1.0]), float(top[1] - 2)),
        (np.array([1.0, -right[0]]), float(right[1] + 2)),
        (np.array([-bottom[0], 1.0]), float(bottom[1] + 2)),
        (np.array([1.0, -left[0]]), float(left[1] - 2)),
    ]

    corners = []
    for index in range(4):
        previous_normal, previous_rho = fitted_lines[index - 1]
        current_normal, current_rho = fitted_lines[index]
        matrix = np.vstack((previous_normal, current_normal))
        if abs(float(np.linalg.det(matrix))) < 0.15:
            return ordered
        corner = np.linalg.solve(matrix, np.array([previous_rho, current_rho]))
        corner[0] = np.clip(corner[0], 0, image.shape[1] - 1)
        corner[1] = np.clip(corner[1], 0, image.shape[0] - 1)
        corners.append(corner)
    return np.asarray(corners, dtype=np.float32)


def _pokemon_back_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Locate a Pokémon card back from its connected blue printed region."""
    height, width = image.shape[:2]
    scale = min(1.0, 1200.0 / max(height, width))
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (96, 85, 35), (132, 255, 255))
    opening_size = max(3, int(min(resized.shape[:2]) * 0.009))
    closing_size = max(7, int(min(resized.shape[:2]) * 0.022))
    if opening_size % 2 == 0:
        opening_size += 1
    if closing_size % 2 == 0:
        closing_size += 1
    opening = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (opening_size, opening_size)
    )
    closing = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_size, closing_size)
    )
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, opening, iterations=1)
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, closing, iterations=1)
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = resized.shape[0] * resized.shape[1]
    candidates = []
    for contour in contours:
        rectangle = cv2.minAreaRect(contour)
        short, long = sorted(rectangle[1])
        rectangle_area = short * long
        if short <= 0 or not 0.45 <= short / long <= 0.90:
            continue
        if not image_area * 0.08 <= rectangle_area <= image_area * 0.78:
            continue
        blue_coverage = cv2.contourArea(contour) / rectangle_area
        if blue_coverage < 0.20:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, 0.025 * perimeter, True)
        boundary = polygon.reshape(4, 2) if len(polygon) == 4 else cv2.boxPoints(rectangle)
        candidates.append((rectangle_area * blue_coverage, boundary))
    if not candidates:
        return None
    _, box = max(candidates, key=lambda candidate: candidate[0])
    box = _refine_card_edges(resized, box.astype(np.float32), blue)
    return box / scale


def normalize_card(
    image: np.ndarray, side: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[str, ...], np.ndarray]:
    """Detect, perspective-correct, and orient a card image."""
    contour = _pokemon_back_contour(image) if side == "back" else None
    if contour is None and side == "front":
        color_candidates = [
            candidate
            for candidate in (
                _pokemon_front_contour(image),
                _pokemon_silver_front_contour(image),
            )
            if candidate is not None
        ]
        contour = (
            max(color_candidates, key=lambda candidate: cv2.contourArea(candidate))
            if color_candidates
            else None
        )
    if contour is None and side == "front":
        contour = _foreground_card_contour(image)
    if contour is None:
        contour = _card_contour(image)
    warnings = []
    if contour is None:
        height, width = image.shape[:2]
        ratio = min(height, width) / max(height, width)
        if not 0.62 <= ratio <= 0.78:
            raise CardImageError(
                "Could not find the card. Place one unsleeved card on a plain, "
                "contrasting background with all four corners visible."
            )
        card = image.copy()
        boundary = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        warnings.append("Card boundary was inferred from the image edges.")
    else:
        source = _order_points(contour)
        boundary = source.copy()
        destination = np.array(
            [[0, 0], [CARD_WIDTH - 1, 0], [CARD_WIDTH - 1, CARD_HEIGHT - 1], [0, CARD_HEIGHT - 1]],
            dtype=np.float32,
        )
        card = cv2.warpPerspective(
            image, cv2.getPerspectiveTransform(source, destination), (CARD_WIDTH, CARD_HEIGHT)
        )

    if card.shape[1] > card.shape[0]:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    card = cv2.resize(card, (CARD_WIDTH, CARD_HEIGHT), interpolation=cv2.INTER_AREA)
    return card, tuple(warnings), boundary


def _border_measurements(
    card: np.ndarray, side: Optional[str] = None
) -> Dict[str, object]:
    height, width = card.shape[:2]
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    y1, y2 = int(height * 0.08), int(height * 0.92)
    x1, x2 = int(width * 0.08), int(width * 0.92)
    x_profile = edges[y1:y2].mean(axis=0) / 255.0
    y_profile = edges[:, x1:x2].mean(axis=1) / 255.0

    def search_bounds(size: int, physical_size_mm: float) -> Tuple[int, int]:
        start = max(2, round(size * 0.8 / physical_size_mm))
        stop = min(size, round(size * MAX_BORDER_MM / physical_size_mm) + 1)
        return start, max(start + 1, stop)

    def guide(
        profile: np.ndarray, size: int, physical_size_mm: float
    ) -> int:
        # Pokémon's printable outer border is narrow. Restricting the search
        # prevents artwork frames, text rules, and the Poké Ball from being
        # mistaken for centering boundaries.
        start, stop = search_bounds(size, physical_size_mm)
        smoothed = np.convolve(profile, np.ones(5) / 5.0, mode="same")
        search = smoothed[start:stop]
        if search.size == 0 or float(search.max()) < 0.035:
            return int(size * 0.04)
        threshold = max(0.035, float(search.max()) * 0.45)
        sustained = np.convolve(
            (search >= threshold).astype(np.uint8), np.ones(3, dtype=np.uint8), mode="same"
        )
        matches = np.flatnonzero(sustained >= 2)
        return start + int(matches[0]) if matches.size else start + int(np.argmax(search))

    def full_span_guide(axis: int, reverse: bool) -> int:
        """Select a continuous frame edge instead of nearby text or artwork."""
        size = height if axis == 0 else width
        physical_size_mm = CARD_HEIGHT_MM if axis == 0 else CARD_WIDTH_MM
        span_start, span_stop = (x1, x2) if axis == 0 else (y1, y2)
        start, stop = search_bounds(size, physical_size_mm)
        scores = []
        for distance in range(start, stop):
            coordinate = size - 1 - distance if reverse else distance
            lower = max(0, coordinate - 2)
            upper = min(size, coordinate + 3)
            band = (
                edges[lower:upper, span_start:span_stop]
                if axis == 0
                else edges[span_start:span_stop, lower:upper].T
            )
            support = (band.max(axis=0) > 0).astype(np.float32)
            end_width = max(8, round(support.size * 0.16))
            end_support = min(
                float(support[:end_width].mean()),
                float(support[-end_width:].mean()),
            )
            scores.append(0.75 * end_support + 0.25 * float(support.mean()))
        if not scores:
            return int(size * 0.04)
        smoothed = np.convolve(np.asarray(scores), np.ones(3) / 3.0, mode="same")
        if float(smoothed.max()) < 0.04:
            profile = y_profile if axis == 0 else x_profile
            return guide(
                profile[::-1] if reverse else profile,
                size,
                physical_size_mm,
            )
        return start + int(np.argmax(smoothed))

    if side == "front":
        left = full_span_guide(axis=1, reverse=False)
        right_distance = full_span_guide(axis=1, reverse=True)
        top = full_span_guide(axis=0, reverse=False)
        bottom_distance = full_span_guide(axis=0, reverse=True)
    else:
        left = guide(x_profile, width, CARD_WIDTH_MM)
        right_distance = guide(x_profile[::-1], width, CARD_WIDTH_MM)
        top = guide(y_profile, height, CARD_HEIGHT_MM)
        bottom_distance = guide(y_profile[::-1], height, CARD_HEIGHT_MM)
    right = width - 1 - right_distance
    bottom = height - 1 - bottom_distance

    left_border = max(left, 1)
    right_border = max(width - 1 - right, 1)
    top_border = max(top, 1)
    bottom_border = max(height - 1 - bottom, 1)
    horizontal = min(left_border, right_border) / max(left_border, right_border)
    vertical = min(top_border, bottom_border) / max(top_border, bottom_border)
    horizontal_total = left_border + right_border
    vertical_total = top_border + bottom_border
    distance_mm = {
        "left": round(left_border / width * CARD_WIDTH_MM, 1),
        "right": round(right_border / width * CARD_WIDTH_MM, 1),
        "top": round(top_border / height * CARD_HEIGHT_MM, 1),
        "bottom": round(bottom_border / height * CARD_HEIGHT_MM, 1),
    }
    near_limit = any(value >= MAX_BORDER_MM - 0.2 for value in distance_mm.values())
    return {
        "balance_x": float(horizontal),
        "balance_y": float(vertical),
        "guides": {"left": left, "right": right, "top": top, "bottom": bottom},
        "distances": {
            "left": left_border,
            "right": right_border,
            "top": top_border,
            "bottom": bottom_border,
        },
        "distance_mm": distance_mm,
        "measurement_limit_mm": MAX_BORDER_MM,
        "retest_recommended": near_limit,
        "manual_override": False,
        "adjusted_distance_mm": None,
        "layout_adjustment": None,
        "card_dimensions": {"width": width, "height": height},
        "offset": {
            "horizontal": round((left_border - right_border) / 2.0, 1),
            "vertical": round((top_border - bottom_border) / 2.0, 1),
        },
        "left_percent": round(left_border / horizontal_total * 100),
        "right_percent": round(right_border / horizontal_total * 100),
        "top_percent": round(top_border / vertical_total * 100),
        "bottom_percent": round(bottom_border / vertical_total * 100),
    }


def _region_stats(
    card: np.ndarray, side: Optional[str] = None
) -> Tuple[Dict[str, float], Dict[str, object]]:
    height, width = card.shape[:2]
    strip = max(8, int(min(height, width) * 0.025))
    corner_fraction = 0.055 if side == "back" else 0.10
    corner = max(24, int(min(height, width) * corner_fraction))
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)

    if side == "back":
        hue, saturation, value = cv2.split(hsv)
        pale = (value > 105) & (
            (saturation < 105) | (hue < 90) | (hue > 140)
        )
    else:
        pale = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 185)

    edge_inset = max(3, round(min(height, width) * 0.006))
    edge_regions = {
        "top edge": pale[edge_inset : edge_inset + strip, corner:-corner],
        "right edge": pale[corner:-corner, -edge_inset - strip : -edge_inset],
        "bottom edge": pale[-edge_inset - strip : -edge_inset, corner:-corner],
        "left edge": pale[corner:-corner, edge_inset : edge_inset + strip],
    }
    edge_pale = {name: float(mask.mean()) for name, mask in edge_regions.items()}
    corner_masks = [
        pale[:corner, :corner],
        pale[:corner, -corner:],
        pale[-corner:, :corner],
        pale[-corner:, -corner:],
    ]
    corner_pale = [float(mask.mean()) for mask in corner_masks]

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    glare = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 245)
    dark = hsv[:, :, 2] < 25
    centering = _border_measurements(card, side)

    features = {
        "centering_x": float(centering["balance_x"]),
        "centering_y": float(centering["balance_y"]),
        "edge_pale": float(np.mean(list(edge_pale.values()))),
        "corner_pale_mean": float(np.mean(corner_pale)),
        "corner_pale_max": float(np.max(corner_pale)),
        "edge_defect_load": 0.0,
        "corner_defect_load": 0.0,
        "surface_glare": float(glare.mean()),
        "surface_dark": float(dark.mean()),
        "surface_damage": 0.0,
        "surface_assessed": 0.0,
        "sharpness": float(np.clip(laplacian.var() / 4000.0, 0.0, 1.0)),
        "exposure": float(gray.mean() / 255.0),
        "contrast": float(np.clip(gray.std() / 80.0, 0.0, 1.0)),
    }
    defects = []
    confirmed_edge_signals = {name: 0.0 for name in edge_regions}
    confirmed_corner_signals = {
        "top-left": 0.0,
        "top-right": 0.0,
        "bottom-left": 0.0,
        "bottom-right": 0.0,
    }
    edge_defect_count = 0
    corner_defect_count = 0
    edge_defect_weight = 0.0
    corner_defect_weight = 0.0
    surface_anomaly_count = 0
    surface_localized_inspection = False
    edge_baseline = float(np.median(list(edge_pale.values())))
    if side == "back":
        corner_radius = max(18, round(min(height, width) * 0.055))

        def fallback_card_mask() -> np.ndarray:
            mask = np.zeros((height, width), dtype=np.uint8)
            radius = corner_radius
            cv2.rectangle(mask, (radius, 0), (width - 1 - radius, height - 1), 255, -1)
            cv2.rectangle(mask, (0, radius), (width - 1, height - 1 - radius), 255, -1)
            for center in (
                (radius, radius),
                (width - 1 - radius, radius),
                (radius, height - 1 - radius),
                (width - 1 - radius, height - 1 - radius),
            ):
                cv2.circle(mask, center, radius, 255, -1)
            return mask

        # Recover the real rounded card silhouette from its blue outer ink.
        # Closing bridges small white chips so they remain inside the shape.
        blue_border = cv2.inRange(hsv, (90, 55, 25), (145, 255, 255))
        blue_border = cv2.morphologyEx(
            blue_border,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        )
        contours, _ = cv2.findContours(
            blue_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        card_mask = fallback_card_mask()
        if contours:
            candidate = max(contours, key=cv2.contourArea)
            if cv2.contourArea(candidate) >= width * height * 0.25:
                card_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(
                    card_mask, [cv2.convexHull(candidate)], -1, 255, -1
                )

        padded_mask = cv2.copyMakeBorder(
            card_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
        )
        distance = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[
            1:-1, 1:-1
        ]
        # Only damage touching the cut edge is edge whitening. White marks
        # farther inside the blue border are surface defects, not edge wear.
        cut_edge_depth = max(6, round(min(height, width) * 0.012))
        surface_depth = max(cut_edge_depth + 1, round(min(height, width) * 0.040))
        perimeter_mask = (distance > 0) & (distance <= cut_edge_depth)
        inner_border_mask = (distance > cut_edge_depth) & (
            distance <= surface_depth
        )

        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)
        local_saturation = cv2.GaussianBlur(saturation, (0, 0), 7.0)
        local_value = cv2.GaussianBlur(value, (0, 0), 7.0)
        localized_change = (
            ((local_saturation - saturation) > 18) & (value > local_value + 3)
        ) | (((value - local_value) > 25) & (saturation < 135))
        strong_white = (saturation < 55) & (value > 155)
        strong_local_white = (
            strong_white
            & (local_saturation > 80)
            & ((local_saturation - saturation) > 8)
        )
        localized = (
            (value > 75)
            & perimeter_mask
            & (localized_change | strong_local_white)
        ).astype(np.uint8) * 255
        localized = cv2.morphologyEx(
            localized, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(localized)
        components = []
        for index in range(1, component_count):
            x, y, box_width, box_height, area = stats[index]
            if area < 5 or area > width * height * 0.012:
                continue
            near_corner = (
                (x < corner_radius * 2 or x + box_width > width - corner_radius * 2)
                and (
                    y < corner_radius * 2
                    or y + box_height > height - corner_radius * 2
                )
            )
            size_limit = 0.20 if near_corner else 0.12
            if box_width > width * size_limit or box_height > height * size_limit:
                continue
            components.append((area, x, y, box_width, box_height))
        for area, x, y, box_width, box_height in sorted(
            components, reverse=True
        )[:60]:
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            nearest_edge = min(
                (
                    (center_y, "top edge"),
                    (width - center_x, "right edge"),
                    (height - center_y, "bottom edge"),
                    (center_x, "left edge"),
                ),
                key=lambda item: item[0],
            )[1]
            edge_length = width if nearest_edge in {"top edge", "bottom edge"} else height
            confirmed_edge_signals[nearest_edge] += area / max(1, strip * edge_length)
            edge_defect_count += 1
            edge_defect_weight += 1.0 if area >= 60 else 0.25
            horizontal_corner = "left" if center_x < width / 2 else "right"
            vertical_corner = "top" if center_y < height / 2 else "bottom"
            if min(center_x, width - center_x) < corner_radius and min(
                center_y, height - center_y
            ) < corner_radius:
                corner_name = f"{vertical_corner}-{horizontal_corner}"
                confirmed_corner_signals[corner_name] += area / max(
                    1, strip * corner_radius
                )
                corner_defect_count += 1
                corner_defect_weight += 1.0 if area >= 60 else 0.25
            padding = 5
            defects.append(
                {
                    "type": "Localized whitening",
                    "location": "back perimeter",
                    "severity": "high" if area >= 60 else "medium",
                    "evidence": f"{int(area)} highlighted edge pixels",
                    "bbox": (
                        int(max(0, x - padding)),
                        int(max(0, y - padding)),
                        int(min(width, x + box_width + padding)),
                        int(min(height, y + box_height + padding)),
                    ),
                }
            )

        surface_mask = (
            (value > 75)
            & inner_border_mask
            & (localized_change | strong_local_white)
            & (local_saturation > 70)
        ).astype(np.uint8) * 255
        surface_mask = cv2.morphologyEx(
            surface_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        surface_count, _, surface_stats, _ = cv2.connectedComponentsWithStats(
            surface_mask
        )
        surface_components = []
        for index in range(1, surface_count):
            x, y, box_width, box_height, area = surface_stats[index]
            short, long = sorted((box_width, box_height))
            if area < 5 or short <= 0 or long < 7 or long / short < 2.0:
                continue
            if box_width > width * 0.15 or box_height > height * 0.15:
                continue
            surface_components.append((area, x, y, box_width, box_height))
        for area, x, y, box_width, box_height in sorted(
            surface_components, reverse=True
        )[:20]:
            padding = 5
            defects.append(
                {
                    "type": "Surface scratch/print-line candidate",
                    "location": "back blue border",
                    "severity": "high" if area >= 60 else "medium",
                    "evidence": f"{int(area)} localized surface pixels",
                    "bbox": (
                        int(max(0, x - padding)),
                        int(max(0, y - padding)),
                        int(min(width, x + box_width + padding)),
                        int(min(height, y + box_height + padding)),
                    ),
                }
            )
        if surface_components:
            surface_anomaly_count = len(surface_components)
            surface_localized_inspection = True
            features["surface_assessed"] = 1.0
            features["surface_damage"] = float(
                np.clip(
                    sum(item[0] for item in surface_components)
                    / max(1.0, float(inner_border_mask.sum()) * 0.01),
                    0.0,
                    1.0,
                )
            )

    edge_boxes = {
        "top edge": (0, 0, width, strip),
        "right edge": (width - strip, 0, width, height),
        "bottom edge": (0, height - strip, width, height),
        "left edge": (0, 0, strip, height),
    }
    for name, amount in edge_pale.items():
        if side == "back":
            continue
        threshold = (
            max(0.16, edge_baseline * 1.35)
        )
        if amount > threshold:
            confirmed_edge_signals[name] = amount
            edge_defect_count += 1
            edge_defect_weight += 1.0 if amount > 0.40 else 0.25
            defects.append(
                {
                    "type": "Edge whitening signal",
                    "location": name,
                    "severity": "high" if amount > (0.08 if side == "back" else 0.40) else "medium",
                    "evidence": f"{round(amount * 100)}% pale pixels along edge",
                    "bbox": edge_boxes[name],
                }
            )

    confirmed_edges = list(confirmed_edge_signals.values())
    confirmed_corners = list(confirmed_corner_signals.values())
    features["edge_pale"] = float(np.clip(max(confirmed_edges), 0.0, 1.0))
    features["corner_pale_mean"] = float(
        np.clip(np.mean(confirmed_corners), 0.0, 1.0)
    )
    features["corner_pale_max"] = float(np.clip(max(confirmed_corners), 0.0, 1.0))
    features["edge_defect_load"] = float(
        np.clip(edge_defect_weight / 12.0, 0.0, 1.0)
    )
    features["corner_defect_load"] = float(
        np.clip(corner_defect_weight / 4.0, 0.0, 1.0)
    )

    diagnostics = {
        "centering": centering,
        "defects": defects,
        "condition_signals": {
            "corners": {
                name: round(amount * 100, 1)
                for name, amount in confirmed_corner_signals.items()
            },
            "edges": {
                name: round(amount * 100, 1)
                for name, amount in confirmed_edge_signals.items()
            },
            "defect_counts": {
                "edges": edge_defect_count,
                "corners": corner_defect_count,
                "edge_weight": round(edge_defect_weight, 2),
                "corner_weight": round(corner_defect_weight, 2),
            },
            "surface": {
                "glare_percent": round(float(glare.mean()) * 100, 1),
                "dark_percent": round(float(dark.mean()) * 100, 1),
                "sharpness_percent": round(features["sharpness"] * 100, 1),
                "reference_compared": False,
                "localized_border_inspection": surface_localized_inspection,
                "anomaly_count": surface_anomaly_count,
            },
        },
        "image_width": width,
        "image_height": height,
        "side": side,
    }
    return features, diagnostics


def apply_manual_centering(
    analysis: CardAnalysis, distances_mm: Dict[str, float]
) -> None:
    """Replace automatic inner guides with explicit user measurements."""
    required = {"left", "right", "top", "bottom"}
    if set(distances_mm) != required:
        raise CardImageError("Manual centering requires left, right, top, and bottom.")
    for name, value in distances_mm.items():
        if not 0.2 <= float(value) <= MAX_BORDER_MM:
            raise CardImageError(
                f"Manual {name} border must be between 0.2 and {MAX_BORDER_MM:.1f} mm."
            )

    centering = analysis.diagnostics["centering"]
    width = int(centering["card_dimensions"]["width"])
    height = int(centering["card_dimensions"]["height"])
    pixels = {
        "left": max(1, round(float(distances_mm["left"]) / CARD_WIDTH_MM * width)),
        "right": max(1, round(float(distances_mm["right"]) / CARD_WIDTH_MM * width)),
        "top": max(1, round(float(distances_mm["top"]) / CARD_HEIGHT_MM * height)),
        "bottom": max(1, round(float(distances_mm["bottom"]) / CARD_HEIGHT_MM * height)),
    }
    horizontal_total = pixels["left"] + pixels["right"]
    vertical_total = pixels["top"] + pixels["bottom"]
    balance_x = min(pixels["left"], pixels["right"]) / max(
        pixels["left"], pixels["right"]
    )
    balance_y = min(pixels["top"], pixels["bottom"]) / max(
        pixels["top"], pixels["bottom"]
    )
    centering.update(
        {
            "balance_x": float(balance_x),
            "balance_y": float(balance_y),
            "guides": {
                "left": pixels["left"],
                "right": width - 1 - pixels["right"],
                "top": pixels["top"],
                "bottom": height - 1 - pixels["bottom"],
            },
            "distances": pixels,
            "distance_mm": {
                name: round(float(value), 1) for name, value in distances_mm.items()
            },
            "offset": {
                "horizontal": round((pixels["left"] - pixels["right"]) / 2.0, 1),
                "vertical": round((pixels["top"] - pixels["bottom"]) / 2.0, 1),
            },
            "left_percent": round(pixels["left"] / horizontal_total * 100),
            "right_percent": round(pixels["right"] / horizontal_total * 100),
            "top_percent": round(pixels["top"] / vertical_total * 100),
            "bottom_percent": round(pixels["bottom"] / vertical_total * 100),
            "manual_override": True,
            "reference_calibrated": False,
            "retest_recommended": False,
        }
    )
    analysis.features["centering_x"] = float(balance_x)
    analysis.features["centering_y"] = float(balance_y)


def annotated_image(analysis: CardAnalysis) -> np.ndarray:
    """Render centering guides and suspected defect boxes on a card image."""
    margin = 90
    output = np.full(
        (CARD_HEIGHT + margin * 2, CARD_WIDTH + margin * 2, 3),
        (238, 235, 228),
        dtype=np.uint8,
    )
    output[margin : margin + CARD_HEIGHT, margin : margin + CARD_WIDTH] = analysis.image
    overlay = output.copy()
    guides = analysis.diagnostics["centering"]["guides"]
    green = (40, 255, 80)
    black = (0, 0, 0)
    red = (65, 75, 240)
    cyan = (255, 220, 40)
    labeled_defect_types = set()

    for defect in analysis.diagnostics["defects"]:
        x1, y1, x2, y2 = defect["bbox"]
        x1, x2 = x1 + margin, x2 + margin
        y1, y2 = y1 + margin, y2 + margin
        cv2.rectangle(overlay, (x1, y1), (x2, y2), red, -1)
        cv2.rectangle(output, (x1, y1), (x2, y2), red, 4)
        if defect["type"] not in labeled_defect_types:
            label_y = max(28, y1 - 8)
            cv2.putText(
                output,
                defect["type"],
                (max(4, x1), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                defect["type"],
                (max(4, x1), label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                red,
                2,
                cv2.LINE_AA,
            )
            labeled_defect_types.add(defect["type"])
    rendered = cv2.addWeighted(overlay, 0.22, output, 0.78, 0)
    outer_left, outer_top = margin, margin
    outer_right = margin + CARD_WIDTH - 1
    outer_bottom = margin + CARD_HEIGHT - 1
    inner_left = margin + guides["left"]
    inner_right = margin + guides["right"]
    inner_top = margin + guides["top"]
    inner_bottom = margin + guides["bottom"]

    # White/black outer rectangle is the physical card edge. Green is the
    # detected meeting point between the dark-blue border and inner design.
    cv2.rectangle(
        rendered, (outer_left, outer_top), (outer_right, outer_bottom), black, 5
    )
    cv2.rectangle(
        rendered, (outer_left, outer_top), (outer_right, outer_bottom), cyan, 3
    )
    cv2.rectangle(
        rendered, (inner_left, inner_top), (inner_right, inner_bottom), black, 5
    )
    cv2.rectangle(
        rendered, (inner_left, inner_top), (inner_right, inner_bottom), green, 3
    )

    distance_mm = analysis.diagnostics["centering"]["distance_mm"]
    adjusted_distance_mm = analysis.diagnostics["centering"].get(
        "adjusted_distance_mm"
    )
    center_x = margin + CARD_WIDTH // 2
    center_y = margin + CARD_HEIGHT // 2
    segments = (
        ((outer_left, center_y), (inner_left, center_y)),
        ((inner_right, center_y), (outer_right, center_y)),
        ((center_x, outer_top), (center_x, inner_top)),
        ((center_x, inner_bottom), (center_x, outer_bottom)),
    )
    for start, end in segments:
        cv2.line(rendered, start, end, black, 5)
        cv2.line(rendered, start, end, cyan, 3)
        cv2.circle(rendered, start, 5, cyan, -1)
        cv2.circle(rendered, end, 5, cyan, -1)

    labels = (
        (f"L {distance_mm['left']:.1f} mm", (8, center_y + 8)),
        (
            f"R {distance_mm['right']:.1f} mm",
            (CARD_WIDTH + margin * 2 - 155, center_y + 8),
        ),
        (f"T {distance_mm['top']:.1f} mm", (center_x - 65, 48)),
        (
            (
                f"B {distance_mm['bottom']:.1f} mm"
                + (
                    f" ({adjusted_distance_mm['bottom']:.1f} adjusted)"
                    if adjusted_distance_mm
                    else ""
                )
            ),
            (center_x - 65, margin + CARD_HEIGHT + 52),
        ),
    )
    for text, origin in labels:
        cv2.putText(
            rendered,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            black,
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            cyan,
            2,
            cv2.LINE_AA,
        )
    return rendered


def source_boundary_image(analysis: CardAnalysis) -> np.ndarray:
    """Show the uncropped upload and the physical boundary used for warping."""
    padding = 55
    source = analysis.source_image
    canvas = cv2.copyMakeBorder(
        source,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(238, 235, 228),
    )
    boundary = np.rint(analysis.source_boundary + padding).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [boundary], True, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.polylines(canvas, [boundary], True, (40, 255, 80), 4, cv2.LINE_AA)
    for point in boundary.reshape(-1, 2):
        cv2.circle(canvas, tuple(point), 8, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(point), 5, (40, 255, 80), -1, cv2.LINE_AA)
    maximum = max(canvas.shape[:2])
    if maximum > 1400:
        scale = 1400.0 / maximum
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return canvas


def analyze_image(data: bytes, side: Optional[str] = None) -> CardAnalysis:
    source_image = decode_image(data)
    card, warnings, source_boundary = normalize_card(source_image, side)
    features, diagnostics = _region_stats(card, side)
    quality_warnings = list(warnings)
    if min(source_image.shape[:2]) < 1000:
        quality_warnings.append(
            "Image resolution is below the recommended 1000 pixels on the "
            "short edge; fine whitening and surface defects may be missed."
        )
    if features["sharpness"] < 0.08:
        quality_warnings.append("The image is blurry; use a tripod or brighter light.")
    if features["surface_glare"] > 0.08:
        quality_warnings.append("Strong glare may hide scratches or surface wear.")
    if not 0.20 <= features["exposure"] <= 0.85:
        quality_warnings.append("The image exposure is too dark or too bright.")
    if diagnostics["centering"].get("retest_recommended"):
        quality_warnings.append(
            "A centering edge was near the 5 mm detection limit; retake the "
            "photo on a contrasting background before relying on centering."
        )
    return CardAnalysis(
        card,
        features,
        tuple(quality_warnings),
        diagnostics,
        source_image,
        source_boundary,
    )
