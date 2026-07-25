"""Pretrained MobileNet artwork embeddings for visual card matching."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Optional, Sequence

import cv2
import httpx
import numpy as np


MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/mobilenetv2-12/"
    "resolve/main/mobilenetv2-12.onnx"
)
MODEL_SHA256 = "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
DEFAULT_MODEL_PATH = Path("data/grading/models/mobilenetv2-12.onnx")
_MODEL_LOCK = threading.Lock()
_MODEL = None


def _model_path() -> Path:
    return Path(os.getenv("CARD_EMBEDDING_MODEL", str(DEFAULT_MODEL_PATH)))


def _valid_model(path: Path) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == MODEL_SHA256


def _download_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    digest = hashlib.sha256()
    size = 0
    with httpx.stream(
        "GET",
        MODEL_URL,
        follow_redirects=True,
        timeout=90.0,
        headers={"User-Agent": "CardLens/0.1 (Apache-2.0 MobileNet model)"},
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > 20 * 1024 * 1024:
                    raise ValueError("Embedding model download exceeded 20 MB.")
                digest.update(chunk)
                handle.write(chunk)
    if digest.hexdigest() != MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError("Embedding model checksum did not match.")
    temporary.replace(path)


def _load_model(allow_download: bool):
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        path = _model_path()
        if not _valid_model(path):
            if not allow_download:
                return None
            _download_model(path)
        _MODEL = cv2.dnn.readNetFromONNX(str(path))
        return _MODEL


def visual_embedding(
    image: np.ndarray, allow_download: bool = False
) -> Optional[np.ndarray]:
    """Return a normalized learned visual descriptor, or None if unavailable."""
    try:
        model = _load_model(allow_download)
    except (cv2.error, httpx.HTTPError, OSError, ValueError):
        return None
    if model is None:
        return None
    rgb = cv2.cvtColor(
        cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    blob = np.transpose(rgb, (2, 0, 1))[None, ...]
    try:
        with _MODEL_LOCK:
            model.setInput(blob)
            output = model.forward().reshape(-1).astype(np.float32)
    except cv2.error:
        return None
    norm = float(np.linalg.norm(output))
    return output / norm if norm > 0 else None


def cosine_similarity(
    first: Sequence[float], second: Sequence[float]
) -> float:
    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    if left.shape != right.shape or left.size == 0:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
