"""
LEGO piece detection using SAM2 (Segment Anything Model) or YOLO.

SAM2 is used as the default — it segments any object in the image without
needing LEGO-specific training. YOLO is kept as a fallback.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MAX_SAM_SIZE = 640  # Resize images to this max dimension before SAM inference
DEFAULT_SAM_MODEL = "sam2.1_t.pt"  # Tiny model — fastest on CPU


@dataclass
class Detection:
    """A single detected object in the image."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    confidence: float
    crop: np.ndarray  # RGB uint8 array of the cropped region


# ---------------------------------------------------------------------------
# Thread-safe model cache
# ---------------------------------------------------------------------------
_models: dict[str, object] = {}
_model_lock = threading.Lock()


def _get_model(model_path: str):
    """Load a model once per path and cache it."""
    if model_path not in _models:
        with _model_lock:
            if model_path not in _models:
                if "sam" in model_path.lower():
                    from ultralytics import SAM
                    model = SAM(model_path)
                    # Use float16 for faster CPU inference
                    try:
                        model.model.half()
                    except Exception:
                        pass
                    _models[model_path] = model
                else:
                    from ultralytics import YOLO
                    _models[model_path] = YOLO(model_path)
    return _models[model_path]


def preload_sam(model_path: str = DEFAULT_SAM_MODEL):
    """Preload SAM model (call at startup to avoid first-request delay)."""
    _get_model(model_path)


def _resize_for_sam(image: Image.Image, max_size: int = MAX_SAM_SIZE) -> tuple[Image.Image, float]:
    """Resize image so longest side is max_size. Returns (resized_image, scale_factor)."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image, 1.0
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS), scale


def detect_objects_sam(
    image: Image.Image | str | Path,
    model_path: str = DEFAULT_SAM_MODEL,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.85,
    min_dimension: int = 15,
    max_detections: int = 50,
) -> list[Detection]:
    """
    Detect objects using SAM2 (Segment Anything).

    SAM segments every object in the image. We filter by size to keep
    only brick-sized segments and discard background/noise.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Keep original for cropping at full resolution
    orig_array = np.array(image)
    orig_h, orig_w = orig_array.shape[:2]

    # Resize for faster SAM inference
    resized, scale = _resize_for_sam(image)
    sam_array = np.array(resized)
    sam_h, sam_w = sam_array.shape[:2]
    sam_area = sam_h * sam_w

    model = _get_model(model_path)
    results = model.predict(source=sam_array, verbose=False)

    detections: list[Detection] = []
    if not results or results[0].masks is None:
        return detections

    r = results[0]
    masks = r.masks.data.cpu().numpy()
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)

    scored = []
    for i in range(len(masks)):
        mask_area = float(masks[i].sum())
        area_ratio = mask_area / sam_area

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        # Boxes are in resized coordinates
        x1, y1, x2, y2 = boxes[i]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(sam_w, x2), min(sam_h, y2)

        bw = x2 - x1
        bh = y2 - y1

        if bw < min_dimension or bh < min_dimension:
            continue

        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 6.0:
            continue

        scored.append((area_ratio, i, x1, y1, x2, y2))

    scored.sort(key=lambda s: -s[0])

    for _, i, x1, y1, x2, y2 in scored[:max_detections]:
        # Scale boxes back to original image coordinates for full-res crops
        ox1 = int(x1 / scale)
        oy1 = int(y1 / scale)
        ox2 = int(x2 / scale)
        oy2 = int(y2 / scale)
        ox1, oy1 = max(0, ox1), max(0, oy1)
        ox2, oy2 = min(orig_w, ox2), min(orig_h, oy2)

        crop = orig_array[oy1:oy2, ox1:ox2].copy()
        conf = min(1.0, float(masks[i].sum()) / sam_area * 10)
        detections.append(Detection(
            bbox=(ox1, oy1, ox2, oy2),
            confidence=conf,
            crop=crop,
        ))

    return detections


def detect_objects(
    image: Image.Image | str | Path,
    conf_threshold: float = 0.25,
    max_detections: int = 50,
    model_path: str = "yolov8m.pt",
) -> list[Detection]:
    """
    Detect objects using YOLO (legacy fallback).
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    img_array = np.array(image)

    from ultralytics import YOLO
    model = _get_model(model_path)

    results = model.predict(
        source=img_array,
        conf=conf_threshold,
        max_det=max_detections,
        verbose=False,
    )

    detections: list[Detection] = []
    if not results or results[0].boxes is None:
        return detections

    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    sorted_indices = np.argsort(-confs)

    h, w = img_array.shape[:2]
    for idx in sorted_indices[:max_detections]:
        x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().astype(int)
        conf = float(confs[idx])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        crop = img_array[y1:y2, x1:x2].copy()
        detections.append(Detection(
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=conf,
            crop=crop,
        ))

    return detections
