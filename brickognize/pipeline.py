"""
LEGO detection + identification pipeline using YOLO + Brickognize API.

Simple two-stage approach:
1. YOLO detects bounding boxes for LEGO pieces
2. Each crop is sent to Brickognize API for part identification

No local embeddings, no CLIP/DINOv2 — just YOLO + API calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from brickognize.api import BrickResult, identify


@dataclass
class Detection:
    """A detected piece with its API identification results."""

    detection_id: str
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    results: list[BrickResult]


@dataclass
class GroupedPart:
    """A unique part with count of how many times it was detected."""

    part_id: str
    name: str
    count: int
    best_score: float
    image_url: str
    bricklink_url: str
    detection_ids: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Complete output of a single image analysis."""

    detections: list[Detection]
    grouped_parts: list[GroupedPart]
    total_pieces: int
    unique_parts: int
    processing_time_ms: float
    image_width: int
    image_height: int


def _iou(a: tuple, b: tuple) -> float:
    """Intersection over union of two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _containment(inner: tuple, outer: tuple) -> float:
    """Fraction of inner box area that lies inside outer box."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_inner = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area_inner if area_inner > 0 else 0.0


def _merge_overlapping_detections(
    detections: list,
    img_array: np.ndarray,
    iou_threshold: float = 0.3,
    containment_threshold: float = 0.7,
    proximity_ratio: float = 0.0,
) -> list:
    """
    Merge overlapping, contained, or closely adjacent detections.

    Fixes YOLO splitting a single brick into multiple sub-detections
    (e.g. detecting individual studs or halves of one brick).
    """
    from ml.detector import Detection as YoloDet

    if len(detections) <= 1:
        return detections

    h, w = img_array.shape[:2]
    boxes = [list(d.bbox) for d in detections]
    confs = [d.confidence for d in detections]
    alive = [True] * len(detections)

    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(boxes)):
                if not alive[j]:
                    continue

                bi = tuple(boxes[i])
                bj = tuple(boxes[j])

                should_merge = False

                # Overlapping boxes
                if _iou(bi, bj) > iou_threshold:
                    should_merge = True

                # One box mostly inside the other
                if not should_merge:
                    if (_containment(bj, bi) > containment_threshold or
                            _containment(bi, bj) > containment_threshold):
                        should_merge = True

                # Close proximity (nearby fragments of same brick)
                if not should_merge and proximity_ratio > 0:
                    avg_w = ((bi[2]-bi[0]) + (bj[2]-bj[0])) / 2
                    avg_h = ((bi[3]-bi[1]) + (bj[3]-bj[1])) / 2
                    avg_size = (avg_w + avg_h) / 2

                    gap_x = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
                    gap_y = max(0, max(bi[1], bj[1]) - min(bi[3], bj[3]))
                    gap = max(gap_x, gap_y)

                    if avg_size > 0 and gap / avg_size < proximity_ratio:
                        should_merge = True

                if should_merge:
                    boxes[i] = [
                        min(boxes[i][0], boxes[j][0]),
                        min(boxes[i][1], boxes[j][1]),
                        max(boxes[i][2], boxes[j][2]),
                        max(boxes[i][3], boxes[j][3]),
                    ]
                    confs[i] = max(confs[i], confs[j])
                    alive[j] = False
                    changed = True

    result = []
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        x1, y1, x2, y2 = boxes[i]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img_array[y1:y2, x1:x2].copy()
        result.append(YoloDet(
            bbox=(x1, y1, x2, y2),
            confidence=confs[i],
            crop=crop,
        ))

    return result


def _find_sam_model(project_root: Path) -> Optional[str]:
    """Find SAM model locally, or return name for auto-download."""
    for name in ("sam2.1_b.pt", "sam2.1_l.pt", "sam_b.pt"):
        path = project_root / name
        if path.exists():
            return str(path)
    # On cloud (HF Spaces etc.), ultralytics auto-downloads by name
    import os
    if os.environ.get("HF_SPACES") or os.environ.get("SPACE_ID"):
        return "sam2.1_b.pt"
    return None


def _find_yolo_model(project_root: Path) -> str:
    """Find the best available YOLO model (fallback if no SAM)."""
    for name in ("lego_yolo11.pt", "lego_yolov8.pt", "lego_yolov8_v2.pt", "yolov8m.pt"):
        path = project_root / name
        if path.exists():
            return str(path)
    return "yolov8m.pt"


def analyze_image(
    image: Image.Image | str | Path,
    yolo_model_path: Optional[str] = None,
    conf_threshold: float = 0.10,
    max_detections: int = 50,
    top_k: int = 5,
) -> PipelineResult:
    """
    Run the full detection -> Brickognize identification pipeline.

    Uses SAM2 for detection if available, falls back to YOLO.

    Parameters
    ----------
    image : PIL Image or path
        Input image containing LEGO pieces.
    yolo_model_path : str, optional
        Path to YOLO weights (used as fallback if SAM not available).
    conf_threshold : float
        YOLO detection confidence threshold (ignored when using SAM).
    max_detections : int
        Max number of detections to process.
    top_k : int
        Number of Brickognize results to keep per detection.

    Returns
    -------
    PipelineResult
    """
    start_time = time.perf_counter()

    # Load image
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    img_w, img_h = image.size
    img_array = np.array(image)

    # --- Step 1: Detection (SAM preferred, YOLO fallback) ---
    project_root = Path(__file__).resolve().parent.parent
    sam_path = _find_sam_model(project_root)

    if sam_path:
        from ml.detector import detect_objects_sam
        raw_detections = detect_objects_sam(
            image=image,
            model_path=sam_path,
            max_detections=max_detections,
        )
    else:
        from ml.detector import detect_objects
        if yolo_model_path is None:
            yolo_model_path = _find_yolo_model(project_root)
        raw_detections = detect_objects(
            image=image,
            conf_threshold=conf_threshold,
            max_detections=max_detections,
            model_path=yolo_model_path,
        )

    # Fallback: if YOLO finds nothing, treat whole image as one detection
    if not raw_detections:
        from ml.detector import Detection as YoloDetection
        raw_detections = [YoloDetection(
            bbox=(0, 0, img_w, img_h),
            confidence=1.0,
            crop=img_array,
        )]

    # Filter noise: remove tiny detections (< 1.5% of image) and thin slivers
    img_area = img_w * img_h
    min_det_area = img_area * 0.015
    filtered = []
    for det in raw_detections:
        dw = det.bbox[2] - det.bbox[0]
        dh = det.bbox[3] - det.bbox[1]
        det_area = dw * dh
        det_aspect = max(dw, dh) / max(min(dw, dh), 1)
        if det_area >= min_det_area and det_aspect < 5.0:
            filtered.append(det)
    if filtered:
        raw_detections = filtered

    # --- Step 1b: Merge overlapping/adjacent detections ---
    raw_detections = _merge_overlapping_detections(raw_detections, img_array)

    # --- Step 2: Identify each crop via Brickognize API ---
    detections: list[Detection] = []

    for i, det in enumerate(raw_detections):
        crop_img = Image.fromarray(det.crop)

        try:
            results = identify(crop_img)
        except Exception as e:
            print(f"  Brickognize API error for det_{i}: {e}")
            results = []

        detections.append(Detection(
            detection_id=f"det_{i}",
            bbox=det.bbox,
            detection_confidence=det.confidence,
            results=results[:top_k],
        ))

    # --- Step 2b: Fallback — if all scores are low, try the full image ---
    best_score = 0.0
    for det in detections:
        if det.results:
            best_score = max(best_score, det.results[0].score)

    if best_score < 0.65:
        try:
            full_results = identify(image)
            if full_results and full_results[0].score > best_score:
                detections = [Detection(
                    detection_id="det_full",
                    bbox=(0, 0, img_w, img_h),
                    detection_confidence=1.0,
                    results=full_results[:top_k],
                )]
        except Exception as e:
            print(f"  Brickognize full-image fallback error: {e}")

    # --- Step 3: Group by part_id ---
    groups: dict[str, GroupedPart] = {}

    for det in detections:
        if not det.results:
            continue
        top = det.results[0]
        if not top.part_id:
            continue

        if top.part_id in groups:
            g = groups[top.part_id]
            g.count += 1
            g.detection_ids.append(det.detection_id)
            if top.score > g.best_score:
                g.best_score = top.score
        else:
            groups[top.part_id] = GroupedPart(
                part_id=top.part_id,
                name=top.name,
                count=1,
                best_score=top.score,
                image_url=top.image_url,
                bricklink_url=top.bricklink_url,
                detection_ids=[det.detection_id],
            )

    grouped = sorted(groups.values(), key=lambda g: (-g.count, g.name))
    elapsed = (time.perf_counter() - start_time) * 1000

    return PipelineResult(
        detections=detections,
        grouped_parts=grouped,
        total_pieces=sum(g.count for g in grouped),
        unique_parts=len(grouped),
        processing_time_ms=elapsed,
        image_width=img_w,
        image_height=img_h,
    )
