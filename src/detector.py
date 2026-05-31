"""
detector.py - YOLOv8-based vehicle detection for a single traffic feed.

Handles both image and video inputs.
For videos, samples up to max_frames evenly across the clip and
returns the average per-frame vehicle count (representative of
the intersection state, not a cumulative total).
"""

import os
import cv2
from collections import defaultdict

from src.utils import normalize_class_name
from src.pcu_logic import calculate_pcu_score

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Supported file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def analyze_feed(
    source_path: str,
    direction_name: str,
    model_path: str,
    conf: float = 0.35,
    max_frames: int = 60,
) -> dict:
    """
    Run YOLOv8 detection on one traffic feed (image or video).

    Args:
        source_path:    path to image or video file
        direction_name: "North", "South", "East", or "West"
        model_path:     path to trained best.pt weights
        conf:           detection confidence threshold (0.0 - 1.0)
        max_frames:     max frames to sample from a video feed

    Returns:
        dict with keys:
            direction         - direction name string
            raw_counts        - {class_name: count} (avg per frame for video)
            total_vehicles    - sum of all vehicle counts
            frames_analyzed   - number of frames processed
            detected_emergency - True if any EMV was detected
            pcu_score         - PCU-weighted traffic load
            source_type       - "image" or "video"

    Raises:
        RuntimeError: if ultralytics is not installed
        FileNotFoundError: if model or source file does not exist
    """
    if not YOLO_AVAILABLE:
        raise RuntimeError(
            "ultralytics is not installed. Run: pip install ultralytics"
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Feed file not found: {source_path}")

    model = YOLO(model_path)

    ext = os.path.splitext(source_path)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS

    if not is_video:
        raw_counts = _process_image(model, source_path, conf)
        frames_analyzed = 1
    else:
        raw_counts, frames_analyzed = _process_video(model, source_path, conf, max_frames)

    total_vehicles = sum(raw_counts.values())
    detected_emergency = raw_counts.get("emv", 0) > 0
    pcu_score = calculate_pcu_score(raw_counts)

    return {
        "direction": direction_name,
        "raw_counts": raw_counts,
        "total_vehicles": total_vehicles,
        "frames_analyzed": frames_analyzed,
        "detected_emergency": detected_emergency,
        "pcu_score": pcu_score,
        "source_type": "video" if is_video else "image",
    }


def _process_image(model, image_path: str, conf: float) -> dict:
    """Run detection on a single image and return class counts."""
    counts = defaultdict(int)
    results = model.predict(image_path, conf=conf, verbose=False)
    for r in results:
        for cls_id in r.boxes.cls.tolist():
            cls_name = normalize_class_name(model.names[int(cls_id)])
            counts[cls_name] += 1
    return dict(counts)


def _process_video(model, video_path: str, conf: float, max_frames: int) -> tuple:
    """
    Sample frames evenly from a video and return average per-frame counts.

    Returns:
        (avg_counts_dict, frames_analyzed_int)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate frame step so we sample evenly across the whole video
    step = max(1, total_frames // max_frames)

    per_frame_counts = []
    frame_idx = 0
    frames_analyzed = 0

    while cap.isOpened() and frames_analyzed < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame_counts = defaultdict(int)
        results = model.predict(frame, conf=conf, verbose=False)
        for r in results:
            for cls_id in r.boxes.cls.tolist():
                cls_name = normalize_class_name(model.names[int(cls_id)])
                frame_counts[cls_name] += 1

        per_frame_counts.append(dict(frame_counts))
        frame_idx += step
        frames_analyzed += 1

    cap.release()

    if frames_analyzed == 0:
        return {}, 0

    # Average counts across all sampled frames
    all_classes = set()
    for fc in per_frame_counts:
        all_classes.update(fc.keys())

    avg_counts = {}
    for cls in all_classes:
        avg = sum(fc.get(cls, 0) for fc in per_frame_counts) / frames_analyzed
        avg_counts[cls] = max(1, round(avg))  # at least 1 if ever seen

    return avg_counts, frames_analyzed
