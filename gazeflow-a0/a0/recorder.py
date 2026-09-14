"""Incremental writers for raw features, selected landmarks, and debug video.

Data durability (section 41) matters more than code quality here: rows are
flushed at least once per second so an interrupted run still leaves usable
data on disk.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from . import landmarks as lm

RAW_FEATURE_COLUMNS = [
    "frame_index", "timestamp_ms", "phase", "point_id", "target_x_norm", "target_y_norm",
    "face_detected", "sample_valid",
    "left_iris_x_rel", "left_iris_y_rel", "right_iris_x_rel", "right_iris_y_rel",
    "left_eye_width_norm", "left_eye_height_norm", "right_eye_width_norm", "right_eye_height_norm",
    "left_ear", "right_ear", "inter_eye_distance_norm",
    "face_center_x_norm", "face_center_y_norm", "face_scale",
    "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
    "prediction_x_norm", "prediction_y_norm",
]

# Landmark indices preserved in a0_landmarks.jsonl (section 18): enough to
# reconstruct the eye/iris/face-scale geometry offline without every point.
_SELECTED_LANDMARK_INDICES = {
    "nose": lm.NOSE_TIP,
    "chin": lm.CHIN,
    "left_eye_outer": lm.LEFT_EYE_OUTER,
    "left_eye_inner": lm.LEFT_EYE_INNER,
    "right_eye_outer": lm.RIGHT_EYE_OUTER,
    "right_eye_inner": lm.RIGHT_EYE_INNER,
    "left_eye_top": lm.LEFT_EYE_TOP,
    "left_eye_bottom": lm.LEFT_EYE_BOTTOM,
    "right_eye_top": lm.RIGHT_EYE_TOP,
    "right_eye_bottom": lm.RIGHT_EYE_BOTTOM,
    "left_iris": lm.LEFT_IRIS_CENTER,
    "right_iris": lm.RIGHT_IRIS_CENTER,
    "mouth_left": lm.MOUTH_LEFT,
    "mouth_right": lm.MOUTH_RIGHT,
}


class RawFeatureWriter:
    def __init__(self, path: Path, flush_interval_s: float = 1.0):
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=RAW_FEATURE_COLUMNS)
        self._writer.writeheader()
        self._flush_interval_s = flush_interval_s
        self._last_flush = time.monotonic()

    def write(self, row: dict) -> None:
        clean = {k: row.get(k, "") for k in RAW_FEATURE_COLUMNS}
        self._writer.writerow(clean)
        if time.monotonic() - self._last_flush >= self._flush_interval_s:
            self._file.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        self._file.flush()
        self._file.close()


class LandmarksWriter:
    def __init__(self, path: Path, flush_interval_s: float = 1.0):
        self._file = open(path, "w")
        self._flush_interval_s = flush_interval_s
        self._last_flush = time.monotonic()

    def write(self, frame_index: int, timestamp_ms: float, phase: str, point_id: str,
              target: tuple[float, float] | None, landmarks_norm: np.ndarray | None) -> None:
        selected = {}
        if landmarks_norm is not None:
            for name, idx in _SELECTED_LANDMARK_INDICES.items():
                p = landmarks_norm[idx]
                selected[name] = [round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5)]
        record = {
            "frame_index": frame_index,
            "timestamp_ms": round(timestamp_ms, 2),
            "phase": phase,
            "point_id": point_id,
            "target": list(target) if target is not None else None,
            "landmarks": selected,
        }
        self._file.write(json.dumps(record) + "\n")
        if time.monotonic() - self._last_flush >= self._flush_interval_s:
            self._file.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        self._file.flush()
        self._file.close()


class DebugVideoWriter:
    """Writes a0_capture.mp4, frame-aligned with the raw feature CSV via
    frame_index/timestamp_ms. Disabled by default in production GazeFlow —
    see README; A0 enables it because the developer is the initial participant."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), (width, height))

    def write(self, image_bgr: np.ndarray) -> None:
        self._writer.write(image_bgr)

    def close(self) -> None:
        self._writer.release()
