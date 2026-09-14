"""OpenCV webcam capture with monotonic timestamps and effective-FPS tracking."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    frame_index: int
    timestamp_ms: float
    image_bgr: np.ndarray


class Camera:
    def __init__(self, index: int, width: int, height: int, fps: int):
        self.index = index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._start_time: float | None = None
        self._recent_timestamps: deque[float] = deque(maxlen=30)

    def open(self) -> None:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
        self._cap = cap
        self._start_time = time.perf_counter_ns()

    @property
    def actual_width(self) -> int:
        assert self._cap is not None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        assert self._cap is not None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> Frame | None:
        assert self._cap is not None and self._start_time is not None
        ok, image = self._cap.read()
        if not ok or image is None:
            return None
        now = time.perf_counter_ns()
        timestamp_ms = (now - self._start_time) / 1e6
        frame = Frame(frame_index=self._frame_index, timestamp_ms=timestamp_ms, image_bgr=image)
        self._frame_index += 1
        self._recent_timestamps.append(timestamp_ms)
        return frame

    @property
    def effective_fps(self) -> float:
        """FPS measured from actual captured-frame timestamps, not the
        camera-reported value (which is often inaccurate)."""
        if len(self._recent_timestamps) < 2:
            return 0.0
        span_ms = self._recent_timestamps[-1] - self._recent_timestamps[0]
        if span_ms <= 0:
            return 0.0
        return (len(self._recent_timestamps) - 1) / (span_ms / 1000.0)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
