"""MediaPipe Face Landmarker wrapper.

A0 uses the current MediaPipe Tasks `FaceLandmarker` API (the legacy
`mediapipe.solutions.face_mesh` API is not present in the installed
MediaPipe version). The model bundle ships in `a0/assets/face_landmarker.task`
so the experiment runs without a network dependency after setup.

Output is 478 normalized landmarks per face: 468 face-mesh points plus 10
iris points (indices 468-477), matching the legacy face-mesh topology.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSET_MODEL_PATH = Path(__file__).parent / "assets" / "face_landmarker.task"

# Stable landmark indices (face-mesh topology), documented per feature use.
NOSE_TIP = 1
CHIN = 152
MOUTH_LEFT = 291   # subject's left (image right)
MOUTH_RIGHT = 61   # subject's right (image left)

# Subject's left eye (image right side of frame).
LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
LEFT_IRIS_CENTER = 473

# Subject's right eye (image left side of frame).
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_IRIS_CENTER = 468

HEAD_POSE_ANCHOR_INDICES = [
    NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, MOUTH_LEFT, MOUTH_RIGHT,
]


@dataclass
class LandmarkResult:
    face_detected: bool
    landmarks_norm: np.ndarray | None  # (478, 3) normalized x, y, z


class FaceLandmarkerWrapper:
    def __init__(self, model_path: Path = ASSET_MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face landmarker model not found at {model_path}. "
                "Run scripts/download_model.py or see README setup."
            )

        options = vision.FaceLandmarkerOptions(
            # CPU delegate: the GPU/Metal delegate crashes in headless/sandboxed
            # environments without a display-linked Metal service.
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def detect(self, image_bgr: np.ndarray, timestamp_ms: float) -> LandmarkResult:
        import cv2

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        if not result.face_landmarks:
            return LandmarkResult(face_detected=False, landmarks_norm=None)

        face = result.face_landmarks[0]
        arr = np.array([[p.x, p.y, p.z] for p in face], dtype=np.float64)
        return LandmarkResult(face_detected=True, landmarks_norm=arr)

    def close(self) -> None:
        self._landmarker.close()
