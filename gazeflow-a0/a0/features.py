"""Per-frame feature vector extraction from MediaPipe landmarks.

Feature column order is fixed and must not change without updating every
consumer (recorder, calibration, model, validation). This is the single
source of truth for that order.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from . import landmarks as lm
from .geometry import HeadPoseEstimator

FEATURE_NAMES: list[str] = [
    "left_iris_x_rel", "left_iris_y_rel",
    "right_iris_x_rel", "right_iris_y_rel",
    "left_eye_width_norm", "left_eye_height_norm",
    "right_eye_width_norm", "right_eye_height_norm",
    "left_ear", "right_ear",
    "inter_eye_distance_norm",
    "face_center_x_norm", "face_center_y_norm", "face_scale",
    "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
]

# Forehead-top anchor used only for the face_scale proxy (distinct from the
# eye-corner landmarks used for inter_eye_distance_norm).
_FOREHEAD_TOP = 10

_MIN_EYE_SPAN = 1e-4  # guards against division by ~0 for degenerate geometry


@dataclass
class FeatureVector:
    left_iris_x_rel: float
    left_iris_y_rel: float
    right_iris_x_rel: float
    right_iris_y_rel: float
    left_eye_width_norm: float
    left_eye_height_norm: float
    right_eye_width_norm: float
    right_eye_height_norm: float
    left_ear: float
    right_ear: float
    inter_eye_distance_norm: float
    face_center_x_norm: float
    face_center_y_norm: float
    face_scale: float
    head_yaw_deg: float
    head_pitch_deg: float
    head_roll_deg: float

    def to_array(self) -> np.ndarray:
        return np.array([getattr(self, f.name) for f in fields(self)], dtype=np.float64)

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.to_array())))


def _eye_relative_iris(points: np.ndarray, corner_a: int, corner_b: int, top: int, bottom: int, iris: int):
    ax, ay = points[corner_a, 0], points[corner_a, 1]
    bx, by = points[corner_b, 0], points[corner_b, 1]
    ty = points[top, 1]
    byy = points[bottom, 1]
    ix, iy = points[iris, 0], points[iris, 1]

    x_lo, x_hi = min(ax, bx), max(ax, bx)
    width = x_hi - x_lo
    height = byy - ty

    rel_x = (ix - x_lo) / width if width > _MIN_EYE_SPAN else float("nan")
    rel_y = (iy - ty) / height if height > _MIN_EYE_SPAN else float("nan")
    ear = height / width if width > _MIN_EYE_SPAN else float("nan")
    return rel_x, rel_y, width, height, ear


class FeatureExtractor:
    def __init__(self, image_width: int, image_height: int):
        self._head_pose = HeadPoseEstimator(image_width, image_height)
        self._image_width = image_width
        self._image_height = image_height

    def extract(self, landmarks_norm: np.ndarray) -> FeatureVector | None:
        """landmarks_norm: (478, 3) array, x/y normalized to [0,1] by MediaPipe."""
        w, h = self._image_width, self._image_height
        points = landmarks_norm[:, :2].copy()
        # x already normalized by width, y already normalized by height, so
        # distances below are already frame-size-relative (section 14).

        left_x_rel, left_y_rel, left_w, left_h, left_ear = _eye_relative_iris(
            points, lm.LEFT_EYE_INNER, lm.LEFT_EYE_OUTER, lm.LEFT_EYE_TOP, lm.LEFT_EYE_BOTTOM, lm.LEFT_IRIS_CENTER
        )
        right_x_rel, right_y_rel, right_w, right_h, right_ear = _eye_relative_iris(
            points, lm.RIGHT_EYE_INNER, lm.RIGHT_EYE_OUTER, lm.RIGHT_EYE_TOP, lm.RIGHT_EYE_BOTTOM, lm.RIGHT_IRIS_CENTER
        )

        left_eye_center = points[lm.LEFT_EYE_INNER] + points[lm.LEFT_EYE_OUTER]
        left_eye_center /= 2
        right_eye_center = points[lm.RIGHT_EYE_INNER] + points[lm.RIGHT_EYE_OUTER]
        right_eye_center /= 2
        inter_eye_distance_norm = float(np.linalg.norm(left_eye_center - right_eye_center))

        face_xy = points[:468]  # exclude iris points from the centroid
        face_center_x_norm = float(np.mean(face_xy[:, 0]))
        face_center_y_norm = float(np.mean(face_xy[:, 1]))
        face_scale = float(np.linalg.norm(points[_FOREHEAD_TOP] - points[lm.CHIN]))

        anchors_px = np.array([
            [points[i, 0] * w, points[i, 1] * h] for i in lm.HEAD_POSE_ANCHOR_INDICES
        ])
        pose = self._head_pose.estimate(anchors_px)

        fv = FeatureVector(
            left_iris_x_rel=left_x_rel,
            left_iris_y_rel=left_y_rel,
            right_iris_x_rel=right_x_rel,
            right_iris_y_rel=right_y_rel,
            left_eye_width_norm=left_w,
            left_eye_height_norm=left_h,
            right_eye_width_norm=right_w,
            right_eye_height_norm=right_h,
            left_ear=left_ear,
            right_ear=right_ear,
            inter_eye_distance_norm=inter_eye_distance_norm,
            face_center_x_norm=face_center_x_norm,
            face_center_y_norm=face_center_y_norm,
            face_scale=face_scale,
            head_yaw_deg=pose.yaw_deg,
            head_pitch_deg=pose.pitch_deg,
            head_roll_deg=pose.roll_deg,
        )
        if not fv.is_finite() or not pose.success:
            return None
        return fv
