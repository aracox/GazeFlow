"""Independent validation target aggregation (sections 29-31)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import compute_error


@dataclass
class ValidationTargetResult:
    point_id: str
    target_x_norm: float
    target_y_norm: float
    predicted_x_norm: float
    predicted_y_norm: float
    error_norm: float
    error_px: float
    error_mm: float | None
    error_deg: float | None
    valid_frames: int
    head_yaw_median: float
    head_pitch_median: float
    head_roll_median: float
    face_scale_median: float


def aggregate_target(
    point_id: str,
    target_x_norm: float,
    target_y_norm: float,
    predicted_x: list[float],
    predicted_y: list[float],
    head_yaw: list[float],
    head_pitch: list[float],
    head_roll: list[float],
    face_scale: list[float],
    screen_width_px: int,
    screen_height_px: int,
    screen_width_mm: float | None,
    screen_height_mm: float | None,
    viewing_distance_mm: float,
) -> ValidationTargetResult:
    median_pred_x = float(np.median(predicted_x))
    median_pred_y = float(np.median(predicted_y))

    err = compute_error(
        median_pred_x, median_pred_y, target_x_norm, target_y_norm,
        screen_width_px, screen_height_px, screen_width_mm, screen_height_mm,
        viewing_distance_mm,
    )

    return ValidationTargetResult(
        point_id=point_id,
        target_x_norm=target_x_norm,
        target_y_norm=target_y_norm,
        predicted_x_norm=median_pred_x,
        predicted_y_norm=median_pred_y,
        error_norm=err.error_norm,
        error_px=err.error_px,
        error_mm=err.error_mm,
        error_deg=err.error_deg,
        valid_frames=len(predicted_x),
        head_yaw_median=float(np.median(head_yaw)) if head_yaw else float("nan"),
        head_pitch_median=float(np.median(head_pitch)) if head_pitch else float("nan"),
        head_roll_median=float(np.median(head_roll)) if head_roll else float("nan"),
        face_scale_median=float(np.median(face_scale)) if face_scale else float("nan"),
    )
