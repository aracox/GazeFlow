"""Screen coordinate math, target generation, error metrics, and head pose.

All screen positions are normalized: (0,0) = top-left, (0.5,0.5) = center,
(1,1) = bottom-right. Pixel/mm/degree conversions happen only at the edges
(drawing, error reporting).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 9-point calibration grid (section 21). Order here is canonical; actual
# on-screen presentation order is shuffled deterministically per run.
CALIBRATION_POINTS: list[tuple[str, float, float]] = [
    ("c1", 0.1, 0.1), ("c2", 0.5, 0.1), ("c3", 0.9, 0.1),
    ("c4", 0.1, 0.5), ("c5", 0.5, 0.5), ("c6", 0.9, 0.5),
    ("c7", 0.1, 0.9), ("c8", 0.5, 0.9), ("c9", 0.9, 0.9),
]


def shuffled_calibration_points(seed: int) -> list[tuple[str, float, float]]:
    """Deterministically shuffle calibration point presentation order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(CALIBRATION_POINTS))
    return [CALIBRATION_POINTS[i] for i in order]


def generate_validation_points(
    seed: int,
    count: int = 20,
    margin: float = 0.1,
    exclude: list[tuple[float, float]] | None = None,
    min_separation: float = 0.06,
) -> list[tuple[str, float, float]]:
    """Deterministically generate validation targets.

    Points are drawn from a jittered grid (stratified sampling) so they
    spread across the screen instead of clustering near the center, stay
    within [margin, 1-margin], and avoid calibration point locations.
    """
    exclude = exclude or [(x, y) for _, x, y in CALIBRATION_POINTS]
    rng = np.random.default_rng(seed)

    n_cols = math.ceil(math.sqrt(count * 1.4))
    n_rows = math.ceil(count / n_cols)
    lo, hi = margin, 1.0 - margin
    cell_w = (hi - lo) / n_cols
    cell_h = (hi - lo) / n_rows

    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    rng.shuffle(cells)

    points: list[tuple[str, float, float]] = []
    idx = 0
    attempts = 0
    while len(points) < count and attempts < len(cells) * 5:
        r, c = cells[idx % len(cells)]
        idx += 1
        attempts += 1
        cx = lo + (c + 0.5) * cell_w
        cy = lo + (r + 0.5) * cell_h
        jx = cx + rng.uniform(-cell_w * 0.3, cell_w * 0.3)
        jy = cy + rng.uniform(-cell_h * 0.3, cell_h * 0.3)
        jx = float(np.clip(jx, lo, hi))
        jy = float(np.clip(jy, lo, hi))

        too_close = any(
            math.hypot(jx - ex, jy - ey) < min_separation for ex, ey in exclude
        )
        if too_close:
            continue
        exclude = exclude + [(jx, jy)]
        points.append((f"v{len(points) + 1}", jx, jy))

    if len(points) < count:
        raise RuntimeError(
            f"Could not generate {count} validation points "
            f"(got {len(points)}); loosen margin or min_separation."
        )
    return points


@dataclass
class ErrorResult:
    error_norm: float
    error_px: float
    error_mm: float | None
    error_deg: float | None


def compute_error(
    predicted_x_norm: float,
    predicted_y_norm: float,
    target_x_norm: float,
    target_y_norm: float,
    screen_width_px: int,
    screen_height_px: int,
    screen_width_mm: float | None,
    screen_height_mm: float | None,
    viewing_distance_mm: float,
) -> ErrorResult:
    """Compute normalized, pixel, mm, and angular error (section 32).

    mm/degree error are None unless physical screen dimensions are known;
    degree values are never fabricated from an assumed screen size.
    """
    dx = predicted_x_norm - target_x_norm
    dy = predicted_y_norm - target_y_norm
    error_norm = math.hypot(dx, dy)

    dx_px = dx * screen_width_px
    dy_px = dy * screen_height_px
    error_px = math.hypot(dx_px, dy_px)

    error_mm: float | None = None
    error_deg: float | None = None
    if screen_width_mm and screen_height_mm:
        dx_mm = dx * screen_width_mm
        dy_mm = dy * screen_height_mm
        error_mm = math.hypot(dx_mm, dy_mm)
        error_deg = math.degrees(math.atan(error_mm / viewing_distance_mm))

    return ErrorResult(error_norm, error_px, error_mm, error_deg)


def normalized_to_pixel(x_norm: float, y_norm: float, width_px: int, height_px: int) -> tuple[int, int]:
    return int(round(x_norm * width_px)), int(round(y_norm * height_px))


# --- Head pose (section 15) ---------------------------------------------
#
# Generic 3D face model (arbitrary units, not measured from the user) used
# with solvePnP purely to obtain a stable yaw/pitch/roll signal. This is a
# widely used approximate anchor set; A0 does not need clinical accuracy.
#
# Axes are aligned to OpenCV's camera convention: +X toward larger
# image-pixel x (image right), +Y toward larger image-pixel y (image
# down), +Z away from the camera. A frontal face's subject-left features
# (eye/mouth corners) appear on the image's *right* side (see
# landmarks.py) so they get positive X; the chin sits below the nose in
# the image so it gets positive Y; eyes/mouth sit behind the nose tip
# (which protrudes toward the camera) so they get positive Z.
#
# Verified empirically against a real face image: this sign convention is
# the one where two independent PnP solvers (SOLVEPNP_ITERATIVE and
# SOLVEPNP_SQPNP) agree closely and reprojection error is lowest. Getting
# any one axis backwards still lets solvePnP converge (the anchors are
# roughly bilaterally symmetric and near-planar) but yields a mirrored or
# solver-dependent pose -- e.g. roll or pitch parked near +-180 deg, or
# flipping between solvers.
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),      # nose tip
    (0.0, 330.0, 65.0),   # chin
    (225.0, -170.0, 135.0),   # subject's left eye, outer corner (image right)
    (-225.0, -170.0, 135.0),  # subject's right eye, outer corner (image left)
    (150.0, 150.0, 125.0),    # subject's left mouth corner (image right)
    (-150.0, 150.0, 125.0),   # subject's right mouth corner (image left)
], dtype=np.float64)


@dataclass
class HeadPose:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    success: bool


class HeadPoseEstimator:
    """solvePnP-based head pose using six stable MediaPipe landmark anchors."""

    def __init__(self, image_width: int, image_height: int):
        self.image_width = image_width
        self.image_height = image_height
        focal_length = image_width
        center = (image_width / 2, image_height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1))

    def estimate(self, anchor_points_px: np.ndarray) -> HeadPose:
        """anchor_points_px: (6,2) array in pixel coords, ordered to match
        _MODEL_POINTS_3D: nose tip, chin, left-eye-outer, right-eye-outer,
        mouth-left, mouth-right."""
        import cv2

        if not np.all(np.isfinite(anchor_points_px)):
            return HeadPose(0.0, 0.0, 0.0, success=False)

        # SQPNP: a globally-optimal, closed-form solver with no initial-guess
        # dependency, so it doesn't fall into the local-minima pose ambiguity
        # that SOLVEPNP_ITERATIVE is prone to for this near-planar 6-point
        # anchor set (verified empirically -- see _MODEL_POINTS_3D comment).
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            anchor_points_px.astype(np.float64),
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            return HeadPose(0.0, 0.0, 0.0, success=False)

        rmat, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = _rotation_matrix_to_euler_deg(rmat)
        return HeadPose(yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll, success=True)


def _rotation_matrix_to_euler_deg(rmat: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
