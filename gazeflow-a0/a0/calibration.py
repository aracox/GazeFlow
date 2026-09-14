"""Blink baseline, blink rejection, and Ridge alpha selection (sections 19, 27-28)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from .model import build_pipeline

ALPHA_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]


def compute_baseline_ear(left_ear_samples: list[float], right_ear_samples: list[float]) -> float:
    """Median EAR (both eyes pooled) during the open-eye precheck."""
    samples = [v for v in (*left_ear_samples, *right_ear_samples) if np.isfinite(v)]
    if not samples:
        raise ValueError("No valid EAR samples collected during baseline precheck")
    return float(np.median(samples))


def blink_threshold(baseline_ear: float, factor: float = 0.65) -> float:
    return baseline_ear * factor


def is_blinking(left_ear: float, right_ear: float, threshold: float) -> bool:
    """Blink if the average eye-openness drops below the per-user threshold."""
    if not (np.isfinite(left_ear) and np.isfinite(right_ear)):
        return True
    return ((left_ear + right_ear) / 2.0) < threshold


@dataclass
class CalibrationCVResult:
    selected_alpha: float
    median_error_norm: float
    mean_error_norm: float
    worst_point: str
    per_point_error: dict[str, float]


def select_ridge_alpha(
    X: np.ndarray,
    y_x: np.ndarray,
    y_y: np.ndarray,
    point_ids: np.ndarray,
    alphas: list[float] = ALPHA_GRID,
) -> CalibrationCVResult:
    """Leave-one-calibration-point-out CV: for each alpha, hold out all
    frames belonging to one calibration point, fit on the remaining 8
    points, and score the held-out point. Select the alpha minimizing the
    median normalized Euclidean error across held-out points (section 27)."""
    logo = LeaveOneGroupOut()
    unique_points = np.unique(point_ids)

    best_alpha = alphas[0]
    best_median = np.inf
    best_per_point: dict[str, float] = {}

    for alpha in alphas:
        per_point_errors: dict[str, list[float]] = {p: [] for p in unique_points}
        for train_idx, test_idx in logo.split(X, y_x, groups=point_ids):
            mx = build_pipeline(alpha).fit(X[train_idx], y_x[train_idx])
            my = build_pipeline(alpha).fit(X[train_idx], y_y[train_idx])
            pred_x = mx.predict(X[test_idx])
            pred_y = my.predict(X[test_idx])
            err = np.hypot(pred_x - y_x[test_idx], pred_y - y_y[test_idx])
            held_out_point = point_ids[test_idx[0]]
            per_point_errors[held_out_point].extend(err.tolist())

        point_medians = {p: float(np.median(v)) for p, v in per_point_errors.items() if v}
        overall_median = float(np.median(list(point_medians.values())))

        if overall_median < best_median:
            best_median = overall_median
            best_alpha = alpha
            best_per_point = point_medians

    mean_error = float(np.mean(list(best_per_point.values())))
    worst_point = max(best_per_point, key=best_per_point.get)

    return CalibrationCVResult(
        selected_alpha=best_alpha,
        median_error_norm=best_median,
        mean_error_norm=mean_error,
        worst_point=worst_point,
        per_point_error=best_per_point,
    )
