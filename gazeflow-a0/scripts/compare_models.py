"""Offline follow-up experiment (not part of the A0 CLI): does a stronger
regressor beat Ridge on the SAME calibration/validation data already
collected by `a0.main run`?

No new capture session. Loads a completed run's a0_raw_features.csv,
retrains several candidate models on its calibration-phase frames, and
evaluates them the same way a0/report.py does: predict every valid
validation-phase frame, take the median prediction per validation target,
and compute error against the true target. This isolates "would a smarter
model help" from "would better tracking/data collection help" (already
addressed separately).

Usage:
    python scripts/compare_models.py outputs/<run-id> [outputs/<run-id> ...]

Disposable, like the rest of A0 -- not meant to be extended into product code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from a0.features import FEATURE_NAMES  # noqa: E402
from a0.geometry import compute_error  # noqa: E402

CANDIDATES = {
    "ridge_baseline": lambda: Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
    "poly2_ridge": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=10.0)),
    ]),
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=0),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0),
    "mlp": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-2,
                              max_iter=5000, random_state=0)),
    ]),
}


def evaluate_run(run_dir: Path) -> pd.DataFrame:
    raw = pd.read_csv(run_dir / "a0_raw_features.csv")
    results = json.loads((run_dir / "a0_results.json").read_text())
    screen = results["screen"]

    calib = raw[(raw["phase"] == "calibration") & (raw["sample_valid"] == True)]  # noqa: E712
    valid_rows = raw[(raw["phase"] == "validation") & (raw["sample_valid"] == True)]  # noqa: E712

    X_train = calib[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_train_x = calib["target_x_norm"].to_numpy(dtype=np.float64)
    y_train_y = calib["target_y_norm"].to_numpy(dtype=np.float64)

    rows = []
    for name, factory in CANDIDATES.items():
        model_x, model_y = factory(), factory()
        model_x.fit(X_train, y_train_x)
        model_y.fit(X_train, y_train_y)

        errors_deg, errors_norm = [], []
        for point_id, group in valid_rows.groupby("point_id"):
            X_val = group[FEATURE_NAMES].to_numpy(dtype=np.float64)
            pred_x = np.median(model_x.predict(X_val))
            pred_y = np.median(model_y.predict(X_val))
            target_x = group["target_x_norm"].iloc[0]
            target_y = group["target_y_norm"].iloc[0]
            err = compute_error(
                pred_x, pred_y, target_x, target_y,
                screen["width_px"], screen["height_px"],
                screen["width_mm"], screen["height_mm"], screen["viewing_distance_mm"],
            )
            errors_deg.append(err.error_deg)
            errors_norm.append(err.error_norm)

        rows.append({
            "run": run_dir.name,
            "model": name,
            "median_error_deg": float(np.median(errors_deg)),
            "p95_error_deg": float(np.percentile(errors_deg, 95)),
            "median_error_norm": float(np.median(errors_norm)),
            "p95_error_norm": float(np.percentile(errors_norm, 95)),
        })
    return pd.DataFrame(rows)


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python scripts/compare_models.py outputs/<run-id> [outputs/<run-id> ...]")
        return 1

    all_results = []
    for run_arg in argv:
        run_dir = Path(run_arg)
        print(f"Evaluating {run_dir.name}...")
        all_results.append(evaluate_run(run_dir))

    combined = pd.concat(all_results, ignore_index=True)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print()
    print(combined[["run", "model", "median_error_deg", "p95_error_deg"]].to_string(index=False))

    print()
    print("Reference gate: PASS median<=3.0 p95<=4.5 | BORDERLINE median<=4.5 p95<=6.75 | else FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
