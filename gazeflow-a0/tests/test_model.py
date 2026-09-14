import numpy as np

from a0.calibration import select_ridge_alpha
from a0.model import GazeModel


def _synthetic_calibration_data(seed: int = 0):
    """9 calibration points x 20 correlated frames each, with a known
    linear feature->target relationship plus small noise."""
    rng = np.random.default_rng(seed)
    grid = [(x, y) for x in (0.1, 0.5, 0.9) for y in (0.1, 0.5, 0.9)]
    true_w_x = np.array([1.0, 0.0, 0.3])
    true_w_y = np.array([0.0, 1.0, -0.2])

    X, y_x, y_y, point_ids = [], [], [], []
    for i, (tx, ty) in enumerate(grid):
        for _ in range(20):
            base = np.array([tx, ty, 1.0]) + rng.normal(scale=0.01, size=3)
            X.append(base)
            y_x.append(base @ true_w_x)
            y_y.append(base @ true_w_y)
            point_ids.append(f"c{i}")
    return np.array(X), np.array(y_x), np.array(y_y), np.array(point_ids)


def test_ridge_pipeline_recovers_linear_mapping():
    X, y_x, y_y, point_ids = _synthetic_calibration_data()
    cv_result = select_ridge_alpha(X, y_x, y_y, point_ids, alphas=[0.01, 0.1, 1.0])
    gaze_model = GazeModel.fit(X, y_x, y_y, cv_result.selected_alpha)

    test_point = np.array([[0.3, 0.7, 1.0]])
    pred_x, pred_y = gaze_model.predict(test_point)

    expected_x = test_point[0] @ np.array([1.0, 0.0, 0.3])
    expected_y = test_point[0] @ np.array([0.0, 1.0, -0.2])
    assert abs(pred_x[0] - expected_x) < 0.05
    assert abs(pred_y[0] - expected_y) < 0.05


def test_select_ridge_alpha_reports_all_calibration_points():
    X, y_x, y_y, point_ids = _synthetic_calibration_data()
    cv_result = select_ridge_alpha(X, y_x, y_y, point_ids, alphas=[0.01, 0.1, 1.0])
    assert set(cv_result.per_point_error.keys()) == set(point_ids.tolist())
    assert cv_result.median_error_norm < 0.1
