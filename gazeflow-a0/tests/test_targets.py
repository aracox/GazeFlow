import math

import pytest

from a0.geometry import (
    CALIBRATION_POINTS,
    generate_calibration_points,
    generate_validation_points,
    shuffled_calibration_points,
)


def test_nine_calibration_positions_correct():
    expected = {
        (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
        (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
        (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
    }
    actual = {(x, y) for _, x, y in CALIBRATION_POINTS}
    assert actual == expected
    assert len(CALIBRATION_POINTS) == 9


def test_calibration_shuffle_is_deterministic_and_not_left_to_right():
    order_a = shuffled_calibration_points(seed=42)
    order_b = shuffled_calibration_points(seed=42)
    assert order_a == order_b
    assert [p[0] for p in order_a] != [p[0] for p in CALIBRATION_POINTS]


def test_twenty_validation_points_deterministic():
    points_a = generate_validation_points(seed=42, count=20)
    points_b = generate_validation_points(seed=42, count=20)
    assert points_a == points_b
    assert len(points_a) == 20


def test_validation_points_inside_margins():
    points = generate_validation_points(seed=42, count=20, margin=0.1)
    for _, x, y in points:
        assert 0.1 - 1e-9 <= x <= 0.9 + 1e-9
        assert 0.1 - 1e-9 <= y <= 0.9 + 1e-9


def test_validation_points_not_equal_to_calibration_points():
    calib_xy = [(x, y) for _, x, y in CALIBRATION_POINTS]
    points = generate_validation_points(seed=42, count=20, exclude=calib_xy)
    for _, x, y in points:
        for cx, cy in calib_xy:
            assert math.hypot(x - cx, y - cy) > 1e-6


def test_validation_points_not_all_clustered_in_center():
    points = generate_validation_points(seed=42, count=20)
    center_count = sum(1 for _, x, y in points if 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6)
    assert center_count < len(points)


def test_generate_calibration_points_supports_other_square_counts():
    points = generate_calibration_points(16, margin=0.10)
    assert len(points) == 16
    xs = sorted({round(x, 6) for _, x, _ in points})
    ys = sorted({round(y, 6) for _, _, y in points})
    assert xs == ys
    assert math.isclose(xs[0], 0.10, rel_tol=1e-9)
    assert math.isclose(xs[-1], 0.90, rel_tol=1e-9)
    assert len(xs) == 4


def test_generate_calibration_points_default_matches_nine_point_grid():
    assert generate_calibration_points(9) == CALIBRATION_POINTS


def test_generate_calibration_points_rejects_non_square_count():
    with pytest.raises(ValueError):
        generate_calibration_points(10)


def test_generate_calibration_points_rejects_too_few_points():
    with pytest.raises(ValueError):
        generate_calibration_points(1)


def test_shuffled_calibration_points_honors_count():
    order = shuffled_calibration_points(seed=42, count=16)
    assert len(order) == 16
    assert {p[0] for p in order} == {f"c{i}" for i in range(1, 17)}
