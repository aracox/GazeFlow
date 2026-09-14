import math

from a0.geometry import compute_error


def test_normalized_error_is_euclidean_distance():
    err = compute_error(
        predicted_x_norm=0.6, predicted_y_norm=0.5,
        target_x_norm=0.5, target_y_norm=0.5,
        screen_width_px=1000, screen_height_px=1000,
        screen_width_mm=None, screen_height_mm=None,
        viewing_distance_mm=600,
    )
    assert math.isclose(err.error_norm, 0.1, rel_tol=1e-9)


def test_pixel_conversion():
    err = compute_error(
        predicted_x_norm=0.6, predicted_y_norm=0.5,
        target_x_norm=0.5, target_y_norm=0.5,
        screen_width_px=1000, screen_height_px=800,
        screen_width_mm=None, screen_height_mm=None,
        viewing_distance_mm=600,
    )
    # dx_px = 0.1 * 1000 = 100, dy_px = 0
    assert math.isclose(err.error_px, 100.0, rel_tol=1e-9)


def test_mm_and_degree_conversion():
    err = compute_error(
        predicted_x_norm=0.6, predicted_y_norm=0.5,
        target_x_norm=0.5, target_y_norm=0.5,
        screen_width_px=1000, screen_height_px=800,
        screen_width_mm=300.0, screen_height_mm=200.0,
        viewing_distance_mm=600,
    )
    # dx_mm = 0.1 * 300 = 30
    assert math.isclose(err.error_mm, 30.0, rel_tol=1e-9)
    expected_deg = math.degrees(math.atan(30.0 / 600))
    assert math.isclose(err.error_deg, expected_deg, rel_tol=1e-9)


def test_degree_and_mm_are_none_without_physical_geometry():
    err = compute_error(
        predicted_x_norm=0.6, predicted_y_norm=0.5,
        target_x_norm=0.5, target_y_norm=0.5,
        screen_width_px=1000, screen_height_px=800,
        screen_width_mm=None, screen_height_mm=None,
        viewing_distance_mm=600,
    )
    assert err.error_mm is None
    assert err.error_deg is None


def test_zero_error_at_target():
    err = compute_error(
        predicted_x_norm=0.42, predicted_y_norm=0.73,
        target_x_norm=0.42, target_y_norm=0.73,
        screen_width_px=1920, screen_height_px=1080,
        screen_width_mm=345.0, screen_height_mm=194.0,
        viewing_distance_mm=600,
    )
    assert err.error_norm == 0.0
    assert err.error_px == 0.0
    assert err.error_mm == 0.0
    assert err.error_deg == 0.0
