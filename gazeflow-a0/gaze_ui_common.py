"""Shared setup code for the gaze-driven sample apps (sample_app.py,
numberpad_app.py): a small windowed UI helper, baseline/blink-threshold
collection, and the fresh in-app calibration flow. Not part of the a0/
package -- these are demo scripts, not production code, but the two demos
share enough setup that duplicating it would just be a maintenance trap.
"""
from __future__ import annotations

import time

import numpy as np

from a0 import calibration, ui as a0_ui
from a0.calibration import select_ridge_alpha
from a0.camera import Camera
from a0.features import FeatureExtractor
from a0.landmarks import FaceLandmarkerWrapper
from a0.model import GazeModel

BASELINE_DURATION_SECONDS = 2.0
WINDOW_SCALE = 0.5  # a centered_window() is WINDOW_SCALE x WINDOW_SCALE of the screen (quarter-area at 0.5)
WINDOW_TOP_FRAC = 0.05  # centered_window()'s top edge sits this far down the screen
BLINK_THRESHOLD_FACTOR = 0.55  # stricter than a0.calibration's default 0.65: requires a more definite closure
BLINK_MIN_CONSECUTIVE_FRAMES = 3  # a single noisy low-EAR frame (e.g. from looking to the side) isn't a blink

CALIB_MARGIN = 0.15  # corner targets sit this far in from the window's edges
CALIB_TARGET_VALID_FRAMES = 20
CALIB_MIN_VALID_FRAMES = 10
CALIB_MAX_MS = 2500
CALIB_SETTLE_MS = 300
CALIB_TRANSITION_MS = 200
CALIB_MIN_SUCCESSFUL_POINTS = 3


class WindowedUI:
    """A normal (non-fullscreen) OpenCV window at an explicit position/size.
    All drawing happens in this window's own local normalized [0,1]
    coordinates."""

    def __init__(self, window_name: str, width_px: int, height_px: int, x_px: int, y_px: int):
        import cv2

        self.width = width_px
        self.height = height_px
        self.x_px = x_px
        self.y_px = y_px
        self._window_name = window_name
        self._cv2 = cv2
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, x_px, y_px)

    def new_canvas(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def show(self, canvas: np.ndarray, wait_ms: int = 1) -> int:
        self._cv2.imshow(self._window_name, canvas)
        return self._cv2.waitKey(wait_ms) & 0xFF

    def close(self) -> None:
        self._cv2.destroyWindow(self._window_name)

    def draw_target(self, canvas: np.ndarray, x_norm: float, y_norm: float, progress: float) -> None:
        cv2 = self._cv2
        cx, cy = int(x_norm * self.width), int(y_norm * self.height)
        max_r, min_r, dot_r = 18, 5, 3
        ring_r = int(max_r - (max_r - min_r) * max(0.0, min(1.0, progress)))
        cv2.circle(canvas, (cx, cy), ring_r, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), dot_r, (255, 255, 255), -1, cv2.LINE_AA)


def centered_window(window_name: str, screen_width_px: int, screen_height_px: int,
                     scale: float = WINDOW_SCALE, top_frac: float = WINDOW_TOP_FRAC) -> WindowedUI:
    """A horizontally-centered window, `scale` x `scale` of the screen, with
    its top edge `top_frac` of the way down the screen."""
    width = int(screen_width_px * scale)
    height = int(screen_height_px * scale)
    x = int(screen_width_px * (1.0 - scale) / 2.0)
    y = int(screen_height_px * top_frac)
    return WindowedUI(window_name, width, height, x, y)


def put_centered(win: WindowedUI, canvas: np.ndarray, text: str, y_frac: float, scale: float = 1.0, color=(255, 255, 255)) -> None:
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    size, _ = cv2.getTextSize(text, font, scale, 2)
    x = win.width // 2 - size[0] // 2
    y = int(win.height * y_frac)
    cv2.putText(canvas, text, (x, y), font, scale, color, 2, cv2.LINE_AA)


def collect_baseline_ear(camera: Camera, landmarker: FaceLandmarkerWrapper, extractor: FeatureExtractor, win: WindowedUI) -> float:
    left_samples, right_samples = [], []
    start = time.monotonic()
    while (time.monotonic() - start) < BASELINE_DURATION_SECONDS:
        frame = camera.read()
        if frame is None:
            continue
        result = landmarker.detect(frame.image_bgr, frame.timestamp_ms)
        fv = extractor.extract(result.landmarks_norm) if result.face_detected else None
        if fv is not None:
            left_samples.append(fv.left_ear)
            right_samples.append(fv.right_ear)

        canvas = win.new_canvas()
        put_centered(win, canvas, "Look naturally,", y_frac=0.45, scale=0.7)
        put_centered(win, canvas, "eyes open...", y_frac=0.58, scale=0.7)
        key = win.show(canvas, wait_ms=1)
        if key == a0_ui.ESC_KEY:
            raise KeyboardInterrupt
    return calibration.compute_baseline_ear(left_samples, right_samples)


def cross_calibration_points(count: int) -> list[tuple[str, float, float]]:
    """4 corners, optionally + center (5), in the window's own normalized
    coordinates. Order is shuffled deterministically."""
    m = CALIB_MARGIN
    if count == 5:
        points = [("center", 0.5, 0.5), ("tl", m, m), ("tr", 1 - m, m), ("bl", m, 1 - m), ("br", 1 - m, 1 - m)]
    elif count == 4:
        points = [("tl", m, m), ("tr", 1 - m, m), ("bl", m, 1 - m), ("br", 1 - m, 1 - m)]
    else:
        raise ValueError(f"calibration point count must be 4 or 5, got {count}")
    rng = np.random.default_rng(42)
    order = rng.permutation(len(points))
    return [points[i] for i in order]


def wait_for_start(win: WindowedUI) -> None:
    canvas = win.new_canvas()
    put_centered(win, canvas, "Follow the dot with your eyes.", y_frac=0.4, scale=0.6)
    put_centered(win, canvas, "SPACE to start, ESC to cancel.", y_frac=0.55, scale=0.6)
    while True:
        key = win.show(canvas, wait_ms=30)
        if key == a0_ui.SPACE_KEY:
            return
        if key == a0_ui.ESC_KEY:
            raise KeyboardInterrupt


def _collect_points_at(camera: Camera, landmarker: FaceLandmarkerWrapper, extractor: FeatureExtractor,
                        win: WindowedUI, xy: tuple[float, float], blink_threshold: float) -> list:
    start = time.monotonic()
    collected = []
    while len(collected) < CALIB_TARGET_VALID_FRAMES and (time.monotonic() - start) * 1000 < CALIB_MAX_MS:
        frame = camera.read()
        if frame is None:
            continue
        result = landmarker.detect(frame.image_bgr, frame.timestamp_ms)
        fv = extractor.extract(result.landmarks_norm) if result.face_detected else None
        if fv is not None and not calibration.is_blinking(fv.left_ear, fv.right_ear, blink_threshold):
            collected.append(fv)

        canvas = win.new_canvas()
        win.draw_target(canvas, xy[0], xy[1], len(collected) / CALIB_TARGET_VALID_FRAMES)
        key = win.show(canvas, wait_ms=1)
        if key == a0_ui.ESC_KEY:
            raise KeyboardInterrupt
    return collected


def _settle_at(camera: Camera, win: WindowedUI, xy: tuple[float, float], duration_ms: float) -> None:
    start = time.monotonic()
    while (time.monotonic() - start) * 1000 < duration_ms:
        camera.read()  # drain the camera buffer while the target holds still
        canvas = win.new_canvas()
        win.draw_target(canvas, xy[0], xy[1], progress=0.0)
        key = win.show(canvas, wait_ms=1)
        if key == a0_ui.ESC_KEY:
            raise KeyboardInterrupt


def _transition_to(camera: Camera, win: WindowedUI, prev_xy: tuple[float, float], next_xy: tuple[float, float]) -> None:
    start = time.monotonic()
    while True:
        t = min(1.0, (time.monotonic() - start) * 1000 / CALIB_TRANSITION_MS)
        x = prev_xy[0] + (next_xy[0] - prev_xy[0]) * t
        y = prev_xy[1] + (next_xy[1] - prev_xy[1]) * t
        camera.read()
        canvas = win.new_canvas()
        win.draw_target(canvas, x, y, progress=0.0)
        key = win.show(canvas, wait_ms=1)
        if key == a0_ui.ESC_KEY:
            raise KeyboardInterrupt
        if t >= 1.0:
            return


def run_calibration(camera: Camera, landmarker: FaceLandmarkerWrapper, extractor: FeatureExtractor,
                     win: WindowedUI, blink_threshold: float, points: list[tuple[str, float, float]]) -> GazeModel:
    """Runs the calibration flow (instructions -> follow each target ->
    collect samples) against the given (id, x_norm, y_norm) target list and
    returns a fitted GazeModel predicting in that same coordinate space."""
    wait_for_start(win)

    all_X, all_yx, all_yy, all_pid = [], [], [], []
    prev_xy = (0.5, 0.5)
    for point_id, tx, ty in points:
        _transition_to(camera, win, prev_xy, (tx, ty))
        prev_xy = (tx, ty)
        _settle_at(camera, win, (tx, ty), CALIB_SETTLE_MS)
        collected = _collect_points_at(camera, landmarker, extractor, win, (tx, ty), blink_threshold)

        if len(collected) < CALIB_MIN_VALID_FRAMES:
            _settle_at(camera, win, (tx, ty), CALIB_SETTLE_MS)
            collected = _collect_points_at(camera, landmarker, extractor, win, (tx, ty), blink_threshold)

        if len(collected) < CALIB_MIN_VALID_FRAMES:
            print(f"Calibration point {point_id} failed ({len(collected)} valid frames), skipping")
            continue

        print(f"Calibration point {point_id}: {len(collected)} valid frames")
        for fv in collected:
            all_X.append(fv.to_array())
            all_yx.append(tx)
            all_yy.append(ty)
            all_pid.append(point_id)

    if len(set(all_pid)) < CALIB_MIN_SUCCESSFUL_POINTS:
        raise RuntimeError(f"Only {len(set(all_pid))} calibration points succeeded; need at least {CALIB_MIN_SUCCESSFUL_POINTS}")

    X, yx, yy = np.array(all_X), np.array(all_yx), np.array(all_yy)
    cv_result = select_ridge_alpha(X, yx, yy, np.array(all_pid))
    print(f"Calibration fit: alpha={cv_result.selected_alpha}, CV median error (norm)={cv_result.median_error_norm:.4f}")
    return GazeModel.fit(X, yx, yy, cv_result.selected_alpha)
