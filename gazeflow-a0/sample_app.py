"""Sample app: a windowed YES/NO picker driven by gaze + blink.

Runs in a normal (non-fullscreen) window sized to a quarter of the
screen's width/height, near the top of the screen. Look at the left half
(NO) or right half (YES) of the window; blink while looking at a side to
select it.

Does its OWN short calibration at startup (a 5-point cross: four corners +
center of the window, or 4 with --calibration-points 4) rather than reusing
an old `a0.main run` session's model. Gaze accuracy depends on the current
seating distance/position, which a reused calibration from a different
session won't match -- this calibrates fresh, right before use, against
the window's own bounds (so the fitted model predicts window-local
coordinates directly, not full-screen ones). A short eyes-open baseline is
also collected for this session's blink threshold.

Usage:
    python sample_app.py [--camera N] [--calibration-points 4|5] [--question "..."]

Like the rest of A0, this is a sample/demo script, not production code.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from a0 import calibration, ui as a0_ui
from a0.camera import Camera
from a0.features import FeatureExtractor
from a0.landmarks import FaceLandmarkerWrapper
from a0.model import GazeModel

GAZE_SMOOTHING_ALPHA = 0.12  # EMA weight on each new prediction; lower = smoother/laggier
CONFIRMATION_DISPLAY_SECONDS = 1.5
BASELINE_DURATION_SECONDS = 2.0
FACE_LOST_GRACE_SECONDS = 0.4  # tolerate brief tracking dropouts without losing the highlighted side
WINDOW_SCALE = 0.5  # window is WINDOW_SCALE x WINDOW_SCALE of the screen (0.5x0.5 = quarter-area)
WINDOW_TOP_FRAC = 0.05  # window's top edge sits this far down the screen (horizontally still centered)
BLINK_THRESHOLD_FACTOR = 0.55  # stricter than a0.calibration's default 0.65: requires a more definite closure
BLINK_MIN_CONSECUTIVE_FRAMES = 3  # a single noisy low-EAR frame (e.g. from looking to the side) isn't a blink

CALIB_MARGIN = 0.15  # corner targets sit this far in from the window's edges
CALIB_TARGET_VALID_FRAMES = 20
CALIB_MIN_VALID_FRAMES = 10
CALIB_MAX_MS = 2500
CALIB_SETTLE_MS = 300
CALIB_TRANSITION_MS = 200
CALIB_MIN_SUCCESSFUL_POINTS = 3

ZONE_NO, ZONE_YES = "NO", "YES"
NO_COLOR, YES_COLOR = (60, 60, 180), (60, 160, 60)     # idle background (BGR)
NO_COLOR_ACTIVE, YES_COLOR_ACTIVE = (80, 80, 240), (80, 210, 80)  # while gaze is on that side

WINDOW_NAME = "GazeFlow Sample: Yes/No"


class WindowedUI:
    """A normal window (not fullscreen), sized to a fraction of the screen.
    All drawing happens in this window's own local normalized [0,1]
    coordinates -- calibration targets and gaze predictions both live in
    that same space, so no screen<->window remapping is needed anywhere."""

    def __init__(self, screen_width_px: int, screen_height_px: int, scale: float = WINDOW_SCALE,
                 top_frac: float = WINDOW_TOP_FRAC):
        import cv2

        self.width = int(screen_width_px * scale)
        self.height = int(screen_height_px * scale)
        self._cv2 = cv2
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        x = int(screen_width_px * (1.0 - scale) / 2.0)  # horizontally centered
        y = int(screen_height_px * top_frac)
        cv2.moveWindow(WINDOW_NAME, x, y)

    def new_canvas(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def show(self, canvas: np.ndarray, wait_ms: int = 1) -> int:
        self._cv2.imshow(WINDOW_NAME, canvas)
        return self._cv2.waitKey(wait_ms) & 0xFF

    def close(self) -> None:
        self._cv2.destroyWindow(WINDOW_NAME)

    def draw_target(self, canvas: np.ndarray, x_norm: float, y_norm: float, progress: float) -> None:
        cv2 = self._cv2
        cx, cy = int(x_norm * self.width), int(y_norm * self.height)
        max_r, min_r, dot_r = 18, 5, 3
        ring_r = int(max_r - (max_r - min_r) * max(0.0, min(1.0, progress)))
        cv2.circle(canvas, (cx, cy), ring_r, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), dot_r, (255, 255, 255), -1, cv2.LINE_AA)


def _put_centered(win: WindowedUI, canvas: np.ndarray, text: str, y_frac: float, scale: float = 1.0, color=(255, 255, 255)) -> None:
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
        _put_centered(win, canvas, "Look naturally,", y_frac=0.45, scale=0.7)
        _put_centered(win, canvas, "eyes open...", y_frac=0.58, scale=0.7)
        key = win.show(canvas, wait_ms=1)
        if key == a0_ui.ESC_KEY:
            raise KeyboardInterrupt
    return calibration.compute_baseline_ear(left_samples, right_samples)


def calibration_points(count: int) -> list[tuple[str, float, float]]:
    m = CALIB_MARGIN
    if count == 5:
        points = [("center", 0.5, 0.5), ("tl", m, m), ("tr", 1 - m, m), ("bl", m, 1 - m), ("br", 1 - m, 1 - m)]
    elif count == 4:
        points = [("tl", m, m), ("tr", 1 - m, m), ("bl", m, 1 - m), ("br", 1 - m, 1 - m)]
    else:
        raise ValueError(f"--calibration-points must be 4 or 5, got {count}")
    rng = np.random.default_rng(42)
    order = rng.permutation(len(points))
    return [points[i] for i in order]


def _wait_for_start(win: WindowedUI) -> None:
    canvas = win.new_canvas()
    _put_centered(win, canvas, "Follow the dot with your eyes.", y_frac=0.4, scale=0.6)
    _put_centered(win, canvas, "SPACE to start, ESC to cancel.", y_frac=0.55, scale=0.6)
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
                     win: WindowedUI, blink_threshold: float, count: int) -> GazeModel:
    _wait_for_start(win)

    all_X, all_yx, all_yy, all_pid = [], [], [], []
    prev_xy = (0.5, 0.5)
    for point_id, tx, ty in calibration_points(count):
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
    from a0.calibration import select_ridge_alpha
    cv_result = select_ridge_alpha(X, yx, yy, np.array(all_pid))
    print(f"Calibration fit: alpha={cv_result.selected_alpha}, CV median error (norm)={cv_result.median_error_norm:.4f}")
    return GazeModel.fit(X, yx, yy, cv_result.selected_alpha)


def draw_idle_screen(win: WindowedUI, question: str, zone: str | None, gaze_xy: tuple[float, float] | None) -> np.ndarray:
    import cv2

    canvas = win.new_canvas()
    mid = win.width // 2
    no_color = NO_COLOR_ACTIVE if zone == ZONE_NO else NO_COLOR
    yes_color = YES_COLOR_ACTIVE if zone == ZONE_YES else YES_COLOR
    canvas[:, :mid] = no_color
    canvas[:, mid:] = yes_color

    _put_centered(win, canvas, question, y_frac=0.15, scale=0.7)
    _put_centered(win, canvas, "NO", y_frac=0.5, scale=1.6)
    x = 3 * win.width // 4
    size, _ = cv2.getTextSize("YES", cv2.FONT_HERSHEY_SIMPLEX, 1.6, 2)
    cv2.putText(canvas, "YES", (x - size[0] // 2, int(win.height * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA)

    if gaze_xy is not None:
        x_clamped, y_clamped = min(1.0, max(0.0, gaze_xy[0])), min(1.0, max(0.0, gaze_xy[1]))
        cv2.circle(canvas, (int(x_clamped * win.width), int(y_clamped * win.height)), 6, (0, 0, 255), 2, cv2.LINE_AA)

    _put_centered(win, canvas, "Look + blink to select. ESC to quit.", y_frac=0.93, scale=0.45)
    return canvas


def draw_confirmation_screen(win: WindowedUI, zone: str) -> np.ndarray:
    canvas = win.new_canvas()
    canvas[:] = NO_COLOR_ACTIVE if zone == ZONE_NO else YES_COLOR_ACTIVE
    _put_centered(win, canvas, f"Selected: {zone}", y_frac=0.5, scale=1.2)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--calibration-points", type=int, default=5, choices=[4, 5])
    parser.add_argument("--question", type=str, default="Yes or No?")
    args = parser.parse_args()

    screen_w, screen_h = a0_ui.get_primary_screen_size_px()

    camera = Camera(args.camera, args.camera_width, args.camera_height, args.camera_fps)
    camera.open()
    landmarker = FaceLandmarkerWrapper()
    extractor = FeatureExtractor(camera.actual_width, camera.actual_height)
    win = WindowedUI(screen_w, screen_h)

    try:
        blink_threshold = calibration.blink_threshold(
            collect_baseline_ear(camera, landmarker, extractor, win), factor=BLINK_THRESHOLD_FACTOR
        )
        print(f"Blink threshold set: {blink_threshold:.4f}")

        gaze_model = run_calibration(camera, landmarker, extractor, win, blink_threshold, args.calibration_points)

        blink_streak = 0
        smoothed_xy: tuple[float, float] | None = None
        current_zone: str | None = None
        last_face_seen = time.monotonic()
        confirming_until: float | None = None
        confirmed_zone: str | None = None

        while True:
            frame = camera.read()
            if frame is None:
                continue

            now = time.monotonic()
            if confirming_until is not None:
                if now >= confirming_until:
                    confirming_until = None
                    confirmed_zone = None
                    current_zone = None
                else:
                    key = win.show(draw_confirmation_screen(win, confirmed_zone), wait_ms=1)
                    if key == a0_ui.ESC_KEY:
                        break
                    continue

            result = landmarker.detect(frame.image_bgr, frame.timestamp_ms)
            fv = extractor.extract(result.landmarks_norm) if result.face_detected else None

            zone = None
            blinking = False
            if fv is not None:
                pred_x, pred_y = gaze_model.predict(fv.to_array().reshape(1, -1))
                raw_xy = (float(pred_x[0]), float(pred_y[0]))
                smoothed_xy = raw_xy if smoothed_xy is None else (
                    GAZE_SMOOTHING_ALPHA * raw_xy[0] + (1 - GAZE_SMOOTHING_ALPHA) * smoothed_xy[0],
                    GAZE_SMOOTHING_ALPHA * raw_xy[1] + (1 - GAZE_SMOOTHING_ALPHA) * smoothed_xy[1],
                )
                zone = ZONE_YES if smoothed_xy[0] >= 0.5 else ZONE_NO
                blinking = calibration.is_blinking(fv.left_ear, fv.right_ear, blink_threshold)
                last_face_seen = now
            elif current_zone is not None and (now - last_face_seen) < FACE_LOST_GRACE_SECONDS:
                zone = current_zone  # tolerate a brief tracking dropout

            current_zone = zone
            blink_streak = blink_streak + 1 if blinking else 0

            if current_zone is not None and blink_streak >= BLINK_MIN_CONSECUTIVE_FRAMES:
                confirmed_zone = current_zone
                confirming_until = now + CONFIRMATION_DISPLAY_SECONDS
                blink_streak = 0
                print(f"Selected: {confirmed_zone}")
                continue

            canvas = draw_idle_screen(win, args.question, current_zone, smoothed_xy)
            key = win.show(canvas, wait_ms=1)
            if key == a0_ui.ESC_KEY:
                break

    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f"Calibration failed: {exc}")
    finally:
        camera.release()
        landmarker.close()
        win.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
