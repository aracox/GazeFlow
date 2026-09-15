"""Sample app: a windowed YES/NO picker driven by gaze + blink.

Runs in a normal (non-fullscreen) window, centered on screen, sized to a
quarter of the screen's width and height. Look at the left half (NO) or
right half (YES) of the window; blink while looking at a side to select it.

Reuses A0's existing camera/landmark/feature/calibration/model code rather
than reimplementing gaze tracking. Calibration is NOT redone here -- it
reuses the Ridge model already fit for a completed `a0.main run` session
(same approach as `python -m a0.main live`), so run A0 first. A fresh,
short eyes-open baseline is collected at startup to set this session's
blink threshold, since that's sensitive to current lighting.

Usage:
    python sample_app.py outputs/<run-id> [--camera N] [--question "..."]

Like the rest of A0, this is a sample/demo script, not production code.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from a0 import calibration, ui
from a0.camera import Camera
from a0.features import FEATURE_NAMES, FeatureExtractor
from a0.landmarks import FaceLandmarkerWrapper
from a0.model import GazeModel

GAZE_SMOOTHING_ALPHA = 0.3  # EMA weight on each new prediction; lower = smoother/laggier
CONFIRMATION_DISPLAY_SECONDS = 1.5
BASELINE_DURATION_SECONDS = 2.0
FACE_LOST_GRACE_SECONDS = 0.4  # tolerate brief tracking dropouts without losing the highlighted side
WINDOW_SCALE = 0.5  # window is WINDOW_SCALE x WINDOW_SCALE of the screen (0.5x0.5 = quarter-area)
WINDOW_TOP_FRAC = 0.05  # window's top edge sits this far down the screen (horizontally still centered)
BLINK_THRESHOLD_FACTOR = 0.55  # stricter than a0.calibration's default 0.65: requires a more definite closure
BLINK_MIN_CONSECUTIVE_FRAMES = 3  # a single noisy low-EAR frame (e.g. from looking to the side) isn't a blink

ZONE_NO, ZONE_YES = "NO", "YES"
NO_COLOR, YES_COLOR = (60, 60, 180), (60, 160, 60)     # idle background (BGR)
NO_COLOR_ACTIVE, YES_COLOR_ACTIVE = (80, 80, 240), (80, 210, 80)  # while gaze is on that side

WINDOW_NAME = "GazeFlow Sample: Yes/No"


class WindowedUI:
    """Same drawing interface as a0.ui.FullscreenUI, but a normal, centered
    window sized to a fraction of the screen instead of fullscreen."""

    def __init__(self, screen_width_px: int, screen_height_px: int, scale: float = WINDOW_SCALE,
                 top_frac: float = WINDOW_TOP_FRAC):
        import cv2

        self.width = int(screen_width_px * scale)
        self.height = int(screen_height_px * scale)
        self._cv2 = cv2
        self._x_offset_norm = (1.0 - scale) / 2.0  # horizontally centered
        self._y_offset_norm = top_frac
        self._scale = scale
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        x = int(screen_width_px * self._x_offset_norm)
        y = int(screen_height_px * self._y_offset_norm)
        cv2.moveWindow(WINDOW_NAME, x, y)

    def new_canvas(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def show(self, canvas: np.ndarray, wait_ms: int = 1) -> int:
        self._cv2.imshow(WINDOW_NAME, canvas)
        return self._cv2.waitKey(wait_ms) & 0xFF

    def close(self) -> None:
        self._cv2.destroyWindow(WINDOW_NAME)

    def to_local_norm(self, screen_x_norm: float, screen_y_norm: float) -> tuple[float, float]:
        """The gaze model predicts positions normalized to the FULL screen
        (that's what it was calibrated against); this window only covers a
        `self._scale` fraction of it, offset from the screen's top-left by
        (_x_offset_norm, _y_offset_norm), so cursor drawing needs to remap
        into the window's own local normalized coordinates."""
        return (
            (screen_x_norm - self._x_offset_norm) / self._scale,
            (screen_y_norm - self._y_offset_norm) / self._scale,
        )


def load_gaze_model(run_dir: Path) -> tuple[GazeModel, dict]:
    results = json.loads((run_dir / "a0_results.json").read_text())
    raw = pd.read_csv(run_dir / "a0_raw_features.csv")
    calib = raw[(raw["phase"] == "calibration") & (raw["sample_valid"] == True)]  # noqa: E712
    if calib.empty:
        raise RuntimeError(f"No valid calibration rows found in {run_dir}/a0_raw_features.csv")

    X = calib[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_x = calib["target_x_norm"].to_numpy(dtype=np.float64)
    y_y = calib["target_y_norm"].to_numpy(dtype=np.float64)
    model = GazeModel.fit(X, y_x, y_y, results["calibration"]["selected_ridge_alpha"])
    return model, results


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
        if key == ui.ESC_KEY:
            raise KeyboardInterrupt
    return calibration.compute_baseline_ear(left_samples, right_samples)


def _put_centered(win: WindowedUI, canvas: np.ndarray, text: str, y_frac: float, scale: float = 1.0, color=(255, 255, 255)) -> None:
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    size, _ = cv2.getTextSize(text, font, scale, 2)
    x = win.width // 2 - size[0] // 2
    y = int(win.height * y_frac)
    cv2.putText(canvas, text, (x, y), font, scale, color, 2, cv2.LINE_AA)


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
        local_x, local_y = win.to_local_norm(*gaze_xy)
        local_x, local_y = min(1.0, max(0.0, local_x)), min(1.0, max(0.0, local_y))
        cv2.circle(canvas, (int(local_x * win.width), int(local_y * win.height)), 6, (0, 0, 255), 2, cv2.LINE_AA)

    _put_centered(win, canvas, "Look + blink to select. ESC to quit.", y_frac=0.93, scale=0.45)
    return canvas


def draw_confirmation_screen(win: WindowedUI, zone: str) -> np.ndarray:
    canvas = win.new_canvas()
    canvas[:] = NO_COLOR_ACTIVE if zone == ZONE_NO else YES_COLOR_ACTIVE
    _put_centered(win, canvas, f"Selected: {zone}", y_frac=0.5, scale=1.2)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed a0.main run directory to reuse calibration from")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--question", type=str, default="Yes or No?")
    args = parser.parse_args()

    gaze_model, results = load_gaze_model(args.run_dir)
    screen_w, screen_h = results["screen"]["width_px"], results["screen"]["height_px"]

    camera = Camera(results["camera"]["index"] if args.camera is None else args.camera,
                     results["camera"]["width"], results["camera"]["height"], results["camera"]["requested_fps"])
    camera.open()
    landmarker = FaceLandmarkerWrapper()
    extractor = FeatureExtractor(camera.actual_width, camera.actual_height)
    win = WindowedUI(screen_w, screen_h)

    try:
        blink_threshold = calibration.blink_threshold(
            collect_baseline_ear(camera, landmarker, extractor, win), factor=BLINK_THRESHOLD_FACTOR
        )
        print(f"Blink threshold set: {blink_threshold:.4f}")
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
                    if key == ui.ESC_KEY:
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
            if key == ui.ESC_KEY:
                break

    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        landmarker.close()
        win.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
