"""Sample app: a fullscreen YES/NO picker driven by gaze + blink.

Look at the left half (NO) or right half (YES) of the screen. Selecting
either side is confirmed by:
  - blinking while looking at that side, or
  - holding your gaze on that side for 2+ seconds (dwell selection).

Reuses A0's existing camera/landmark/feature/calibration/model/UI code
rather than reimplementing gaze tracking. Calibration is NOT redone here --
it reuses the Ridge model already fit for a completed `a0.main run` session
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

DWELL_SECONDS = 2.0
GAZE_SMOOTHING_ALPHA = 0.3  # EMA weight on each new prediction; lower = smoother/laggier
CONFIRMATION_DISPLAY_SECONDS = 1.5
BASELINE_DURATION_SECONDS = 2.0
FACE_LOST_GRACE_SECONDS = 0.4  # tolerate brief tracking dropouts without resetting dwell progress

ZONE_NO, ZONE_YES = "NO", "YES"
NO_COLOR, YES_COLOR = (60, 60, 180), (60, 160, 60)     # idle background (BGR)
NO_COLOR_ACTIVE, YES_COLOR_ACTIVE = (80, 80, 240), (80, 210, 80)  # while gaze is on that side


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


def collect_baseline_ear(camera: Camera, landmarker: FaceLandmarkerWrapper, extractor: FeatureExtractor, fui: ui.FullscreenUI) -> float:
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

        canvas = fui.new_canvas()
        _put_centered(fui, canvas, "Look naturally at the center, eyes open...", y_frac=0.5)
        key = fui.show(canvas, wait_ms=1)
        if key == ui.ESC_KEY:
            raise KeyboardInterrupt
    return calibration.compute_baseline_ear(left_samples, right_samples)


def _put_centered(fui: ui.FullscreenUI, canvas: np.ndarray, text: str, y_frac: float, scale: float = 1.0, color=(255, 255, 255)) -> None:
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    size, _ = cv2.getTextSize(text, font, scale, 2)
    x = fui.width // 2 - size[0] // 2
    y = int(fui.height * y_frac)
    cv2.putText(canvas, text, (x, y), font, scale, color, 2, cv2.LINE_AA)


def draw_idle_screen(fui: ui.FullscreenUI, question: str, zone: str | None, dwell_progress: float, gaze_xy: tuple[float, float] | None) -> np.ndarray:
    import cv2

    canvas = fui.new_canvas()
    mid = fui.width // 2
    no_color = NO_COLOR_ACTIVE if zone == ZONE_NO else NO_COLOR
    yes_color = YES_COLOR_ACTIVE if zone == ZONE_YES else YES_COLOR
    canvas[:, :mid] = no_color
    canvas[:, mid:] = yes_color

    _put_centered(fui, canvas, question, y_frac=0.12, scale=1.1)
    _put_centered(fui, canvas, "NO", y_frac=0.5, scale=3.0)
    x = 3 * fui.width // 4
    size, _ = cv2.getTextSize("YES", cv2.FONT_HERSHEY_SIMPLEX, 3.0, 2)
    cv2.putText(canvas, "YES", (x - size[0] // 2, int(fui.height * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 2, cv2.LINE_AA)

    if zone is not None:
        cx = fui.width // 4 if zone == ZONE_NO else 3 * fui.width // 4
        cy = int(fui.height * 0.72)
        fui.draw_target(canvas, cx / fui.width, cy / fui.height, dwell_progress)

    if gaze_xy is not None:
        cv2.circle(canvas, (int(gaze_xy[0] * fui.width), int(gaze_xy[1] * fui.height)), 8, (0, 0, 255), 2, cv2.LINE_AA)

    _put_centered(fui, canvas, "Look + blink, or hold gaze 2s to select. ESC to quit.", y_frac=0.95, scale=0.6)
    return canvas


def draw_confirmation_screen(fui: ui.FullscreenUI, zone: str) -> np.ndarray:
    canvas = fui.new_canvas()
    canvas[:] = NO_COLOR_ACTIVE if zone == ZONE_NO else YES_COLOR_ACTIVE
    _put_centered(fui, canvas, f"Selected: {zone}", y_frac=0.5, scale=2.5)
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
    fui = ui.FullscreenUI(screen_w, screen_h)

    try:
        blink_threshold = calibration.blink_threshold(collect_baseline_ear(camera, landmarker, extractor, fui))
        print(f"Blink threshold set: {blink_threshold:.4f}")

        smoothed_xy: tuple[float, float] | None = None
        current_zone: str | None = None
        zone_enter_time = time.monotonic()
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
                    zone_enter_time = now
                else:
                    key = fui.show(draw_confirmation_screen(fui, confirmed_zone), wait_ms=1)
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
                zone = current_zone  # tolerate a brief tracking dropout without losing dwell progress

            if zone != current_zone:
                current_zone = zone
                zone_enter_time = now
            dwell_elapsed = (now - zone_enter_time) if current_zone is not None else 0.0
            dwell_progress = min(1.0, dwell_elapsed / DWELL_SECONDS)

            if current_zone is not None and (blinking or dwell_elapsed >= DWELL_SECONDS):
                confirmed_zone = current_zone
                confirming_until = now + CONFIRMATION_DISPLAY_SECONDS
                print(f"Selected: {confirmed_zone}")
                continue

            canvas = draw_idle_screen(fui, args.question, current_zone, dwell_progress, smoothed_xy)
            key = fui.show(canvas, wait_ms=1)
            if key == ui.ESC_KEY:
                break

    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        landmarker.close()
        fui.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
