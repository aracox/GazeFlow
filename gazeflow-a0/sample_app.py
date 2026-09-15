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

Shared setup code (windowing, baseline, calibration flow) lives in
gaze_ui_common.py -- also used by numberpad_app.py.

Usage:
    python sample_app.py [--camera N] [--calibration-points 4|5] [--question "..."]

Like the rest of A0, this is a sample/demo script, not production code.
"""
from __future__ import annotations

import argparse
import time

from a0 import calibration, ui as a0_ui
from a0.camera import Camera
from a0.features import FeatureExtractor
from a0.landmarks import FaceLandmarkerWrapper
from gaze_ui_common import (
    BLINK_MIN_CONSECUTIVE_FRAMES,
    BLINK_THRESHOLD_FACTOR,
    centered_window,
    collect_baseline_ear,
    cross_calibration_points,
    put_centered,
    run_calibration,
)

GAZE_SMOOTHING_ALPHA = 0.12  # EMA weight on each new prediction; lower = smoother/laggier
CONFIRMATION_DISPLAY_SECONDS = 1.5
FACE_LOST_GRACE_SECONDS = 0.4  # tolerate brief tracking dropouts without losing the highlighted side

ZONE_NO, ZONE_YES = "NO", "YES"
NO_COLOR, YES_COLOR = (60, 60, 180), (60, 160, 60)     # idle background (BGR)
NO_COLOR_ACTIVE, YES_COLOR_ACTIVE = (80, 80, 240), (80, 210, 80)  # while gaze is on that side

WINDOW_NAME = "GazeFlow Sample: Yes/No"


def draw_idle_screen(win, question: str, zone: str | None, gaze_xy: tuple[float, float] | None):
    import cv2

    canvas = win.new_canvas()
    mid = win.width // 2
    no_color = NO_COLOR_ACTIVE if zone == ZONE_NO else NO_COLOR
    yes_color = YES_COLOR_ACTIVE if zone == ZONE_YES else YES_COLOR
    canvas[:, :mid] = no_color
    canvas[:, mid:] = yes_color

    put_centered(win, canvas, question, y_frac=0.15, scale=0.7)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, half_center_x in (("NO", win.width // 4), ("YES", 3 * win.width // 4)):
        size, _ = cv2.getTextSize(label, font, 1.6, 2)
        cv2.putText(canvas, label, (half_center_x - size[0] // 2, int(win.height * 0.5)), font, 1.6, (255, 255, 255), 2, cv2.LINE_AA)

    if gaze_xy is not None:
        x_clamped, y_clamped = min(1.0, max(0.0, gaze_xy[0])), min(1.0, max(0.0, gaze_xy[1]))
        cv2.circle(canvas, (int(x_clamped * win.width), int(y_clamped * win.height)), 6, (0, 0, 255), 2, cv2.LINE_AA)

    put_centered(win, canvas, "Look + blink to select. ESC to quit.", y_frac=0.93, scale=0.45)
    return canvas


def draw_confirmation_screen(win, zone: str):
    canvas = win.new_canvas()
    canvas[:] = NO_COLOR_ACTIVE if zone == ZONE_NO else YES_COLOR_ACTIVE
    put_centered(win, canvas, f"Selected: {zone}", y_frac=0.5, scale=1.2)
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
    win = centered_window(WINDOW_NAME, screen_w, screen_h)

    try:
        blink_threshold = calibration.blink_threshold(
            collect_baseline_ear(camera, landmarker, extractor, win), factor=BLINK_THRESHOLD_FACTOR
        )
        print(f"Blink threshold set: {blink_threshold:.4f}")

        gaze_model = run_calibration(camera, landmarker, extractor, win, blink_threshold,
                                      cross_calibration_points(args.calibration_points))

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
