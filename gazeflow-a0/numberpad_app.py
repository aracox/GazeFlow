"""Sample app: a 0-9 number pad driven by gaze + double-blink, with a
small output window showing the digits picked so far.

Ten boxes (0-4 on top, 5-9 on bottom) in a normal (non-fullscreen) window
near the top of the screen, plus a second small "output" window directly
below it showing the accumulated digit string. Look at a box and blink
TWICE in quick succession to select it -- a single blink does nothing here
(unlike sample_app.py's YES/NO picker), so a normal blink while just
looking around a box doesn't accidentally enter a digit.

Does its own fresh in-app calibration at startup, same as sample_app.py --
see gaze_ui_common.py.

Usage:
    python numberpad_app.py [--camera N] [--calibration-points 4|5]

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
from gaze_ui_common import (
    BLINK_MIN_CONSECUTIVE_FRAMES,
    BLINK_THRESHOLD_FACTOR,
    WindowedUI,
    centered_window,
    collect_baseline_ear,
    cross_calibration_points,
    put_centered,
    run_calibration,
)

GAZE_SMOOTHING_ALPHA = 0.12
FACE_LOST_GRACE_SECONDS = 0.4
DOUBLE_BLINK_WINDOW_SECONDS = 0.8  # max gap between the two blinks of a double-blink
SELECTION_FLASH_SECONDS = 0.4      # how long a just-selected box stays highlighted

DIGIT_GRID = [str(d) for d in range(10)]  # index -> label; row = idx // 5, col = idx % 5
GRID_MARGIN = 0.14

BG_COLOR = (30, 30, 30)
BOX_COLOR = (70, 70, 70)
BOX_ACTIVE_COLOR = (110, 110, 60)     # gaze currently on this box
BOX_ARMED_COLOR = (60, 110, 170)      # first blink registered, waiting for the second
BOX_FLASH_COLOR = (60, 200, 60)       # just confirmed

MAIN_WINDOW_NAME = "GazeFlow Sample: Number Pad"
OUTPUT_WINDOW_NAME = "GazeFlow Sample: Output"


def digit_box_centers() -> list[tuple[float, float]]:
    xs = np.linspace(GRID_MARGIN, 1 - GRID_MARGIN, 5)
    ys = [0.32, 0.72]
    return [(float(xs[i % 5]), ys[i // 5]) for i in range(10)]


def nearest_digit(gaze_xy: tuple[float, float]) -> int:
    centers = digit_box_centers()
    dists = [((gaze_xy[0] - cx) ** 2 + (gaze_xy[1] - cy) ** 2) for cx, cy in centers]
    return min(range(10), key=lambda i: dists[i])


def draw_numberpad(win: WindowedUI, gaze_xy: tuple[float, float] | None, active_digit: int | None,
                    armed_digit: int | None, flash_digit: int | None) -> np.ndarray:
    import cv2

    canvas = win.new_canvas()
    canvas[:] = BG_COLOR
    box_w = int(win.width / 5 * 0.8)
    box_h = int(win.height / 2 * 0.55)

    for i, (cx_norm, cy_norm) in enumerate(digit_box_centers()):
        cx, cy = int(cx_norm * win.width), int(cy_norm * win.height)
        color = BOX_COLOR
        if flash_digit == i:
            color = BOX_FLASH_COLOR
        elif armed_digit == i:
            color = BOX_ARMED_COLOR
        elif active_digit == i:
            color = BOX_ACTIVE_COLOR
        cv2.rectangle(canvas, (cx - box_w // 2, cy - box_h // 2), (cx + box_w // 2, cy + box_h // 2), color, -1)
        cv2.rectangle(canvas, (cx - box_w // 2, cy - box_h // 2), (cx + box_w // 2, cy + box_h // 2), (200, 200, 200), 1)
        label = DIGIT_GRID[i]
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)
        cv2.putText(canvas, label, (cx - size[0] // 2, cy + size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)

    if gaze_xy is not None:
        x_clamped, y_clamped = min(1.0, max(0.0, gaze_xy[0])), min(1.0, max(0.0, gaze_xy[1]))
        cv2.circle(canvas, (int(x_clamped * win.width), int(y_clamped * win.height)), 5, (0, 0, 255), 2, cv2.LINE_AA)

    put_centered(win, canvas, "Look at a box, blink TWICE quickly to select. ESC to quit.", y_frac=0.95, scale=0.42)
    return canvas


def draw_output(win: WindowedUI, digits: str) -> np.ndarray:
    canvas = win.new_canvas()
    canvas[:] = (20, 20, 20)
    put_centered(win, canvas, digits if digits else "-", y_frac=0.65, scale=1.3)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--calibration-points", type=int, default=5, choices=[4, 5])
    args = parser.parse_args()

    screen_w, screen_h = a0_ui.get_primary_screen_size_px()

    camera = Camera(args.camera, args.camera_width, args.camera_height, args.camera_fps)
    camera.open()
    landmarker = FaceLandmarkerWrapper()
    extractor = FeatureExtractor(camera.actual_width, camera.actual_height)
    win = centered_window(MAIN_WINDOW_NAME, screen_w, screen_h)
    output_win = WindowedUI(OUTPUT_WINDOW_NAME, width_px=win.width, height_px=int(screen_h * 0.08),
                             x_px=win.x_px, y_px=win.y_px + win.height + 10)

    try:
        blink_threshold = calibration.blink_threshold(
            collect_baseline_ear(camera, landmarker, extractor, win), factor=BLINK_THRESHOLD_FACTOR
        )
        print(f"Blink threshold set: {blink_threshold:.4f}")

        gaze_model = run_calibration(camera, landmarker, extractor, win, blink_threshold,
                                      cross_calibration_points(args.calibration_points))

        eyes_closed_run = 0
        first_blink_time: float | None = None
        armed_digit: int | None = None
        flash_digit: int | None = None
        flash_until = 0.0
        smoothed_xy: tuple[float, float] | None = None
        active_digit: int | None = None
        last_active_digit: int | None = None
        last_face_seen = time.monotonic()
        entered_digits = ""

        while True:
            frame = camera.read()
            if frame is None:
                continue
            now = time.monotonic()

            result = landmarker.detect(frame.image_bgr, frame.timestamp_ms)
            fv = extractor.extract(result.landmarks_norm) if result.face_detected else None

            active_digit = None
            blinking = False
            if fv is not None:
                pred_x, pred_y = gaze_model.predict(fv.to_array().reshape(1, -1))
                raw_xy = (float(pred_x[0]), float(pred_y[0]))
                smoothed_xy = raw_xy if smoothed_xy is None else (
                    GAZE_SMOOTHING_ALPHA * raw_xy[0] + (1 - GAZE_SMOOTHING_ALPHA) * smoothed_xy[0],
                    GAZE_SMOOTHING_ALPHA * raw_xy[1] + (1 - GAZE_SMOOTHING_ALPHA) * smoothed_xy[1],
                )
                active_digit = nearest_digit(smoothed_xy)
                blinking = calibration.is_blinking(fv.left_ear, fv.right_ear, blink_threshold)
                last_face_seen = now
            elif last_active_digit is not None and (now - last_face_seen) < FACE_LOST_GRACE_SECONDS:
                active_digit = last_active_digit  # tolerate a brief tracking dropout
            last_active_digit = active_digit

            # completed-blink edge detection (transition out of a sustained closure)
            blink_event = False
            if blinking:
                eyes_closed_run += 1
            else:
                if eyes_closed_run >= BLINK_MIN_CONSECUTIVE_FRAMES:
                    blink_event = True
                eyes_closed_run = 0

            if blink_event:
                if active_digit is not None:
                    if (first_blink_time is not None
                            and (now - first_blink_time) <= DOUBLE_BLINK_WINDOW_SECONDS
                            and armed_digit == active_digit):
                        entered_digits += DIGIT_GRID[active_digit]
                        print(f"Selected: {DIGIT_GRID[active_digit]}  (output so far: {entered_digits})")
                        flash_digit, flash_until = active_digit, now + SELECTION_FLASH_SECONDS
                        first_blink_time, armed_digit = None, None
                    else:
                        first_blink_time, armed_digit = now, active_digit
                else:
                    first_blink_time, armed_digit = None, None

            if flash_digit is not None and now >= flash_until:
                flash_digit = None

            win.show(draw_numberpad(win, smoothed_xy, active_digit, armed_digit, flash_digit), wait_ms=1)
            key = output_win.show(draw_output(output_win, entered_digits), wait_ms=1)
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
        output_win.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
