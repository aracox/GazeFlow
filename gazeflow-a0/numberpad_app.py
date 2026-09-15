"""Sample app: a 0-9 number pad driven by gaze + double-blink, with the
digits picked so far shown as a footer line in the same window.

Two-stage selection instead of one 10-way grid, because a single-stage 5x2
grid turned out too imprecise in practice (mixing a left/right AND a
top/bottom discrimination per pick, and the vertical axis is the weaker
one for this gaze model per A0's findings):

  Stage 1: look left ("0-4") or right ("5-9") -- the same big two-zone
           split validated in sample_app.py's YES/NO picker.
  Stage 2: the five digits in that group, in a single row spanning the
           full window width -- only left/right position matters, no row
           ambiguity.

Look at a box and blink TWICE in quick succession to select it at either
stage; a single blink does nothing (unlike sample_app.py's single-blink
YES/NO), so it takes a deliberate double-blink to confirm.

Does its own fresh in-app calibration at startup, same as sample_app.py --
see gaze_ui_common.py.

Usage:
    python numberpad_app.py [--camera N] [--calibration-points 4|5]

Like the rest of A0, this is a sample/demo script, not production code.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

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
SELECTION_FLASH_SECONDS = 0.4      # how long a just-confirmed box stays highlighted

ROW_MARGIN = 0.10  # digit boxes in stage 2 sit this far in from the window's left/right edges

BG_COLOR = (30, 30, 30)
BOX_COLOR = (70, 70, 70)
BOX_ACTIVE_COLOR = (110, 110, 60)     # gaze currently on this box
BOX_ARMED_COLOR = (60, 110, 170)      # first blink registered, waiting for the second
BOX_FLASH_COLOR = (60, 200, 60)       # just confirmed

MAIN_WINDOW_NAME = "GazeFlow Sample: Number Pad"

GROUPS = {"0-4": ["0", "1", "2", "3", "4"], "5-9": ["5", "6", "7", "8", "9"]}


@dataclass
class Box:
    label: str
    cx: float
    cy: float
    w: float
    h: float


def group_boxes() -> list[Box]:
    labels = list(GROUPS.keys())
    return [Box(labels[0], 0.25, 0.5, 0.46, 0.7), Box(labels[1], 0.75, 0.5, 0.46, 0.7)]


def digit_boxes(group_label: str) -> list[Box]:
    digits = GROUPS[group_label]
    xs = np.linspace(ROW_MARGIN, 1 - ROW_MARGIN, len(digits))
    box_w = (1 - 2 * ROW_MARGIN) / len(digits) * 0.85
    return [Box(d, float(x), 0.5, box_w, 0.6) for d, x in zip(digits, xs)]


def nearest_box(gaze_xy: tuple[float, float], boxes: list[Box]) -> int:
    dists = [((gaze_xy[0] - b.cx) ** 2 + (gaze_xy[1] - b.cy) ** 2) for b in boxes]
    return min(range(len(boxes)), key=lambda i: dists[i])


def draw_boxes(win: WindowedUI, boxes: list[Box], gaze_xy: tuple[float, float] | None,
               active_idx: int | None, armed_idx: int | None, flash_idx: int | None, entered_digits: str, footer: str) -> np.ndarray:
    import cv2

    canvas = win.new_canvas()
    canvas[:] = BG_COLOR

    for i, box in enumerate(boxes):
        cx, cy = int(box.cx * win.width), int(box.cy * win.height)
        bw, bh = int(box.w * win.width), int(box.h * win.height)
        color = BOX_COLOR
        if flash_idx == i:
            color = BOX_FLASH_COLOR
        elif armed_idx == i:
            color = BOX_ARMED_COLOR
        elif active_idx == i:
            color = BOX_ACTIVE_COLOR
        cv2.rectangle(canvas, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), color, -1)
        cv2.rectangle(canvas, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), (200, 200, 200), 1)
        size, _ = cv2.getTextSize(box.label, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)
        cv2.putText(canvas, box.label, (cx - size[0] // 2, cy + size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)

    if gaze_xy is not None:
        x_clamped, y_clamped = min(1.0, max(0.0, gaze_xy[0])), min(1.0, max(0.0, gaze_xy[1]))
        cv2.circle(canvas, (int(x_clamped * win.width), int(y_clamped * win.height)), 5, (0, 0, 255), 2, cv2.LINE_AA)

    put_centered(win, canvas, f"Output: {entered_digits if entered_digits else '-'}", y_frac=0.87, scale=0.6)
    put_centered(win, canvas, footer, y_frac=0.95, scale=0.42)
    return canvas


@dataclass
class BlinkPicker:
    """Double-blink-to-select state machine over whatever `boxes` are
    currently shown. Call `.step(gaze_xy, blinking, now)` once per frame;
    returns the selected box index, or None."""
    eyes_closed_run: int = 0
    first_blink_time: float | None = None
    armed_idx: int | None = None
    flash_idx: int | None = None
    flash_until: float = 0.0
    active_idx: int | None = None
    last_active_idx: int | None = None
    last_seen: float = field(default_factory=time.monotonic)

    def reset(self) -> None:
        self.__init__()

    def step(self, boxes: list[Box], gaze_xy: tuple[float, float] | None, blinking: bool, now: float) -> int | None:
        self.active_idx = None
        if gaze_xy is not None:
            self.active_idx = nearest_box(gaze_xy, boxes)
            self.last_seen = now
        elif self.last_active_idx is not None and (now - self.last_seen) < FACE_LOST_GRACE_SECONDS:
            self.active_idx = self.last_active_idx
        self.last_active_idx = self.active_idx

        blink_event = False
        if blinking:
            self.eyes_closed_run += 1
        else:
            if self.eyes_closed_run >= BLINK_MIN_CONSECUTIVE_FRAMES:
                blink_event = True
            self.eyes_closed_run = 0

        selected = None
        if blink_event:
            if self.active_idx is not None:
                if (self.first_blink_time is not None
                        and (now - self.first_blink_time) <= DOUBLE_BLINK_WINDOW_SECONDS
                        and self.armed_idx == self.active_idx):
                    selected = self.active_idx
                    self.flash_idx, self.flash_until = self.active_idx, now + SELECTION_FLASH_SECONDS
                    self.first_blink_time, self.armed_idx = None, None
                else:
                    self.first_blink_time, self.armed_idx = now, self.active_idx
            else:
                self.first_blink_time, self.armed_idx = None, None

        if self.flash_idx is not None and now >= self.flash_until:
            self.flash_idx = None
        return selected


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

    try:
        blink_threshold = calibration.blink_threshold(
            collect_baseline_ear(camera, landmarker, extractor, win), factor=BLINK_THRESHOLD_FACTOR
        )
        print(f"Blink threshold set: {blink_threshold:.4f}")

        gaze_model = run_calibration(camera, landmarker, extractor, win, blink_threshold,
                                      cross_calibration_points(args.calibration_points))

        smoothed_xy: tuple[float, float] | None = None
        entered_digits = ""
        stage = "group"
        boxes = group_boxes()
        picker = BlinkPicker()

        while True:
            frame = camera.read()
            if frame is None:
                continue
            now = time.monotonic()

            result = landmarker.detect(frame.image_bgr, frame.timestamp_ms)
            fv = extractor.extract(result.landmarks_norm) if result.face_detected else None

            gaze_xy = None
            blinking = False
            if fv is not None:
                pred_x, pred_y = gaze_model.predict(fv.to_array().reshape(1, -1))
                raw_xy = (float(pred_x[0]), float(pred_y[0]))
                smoothed_xy = raw_xy if smoothed_xy is None else (
                    GAZE_SMOOTHING_ALPHA * raw_xy[0] + (1 - GAZE_SMOOTHING_ALPHA) * smoothed_xy[0],
                    GAZE_SMOOTHING_ALPHA * raw_xy[1] + (1 - GAZE_SMOOTHING_ALPHA) * smoothed_xy[1],
                )
                gaze_xy = smoothed_xy
                blinking = calibration.is_blinking(fv.left_ear, fv.right_ear, blink_threshold)

            selected = picker.step(boxes, gaze_xy, blinking, now)

            if selected is not None:
                if stage == "group":
                    stage = "digit"
                    boxes = digit_boxes(boxes[selected].label)
                else:
                    entered_digits += boxes[selected].label
                    print(f"Selected: {boxes[selected].label}  (output so far: {entered_digits})")
                    stage = "group"
                    boxes = group_boxes()
                picker.reset()

            footer = "Pick a group: blink TWICE to select. ESC to quit." if stage == "group" \
                else "Pick a digit: blink TWICE to select. ESC to quit."
            canvas = draw_boxes(win, boxes, smoothed_xy, picker.active_idx, picker.armed_idx, picker.flash_idx, entered_digits, footer)
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
