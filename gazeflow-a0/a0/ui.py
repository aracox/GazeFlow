"""OpenCV-based fullscreen UI: instructions, calibration/validation targets,
camera preview with debug overlays, and the completion summary.

This is intentionally minimal (section 2): a black canvas, a ring-and-dot
target, and text overlays. No polish beyond what's needed to run A0.
"""
from __future__ import annotations

import numpy as np

WINDOW_NAME = "GazeFlow A0"

ESC_KEY = 27
SPACE_KEY = 32

_TARGET_COLOR = (255, 255, 255)
_TEXT_COLOR = (255, 255, 255)
_BG_COLOR = (20, 20, 20)


def get_primary_screen_size_px() -> tuple[int, int]:
    try:
        import screeninfo
        monitors = screeninfo.get_monitors()
        if monitors:
            m = monitors[0]
            return int(m.width), int(m.height)
    except Exception:
        pass
    return 1280, 800


class FullscreenUI:
    def __init__(self, screen_width_px: int, screen_height_px: int):
        import cv2

        self.width = screen_width_px
        self.height = screen_height_px
        self._cv2 = cv2
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def new_canvas(self) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = _BG_COLOR
        return canvas

    def show(self, canvas: np.ndarray, wait_ms: int = 1) -> int:
        self._cv2.imshow(WINDOW_NAME, canvas)
        return self._cv2.waitKey(wait_ms) & 0xFF

    def show_text_screen(self, lines: list[str], wait_for_key: bool = True) -> int:
        cv2 = self._cv2
        canvas = self.new_canvas()
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_height = 40
        total_height = line_height * len(lines)
        y0 = self.height // 2 - total_height // 2
        for i, line in enumerate(lines):
            size, _ = cv2.getTextSize(line, font, 0.8, 2)
            x = self.width // 2 - size[0] // 2
            y = y0 + i * line_height
            cv2.putText(canvas, line, (x, y), font, 0.8, _TEXT_COLOR, 2, cv2.LINE_AA)
        key = self.show(canvas, wait_ms=1 if not wait_for_key else 0)
        return key

    def draw_target(self, canvas: np.ndarray, x_norm: float, y_norm: float, progress: float) -> None:
        """progress in [0,1]: 0 = outer ring at full size, 1 = collapsed to dot."""
        cv2 = self._cv2
        cx = int(x_norm * self.width)
        cy = int(y_norm * self.height)
        max_r, min_r, dot_r = 26, 6, 4
        ring_r = int(max_r - (max_r - min_r) * max(0.0, min(1.0, progress)))
        cv2.circle(canvas, (cx, cy), ring_r, _TARGET_COLOR, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), dot_r, _TARGET_COLOR, -1, cv2.LINE_AA)

    def draw_gaze_point(self, canvas: np.ndarray, x_norm: float, y_norm: float) -> None:
        cv2 = self._cv2
        cx = int(x_norm * self.width)
        cy = int(y_norm * self.height)
        cv2.circle(canvas, (cx, cy), 12, (0, 0, 255), -1, cv2.LINE_AA)

    def close(self) -> None:
        self._cv2.destroyWindow(WINDOW_NAME)


def draw_preflight_overlay(
    image_bgr: np.ndarray,
    landmarks_norm: np.ndarray | None,
    camera_fps: float,
    vision_fps: float,
    face_detected: bool,
    left_ear: float | None,
    right_ear: float | None,
    head_pose,
    face_scale: float | None,
) -> np.ndarray:
    """Draws face/eye/iris landmarks and diagnostic text for `preflight`."""
    import cv2

    from . import landmarks as lm

    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    if landmarks_norm is not None:
        eye_indices = [
            lm.LEFT_EYE_OUTER, lm.LEFT_EYE_INNER, lm.LEFT_EYE_TOP, lm.LEFT_EYE_BOTTOM,
            lm.RIGHT_EYE_OUTER, lm.RIGHT_EYE_INNER, lm.RIGHT_EYE_TOP, lm.RIGHT_EYE_BOTTOM,
        ]
        for i in range(0, 468, 4):  # sparse face outline dots to keep it readable
            x, y = landmarks_norm[i, 0] * w, landmarks_norm[i, 1] * h
            cv2.circle(canvas, (int(x), int(y)), 1, (80, 200, 80), -1)
        for i in eye_indices:
            x, y = landmarks_norm[i, 0] * w, landmarks_norm[i, 1] * h
            cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 255), -1)
        for i in (lm.LEFT_IRIS_CENTER, lm.RIGHT_IRIS_CENTER):
            x, y = landmarks_norm[i, 0] * w, landmarks_norm[i, 1] * h
            cv2.circle(canvas, (int(x), int(y)), 3, (0, 0, 255), -1)

    lines = [
        f"face_detected: {face_detected}",
        f"camera_fps: {camera_fps:.1f}  vision_fps: {vision_fps:.1f}",
        f"left_ear: {left_ear:.3f}" if left_ear is not None else "left_ear: n/a",
        f"right_ear: {right_ear:.3f}" if right_ear is not None else "right_ear: n/a",
        f"face_scale: {face_scale:.3f}" if face_scale is not None else "face_scale: n/a",
        (f"yaw/pitch/roll: {head_pose.yaw_deg:.1f}/{head_pose.pitch_deg:.1f}/{head_pose.roll_deg:.1f} deg"
         if head_pose is not None and head_pose.success else "head_pose: n/a"),
        "ESC to quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas
