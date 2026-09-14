"""CLI entry point: preflight, run, report, live (section 6)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import platform
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration, features, geometry, landmarks, model, recorder, report, ui
from .camera import Camera

PROJECT_ROOT = Path(__file__).parent.parent
LOG = logging.getLogger("a0")


class UserCancelled(Exception):
    """Raised when ESC is pressed."""


# --- Config ---------------------------------------------------------------

@dataclass
class Config:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    viewing_distance_mm: float = 600
    screen_width_mm: float | None = None
    screen_height_mm: float | None = None

    calibration_target_margin: float = 0.10
    calibration_target_count: int = 9
    calibration_settle_ms: int = 300
    calibration_target_valid_frames: int = 20
    calibration_min_valid_frames: int = 15
    calibration_max_ms: int = 2000

    validation_target_count: int = 20
    validation_settle_ms: int = 300
    validation_valid_frames: int = 15

    random_seed: int = 42

    save_debug_video: bool = True

    @classmethod
    def load(cls, config_path: Path | None) -> "Config":
        data = {}
        if config_path is not None:
            data = json.loads(Path(config_path).read_text())
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def apply_cli_overrides(self, args: argparse.Namespace) -> None:
        if getattr(args, "camera", None) is not None:
            self.camera_index = args.camera
        if getattr(args, "viewing_distance_mm", None) is not None:
            self.viewing_distance_mm = args.viewing_distance_mm
        if getattr(args, "screen_width_mm", None) is not None:
            self.screen_width_mm = args.screen_width_mm
        if getattr(args, "screen_height_mm", None) is not None:
            self.screen_height_mm = args.screen_height_mm
        if getattr(args, "calibration_points", None) is not None:
            self.calibration_target_count = args.calibration_points
        if getattr(args, "save_debug_video", False):
            self.save_debug_video = True
        if getattr(args, "no_debug_video", False):
            self.save_debug_video = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --- Logging / setup -------------------------------------------------------

def setup_logging(run_dir: Path) -> None:
    LOG.setLevel(logging.DEBUG)
    LOG.handlers.clear()
    fh = logging.FileHandler(run_dir / "a0.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    LOG.addHandler(fh)
    LOG.addHandler(ch)


def log_environment() -> None:
    import importlib.metadata as im
    LOG.info("Python version: %s", platform.python_version())
    for pkg in ["opencv-python", "mediapipe", "numpy", "pandas", "scikit-learn", "matplotlib", "screeninfo"]:
        try:
            LOG.info("%s version: %s", pkg, im.version(pkg))
        except im.PackageNotFoundError:
            LOG.info("%s version: not found", pkg)


class FpsTracker:
    def __init__(self, window: int = 30):
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / span if span > 0 else 0.0


# --- Run context ------------------------------------------------------------

@dataclass
class Counters:
    face_detection_failures: int = 0
    blink_rejected_frames: int = 0
    valid_frames: int = 0
    total_frames: int = 0


@dataclass
class RunContext:
    config: Config
    camera: Camera
    landmarker: landmarks.FaceLandmarkerWrapper
    extractor: features.FeatureExtractor
    fui: ui.FullscreenUI
    raw_writer: recorder.RawFeatureWriter
    lm_writer: recorder.LandmarksWriter
    video_writer: recorder.DebugVideoWriter | None
    vision_fps: FpsTracker
    counters: Counters = field(default_factory=Counters)
    blink_threshold: float = 0.0


def _process_frame(ctx: RunContext):
    frame = ctx.camera.read()
    if frame is None:
        raise RuntimeError("Camera read failed (no frame returned)")
    result = ctx.landmarker.detect(frame.image_bgr, frame.timestamp_ms)
    ctx.vision_fps.tick()
    ctx.counters.total_frames += 1
    fv = None
    if result.face_detected:
        fv = ctx.extractor.extract(result.landmarks_norm)
    else:
        ctx.counters.face_detection_failures += 1
    return frame, result, fv


def _record(ctx: RunContext, frame, phase: str, point_id: str, target, result, fv, sample_valid: bool, prediction=None) -> None:
    row = {
        "frame_index": frame.frame_index,
        "timestamp_ms": round(frame.timestamp_ms, 2),
        "phase": phase,
        "point_id": point_id,
        "target_x_norm": target[0] if target else "",
        "target_y_norm": target[1] if target else "",
        "face_detected": result.face_detected,
        "sample_valid": sample_valid,
    }
    if fv is not None:
        row.update(dataclasses.asdict(fv))
    if prediction is not None:
        row["prediction_x_norm"], row["prediction_y_norm"] = prediction
    ctx.raw_writer.write(row)
    ctx.lm_writer.write(frame.frame_index, frame.timestamp_ms, phase, point_id, target, result.landmarks_norm)
    if ctx.video_writer is not None:
        ctx.video_writer.write(frame.image_bgr)


def _check_esc(ctx: RunContext, canvas: np.ndarray) -> None:
    key = ctx.fui.show(canvas, wait_ms=1)
    if key == ui.ESC_KEY:
        raise UserCancelled()


def _wait_for_start(ctx: RunContext) -> None:
    lines = [
        "GazeFlow A0", "",
        "Sit approximately 60 cm from the screen.",
        "Keep your normal sitting posture.",
        "Follow the target using your eyes.",
        "Small natural head movements are allowed.",
        "Do not click the target.", "",
        "Press SPACE to start.",
        "Press ESC to cancel.",
    ]
    canvas = ctx.fui.new_canvas()
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 40
    y0 = ctx.fui.height // 2 - (line_height * len(lines)) // 2
    for i, line in enumerate(lines):
        size, _ = cv2.getTextSize(line, font, 0.8, 2)
        x = ctx.fui.width // 2 - size[0] // 2
        cv2.putText(canvas, line, (x, y0 + i * line_height), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    while True:
        key = ctx.fui.show(canvas, wait_ms=30)
        if key == ui.SPACE_KEY:
            return
        if key == ui.ESC_KEY:
            raise UserCancelled()


def _collect_baseline(ctx: RunContext, duration_ms: float = 2000) -> float:
    start = time.monotonic()
    left_samples: list[float] = []
    right_samples: list[float] = []
    while (time.monotonic() - start) * 1000 < duration_ms:
        frame, result, fv = _process_frame(ctx)
        progress = (time.monotonic() - start) * 1000 / duration_ms
        canvas = ctx.fui.new_canvas()
        ctx.fui.draw_target(canvas, 0.5, 0.5, progress)
        if fv is not None:
            left_samples.append(fv.left_ear)
            right_samples.append(fv.right_ear)
        _record(ctx, frame, "precheck", "baseline", (0.5, 0.5), result, fv, sample_valid=fv is not None)
        _check_esc(ctx, canvas)
    return calibration.compute_baseline_ear(left_samples, right_samples)


def _transition(ctx: RunContext, prev_xy: tuple[float, float], next_xy: tuple[float, float], point_id: str, duration_ms: float = 200) -> None:
    start = time.monotonic()
    while True:
        t = min(1.0, (time.monotonic() - start) * 1000 / duration_ms)
        x = prev_xy[0] + (next_xy[0] - prev_xy[0]) * t
        y = prev_xy[1] + (next_xy[1] - prev_xy[1]) * t
        frame, result, fv = _process_frame(ctx)
        canvas = ctx.fui.new_canvas()
        ctx.fui.draw_target(canvas, x, y, progress=0.0)
        _record(ctx, frame, "transition", point_id, (x, y), result, fv, sample_valid=False)
        _check_esc(ctx, canvas)
        if t >= 1.0:
            return


def _settle(ctx: RunContext, xy: tuple[float, float], point_id: str, duration_ms: float, phase: str) -> None:
    start = time.monotonic()
    while (time.monotonic() - start) * 1000 < duration_ms:
        frame, result, fv = _process_frame(ctx)
        canvas = ctx.fui.new_canvas()
        ctx.fui.draw_target(canvas, xy[0], xy[1], progress=0.0)
        _record(ctx, frame, phase, point_id, xy, result, fv, sample_valid=False)
        _check_esc(ctx, canvas)


def _collect_points(ctx: RunContext, xy: tuple[float, float], point_id: str, phase: str, target_frames: int, max_ms: float) -> list:
    """Collects up to `target_frames` blink-free, face-detected samples,
    or stops after `max_ms`. Ring progress pauses on invalid frames."""
    start = time.monotonic()
    collected: list[features.FeatureVector] = []
    while len(collected) < target_frames and (time.monotonic() - start) * 1000 < max_ms:
        frame, result, fv = _process_frame(ctx)
        sample_valid = False
        if fv is not None:
            blinking = calibration.is_blinking(fv.left_ear, fv.right_ear, ctx.blink_threshold)
            if blinking:
                ctx.counters.blink_rejected_frames += 1
            else:
                sample_valid = True
                collected.append(fv)
                ctx.counters.valid_frames += 1
        progress = len(collected) / target_frames
        canvas = ctx.fui.new_canvas()
        ctx.fui.draw_target(canvas, xy[0], xy[1], progress)
        _record(ctx, frame, phase, point_id, xy, result, fv, sample_valid=sample_valid)
        _check_esc(ctx, canvas)
    return collected


def _run_calibration(ctx: RunContext) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    cfg = ctx.config
    order = geometry.shuffled_calibration_points(
        cfg.random_seed, count=cfg.calibration_target_count, margin=cfg.calibration_target_margin
    )
    prev_xy = (0.5, 0.5)
    all_features: list[np.ndarray] = []
    all_x: list[float] = []
    all_y: list[float] = []
    all_point_ids: list[str] = []
    failed_points: list[str] = []

    for point_id, tx, ty in order:
        _transition(ctx, prev_xy, (tx, ty), point_id)
        prev_xy = (tx, ty)

        _settle(ctx, (tx, ty), point_id, cfg.calibration_settle_ms, phase="calibration")
        collected = _collect_points(ctx, (tx, ty), point_id, "calibration", cfg.calibration_target_valid_frames, cfg.calibration_max_ms)

        if len(collected) < cfg.calibration_min_valid_frames:
            LOG.warning("Calibration point %s: only %d/%d valid frames, retrying once", point_id, len(collected), cfg.calibration_min_valid_frames)
            _settle(ctx, (tx, ty), point_id, cfg.calibration_settle_ms, phase="calibration")
            collected = _collect_points(ctx, (tx, ty), point_id, "calibration", cfg.calibration_target_valid_frames, cfg.calibration_max_ms)

        if len(collected) < cfg.calibration_min_valid_frames:
            LOG.error("Calibration point %s failed after retry (%d valid frames)", point_id, len(collected))
            failed_points.append(point_id)
            continue

        LOG.info("Calibration point %s: %d valid frames", point_id, len(collected))
        for fv in collected:
            all_features.append(fv.to_array())
            all_x.append(tx)
            all_y.append(ty)
            all_point_ids.append(point_id)

    if len(set(all_point_ids)) < 3:
        raise RuntimeError(f"Too few successful calibration points ({len(set(all_point_ids))}) to fit a model")

    return (
        np.array(all_features), np.array(all_x), np.array(all_y),
        np.array(all_point_ids), failed_points,
    )


def _run_validation(ctx: RunContext, gaze_model: model.GazeModel, calib_points: list[tuple[float, float]]):
    cfg = ctx.config
    points = geometry.generate_validation_points(
        cfg.random_seed, count=cfg.validation_target_count,
        margin=cfg.calibration_target_margin, exclude=calib_points,
    )
    max_ms = cfg.calibration_max_ms  # reuse as a per-target safety cap
    prev_xy = (0.5, 0.5)
    results = []

    for point_id, tx, ty in points:
        _transition(ctx, prev_xy, (tx, ty), point_id)
        prev_xy = (tx, ty)
        _settle(ctx, (tx, ty), point_id, cfg.validation_settle_ms, phase="validation")

        start = time.monotonic()
        pred_x, pred_y, yaw, pitch, roll, scale = [], [], [], [], [], []
        while len(pred_x) < cfg.validation_valid_frames and (time.monotonic() - start) * 1000 < max_ms:
            frame, result, fv = _process_frame(ctx)
            sample_valid = False
            prediction = None
            if fv is not None and not calibration.is_blinking(fv.left_ear, fv.right_ear, ctx.blink_threshold):
                sample_valid = True
                ctx.counters.valid_frames += 1
                X = fv.to_array().reshape(1, -1)
                px, py = gaze_model.predict(X)
                prediction = (float(px[0]), float(py[0]))
                pred_x.append(prediction[0]); pred_y.append(prediction[1])
                yaw.append(fv.head_yaw_deg); pitch.append(fv.head_pitch_deg)
                roll.append(fv.head_roll_deg); scale.append(fv.face_scale)
            progress = len(pred_x) / cfg.validation_valid_frames
            canvas = ctx.fui.new_canvas()
            ctx.fui.draw_target(canvas, tx, ty, progress)
            _record(ctx, frame, "validation", point_id, (tx, ty), result, fv, sample_valid, prediction)
            _check_esc(ctx, canvas)

        if not pred_x:
            LOG.error("Validation point %s: no valid frames collected, skipping", point_id)
            continue

        from .validation import aggregate_target
        results.append(aggregate_target(
            point_id, tx, ty, pred_x, pred_y, yaw, pitch, roll, scale,
            ctx.fui.width, ctx.fui.height, cfg.screen_width_mm, cfg.screen_height_mm,
            cfg.viewing_distance_mm,
        ))
    return results


# --- Commands ---------------------------------------------------------------

def cmd_preflight(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    cfg.apply_cli_overrides(args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cam = Camera(cfg.camera_index, cfg.camera_width, cfg.camera_height, cfg.camera_fps)
    cam.open()
    lmk = landmarks.FaceLandmarkerWrapper()
    extractor = features.FeatureExtractor(cam.actual_width, cam.actual_height)
    vision_fps = FpsTracker()

    import cv2
    cv2.namedWindow("A0 Preflight")
    print(f"Camera opened: {cam.actual_width}x{cam.actual_height}. Press ESC to quit.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            result = lmk.detect(frame.image_bgr, frame.timestamp_ms)
            vision_fps.tick()
            fv = extractor.extract(result.landmarks_norm) if result.face_detected else None
            canvas = ui.draw_preflight_overlay(
                frame.image_bgr, result.landmarks_norm, cam.effective_fps, vision_fps.fps,
                result.face_detected,
                fv.left_ear if fv else None, fv.right_ear if fv else None,
                _pose_from_feature(fv), fv.face_scale if fv else None,
            )
            cv2.imshow("A0 Preflight", canvas)
            if (cv2.waitKey(1) & 0xFF) == ui.ESC_KEY:
                break
    finally:
        cam.release()
        lmk.close()
        cv2.destroyAllWindows()
    return 0


class _PoseShim:
    def __init__(self, yaw, pitch, roll, success):
        self.yaw_deg, self.pitch_deg, self.roll_deg, self.success = yaw, pitch, roll, success


def _pose_from_feature(fv):
    if fv is None:
        return _PoseShim(0, 0, 0, False)
    return _PoseShim(fv.head_yaw_deg, fv.head_pitch_deg, fv.head_roll_deg, True)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    cfg.apply_cli_overrides(args)
    try:
        geometry.generate_calibration_points(cfg.calibration_target_count, cfg.calibration_target_margin)
    except ValueError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 1

    run_id = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    run_dir = Path(args.outputs_dir) / run_id
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)
    log_environment()
    LOG.info("Run ID: %s", run_id)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    lock_file = PROJECT_ROOT / "requirements-lock.txt"
    if lock_file.exists():
        shutil.copy(lock_file, run_dir / "requirements-lock.txt")
    else:
        LOG.warning("requirements-lock.txt not found at project root; skipping copy")

    screen_w, screen_h = ui.get_primary_screen_size_px()
    LOG.info("Screen resolution: %dx%d", screen_w, screen_h)

    # Created incrementally and torn down in `finally` via `_partial_cleanup`
    # below, so a failure partway through setup (e.g. camera opens but the
    # landmarker model fails to load) still releases whatever did open.
    cam = lmk = fui = raw_writer = lm_writer = video_writer = None
    ctx = None
    try:
        cam = Camera(cfg.camera_index, cfg.camera_width, cfg.camera_height, cfg.camera_fps)
        cam.open()
        LOG.info("Camera %d selected. Actual resolution: %dx%d", cfg.camera_index, cam.actual_width, cam.actual_height)

        lmk = landmarks.FaceLandmarkerWrapper()
        extractor = features.FeatureExtractor(cam.actual_width, cam.actual_height)
        fui = ui.FullscreenUI(screen_w, screen_h)
        raw_writer = recorder.RawFeatureWriter(run_dir / "a0_raw_features.csv")
        lm_writer = recorder.LandmarksWriter(run_dir / "a0_landmarks.jsonl")
        if cfg.save_debug_video:
            video_writer = recorder.DebugVideoWriter(run_dir / "a0_capture.mp4", cam.actual_width, cam.actual_height, cfg.camera_fps)
            LOG.info("Debug video enabled: a0_capture.mp4 (A0-only; do not carry this default into production GazeFlow)")

        ctx = RunContext(cfg, cam, lmk, extractor, fui, raw_writer, lm_writer, video_writer, FpsTracker())
        _wait_for_start(ctx)

        baseline_ear = _collect_baseline(ctx)
        ctx.blink_threshold = calibration.blink_threshold(baseline_ear)
        LOG.info("Baseline EAR: %.4f, blink threshold: %.4f", baseline_ear, ctx.blink_threshold)

        X, yx, yy, point_ids, failed_points = _run_calibration(ctx)
        cv_result = calibration.select_ridge_alpha(X, yx, yy, point_ids)
        LOG.info("Selected Ridge alpha: %s (median CV error: %.4f)", cv_result.selected_alpha, cv_result.median_error_norm)

        gaze_model = model.GazeModel.fit(X, yx, yy, cv_result.selected_alpha)

        calib_points_xy = [
            (x, y) for _, x, y in geometry.generate_calibration_points(cfg.calibration_target_count, cfg.calibration_target_margin)
        ]
        validation_rows = _run_validation(ctx, gaze_model, calib_points_xy)
        if not validation_rows:
            raise RuntimeError("No validation targets produced usable data")

        validation_df = pd.DataFrame([dataclasses.asdict(r) for r in validation_rows])
        validation_df.to_csv(run_dir / "a0_validation.csv", index=False)

        results = report.build_results_json(
            run_id=run_id,
            screen={
                "width_px": screen_w, "height_px": screen_h,
                "width_mm": cfg.screen_width_mm, "height_mm": cfg.screen_height_mm,
                "viewing_distance_mm": cfg.viewing_distance_mm,
            },
            camera={
                "index": cfg.camera_index, "width": cam.actual_width, "height": cam.actual_height,
                "requested_fps": cfg.camera_fps, "effective_fps": cam.effective_fps,
            },
            calibration={
                "points": len(set(point_ids.tolist())),
                "selected_ridge_alpha": cv_result.selected_alpha,
                "cv_median_error_norm": cv_result.median_error_norm,
                "cv_worst_error_norm": max(cv_result.per_point_error.values()),
            },
            validation_df=validation_df,
        )
        report.write_json(run_dir / "a0_results.json", results)

        observations = []
        if failed_points:
            observations.append(f"Calibration points failed after retry: {', '.join(failed_points)}")
        if cam.effective_fps and cam.effective_fps < 15:
            observations.append(f"Low effective camera FPS ({cam.effective_fps:.1f}) may have degraded tracking.")
        if cfg.screen_width_mm is None or cfg.screen_height_mm is None:
            observations.append("Physical screen dimensions were not supplied; only normalized/pixel error is available.")

        tracking = {
            "valid_frame_ratio": ctx.counters.valid_frames / max(ctx.counters.total_frames, 1),
            "face_detection_failures": ctx.counters.face_detection_failures,
            "blink_rejected_frames": ctx.counters.blink_rejected_frames,
        }
        summary_md = report.build_summary_md(results, validation_df, tracking, observations)
        (run_dir / "a0_summary.md").write_text(summary_md)
        report.generate_plots(validation_df, screen_w, screen_h, run_dir / "plots")

        LOG.info("Final decision: %s", results["status"])
        print("\nGazeFlow A0 Complete\n")
        if results["validation"]["median_error_deg"] is not None:
            print(f"Median error: {results['validation']['median_error_deg']:.1f} deg")
            print(f"P95 error:   {results['validation']['p95_error_deg']:.1f} deg")
        else:
            print(f"Median error (normalized): {results['validation']['median_error_norm']:.4f}")
            print(f"P95 error (normalized):    {results['validation']['p95_error_norm']:.4f}")
        print(f"\nDecision:\n{results['status']}\n")
        print(f"Results:\n{run_dir}/\n")
        return 0

    except UserCancelled:
        LOG.warning("Run cancelled by user (ESC)")
        (run_dir / "a0_failure.txt").write_text("Cancelled by user (ESC)\n")
        return 1
    except Exception as exc:
        LOG.exception("Run failed: %s", exc)
        (run_dir / "a0_failure.txt").write_text(f"{exc}\n")
        return 1
    finally:
        for closeable in (raw_writer, lm_writer, video_writer, lmk, fui):
            if closeable is not None:
                try:
                    closeable.close()
                except Exception:
                    LOG.exception("Error closing %r during cleanup", closeable)
        if cam is not None:
            cam.release()


def cmd_report(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = Path(args.run_dir)
    validation_df = pd.read_csv(run_dir / "a0_validation.csv")
    existing = json.loads((run_dir / "a0_results.json").read_text())

    results = report.build_results_json(
        run_id=existing["run_id"], screen=existing["screen"], camera=existing["camera"],
        calibration=existing["calibration"], validation_df=validation_df,
    )
    report.write_json(run_dir / "a0_results.json", results)
    report.generate_plots(validation_df, existing["screen"]["width_px"], existing["screen"]["height_px"], run_dir / "plots")

    # Frame-level tracking counters are not persisted structurally, so the
    # rebuilt summary reports validation-derived stats only (documented
    # limitation of `report`, which works from saved data, not raw frames).
    summary_md = report.build_summary_md(results, validation_df, tracking={}, observations=[])
    (run_dir / "a0_summary.md").write_text(summary_md)
    print(f"Rebuilt report for {run_dir}. Decision: {results['status']}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    results = json.loads((run_dir / "a0_results.json").read_text())
    raw_df = pd.read_csv(run_dir / "a0_raw_features.csv")
    calib_rows = raw_df[(raw_df["phase"] == "calibration") & (raw_df["sample_valid"] == True)]  # noqa: E712
    if calib_rows.empty:
        print("No valid calibration rows found in a0_raw_features.csv")
        return 1

    X = calib_rows[features.FEATURE_NAMES].to_numpy(dtype=np.float64)
    yx = calib_rows["target_x_norm"].to_numpy(dtype=np.float64)
    yy = calib_rows["target_y_norm"].to_numpy(dtype=np.float64)
    gaze_model = model.GazeModel.fit(X, yx, yy, results["calibration"]["selected_ridge_alpha"])

    cfg = Config.load(run_dir / "config.json")
    if args.camera is not None:
        cfg.camera_index = args.camera
    screen_w, screen_h = results["screen"]["width_px"], results["screen"]["height_px"]

    cam = Camera(cfg.camera_index, cfg.camera_width, cfg.camera_height, cfg.camera_fps)
    cam.open()
    lmk = landmarks.FaceLandmarkerWrapper()
    extractor = features.FeatureExtractor(cam.actual_width, cam.actual_height)
    fui = ui.FullscreenUI(screen_w, screen_h)

    print("Live prediction view. Press ESC to quit.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                break
            result = lmk.detect(frame.image_bgr, frame.timestamp_ms)
            canvas = fui.new_canvas()
            if result.face_detected:
                fv = extractor.extract(result.landmarks_norm)
                if fv is not None:
                    px, py = gaze_model.predict(fv.to_array().reshape(1, -1))
                    fui.draw_gaze_point(canvas, float(px[0]), float(py[0]))
            key = fui.show(canvas, wait_ms=1)
            if key == ui.ESC_KEY:
                break
    finally:
        cam.release()
        lmk.close()
        fui.close()
    return 0


# --- Argument parsing --------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m a0.main")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preflight")
    p_pre.add_argument("--camera", type=int, default=None)
    p_pre.add_argument("--config", type=Path, default=None)
    p_pre.set_defaults(func=cmd_preflight)

    p_run = sub.add_parser("run")
    p_run.add_argument("--camera", type=int, default=None)
    p_run.add_argument("--viewing-distance-mm", type=float, default=None)
    p_run.add_argument("--screen-width-mm", type=float, default=None)
    p_run.add_argument("--screen-height-mm", type=float, default=None)
    p_run.add_argument("--calibration-points", type=int, default=None,
                        help="Number of calibration points; must be a perfect square (4, 9, 16, 25, ...). Default: 9.")
    p_run.add_argument("--save-debug-video", action="store_true")
    p_run.add_argument("--no-debug-video", action="store_true")
    p_run.add_argument("--config", type=Path, default=None)
    p_run.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report")
    p_report.add_argument("run_dir", type=Path)
    p_report.set_defaults(func=cmd_report)

    p_live = sub.add_parser("live")
    p_live.add_argument("run_dir", type=Path)
    p_live.add_argument("--camera", type=int, default=None)
    p_live.set_defaults(func=cmd_live)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
