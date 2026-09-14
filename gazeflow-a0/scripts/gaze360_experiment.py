"""Offline experiment: does a pretrained appearance-based gaze model
(ResNet34 trained on Gaze360, vendored in gaze360_resnet.py) beat the A0
geometric-feature Ridge baseline, calibrated on the SAME session?

Reuses a completed run's saved debug video (a0_capture.mp4) instead of a
new capture session:
  1. Re-run MediaPipe on each needed frame (from a0_raw_features.csv:
     calibration/validation phase, sample_valid) to get a face bounding box.
  2. Crop + run the pretrained model to get raw appearance-based gaze
     yaw/pitch (camera-relative, not screen-relative).
  3. Fit a small Ridge calibration mapping from [gaze_yaw, gaze_pitch,
     face_center_x, face_center_y, face_scale] -> screen (x, y), using the
     same 9/16-point calibration frames and leave-one-point-out alpha
     selection as the geometric-feature pipeline.
  4. Evaluate on the held-out validation targets the same way
     a0/report.py does, for an apples-to-apples comparison.

Usage:
    curl -sSL -o scripts/weights/resnet34.pt \
      https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet34.pt
    python scripts/gaze360_experiment.py outputs/<run-id>

Requires torch/torchvision (not in the base A0 requirements -- install
separately: `uv pip install torch torchvision`). Weights are not checked
into git (scripts/weights/ is gitignored, ~85MB); download them with the
command above before running. Model architecture is vendored in
gaze360_resnet.py, MIT-licensed from
https://github.com/yakhyo/gaze-estimation (Copyright (c) 2024 Yakhyokhuja
Valikhujaev) -- see LICENSE terms in that repo if reusing beyond A0.

Disposable, like the rest of A0 -- not meant to be extended into product code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from gaze360_resnet import resnet34_gaze  # noqa: E402

from a0.calibration import select_ridge_alpha  # noqa: E402
from a0.geometry import compute_error  # noqa: E402
from a0.landmarks import FaceLandmarkerWrapper  # noqa: E402
from a0.model import GazeModel  # noqa: E402

WEIGHTS_PATH = Path(__file__).parent / "weights" / "resnet34.pt"
GAZE360_BINS, GAZE360_BINWIDTH, GAZE360_ANGLE = 90, 4, 180
CALIB_FEATURE_NAMES = ["gaze_yaw_deg", "gaze_pitch_deg", "face_center_x_norm", "face_center_y_norm", "face_scale"]

_PREPROCESS = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _face_bbox_px(landmarks_norm: np.ndarray, width: int, height: int, margin: float = 0.25) -> tuple[int, int, int, int]:
    face_xy = landmarks_norm[:468, :2]
    x_min, y_min = face_xy.min(axis=0)
    x_max, y_max = face_xy.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    x_min = max(0.0, x_min - w * margin)
    x_max = min(1.0, x_max + w * margin)
    y_min = max(0.0, y_min - h * margin)
    y_max = min(1.0, y_max + h * margin)
    return int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height)


def extract_appearance_gaze(run_dir: Path, needed: pd.DataFrame) -> pd.DataFrame:
    """Returns needed with gaze_yaw_deg/gaze_pitch_deg columns added
    (rows where MediaPipe couldn't find a face in the re-decoded frame are dropped)."""
    device = _device()
    print(f"Using device: {device}")

    model = resnet34_gaze(num_classes=GAZE360_BINS)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.to(device).eval()
    idx_tensor = torch.arange(GAZE360_BINS, dtype=torch.float32, device=device)

    landmarker = FaceLandmarkerWrapper()
    needed_indices = set(needed["frame_index"].tolist())
    results: dict[int, tuple[float, float]] = {}

    cap = cv2.VideoCapture(str(run_dir / "a0_capture.mp4"))
    frame_index = 0
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index in needed_indices:
                h, w = frame_bgr.shape[:2]
                lm_result = landmarker.detect(frame_bgr, frame_index * 33.0)
                if lm_result.face_detected:
                    x0, y0, x1, y1 = _face_bbox_px(lm_result.landmarks_norm, w, h)
                    crop = frame_bgr[y0:y1, x0:x1]
                    if crop.size > 0:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        tensor = _PREPROCESS(Image.fromarray(rgb)).unsqueeze(0).to(device)
                        yaw_logits, pitch_logits = model(tensor)
                        yaw_deg = (F.softmax(yaw_logits, dim=1) * idx_tensor).sum(dim=1) * GAZE360_BINWIDTH - GAZE360_ANGLE
                        pitch_deg = (F.softmax(pitch_logits, dim=1) * idx_tensor).sum(dim=1) * GAZE360_BINWIDTH - GAZE360_ANGLE
                        results[frame_index] = (float(yaw_deg.item()), float(pitch_deg.item()))
            frame_index += 1
    cap.release()
    landmarker.close()

    print(f"Ran appearance model on {len(results)}/{len(needed_indices)} needed frames "
          f"({len(needed_indices) - len(results)} had no re-detected face)")

    out = needed[needed["frame_index"].isin(results.keys())].copy()
    out["gaze_yaw_deg"] = out["frame_index"].map(lambda i: results[i][0])
    out["gaze_pitch_deg"] = out["frame_index"].map(lambda i: results[i][1])
    return out


def evaluate(run_dir: Path, df: pd.DataFrame, screen: dict) -> dict:
    calib = df[df["phase"] == "calibration"]
    val = df[df["phase"] == "validation"]

    X = calib[CALIB_FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_x = calib["target_x_norm"].to_numpy(dtype=np.float64)
    y_y = calib["target_y_norm"].to_numpy(dtype=np.float64)
    point_ids = calib["point_id"].to_numpy()

    cv_result = select_ridge_alpha(X, y_x, y_y, point_ids)
    gaze_model = GazeModel.fit(X, y_x, y_y, cv_result.selected_alpha)
    print(f"Calibration mapping: alpha={cv_result.selected_alpha}, CV median error (norm)={cv_result.median_error_norm:.4f}")

    errors_deg, errors_norm = [], []
    for point_id, group in val.groupby("point_id"):
        X_val = group[CALIB_FEATURE_NAMES].to_numpy(dtype=np.float64)
        pred_x = np.median(gaze_model.predict(X_val)[0])
        pred_y = np.median(gaze_model.predict(X_val)[1])
        target_x, target_y = group["target_x_norm"].iloc[0], group["target_y_norm"].iloc[0]
        err = compute_error(pred_x, pred_y, target_x, target_y,
                             screen["width_px"], screen["height_px"],
                             screen["width_mm"], screen["height_mm"], screen["viewing_distance_mm"])
        errors_deg.append(err.error_deg)
        errors_norm.append(err.error_norm)

    return {
        "targets": len(errors_deg),
        "median_error_deg": float(np.median(errors_deg)),
        "p95_error_deg": float(np.percentile(errors_deg, 95)),
        "median_error_norm": float(np.median(errors_norm)),
        "p95_error_norm": float(np.percentile(errors_norm, 95)),
    }


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python scripts/gaze360_experiment.py outputs/<run-id>")
        return 1

    run_dir = Path(argv[0])
    raw = pd.read_csv(run_dir / "a0_raw_features.csv")
    existing_results = json.loads((run_dir / "a0_results.json").read_text())
    screen = existing_results["screen"]

    needed = raw[(raw["phase"].isin(["calibration", "validation"])) & (raw["sample_valid"] == True)]  # noqa: E712
    print(f"{len(needed)} frames needed from {run_dir.name}")

    df = extract_appearance_gaze(run_dir, needed)
    result = evaluate(run_dir, df, screen)

    print()
    print(f"{'':20s} {'median_deg':>12s} {'p95_deg':>12s}")
    print(f"{'geometric (Ridge)':20s} {existing_results['validation']['median_error_deg']:12.2f} "
          f"{existing_results['validation']['p95_error_deg']:12.2f}")
    print(f"{'appearance (gaze360)':20s} {result['median_error_deg']:12.2f} {result['p95_error_deg']:12.2f}")
    print()
    print("Reference gate: PASS median<=3.0 p95<=4.5 | BORDERLINE median<=4.5 p95<=6.75 | else FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
