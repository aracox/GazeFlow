# GazeFlow A0 — Disposable Feasibility Spike

**This code is disposable.** It exists to answer one question as quickly as
possible:

> Can a standard RGB webcam, MediaPipe face/eye landmarks, and per-user Ridge
> calibration estimate screen gaze accurately enough to justify continuing
> GazeFlow?

It is a measurement harness, not a product prototype. Do not build on top of
this code for production GazeFlow — see `plan/GazeFlow_A0_Coding_Agent_Plan.md`
for the full specification and rationale.

The permanent output of A0 is the **data** it produces (raw feature streams,
validation results, plots, summaries) — not the Python implementation.

## Environment Setup

Requires Python 3.9–3.13 (MediaPipe does not yet support 3.14). This project
was built and tested against **Python 3.12**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

(If you use [uv](https://docs.astral.sh/uv/), `uv venv --python 3.12 .venv &&
uv pip install -e ".[dev]"` works too, and is what this repo was set up
with.)

After the environment works, snapshot exact versions:

```bash
python -m pip freeze > requirements-lock.txt
```

The MediaPipe Face Landmarker model bundle is checked into
`a0/assets/face_landmarker.task` (~3.6 MB) so the experiment runs without a
network dependency after setup.

## How to Run Preflight

Verifies the camera, face/eye/iris landmark detection, and head pose before
running the full experiment:

```bash
python -m a0.main preflight
```

Shows a live camera window with landmarks, FPS, eye-open estimate, head
pose, and face scale overlaid. Press **ESC** to quit.

## How to Measure Screen Physical Width

Physical screen dimensions are optional but required for mm/degree error
reporting (section 32 of the plan). Measure your screen's visible width and
height in millimeters with a ruler/tape measure, or look up your display's
published active area dimensions. Pass them via `--screen-width-mm` /
`--screen-height-mm`.

## How to Estimate/Set Viewing Distance

Default is 600 mm (a typical desktop viewing distance). Measure your actual
distance from eyes to screen and pass it via `--viewing-distance-mm` if it
differs meaningfully.

## How to Run the Experiment

```bash
python -m a0.main run \
  --camera 0 \
  --viewing-distance-mm 600 \
  --screen-width-mm 345 \
  --screen-height-mm 223 \
  --save-debug-video
```

All flags are optional; defaults come from `config.example.json` (copy it to
your own config and pass `--config path/to/config.json` to override
multiple values at once). If `--screen-width-mm`/`--screen-height-mm` are
omitted, only normalized and pixel error are reported — degree error is
never fabricated from an assumed screen size.

Calibration grid density can be changed with `--calibration-points` (must
be a perfect square: 4, 9, 16, 25, ...; default 9). More points give the
model more spatial coverage to fit at the cost of a longer session:

```bash
python -m a0.main run --calibration-points 16 --screen-width-mm 345 --screen-height-mm 223
```

The full flow: instructions screen → SPACE to start → ~2s open-eye baseline
→ 9-point calibration (follow the target with your eyes only) → Ridge
model fit + cross-validation → 20-point independent validation → results
written to `outputs/<run-id>/`.

## Controls

- **SPACE** — start the experiment from the instructions screen
- **ESC** — cancel at any point (partial data is preserved, not discarded)
- Just follow the on-screen target with your eyes during calibration and
  validation. No clicks or keypresses are needed per target.

## Rebuilding a Report

Regenerates `a0_summary.md` and `plots/` (and recomputes `a0_results.json`)
from a previously saved run's `a0_validation.csv`, without rerunning the
camera/calibration/validation session:

```bash
python -m a0.main report outputs/<run-id>
```

## Optional Live Prediction View

Qualitative-only debug view; not a substitute for the validation metrics.
Retrains a lightweight model from the saved calibration-phase rows and shows
a live predicted gaze dot:

```bash
python -m a0.main live outputs/<run-id>
```

## Sample App: Gaze + Blink YES/NO Picker

A small demo built on the same calibration/model/UI code, showing gaze
tracking driving a real interaction instead of just a debug view. Reuses a
completed run's calibration rather than recalibrating:

```bash
python sample_app.py outputs/<run-id>
```

Look at the left half (NO) or right half (YES) of the screen; select by
blinking while looking at that side, or by holding your gaze there for 2+
seconds. A short eyes-open baseline is collected at startup (blink
threshold is session/lighting-sensitive, unlike the reused gaze
calibration). ESC to quit. See `sample_app.py`'s module docstring for
details.

## Output Files

Each run creates `outputs/<run-id>/`:

- `config.json` — the merged configuration used for this run
- `a0_raw_features.csv` — every processed frame (precheck, calibration,
  transition, validation phases), features, and predictions
- `a0_landmarks.jsonl` — selected raw MediaPipe landmark coordinates per
  frame (nose, chin, eye corners, iris centers, mouth corners)
- `a0_validation.csv` — one row per validation target with predicted vs.
  actual position and error in normalized/pixel/mm/degree units
- `a0_results.json` — machine-readable summary: screen/camera setup,
  calibration cross-validation stats, validation accuracy, and the decision
- `a0_summary.md` — human-readable summary of the above
- `plots/validation_scatter.png`, `error_by_target.png`,
  `error_distribution.png`, `screen_error_map.png`
- `a0_capture.mp4` — raw debug video, frame-aligned with
  `a0_raw_features.csv` via `frame_index`/`timestamp_ms` (only if
  `save_debug_video` is enabled; **this default must not be copied into
  production GazeFlow**, where raw video should not be saved by default)
- `a0.log` — full run log (startup info, package versions, calibration
  point results, failures)
- `requirements-lock.txt` — snapshot of the environment used for this run

## How Decision Status Is Calculated

Gate (section 33 of the plan), using median and P95 angular error across
the 20 independent validation targets:

- **PASS-CANDIDATE**: median ≤ 3.0° and P95 ≤ 4.5°
- **BORDERLINE**: median ≤ 4.5° and P95 ≤ 6.75°
- **FAIL**: worse than BORDERLINE, or tracking is unstable/unusable
- **UNKNOWN-NO-PHYSICAL-GEOMETRY**: `--screen-width-mm`/`--screen-height-mm`
  were not supplied, so degree error (and therefore the gate) cannot be
  computed. Normalized/pixel error are still reported.

This is an A0 working gate based on a ~95 mm working minimum AOI width at
600 mm viewing distance (~4.5°), not a final product claim. It must be
confirmed or rejected by a multi-participant Stage A test.

## Known Limitations

- Single-participant, single-session measurement — not statistically
  powered for a product claim.
- Ridge regression on hand-crafted geometric features only; no
  appearance-based/deep-learning gaze model is used (see
  `plan/GazeFlow_A0_Coding_Agent_Plan.md` section 50 for when that would be
  considered).
- Head pose (`solvePnP` with a generic, unmeasured 3D face model) is a
  stability signal, not clinically accurate pose estimation.
- EAR (eye aspect ratio) is a simplified height/width proxy from face-mesh
  corner/lid landmarks, not the full 6-point formula.
- `face_scale` and `inter_eye_distance_norm` use image-normalized (not
  aspect-corrected) landmark coordinates — documented in `a0/features.py`.
- MediaPipe is pinned to `0.10.21`. The newer `1.0.1` release's
  FaceLandmarker graph crashes with a hard C++ `CHECK` failure (not a
  catchable Python exception) when its GPU/Metal service isn't available,
  even with the CPU delegate requested — reproduced on this machine outside
  any sandbox. `a0/landmarks.py` also explicitly requests the CPU delegate
  for portability.
- Requires MediaPipe-compatible Python (3.9–3.13); does not run on 3.14+
  until MediaPipe adds support. This repo's `.venv` uses 3.12.

## This Code Is Disposable

Do not extend this module structure into production GazeFlow. When Stage A
begins, keep the datasets (`outputs/*/a0_raw_features.csv`,
`a0_validation.csv`, `a0_results.json`, plots, configs) as permanent
research assets for comparing future gaze models — but the `a0/` Python
package itself is not meant to be reused or built upon.
