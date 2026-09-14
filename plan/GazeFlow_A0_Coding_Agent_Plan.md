# GazeFlow A0 Disposable Feasibility Spike
## Implementation Specification for Codex / Claude Code

> **Use this document as the implementation task.**
>
> Implement exactly the A0 feasibility harness described here.
> Do not build the production GazeFlow application.
> Do not add Tauri, React, a database, cloud services, or product UI.
> This is disposable Python research code whose purpose is to produce measured gaze-error data as quickly as possible.

---

# 1. Objective

Build a small Python-only experiment that answers one question:

> Can a standard RGB webcam, MediaPipe face/eye landmarks, and per-user Ridge calibration estimate screen gaze accurately enough to justify continuing GazeFlow?

A0 is not a product prototype.

It is a measurement harness.

The result must tell us whether to:

```text
PASS-CANDIDATE
    ↓
Continue to the broader feasibility stage

BORDERLINE
    ↓
Investigate a stronger gaze model

FAIL
    ↓
Reposition to coarse attention zones,
change tracking hardware,
or stop
```

---

# 2. Hard Scope Rules

A0 must use:

```text
Python
OpenCV
MediaPipe
NumPy
Pandas
scikit-learn
Matplotlib
CSV / JSON
```

A0 must not use:

```text
Tauri
React
TypeScript
Rust
SQLite
Cloud services
REST APIs
Authentication
Product dashboards
Production architecture
```

Do not spend time polishing the UI.

The only UI required is:

```text
camera pre-check
fullscreen calibration target
fullscreen validation target
final result summary
```

---

# 3. Time Box

Target implementation time:

```text
3 to 5 working days
```

Do not redesign GazeFlow during A0.

Do not turn A0 into reusable production code.

If a design choice is not needed to measure gaze error, choose the simplest reasonable implementation.

---

# 4. Development Environment

Primary machine:

```text
macOS
Apple Silicon
built-in or USB RGB webcam
```

Use the user's installed Python if dependencies install successfully.

Create an isolated virtual environment.

Suggested setup:

```bash
python3 --version

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

Install the minimum required packages.

Suggested packages:

```text
opencv-python
mediapipe
numpy
pandas
scikit-learn
matplotlib
screeninfo
```

Do not add large frameworks unless required.

After the environment works, save an exact dependency snapshot:

```bash
python -m pip freeze > requirements-lock.txt
```

---

# 5. Repository Structure

Create a small standalone folder:

```text
gazeflow-a0/
|
+-- README.md
+-- pyproject.toml
+-- requirements-lock.txt
+-- config.example.json
|
+-- a0/
|   +-- __init__.py
|   +-- main.py
|   +-- camera.py
|   +-- landmarks.py
|   +-- features.py
|   +-- calibration.py
|   +-- model.py
|   +-- validation.py
|   +-- geometry.py
|   +-- recorder.py
|   +-- report.py
|   +-- ui.py
|
+-- tests/
|   +-- test_geometry.py
|   +-- test_targets.py
|
+-- outputs/
    +-- .gitkeep
```

Keep the code understandable, but do not over-engineer it.

---

# 6. Required Commands

The implementation must expose these commands.

## Preflight

```bash
python -m a0.main preflight
```

Purpose:

- Open webcam
- Show camera preview
- Show face landmarks
- Show eye and iris landmarks
- Show FPS
- Show face detection status
- Show eye-open estimate
- Show head pose
- Show face scale
- Allow ESC to quit

## Run A0

```bash
python -m a0.main run
```

Optional configuration:

```bash
python -m a0.main run \
  --camera 0 \
  --viewing-distance-mm 600 \
  --screen-width-mm 345 \
  --screen-height-mm 223 \
  --save-debug-video
```

Physical screen dimensions must remain optional.

If physical screen dimensions are not supplied:

```text
calculate normalized error
calculate pixel error
do not report angular error as valid
```

## Rebuild Report

```bash
python -m a0.main report outputs/<run-id>
```

This must rebuild the summary and plots from saved data without rerunning the experiment.

---

# 7. Configuration

Create `config.example.json`.

Example:

```json
{
  "camera_index": 0,
  "camera_width": 1280,
  "camera_height": 720,
  "camera_fps": 30,

  "viewing_distance_mm": 600,
  "screen_width_mm": null,
  "screen_height_mm": null,

  "calibration_target_margin": 0.10,
  "calibration_target_count": 9,
  "calibration_settle_ms": 300,
  "calibration_target_valid_frames": 20,
  "calibration_min_valid_frames": 15,
  "calibration_max_ms": 2000,

  "validation_target_count": 20,
  "validation_settle_ms": 300,
  "validation_valid_frames": 15,

  "random_seed": 42,

  "save_debug_video": true
}
```

Command-line parameters should override the config file.

---

# 8. Screen Coordinate System

Use normalized screen coordinates everywhere internally.

```text
Top-left:
(0.0, 0.0)

Center:
(0.5, 0.5)

Bottom-right:
(1.0, 1.0)
```

Do not train the model directly on OS-specific pixel coordinates.

Convert to pixels only for:

- drawing targets
- visualizing predictions
- calculating pixel error

---

# 9. Camera Capture

Use OpenCV for A0.

Requirements:

```text
request 1280 x 720
request 30 FPS
record actual width
record actual height
measure effective FPS
record frame index
record monotonic timestamp
```

Use:

```python
time.perf_counter_ns()
```

or an equivalent monotonic high-resolution clock for local timestamps.

Do not assume the camera-reported FPS is correct.

Calculate effective FPS from captured frames.

---

# 10. Optional Debug Video

For A0 only, support:

```text
a0_capture.mp4
```

This is useful because later experiments can inspect the exact recording.

The video must be aligned with:

```text
frame_index
timestamp_ms
```

in `a0_raw_features.csv`.

This debug recording is allowed because A0 initially uses the developer as the participant.

Add a clear comment:

> This debug-video default must not be copied into production GazeFlow, where raw video should not be saved by default.

---

# 11. MediaPipe Processing

Use MediaPipe face landmark detection.

The implementation must extract:

```text
face landmarks
left eye landmarks
right eye landmarks
left iris
right iris
```

Prefer the current MediaPipe Face Landmarker API.

If the installed MediaPipe version exposes a different supported API, adapt to it without changing the A0 goal.

Process one face only.

---

# 12. Landmark Debug View

The `preflight` view should show:

```text
face outline or selected face points
left eye
right eye
iris centers
head pose
camera FPS
vision FPS
EAR / eye-open measure
face scale
```

The purpose is only to verify that the input features behave sensibly.

---

# 13. Eye and Iris Features

At minimum compute:

```text
left iris center x relative to left eye
left iris center y relative to left eye

right iris center x relative to right eye
right iris center y relative to right eye

left eye width
left eye height
right eye width
right eye height

left eye aspect ratio
right eye aspect ratio

inter-eye distance
```

Use relative geometry rather than raw pixels where possible.

Example:

```text
relative iris X =
(iris_center_x - eye_left_corner_x)
/
(eye_right_corner_x - eye_left_corner_x)
```

The direction of the denominator must be normalized consistently.

---

# 14. Face Features

At minimum compute:

```text
face center x
face center y
face scale
```

A simple face-scale proxy may use:

```text
distance between selected stable facial landmarks
```

normalized by frame size.

The exact definition must be documented in the source.

---

# 15. Head Pose

Estimate:

```text
yaw
pitch
roll
```

For A0, an OpenCV `solvePnP` implementation using stable MediaPipe facial landmarks is acceptable.

Use one documented generic 3D face model.

Possible landmark anchors include:

```text
nose tip
chin
left eye outer corner
right eye outer corner
left mouth corner
right mouth corner
```

Do not spend time building a perfect 3D head model.

A0 needs a stable head-pose feature, not clinical pose accuracy.

---

# 16. Feature Vector

Create one feature vector per usable frame.

Minimum model features:

```text
left_iris_x_rel
left_iris_y_rel

right_iris_x_rel
right_iris_y_rel

left_eye_width_norm
left_eye_height_norm

right_eye_width_norm
right_eye_height_norm

left_ear
right_ear

inter_eye_distance_norm

face_center_x_norm
face_center_y_norm
face_scale

head_yaw_deg
head_pitch_deg
head_roll_deg
```

Keep the feature column order deterministic.

Document it.

---

# 17. Raw Feature Stream

A0 must save a raw feature stream.

File:

```text
a0_raw_features.csv
```

Every processed frame should include:

```text
frame_index
timestamp_ms
phase
point_id
target_x_norm
target_y_norm

face_detected
sample_valid

left_iris_x_rel
left_iris_y_rel
right_iris_x_rel
right_iris_y_rel

left_eye_width_norm
left_eye_height_norm
right_eye_width_norm
right_eye_height_norm

left_ear
right_ear
inter_eye_distance_norm

face_center_x_norm
face_center_y_norm
face_scale

head_yaw_deg
head_pitch_deg
head_roll_deg

prediction_x_norm
prediction_y_norm
```

Use:

```text
phase =
precheck
calibration
transition
validation
```

Prediction fields may be empty before the model has been trained.

This file is a permanent research asset even though the A0 code is disposable.

---

# 18. Selected Raw Landmark Logging

Also store the selected raw MediaPipe landmark coordinates used by the feature extractor.

Do not necessarily store all landmarks as hundreds of CSV columns.

A reasonable representation is:

```text
a0_landmarks.jsonl
```

Each line:

```json
{
  "frame_index": 123,
  "timestamp_ms": 4100.2,
  "phase": "calibration",
  "point_id": "c5",
  "target": [0.5, 0.5],
  "landmarks": {
    "nose": [0.50, 0.42, -0.03],
    "chin": [0.50, 0.79, 0.01],
    "left_eye_outer": [0.40, 0.38, -0.02],
    "left_eye_inner": [0.47, 0.38, -0.03],
    "right_eye_inner": [0.53, 0.38, -0.03],
    "right_eye_outer": [0.60, 0.38, -0.02],
    "left_iris": [0.44, 0.39, -0.03],
    "right_iris": [0.56, 0.39, -0.03]
  }
}
```

This preserves the original geometry needed for offline feature experiments.

---

# 19. Eye-Open Baseline and Blink Rejection

Before calibration, run a short center-looking precheck.

Duration:

```text
approximately 2 seconds
```

Ask the user to:

> Look at the center target naturally with your eyes open.

Calculate:

```text
baseline_EAR =
median eye aspect ratio during valid open-eye frames
```

Set a provisional blink threshold relative to the user's baseline, for example:

```text
blink_threshold =
0.65 × baseline_EAR
```

Do not hard-code one universal EAR threshold if avoidable.

During calibration:

```text
if blink detected:
    record frame
    mark sample_valid = false
    do not use frame for model fitting
```

---

# 20. Pre-Calibration Instructions

Show one simple instruction screen:

```text
GazeFlow A0

Sit approximately 60 cm from the screen.

Keep your normal sitting posture.

Follow the target using your eyes.

Small natural head movements are allowed.

Do not click the target.

Press SPACE to start.
Press ESC to cancel.
```

Do not clutter the calibration screen with text after the experiment starts.

---

# 21. Calibration Targets

Use 9 points:

```text
(0.1, 0.1)
(0.5, 0.1)
(0.9, 0.1)

(0.1, 0.5)
(0.5, 0.5)
(0.9, 0.5)

(0.1, 0.9)
(0.5, 0.9)
(0.9, 0.9)
```

Randomize the order deterministically using the configured random seed.

Do not always move left-to-right, top-to-bottom.

---

# 22. Calibration Target UX

The user only follows the target.

No click is required.

No keypress is required between points.

No blink is used as confirmation.

Use a simple target:

```text
outer ring
inner dot
```

During collection, shrink the ring toward the dot.

Conceptually:

```text
◎
◉
●
```

The ring represents accepted valid frames, not elapsed time.

If frames are invalid:

```text
pause visual progress
```

---

# 23. Target Transition

Between points:

```text
150 to 250 ms
```

A short movement animation is acceptable.

During transition:

```text
phase = transition
sample_valid = false
do not train on these frames
```

When the target arrives:

```text
wait calibration_settle_ms
then begin collecting
```

---

# 24. Calibration Sample Collection

Per calibration point:

```text
target valid frames = 20
minimum acceptable = 15
maximum collection time = 2000 ms
```

A frame is eligible when:

```text
face detected
both eye regions available
not blinking
features finite
no invalid geometry
```

Do not reject frames based on arbitrary head-pose thresholds in A0 unless detection is clearly broken.

Log the head pose instead.

If fewer than the minimum valid frames are collected:

```text
retry the same point once
```

If it still fails:

```text
record failure
continue if possible
```

---

# 25. Calibration Training Data

Calibration model input:

```text
feature vector
```

Labels:

```text
target_x_norm
target_y_norm
```

The same target generates several correlated frames.

Do not treat 180 frames as 180 independent spatial positions.

---

# 26. Model Pipeline

Use separate X and Y models.

Required pipeline:

```text
StandardScaler
    ↓
Ridge Regression
```

Train:

```text
features -> x_norm
features -> y_norm
```

Do not use polynomial regression for the initial A0 baseline.

---

# 27. Ridge Alpha Selection

Use a small deterministic alpha grid, for example:

```text
0.01
0.1
1.0
10.0
100.0
```

Select alpha using leave-one-calibration-point-out cross-validation.

Group by calibration target.

Example:

```text
train 8 calibration positions
test the held-out 9th position
repeat for all 9 positions
```

Choose the alpha minimizing a clearly documented held-out error metric.

Prefer:

```text
median normalized Euclidean error
```

over training error.

---

# 28. Calibration Cross-Validation Output

Store:

```text
selected alpha
median held-out error
mean held-out error
worst held-out point
per-point error
```

These values belong in:

```text
a0_results.json
```

Cross-validation does not replace independent validation.

---

# 29. Independent Validation Targets

After training, show targets that were not used for fitting.

Use:

```text
20 validation targets
```

Generate them deterministically from the configured random seed.

Constraints:

```text
x between 0.1 and 0.9
y between 0.1 and 0.9
not identical to calibration points
reasonably distributed across the screen
```

Do not cluster most points in the center.

---

# 30. Validation Collection

For each target:

```text
move target
wait validation_settle_ms
collect 15 valid frames
```

Generate predictions for each valid frame.

For each target calculate:

```text
median predicted x
median predicted y
```

Target-level error is calculated from the median prediction.

Also preserve frame-level predictions.

---

# 31. Validation CSV

Create:

```text
a0_validation.csv
```

One row per validation target.

Required columns:

```text
point_id
target_x_norm
target_y_norm

predicted_x_norm
predicted_y_norm

error_norm
error_px

error_mm
error_deg

valid_frames

median_confidence_or_quality
head_yaw_median
head_pitch_median
head_roll_median
face_scale_median
```

If physical screen geometry is unavailable:

```text
error_mm = blank
error_deg = blank
```

Do not fabricate degree values.

---

# 32. Error Calculation

Normalized error:

```text
dx = predicted_x_norm - target_x_norm
dy = predicted_y_norm - target_y_norm

error_norm = sqrt(dx² + dy²)
```

Pixel error:

```text
dx_px = dx × screen_width_px
dy_px = dy × screen_height_px

error_px = sqrt(dx_px² + dy_px²)
```

If physical screen dimensions are known:

```text
dx_mm = dx × screen_width_mm
dy_mm = dy × screen_height_mm

error_mm = sqrt(dx_mm² + dy_mm²)
```

Angular error:

```text
error_deg =
degrees(
  atan(error_mm / viewing_distance_mm)
)
```

Document this approximation.

---

# 33. Initial Product Gate

For A0, use the current working GazeFlow v0.1 positioning:

```text
Target use case:
coarse attention-zone analysis

Working minimum AOI width:
approximately 95 mm

Nominal viewing distance:
600 mm
```

At 600 mm, half of a 95 mm AOI is:

```text
47.5 mm
```

which is approximately:

```text
4.5 degrees
```

Use this initial technical decision guide:

```text
PASS-CANDIDATE

median angular error <= 3.0 degrees
AND
p95 target-level angular error <= 4.5 degrees
```

This is an A0 working gate, not a final product claim.

Stage A with multiple participants must confirm or reject it.

---

# 34. Borderline and Failure Logic

Use:

```text
PASS-CANDIDATE:
median <= 3.0°
and
p95 <= 4.5°
```

```text
BORDERLINE:
performance is worse than PASS-CANDIDATE
but remains within approximately 1.5 × the gate
```

For guidance:

```text
median <= 4.5°
and
p95 <= 6.75°
```

```text
FAIL:
clearly worse than the BORDERLINE range
or tracking is unstable/unusable
```

The report should not silently change these thresholds after seeing results.

---

# 35. Interpretation

## PASS-CANDIDATE

Meaning:

```text
MediaPipe + Ridge is promising enough
to justify broader feasibility testing.
```

Next:

```text
Stage A with multiple participants
```

## BORDERLINE

Meaning:

```text
The webcam signal may contain enough information,
but the simple model may be insufficient.
```

Next:

```text
consider stronger modeling
before product engineering
```

This is where a pretrained appearance-based model may be worth testing.

## FAIL

Meaning:

```text
This approach is too far from the current product requirement.
```

Do not build Tauri/React because of sunk cost.

Consider:

```text
Path A:
reposition to larger attention zones

Path B:
use external eye-tracking hardware

Path C:
stop
```

---

# 36. A0 Result JSON

Create:

```text
a0_results.json
```

Required shape:

```json
{
  "run_id": "2026-09-14T003000",
  "status": "PASS-CANDIDATE",
  "screen": {
    "width_px": 0,
    "height_px": 0,
    "width_mm": null,
    "height_mm": null,
    "viewing_distance_mm": 600
  },
  "camera": {
    "index": 0,
    "width": 1280,
    "height": 720,
    "requested_fps": 30,
    "effective_fps": 29.7
  },
  "calibration": {
    "points": 9,
    "selected_ridge_alpha": 1.0,
    "cv_median_error_norm": 0.0,
    "cv_worst_error_norm": 0.0
  },
  "validation": {
    "targets": 20,
    "median_error_norm": 0.0,
    "p95_error_norm": 0.0,
    "median_error_px": 0.0,
    "p95_error_px": 0.0,
    "median_error_mm": null,
    "p95_error_mm": null,
    "median_error_deg": null,
    "p95_error_deg": null
  }
}
```

Add other diagnostic fields if useful.

Do not remove these core fields.

---

# 37. A0 Summary Markdown

Generate:

```text
a0_summary.md
```

It must be understandable without opening Python.

Required sections:

```text
# GazeFlow A0 Result

## Decision
PASS-CANDIDATE / BORDERLINE / FAIL

## Setup
camera
resolution
screen
viewing distance
date

## Accuracy
median
p95
center
edges/corners
worst target

## Calibration
selected Ridge alpha
cross-validation result

## Tracking Quality
effective FPS
valid frame ratio
face detection failures
blink rejected frames

## Observations
visible problems

## Recommendation
next action
```

---

# 38. Required Plots

Generate PNG plots into:

```text
plots/
```

At minimum:

```text
validation_scatter.png
error_by_target.png
error_distribution.png
screen_error_map.png
```

## `validation_scatter.png`

Show:

```text
actual targets
predicted target medians
line from actual to predicted
```

## `screen_error_map.png`

Place the error value at each validation location.

This helps reveal:

```text
corner bias
vertical bias
left/right asymmetry
systematic offset
```

---

# 39. Optional Live Prediction View

After validation, optionally provide a debug mode:

```bash
python -m a0.main live outputs/<run-id>
```

Show:

```text
full screen
predicted gaze dot
confidence / validity
```

This is useful for qualitative inspection.

It is not a substitute for validation metrics.

Do not spend significant time polishing it.

---

# 40. Output Folder

Every run must create a unique folder.

Example:

```text
outputs/
└── 2026-09-14T003000/
    |
    +-- config.json
    +-- a0_raw_features.csv
    +-- a0_landmarks.jsonl
    +-- a0_validation.csv
    +-- a0_results.json
    +-- a0_summary.md
    +-- requirements-lock.txt
    |
    +-- plots/
    |   +-- validation_scatter.png
    |   +-- error_by_target.png
    |   +-- error_distribution.png
    |   +-- screen_error_map.png
    |
    +-- a0_capture.mp4
```

If debug video is disabled:

```text
a0_capture.mp4
```

is omitted.

---

# 41. Data Durability

A0 data is more important than A0 code.

Write CSV incrementally where practical.

If the experiment is interrupted, preserve collected data.

For the raw feature stream:

```text
append rows continuously
flush at least every 1 second
```

Do not keep all raw features only in memory until the end.

---

# 42. Permanent vs Disposable Assets

Disposable:

```text
A0 Python implementation
temporary UI code
experimental glue code
```

Permanent research assets:

```text
raw feature datasets
validation results
debug recordings when intentionally saved
analysis notebooks
result summaries
plots
experiment configuration
dependency lock file
```

Do not delete the dataset when Stage B begins.

Future gaze models must be comparable against historical A0 runs.

---

# 43. Minimal Tests

Do not create a large test suite.

Create only high-value deterministic tests.

## Geometry test

Verify:

```text
normalized error
pixel conversion
mm conversion
angular conversion
```

## Target generation test

Verify:

```text
9 calibration positions correct
20 validation points deterministic
validation points inside margins
validation points not equal to calibration points
```

## Model smoke test

Synthetic data:

```text
known features
known linear relationship
```

Verify Ridge pipeline approximately recovers the mapping.

---

# 44. Logging

Write a human-readable log:

```text
a0.log
```

Include:

```text
startup
Python version
package versions
camera selected
camera actual resolution
effective FPS
screen resolution
physical screen values
calibration point start/end
failed point retries
model fit
selected alpha
validation completion
final decision
errors/exceptions
```

---

# 45. Failure Handling

Handle these conditions without losing existing data:

```text
camera cannot open
MediaPipe model cannot load
face not detected
too few calibration samples
invalid feature values
user presses ESC
window closes
model fit fails
validation interrupted
```

On failure:

```text
flush current data
write failure reason
exit cleanly
```

---

# 46. Coding Style

Prioritize:

```text
clarity
short functions
explicit data structures
reproducibility
easy debugging
```

Avoid:

```text
complex design patterns
dependency injection frameworks
plugins
generic production abstractions
premature optimization
```

Use type hints where useful.

Use dataclasses for simple structured records if helpful.

---

# 47. README Requirements

The generated `README.md` must contain:

1. A0 purpose
2. Environment setup
3. How to run preflight
4. How to measure screen physical width
5. How to estimate/set viewing distance
6. How to run the experiment
7. Controls
8. Output files
9. How decision status is calculated
10. Known limitations
11. Reminder that A0 code is disposable

---

# 48. User Flow

The complete A0 experience should be:

```text
Start command
    ↓
Camera preflight
    ↓
Face / eye check
    ↓
SPACE
    ↓
Open-eye baseline
    ↓
9-point calibration
    ↓
Ridge fit + cross-validation
    ↓
20-point independent validation
    ↓
Result generated
    ↓
Summary displayed
```

On-screen completion example:

```text
GazeFlow A0 Complete

Median error: 2.8°
P95 error:   4.2°

Decision:
PASS-CANDIDATE

Results:
outputs/2026-09-14T003000/
```

---

# 49. Success Criteria for the Coding Agent

Implementation is complete only when all of the following work:

- [ ] `preflight` opens the camera.
- [ ] MediaPipe face landmarks are visible.
- [ ] Eye and iris features update live.
- [ ] Head pose updates live.
- [ ] Fullscreen 9-point calibration works.
- [ ] User only has to follow the target.
- [ ] Blink frames are rejected.
- [ ] Raw feature stream is saved.
- [ ] Selected landmarks are saved.
- [ ] Ridge alpha is selected using leave-one-point-out CV.
- [ ] Final Ridge model is fitted.
- [ ] 20 unseen validation targets run.
- [ ] Normalized error is calculated.
- [ ] Pixel error is calculated.
- [ ] mm/degree error is calculated only when physical geometry is known.
- [ ] Median and p95 are reported.
- [ ] Plots are generated.
- [ ] `a0_results.json` is generated.
- [ ] `a0_summary.md` is generated.
- [ ] Interrupted runs preserve collected raw data.
- [ ] README explains how to reproduce the run.

---

# 50. Explicit Non-Goals

Do not implement:

- Production GazeFlow desktop app
- Windows support during A0
- Tauri
- React
- Database
- Heatmap product UI
- AOI editor
- Participant management
- Study management
- Login
- Cloud sync
- Auto-update
- Installer
- Tobii integration
- IR tracking
- Deep-learning gaze model
- L2CS-Net integration in the first implementation

The deep-learning baseline is considered only if Ridge produces a BORDERLINE result.

---

# 51. Final Instruction to Codex / Claude Code

Implement this A0 harness end to end.

Do not stop after scaffolding.

Run available automated tests.

Resolve dependency and syntax issues.

Make the experiment runnable on macOS from the command line.

Where a reasonable implementation choice is required, choose the simplest approach that preserves the validity of the experiment.

Do not expand scope.

The final deliverable is not the code itself.

The final deliverable is a reproducible experiment capable of producing:

```text
a0_raw_features.csv
a0_landmarks.jsonl
a0_validation.csv
a0_results.json
a0_summary.md
plots/
```

from a real webcam calibration and validation session.

The question A0 must answer is:

> Is webcam-based gaze estimation promising enough to justify continuing GazeFlow?
