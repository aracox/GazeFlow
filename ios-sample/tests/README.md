# Gaze input regression checks

From the repository root, run `sh ios-sample/tests/run-tests.sh`. This compiles
the production calibration and number-selection logic with a small Swift test
driver; no camera, simulator, package download, or signing is required.

The sequences cover stale/duplicate frames, calibration with normal blinks, outliers and bad anchors,
both horizontal axis directions, blink recovery, double-blink
expiry, dwell pauses/re-arming, contiguous regions, edge overshoot, isolated spikes, and smoothing lag.

## Device check

Build/run GazeFlowSample on the iPad and choose Numbers. Three digits (0–2) share
one row of adjoining equal-width boxes with the existing selection-method controls.
Calibration matches their centers and makes one pass through
five targets (three digits and two vertical targets), then opens Numbers when the
model can be fitted. It does not add validation/retry rounds. A point that cannot
be captured within five seconds shows an error; Back restarts the flow.
Normal blinks pause collection and preserve progress. Expect roughly 10–15 seconds
including the five-second countdown with fresh readings at about 30 Hz. This is
an estimate supported by synthetic timing checks, not a device measurement.

- Pick each digit with a single blink, then repeat with double blink. Confirm the
  highlight settles before blinking and that reopening does not change the digit.
- In Look 2s, keep looking after a selection: it should append only once. Look
  away briefly and return to deliberately enter the same digit again.
- While dwelling, close your eyes, obscure the camera, or open the camera preview.
  Eye closure must not advance the dwell; tracking loss/preview must cancel it.
- The left, middle, and right thirds select 0, 1, and 2 with no gaps. A stable
  reading beyond the left/right edge still highlights 0/2 and can be confirmed.
  Tracking loss must not guess a digit; looking at the output line remains excluded.
- Repeat after normal head movement. Synthetic tests verify the logic, not device
  accuracy; calibration thresholds still need evaluation with pilot participants.

The initial thresholds are 0.2 s maximum sample age, 0.3 s initial calibration
settling (0.12 s after a normal blink), 24 samples with at least 0.65 s of valid
capture, 0.18 s stable target acquisition, and 0.25 s looking away to re-arm dwell.
Median calibration anchors must be ordered and separated; within-point scatter
alone does not block entry. There is no independent accuracy pass in this quick
calibration flow, so entering Numbers does not certify tracking accuracy.
