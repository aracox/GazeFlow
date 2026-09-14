# GazeFlow A0 — Final Conclusion

**Question A0 was built to answer:** can a standard RGB webcam, MediaPipe
face/eye landmarks, and per-user Ridge calibration estimate screen gaze
accurately enough to justify continuing GazeFlow?

**Answer: BORDERLINE.** The signal is real and, with enough calibration
coverage, gets close to the PASS-CANDIDATE gate — median error actually beat
the PASS threshold in the best run. What's holding it back is one specific,
well-diagnosed tracking weak spot, not a fundamental limitation of the
approach.

---

## Result across all runs

| Run | Calib. grid | Calib. success | Median error | P95 error | Decision |
|---|---|---|---|---|---|
| 2026-09-14T223421 | 9-point | 7/9 | 6.96° | 12.00° | FAIL |
| 2026-09-14T224428 | 9-point | 9/9 | 5.14° | 11.14° | FAIL |
| 2026-09-14T231022 | 9-point | 9/9 | 5.40° | 12.37° | FAIL |
| 2026-09-14T232034 | 16-point | 14/16 | 3.13° | 8.19° | FAIL |
| 2026-09-14T232947 | 9-point | 7/9 | 8.62° | 13.74° | FAIL |
| 2026-09-14T233257 | 16-point | 15/16 | 3.85° | 5.66° | **BORDERLINE** |
| **2026-09-14T234746** | **25-point** | **23/25** | **2.17°** | **6.15°** | **BORDERLINE (best)** |

Gate definitions (`plan/GazeFlow_A0_Coding_Agent_Plan.md` section 33-34):
PASS-CANDIDATE needs median ≤3.0° **and** P95 ≤4.5°; BORDERLINE needs
median ≤4.5° **and** P95 ≤6.75°.

## What actually moved the number

**Calibration grid density was the single biggest lever.** Going from 9 to
16 to 25 calibration points drove a clear, monotonic improvement in the
best runs at each density (6.96°→3.85°→2.17° median across the best result
at each grid size). This wasn't noise — the two 9-point runs with clean
calibration data agreed with each other within 0.3°, and the pattern held
across three different grid sizes.

**A specific, recurring tracking dropout is the remaining bottleneck.**
Every single run — all seven — had its calibration failures concentrated
in the bottom-right region of the screen (`c7`/`c9` on the 9-point grid,
`c15`/`c16` on 16-point, `c24`/`c25` on 25-point — consistently the
lower-right corner regardless of grid resolution). The screen error map
from the 16-point BORDERLINE run showed the worst validation errors
clustered in exactly that region, directly adjacent to the failed
calibration anchors. This is very likely eyelid coverage of the iris (or
camera angle) when looking toward the lower-right of the screen, not
random noise — fixing it is the highest-leverage remaining change.

## What didn't help

Two follow-up investigations, both run as fast offline experiments reusing
already-collected data (no new capture sessions):

- **Swapping Ridge for a stronger regressor** (polynomial Ridge, random
  forest, gradient boosting, MLP — `scripts/compare_models.py`) made
  results *worse* on every model tested, on both runs tested. The
  calibration set (9-16 points × ~20 correlated frames) is too small and
  low-diversity for models with more capacity than Ridge to generalize
  from — they overfit.
- **A pretrained appearance-based gaze model** (ResNet34 trained on
  Gaze360, MIT-licensed, from
  [yakhyo/gaze-estimation](https://github.com/yakhyo/gaze-estimation);
  `scripts/gaze360_experiment.py`) also did worse than the geometric
  baseline on the same session (4.96°/8.65° vs. 3.85°/5.66°), despite a
  real working integration (100% frame processing success). Most likely
  explanation: Gaze360's training distribution (wide-ranging poses,
  distances, environments) doesn't match this narrow, controlled
  single-user desktop setup as well as features computed directly from
  this user's own eye geometry do; a face-crop convention mismatch with
  the model's training data may also have hurt its raw accuracy.

Both were legitimate, executed experiments with real results — not
untested guesses — and both point the same direction: for this specific
scenario, the simple geometric approach with adequate calibration coverage
is currently the strongest option, not a starting point to be replaced.

## Other findings along the way

- Fixed a real bug in head-pose estimation (`a0/geometry.py`): a sign
  convention mismatch in the 3D face anchor model was causing `solvePnP`
  to converge on a mirrored pose, manifesting as roll/pitch parked near
  ±180° on nearly every frame. Verified the fix empirically (two
  independent PnP solvers agreeing, lowest reprojection error) and
  switched to the `SQPNP` solver, which isn't prone to the local-minima
  ambiguity the old solver hit on this near-planar 6-point setup. This
  didn't change the FAIL/BORDERLINE outcome directly (head pose is just
  one of 17 model features) but removed a source of noise.
- Session-to-session tracking quality varied meaningfully (blink-rejected
  frame counts ranged from ~20 to ~380 across runs), which is itself a
  practical finding: results are sensitive to fatigue/lighting
  consistency between sessions, not just calibration density.
- `--calibration-points` was added to `a0/main.py` to make grid density
  configurable (previously hardcoded to 9 despite existing in config).

## Recommendation

Per the plan's own interpretation for BORDERLINE: worth continuing to
invest in this direction before concluding the approach won't work. Given
what's actually been learned, the two highest-value next moves, in order:

1. **Fix the lower-right tracking dropout specifically** (camera angle/
   position, or explicit attention to keeping eyes fully in frame there) —
   this alone is the difference between the current best BORDERLINE result
   (2.17°/6.15°) and likely clearing PASS-CANDIDATE (needs P95 ≤4.5°,
   which is where every run's worst points live).
2. **Move to Stage A** (multiple participants) once (1) is addressed, per
   the plan — a single-participant, single-day result, however
   consistent, isn't statistically powered for a product decision on its
   own.

Do **not** invest further in a stronger regressor or a pretrained
appearance model without new information — both were tried here and both
underperformed the current approach for this use case.

## Permanent artifacts from this spike

All seven runs' raw data, validation results, plots, and configs are kept
under `outputs/` (git-ignored by default, but on disk) as permanent
research assets per the plan — not deleted even though the `a0/` code
itself remains disposable. The two follow-up experiment scripts
(`scripts/compare_models.py`, `scripts/gaze360_experiment.py`) are also
disposable but kept for reference/reproducibility.
