#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/gazeflow-tests.XXXXXX")
swiftc -module-cache-path "$test_dir/module-cache" \
    GazeFlowSample/GazeReading.swift \
    GazeFlowSample/CalibrationQuality.swift \
    GazeFlowSample/LinearCalibrationModel.swift \
    GazeFlowSample/SelectionMethod.swift \
    GazeFlowSample/NumberSelectionState.swift \
    tests/GazeInputTests.swift -o "$test_dir/gaze-input-tests"
"$test_dir/gaze-input-tests"
