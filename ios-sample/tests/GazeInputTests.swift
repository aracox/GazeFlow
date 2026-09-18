import Foundation

private let targets: [Float] = [1.0 / 6.0, 0.5, 5.0 / 6.0]

private func check(_ condition: @autoclosure () -> Bool, _ message: String,
                   file: StaticString = #file, line: UInt = #line) {
    if !condition() { fatalError(message, file: file, line: line) }
}

private struct Simulation {
    var state = NumberSelectionState(targets: targets)
    var time: TimeInterval = 100
    var selected: [Int] = []
    var method: SelectionMethod = .singleBlink
    var alpha: Float = 0.08

    mutating func run(_ seconds: Double, x: Float = 0.1, y: Float = 0.5,
                      blink: Bool = false, enabled: Bool = true, hz: Double = 30) {
        for _ in 0..<Int((seconds * hz).rounded(.up)) {
            time += 1 / hz
            let reading = GazeReading(lookAtX: x, lookAtY: y, blinking: blink, timestamp: time)
            if let selected = state.update(reading: reading, position: (x, y), now: time,
                                           enabled: enabled, method: method, smoothingAlpha: alpha) {
                self.selected.append(selected)
            }
        }
    }

    mutating func blink(x: Float = 0.1) {
        run(0.13, x: x, blink: true)
        run(0.04, x: 0.9) // Intentionally corrupted reopen position.
    }
}

@main
enum GazeInputTests {
    static func main() {
        testFreshnessAndCapture()
        testCalibrationModel()
        testCalibrationCompletesWithBlinks()
        testBlinkSelection()
        testDwellAndInterruptions()
        testBoundariesAndSmoothing()
        testContiguousRegionsAndOvershoot()
        print("PASS: gaze freshness, calibration quality, blink/dwell selection, interruptions, and boundaries")
    }

    static func testFreshnessAndCapture() {
        let reading = GazeReading(lookAtX: 0.01, lookAtY: 0, blinking: false, timestamp: 100)
        check(reading.isFresh(at: 100.1), "Fresh sample rejected")
        check(!reading.isFresh(at: 100.3), "Stale sample accepted")
        check(!reading.isFresh(at: 99), "Future sample accepted")
        check(!GazeReading(lookAtX: .nan, lookAtY: 0, blinking: false, timestamp: 100).isFresh(at: 100), "NaN accepted")
        var capture = CalibrationCapture()
        for _ in 0..<100 {
            check(capture.append(reading, now: 100, enabled: true, targetX: 0.5, targetY: 0.5) == nil,
                  "Repeated camera frame completed calibration")
        }
        var accepted: [CalibrationSample]?
        for i in 1...55 {
            let time = 100 + Double(i) / 30
            let sample = GazeReading(lookAtX: 0.01, lookAtY: 0, blinking: false, timestamp: time)
            if let result = capture.append(sample, now: time, enabled: true, targetX: 0.5, targetY: 0.5) {
                accepted = result
                break
            }
        }
        check((accepted?.count ?? 0) >= 18, "Stable fixation did not complete")
        _ = capture.append(nil, now: 102, enabled: false, targetX: 0.5, targetY: 0.5)
        check(capture.samples.isEmpty, "Tracking loss retained partial fixation")
        for i in 1...20 {
            let time = 103 + Double(i) / 30
            _ = capture.append(GazeReading(lookAtX: 0.01, lookAtY: 0, blinking: false, timestamp: time),
                               now: time, enabled: true, targetX: 0.5, targetY: 0.5)
        }
        check(!capture.samples.isEmpty, "Partial fixation not collected")
        let beforeBlink = capture.samples.count
        _ = capture.append(GazeReading(lookAtX: 0.01, lookAtY: 0, blinking: true, timestamp: 103.7),
                           now: 103.7, enabled: true, targetX: 0.5, targetY: 0.5)
        check(capture.samples.count == beforeBlink, "Normal blink erased calibration progress")
        let drift: [CalibrationSample] = (0..<24).map { (0.01 + Float($0) * 0.00001, 0, 0.5, 0.5) }
        check(CalibrationQuality.robustSamples(drift) != nil, "Small natural drift restarted the point")
        let spiked: [CalibrationSample] = (0..<24).map { ($0 == 3 ? 10 : 0.01, 0, 0.5, 0.5) }
        check(CalibrationQuality.robustSamples(spiked)?.count == 23, "Isolated calibration spike was retained")
    }

    static func fixture(reversed: Bool = false) -> [CalibrationSample] {
        var samples: [CalibrationSample] = []
        for target in targets {
            for i in 0..<30 {
                let noise = Float(i % 3 - 1) * 0.00002
                let x = (target - 0.5) * (reversed ? -0.08 : 0.08) + noise
                samples.append((x, noise, target, 0.5))
            }
        }
        // Deliberately different horizontal readings at top/bottom must not
        // move digit 1's horizontal anchor away from the middle of the row.
        for _ in 0..<30 {
            samples.append((0.025, -0.02, 0.5, 0.15))
            samples.append((0.025, 0.02, 0.5, 0.85))
        }
        return samples
    }

    static func testCalibrationModel() {
        for reversed in [false, true] {
            guard let model = LinearCalibrationModel(samples: fixture(reversed: reversed)) else {
                fatalError("Good calibration rejected")
            }
            for target in targets {
                let raw = (target - 0.5) * (reversed ? -0.08 : 0.08)
                check(abs(model.predict(lookAtX: raw, lookAtY: 0).x - target) < 0.001,
                      "Calibrated target missed")
            }
            check(abs(model.predict(lookAtX: 0, lookAtY: 0).x - 0.5) < 0.001,
                  "Off-row samples corrupted horizontal mapping")
        }
        var reversedAnchor = fixture()
        for i in reversedAnchor.indices where reversedAnchor[i].2 == targets[1] {
            reversedAnchor[i].0 = (targets.last! - 0.5) * 0.08 + 0.01
        }
        check(LinearCalibrationModel(samples: reversedAnchor) == nil, "Non-monotonic anchors accepted")
        var duplicateAnchor = fixture()
        for i in duplicateAnchor.indices where duplicateAnchor[i].2 == targets[2] {
            duplicateAnchor[i].0 = (targets[1] - 0.5) * 0.08
        }
        check(LinearCalibrationModel(samples: duplicateAnchor) == nil, "Overlapping anchors accepted")
        var noisy = fixture()
        for i in noisy.indices where noisy[i].3 == 0.5 {
            noisy[i].0 += Float(i % 3 - 1) * 0.01
        }
        check(LinearCalibrationModel(samples: noisy) != nil, "Ordered robust anchors blocked by within-point noise")
        var withSpike = fixture()
        withSpike[0].0 = 1
        check(LinearCalibrationModel(samples: withSpike) != nil, "Single spike ruined robust anchors")
        withSpike[0].0 = .nan
        check(LinearCalibrationModel(samples: withSpike) == nil, "Non-finite calibration accepted")

    }

    static func testCalibrationCompletesWithBlinks() {
        // Reproduce the regression: blink once during EVERY target. The old
        // capture erased progress; full validation then added repeated rounds.
        // A single five-point pass must now produce a model within 15 s at 30 Hz.
        var capture = CalibrationCapture()
        var allSamples: [CalibrationSample] = []
        var time: TimeInterval = 200
        let points: [(Float, Float)] = [(0.5, 0.15), (0.5, 0.85)] + targets.map { ($0, 0.5) }
        for (targetX, targetY) in points {
            capture.reset()
            var completed = false
            for frame in 0..<90 {
                time += 1.0 / 30
                let noise = Float(frame % 3 - 1) * 0.00002
                let sample = GazeReading(lookAtX: (targetX - 0.5) * 0.08 + noise,
                                         lookAtY: (targetY - 0.5) * 0.06 + noise,
                                         blinking: (20...23).contains(frame), timestamp: time)
                if let result = capture.append(sample, now: time, enabled: true, targetX: targetX, targetY: targetY) {
                    allSamples.append(contentsOf: result)
                    completed = true
                    break
                }
            }
            check(completed, "Target failed to complete with a normal blink")
        }
        check(time - 200 + 5 < 15, "Calibration exceeded 15 seconds including countdown")
        check(LinearCalibrationModel(samples: allSamples) != nil, "Completed pass could not open Numbers")

        var closed = CalibrationCapture()
        for i in 0..<60 {
            let time = 300 + Double(i) / 30
            _ = closed.append(GazeReading(lookAtX: 0.01, lookAtY: 0, blinking: i >= 20, timestamp: time),
                              now: time, enabled: true, targetX: 0.5, targetY: 0.5)
        }
        check(closed.samples.isEmpty, "Prolonged eye closure retained old calibration samples")
    }

    static func testBlinkSelection() {
        var sim = Simulation()
        sim.run(0.7)
        sim.blink()
        check(sim.selected == [0], "Blink used noisy reopening gaze instead of stable target")
        sim.run(0.8)
        sim.blink()
        check(sim.selected == [0, 0], "Deliberate repeated digit blocked")

        var early = Simulation()
        early.run(0.12)
        early.blink()
        check(early.selected.isEmpty, "Blink before stable acquisition selected")
        var long = Simulation()
        long.run(0.7)
        long.run(0.9, blink: true)
        long.run(0.1)
        check(long.selected.isEmpty, "Long eye closure selected")

        var twice = Simulation(method: .doubleBlink)
        twice.run(0.7)
        twice.blink()
        check(twice.selected.isEmpty && twice.state.armedIndex == 0, "First double-blink confirmed prematurely")
        twice.run(0.2)
        twice.blink()
        check(twice.selected == [0], "Deliberate double-blink did not select")
        var expired = Simulation(method: .doubleBlink)
        expired.run(0.7)
        expired.blink()
        expired.run(1)
        check(expired.state.armedIndex == nil, "Expired double-blink remained armed")
        expired.blink()
        check(expired.selected.isEmpty, "Expired first blink completed selection")
    }

    static func testDwellAndInterruptions() {
        for hz in [15.0, 30.0, 60.0] {
            var sim = Simulation(method: .dwell)
            sim.run(8, hz: hz)
            check(sim.selected == [0], "Dwell repeated without leaving at \(hz) Hz")
            sim.run(0.7, y: 0.95, hz: hz)
            sim.run(4, hz: hz)
            check(sim.selected == [0, 0], "Look-away did not rearm repeated dwell digit")
        }
        var closed = Simulation(method: .dwell)
        closed.run(1.8)
        closed.run(0.5, blink: true)
        closed.run(0.15)
        check(closed.selected.isEmpty, "Dwell counted eye-closure time")

        var paused = Simulation(method: .dwell)
        paused.run(1.8)
        paused.run(2, enabled: false)
        paused.run(0.7)
        check(paused.selected.isEmpty, "Preview/tracking interruption retained dwell timer")
        paused.run(2)
        check(paused.selected == [0], "Dwell failed after reacquiring")
        paused.run(1, enabled: false)
        paused.run(4)
        check(paused.selected == [0], "Interruption rearmed a latched dwell")

        var stale = Simulation(method: .dwell)
        stale.run(1.8)
        let frozen = GazeReading(lookAtX: 0.1, lookAtY: 0.5, blinking: false, timestamp: stale.time)
        for i in 1...100 {
            let result = stale.state.update(reading: frozen, position: (0.1, 0.5), now: stale.time + Double(i) / 30,
                                            enabled: true, method: .dwell, smoothingAlpha: 0.08)
            check(result == nil, "Frozen frame completed dwell")
        }
        check(stale.state.activeIndex == nil, "Stale reading left an active target")
    }

    static func testBoundariesAndSmoothing() {
        var jitter = Simulation(method: .dwell)
        for i in 0..<120 {
            jitter.run(1.0 / 30, x: i % 10 == 9 ? targets[1] : targets[0])
        }
        check(jitter.selected == [0], "Isolated spikes prevented a deliberate dwell")

        var output = Simulation(method: .dwell)
        output.run(4, y: 0.95)
        check(output.selected.isEmpty, "Output area selected a digit")
        var lag = Simulation()
        lag.run(0.7)
        lag.run(0.04, x: targets[1])
        lag.blink(x: targets[1])
        check(lag.selected.isEmpty, "Slow smoothing confirmed old target after gaze moved")
        lag.run(2, x: targets[1])
        lag.blink(x: targets[1])
        check(lag.selected == [1], "Neighbor could not be acquired after settling")
    }

    static func testContiguousRegionsAndOvershoot() {
        let leftBoundary = (targets[0] + targets[1]) / 2
        let rightBoundary = (targets[1] + targets[2]) / 2
        let cases: [(Float, Int)] = [
            (-3, 0), (-0.3, 0), (0, 0), (0.25, 0), (leftBoundary - 0.001, 0),
            (leftBoundary, 1), (0.5, 1), (rightBoundary - 0.001, 1),
            (rightBoundary, 2), (0.8, 2), (1, 2), (1.3, 2), (3, 2)
        ]
        for (x, expected) in cases {
            var sim = Simulation()
            sim.run(0.7, x: x)
            sim.blink(x: x)
            check(sim.selected == [expected], "Contiguous region or edge blink failed at x=\(x)")
        }
        for (x, expected) in [(Float(-0.3), 0), (Float(1.3), 2)] {
            var twice = Simulation(method: .doubleBlink)
            twice.run(0.7, x: x)
            twice.blink(x: x)
            twice.run(0.2, x: x)
            twice.blink(x: x)
            check(twice.selected == [expected], "Double blink failed beyond screen edge")
            var dwell = Simulation(method: .dwell)
            dwell.run(4, x: x)
            check(dwell.selected == [expected], "Edge dwell failed or repeated automatically")
        }
        var lost = Simulation()
        lost.run(0.7, x: 1.3)
        lost.run(0.1, x: 1.3, enabled: false)
        lost.blink(x: 1.3)
        check(lost.selected.isEmpty, "Tracking loss guessed an edge selection")

        var extreme = Simulation()
        extreme.run(0.7, x: 1000)
        extreme.run(2, x: 0.5)
        extreme.blink(x: 0.5)
        check(extreme.selected == [1], "Extreme overshoot delayed returning to center")
    }
}
