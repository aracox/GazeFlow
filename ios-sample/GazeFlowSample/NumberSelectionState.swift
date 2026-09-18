import Foundation

/// Input state independent of SwiftUI so recorded/synthetic frame sequences can
/// exercise selection without a camera. All durations use camera timestamps.
struct NumberSelectionState {
    let targets: [Float]
    private(set) var smoothedX: Float = 0.5
    private(set) var smoothedY: Float = 0.5
    private(set) var activeIndex: Int?
    private(set) var armedIndex: Int?
    private(set) var dwellProgress: Double = 0

    private var lastTimestamp: TimeInterval?
    private var hasPosition = false
    private var recentPositions: [(x: Float, y: Float)] = []
    private var latestRawTarget: Int?
    private var candidateIndex: Int?
    private var candidateSince: TimeInterval?
    private var openSince: TimeInterval?
    private var closedSince: TimeInterval?
    private var blinkTarget: Int?
    private var firstBlinkTime: TimeInterval?
    private var dwellElapsed: TimeInterval = 0
    private var cooldownUntil: TimeInterval = 0
    private var dwellLatch: Int?
    private var leaveSince: TimeInterval?

    mutating func suspend() {
        clearCandidate()
        openSince = nil
        closedSince = nil
        blinkTarget = nil
        hasPosition = false
        recentPositions = []
        latestRawTarget = nil
        leaveSince = nil
        // Keep dwellLatch across interruptions: losing the camera is not an
        // intentional look away and must not enable another copy of a digit.
    }

    mutating func update(reading: GazeReading?, position: (x: Float, y: Float)?,
                         now: TimeInterval, enabled: Bool, method: SelectionMethod,
                         smoothingAlpha: Float) -> Int? {
        guard enabled, let reading, reading.isFresh(at: now),
              let position, position.x.isFinite, position.y.isFinite else {
            suspend()
            return nil
        }
        guard lastTimestamp.map({ reading.timestamp > $0 }) ?? true else { return nil }
        let time = reading.timestamp
        let elapsed = lastTimestamp.map { time - $0 } ?? 0
        if elapsed > GazeReading.maximumAge { suspend() }
        let dt = min(elapsed, GazeReading.maximumAge)
        lastTimestamp = time

        if let first = firstBlinkTime, time - first > 0.8 {
            firstBlinkTime = nil
            armedIndex = nil
        }

        if reading.blinking {
            if closedSince == nil {
                closedSince = time
                blinkTarget = latestRawTarget == activeIndex ? activeIndex : nil
                if activeIndex == nil { clearCandidate() }
            }
            openSince = nil
            leaveSince = nil
            if time - (closedSince ?? time) > 0.6 {
                clearCandidate()
                blinkTarget = nil
            }
            return nil
        }

        if let closed = closedSince {
            let duration = time - closed
            let target = blinkTarget
            closedSince = nil
            blinkTarget = nil
            openSince = time
            // Freeze the stable PRE-blink target. Reopening gaze can be noisy.
            if duration >= 0.08, duration <= 0.6, time >= cooldownUntil,
               let target, target == activeIndex {
                switch method {
                case .singleBlink:
                    return confirm(target, at: time, method: method)
                case .doubleBlink:
                    if armedIndex == target, let first = firstBlinkTime, time - first <= 0.8 {
                        return confirm(target, at: time, method: method)
                    }
                    armedIndex = target
                    firstBlinkTime = time
                case .dwell:
                    break
                }
            }
            return nil
        }

        if openSince == nil { openSince = time }
        guard time - (openSince ?? time) >= 0.1, time >= cooldownUntil else { return nil }

        latestRawTarget = target(at: position.x, y: position.y)
        // Horizontal overshoot belongs to the outer digit. Bound the value
        // before smoothing so a large overshoot doesn't delay returning inside.
        recentPositions.append((min(1, max(0, position.x)), position.y))
        if recentPositions.count > 3 { recentPositions.removeFirst() }
        // Remove isolated tracking spikes before smoothing. Requiring every
        // raw frame to hit the same digit would make a two-second dwell almost
        // impossible even for an otherwise good, slightly noisy calibration.
        let robustX = CalibrationQuality.median(recentPositions.map { $0.x })
        let robustY = CalibrationQuality.median(recentPositions.map { $0.y })
        if hasPosition {
            // Preserve the existing smoothing choices at 30 Hz while making
            // their response independent of timer jitter and frame rate.
            let alpha = Float(1 - pow(Double(1 - smoothingAlpha), dt * 30))
            smoothedX += alpha * (robustX - smoothedX)
            smoothedY += alpha * (robustY - smoothedY)
        } else {
            smoothedX = robustX
            smoothedY = robustY
            hasPosition = true
        }

        let rawTarget = target(at: robustX, y: robustY)
        let smoothTarget = target(at: smoothedX, y: smoothedY)
        // A slow cursor must not make a blink select the previous digit while
        // the current reading has already moved to its neighbor.
        let candidate = rawTarget == smoothTarget ? rawTarget : nil

        if let latch = dwellLatch {
            if let rawTarget, rawTarget == latch {
                leaveSince = nil
            } else {
                if leaveSince == nil { leaveSince = time }
                if time - (leaveSince ?? time) >= 0.25 {
                    dwellLatch = nil
                    leaveSince = nil
                }
            }
        }

        guard let candidate else { clearCandidate(); return nil }
        if candidate != candidateIndex {
            clearCandidate()
            candidateIndex = candidate
            candidateSince = time
            return nil
        }
        guard time - (candidateSince ?? time) >= 0.18 else { return nil }
        if activeIndex == nil {
            activeIndex = candidate
            dwellElapsed = 0
            return nil
        }

        if method == .dwell, dwellLatch != candidate, latestRawTarget == candidate {
            dwellElapsed += dt
            dwellProgress = min(1, dwellElapsed / 2.0)
            if dwellElapsed >= 2.0 { return confirm(candidate, at: time, method: method) }
        }
        return nil
    }

    private func target(at x: Float, y: Float) -> Int? {
        // Y is only a broad exclusion for the instructions/output area; the
        // actual choice between digits remains horizontal.
        guard x.isFinite, y.isFinite, (0.15...0.8).contains(y),
              targets.count >= 2 else { return nil }
        // Contiguous regions match the visible boxes: no dead zones. Exact
        // boundaries belong to the box on their right; the outer regions extend
        // beyond the screen so a live left/right overshoot still selects 0/2.
        for index in 0..<(targets.count - 1) {
            let boundary = (targets[index] + targets[index + 1]) / 2
            if x < boundary { return index }
        }
        return targets.count - 1
    }

    private mutating func clearCandidate() {
        candidateIndex = nil
        candidateSince = nil
        activeIndex = nil
        armedIndex = nil
        firstBlinkTime = nil
        dwellElapsed = 0
        dwellProgress = 0
    }

    private mutating func confirm(_ target: Int, at time: TimeInterval, method: SelectionMethod) -> Int {
        if method == .dwell { dwellLatch = target }
        cooldownUntil = time + 0.4
        clearCandidate()
        return target
    }
}
