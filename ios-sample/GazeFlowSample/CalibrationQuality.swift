import Foundation

typealias CalibrationSample = (Float, Float, Float, Float)

enum CalibrationQuality {
    static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        return (sorted[(sorted.count - 1) / 2] + sorted[sorted.count / 2]) / 2
    }

    static func spread(_ values: [Float]) -> Float {
        let center = median(values)
        return 1.4826 * median(values.map { abs($0 - center) })
    }

    /// A robust snapshot, not a laboratory fixation test. Natural slow eye
    /// movement must not restart capture; median anchors handle isolated noise.
    static func robustSamples(_ samples: [CalibrationSample]) -> [CalibrationSample]? {
        guard samples.count >= 18 else { return nil }
        let xs = samples.map { $0.0 }, ys = samples.map { $0.1 }
        let mx = median(xs), my = median(ys)
        let sx = max(spread(xs), 0.00001), sy = max(spread(ys), 0.00001)
        let inliers = samples.filter { abs($0.0 - mx) <= 4 * sx && abs($0.1 - my) <= 4 * sy }
        return inliers.count >= 18 ? inliers : nil
    }
}

/// Collects distinct live frames, pausing across normal blinks rather than
/// discarding the entire point. Tracking loss still invalidates the capture.
struct CalibrationCapture {
    static let targetSampleCount = 24
    static let maximumPointDuration: TimeInterval = 5

    private(set) var samples: [CalibrationSample] = []
    private var lastTimestamp: TimeInterval?
    private var openSince: TimeInterval?
    private var closedSince: TimeInterval?
    private var lastAcceptedTimestamp: TimeInterval?
    private var validDuration: TimeInterval = 0

    mutating func reset() {
        samples = []
        openSince = nil
        closedSince = nil
        lastAcceptedTimestamp = nil
        validDuration = 0
        // Retain the last consumed frame so a timer cannot count it twice.
    }

    mutating func append(_ reading: GazeReading?, now: TimeInterval, enabled: Bool,
                         targetX: Float, targetY: Float) -> [CalibrationSample]? {
        guard enabled, let reading, reading.isFresh(at: now) else { reset(); return nil }
        guard lastTimestamp.map({ reading.timestamp > $0 }) ?? true else { return nil }
        if let last = lastTimestamp, reading.timestamp - last > GazeReading.maximumAge { reset() }
        lastTimestamp = reading.timestamp
        if reading.blinking {
            if closedSince == nil { closedSince = reading.timestamp }
            openSince = nil
            lastAcceptedTimestamp = nil
            if reading.timestamp - (closedSince ?? reading.timestamp) > 0.6 {
                samples = []
                validDuration = 0
            }
            return nil
        }
        closedSince = nil
        if openSince == nil { openSince = reading.timestamp }
        // Initial target settling is longer than the recovery after a blink.
        let settling = samples.isEmpty ? 0.3 : 0.12
        guard reading.timestamp - (openSince ?? reading.timestamp) >= settling else { return nil }
        if let last = lastAcceptedTimestamp { validDuration += reading.timestamp - last }
        lastAcceptedTimestamp = reading.timestamp
        samples.append((reading.lookAtX, reading.lookAtY, targetX, targetY))
        if samples.count > Self.targetSampleCount { samples.removeFirst() }
        guard samples.count >= Self.targetSampleCount, validDuration >= 0.65 else { return nil }
        // A rejected window slides forward one fresh frame at a time, instead
        // of resetting progress and starting another full collection cycle.
        guard let result = CalibrationQuality.robustSamples(samples) else { return nil }
        reset()
        return result
    }
}
