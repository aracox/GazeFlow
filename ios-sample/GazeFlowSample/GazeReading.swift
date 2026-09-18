import Foundation

/// One camera frame, measured on the same uptime clock as ARFrame.timestamp.
struct GazeReading {
    let lookAtX: Float
    let lookAtY: Float
    let blinking: Bool
    let timestamp: TimeInterval

    static let maximumAge: TimeInterval = 0.2

    func isFresh(at now: TimeInterval) -> Bool {
        lookAtX.isFinite && lookAtY.isFinite && timestamp.isFinite &&
        now >= timestamp && now - timestamp <= Self.maximumAge
    }
}
