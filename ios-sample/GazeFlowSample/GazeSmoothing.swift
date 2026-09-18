import Foundation

/// How much the gaze cursor's per-frame position is smoothed (exponential
/// moving average). Ordered low->high steadiness, like a Likert scale, so
/// the user can trade cursor responsiveness for less jitter when holding
/// still. Chosen on the start screen; applies to both Yes/No and Numbers.
enum GazeSmoothing: String, CaseIterable, Identifiable {
    case responsive = "Responsive"
    case balanced = "Balanced"
    case steady = "Steady"
    case verySteady = "Very Steady"

    var id: String { rawValue }

    /// Exponential moving average weight given to each new reading.
    /// Lower = more smoothing (steadier, but slower to follow real gaze).
    var alpha: Float {
        switch self {
        case .responsive: return 0.20
        case .balanced: return 0.12
        case .steady: return 0.08
        case .verySteady: return 0.05
        }
    }
}
