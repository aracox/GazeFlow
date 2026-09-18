import Foundation

/// How a YES/NO side gets confirmed once gaze is on it. Chosen on the start
/// screen before calibration; the calibration flow itself is unaffected.
enum SelectionMethod: String, CaseIterable, Identifiable {
    case singleBlink = "Blink"
    case doubleBlink = "Blink Twice"
    case dwell = "Look 3s"

    var id: String { rawValue }

    var instructions: String {
        switch self {
        case .singleBlink: return "Look + blink to select."
        case .doubleBlink: return "Look + blink twice quickly to select."
        case .dwell: return "Hold your gaze for 3s to select."
        }
    }
}
