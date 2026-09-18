import Foundation

/// Which screen calibration leads into, chosen on the start menu.
enum AppMode: String, CaseIterable, Identifiable {
    case yesNo = "Yes/No"
    case numberPad = "Numbers"
    case wordPicker = "Words"

    var id: String { rawValue }

    var instructions: String {
        switch self {
        case .yesNo: return "Follow a dot to calibrate, then look left/right to answer Yes or No."
        case .numberPad: return "Follow a dot to calibrate, then look at a digit (0-2) to pick it."
        case .wordPicker: return "Follow a dot to calibrate, then look left/right to narrow down to a word."
        }
    }
}
