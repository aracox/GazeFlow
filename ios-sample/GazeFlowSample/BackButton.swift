import SwiftUI

/// Small "back to menu" button, positioned top-leading, legible against
/// either a black (calibration) or red/green (Yes/No) background.
struct BackButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label("Menu", systemImage: "chevron.left")
                .font(.system(size: 13, weight: .semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.45))
                .foregroundColor(.white)
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}
