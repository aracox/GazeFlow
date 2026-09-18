import SwiftUI

struct ContentView: View {
    @StateObject private var tracker = GazeTracker()
    @State private var stage: Stage = .starting
    @State private var calibrationModel: LinearCalibrationModel?
    @State private var selectionMethod: SelectionMethod = .singleBlink

    private enum Stage { case starting, notSupported, calibrating, running }

    var body: some View {
        Group {
            switch stage {
            case .starting:
                startScreen
            case .notSupported:
                Text("This device doesn't have a TrueDepth camera, so ARKit face tracking isn't available.")
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.black.ignoresSafeArea())
            case .calibrating:
                CalibrationView(tracker: tracker, onBack: { stage = .starting }) { model in
                    calibrationModel = model
                    stage = .running
                }
            case .running:
                if let model = calibrationModel {
                    YesNoView(tracker: tracker, model: model, question: "Yes or No?",
                              selectionMethod: selectionMethod, onBack: { stage = .starting })
                }
            }
        }
        .onAppear {
            // Calibration/interaction require the user to just look at the
            // screen without touching it for extended stretches -- without
            // this, iOS's idle timer auto-locks/dims the screen mid-session,
            // interrupting the ARSession and resetting tracking state.
            UIApplication.shared.isIdleTimerDisabled = true
        }
        .onDisappear {
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    private var startScreen: some View {
        VStack(spacing: 24) {
            Text("GazeFlow Sample").font(.largeTitle).bold()
            Text("Follow a dot to calibrate, then look left/right to answer Yes or No.")
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            VStack(spacing: 10) {
                Text("Select by:").font(.headline)
                Picker("Selection method", selection: $selectionMethod) {
                    ForEach(SelectionMethod.allCases) { method in
                        Text(method.rawValue).tag(method)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 420)
            }

            Button("Start") {
                guard tracker.isSupported else {
                    stage = .notSupported
                    return
                }
                tracker.start()
                stage = .calibrating
            }
            .font(.title2)
            .buttonStyle(.borderedProminent)
        }
        .foregroundColor(.white)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black.ignoresSafeArea())
    }
}
