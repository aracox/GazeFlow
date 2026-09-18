import SwiftUI

struct ContentView: View {
    @StateObject private var tracker = GazeTracker()
    @State private var stage: Stage = .starting
    @State private var calibrationModel: LinearCalibrationModel?
    @State private var selectionMethod: SelectionMethod = .singleBlink
    @State private var appMode: AppMode = .yesNo
    @State private var showGazeDot: Bool = true
    @State private var gazeSmoothing: GazeSmoothing = .steady

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
                CalibrationView(tracker: tracker, appMode: appMode, onBack: { stage = .starting }) { model in
                    calibrationModel = model
                    stage = .running
                }
            case .running:
                if let model = calibrationModel {
                    switch appMode {
                    case .yesNo:
                        YesNoView(tracker: tracker, model: model, question: "Yes or No?",
                                  selectionMethod: selectionMethod, showGazeDot: showGazeDot,
                                  gazeSmoothing: gazeSmoothing, onBack: { stage = .starting })
                    case .numberPad:
                        NumberPadView(tracker: tracker, model: model, selectionMethod: selectionMethod,
                                      showGazeDot: showGazeDot, gazeSmoothing: gazeSmoothing, onBack: { stage = .starting })
                    case .wordPicker:
                        WordPickerView(tracker: tracker, model: model, selectionMethod: selectionMethod,
                                        showGazeDot: showGazeDot, gazeSmoothing: gazeSmoothing, onBack: { stage = .starting })
                    }
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
            Text(appMode.instructions)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            VStack(spacing: 10) {
                Text("Mode:").font(.headline)
                Picker("Mode", selection: $appMode) {
                    ForEach(AppMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 420)
            }

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

            VStack(spacing: 10) {
                Text("Cursor smoothing:").font(.headline)
                Picker("Cursor smoothing", selection: $gazeSmoothing) {
                    ForEach(GazeSmoothing.allCases) { level in
                        Text(level.rawValue).tag(level)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 420)
                Text("More steady reduces jitter but follows your eyes a bit slower.")
                    .font(.footnote)
                    .foregroundColor(.white.opacity(0.7))
            }

            Toggle("Show gaze dot", isOn: $showGazeDot)
                .frame(maxWidth: 260)

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
