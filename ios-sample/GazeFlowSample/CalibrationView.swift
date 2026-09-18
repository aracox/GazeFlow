import SwiftUI

/// Fresh in-app calibration, same idea as gaze_ui_common.py's
/// cross_calibration_points()/run_calibration(): calibrate against THIS
/// window's own bounds each run rather than reusing a stale session's fit.
///
/// Points are chosen per AppMode to match where that mode's real targets
/// sit, not a generic 5-point cross -- a purely linear calibration model
/// only strictly needs 2 points to define its slope, but concentrating the
/// training data at the exact spots the user will later look at biases the
/// fit's low-error region to land where it's actually needed (e.g. all 6
/// narrow digit x-positions for Numbers, vs. just the 2 wide Yes/No zones).
/// A couple of off-row points are still included in every mode so the
/// per-axis 3-parameter fit (bias + rawX + rawY) has real Y variation to
/// fit against, instead of degenerating from an all-one-row training set.
struct CalibrationPoint: Identifiable {
    let id: String
    let x: CGFloat  // normalized [0,1] within the calibration area
    let y: CGFloat
}

private func calibrationPoints(for mode: AppMode) -> [CalibrationPoint] {
    var points: [CalibrationPoint] = [
        CalibrationPoint(id: "top", x: 0.5, y: 0.15),
        CalibrationPoint(id: "bottom", x: 0.5, y: 0.85),
    ]

    switch mode {
    case .yesNo, .wordPicker:
        points.append(CalibrationPoint(id: "left", x: 0.25, y: 0.5))
        points.append(CalibrationPoint(id: "right", x: 0.75, y: 0.5))
        points.append(CalibrationPoint(id: "center", x: 0.5, y: 0.5))
    case .numberPad:
        for (i, x) in numberPadTargetXPositions().enumerated() {
            points.append(CalibrationPoint(id: "digit\(i)", x: x, y: 0.5))
        }
    }

    return points.shuffled()
}

private let settleSeconds: Double = 0.4
private let targetValidSamples = 30
private let prepSeconds: Double = 5.0

struct CalibrationView: View {
    @ObservedObject var tracker: GazeTracker
    let appMode: AppMode
    let onBack: () -> Void
    let onComplete: (LinearCalibrationModel) -> Void

    @State private var points: [CalibrationPoint]
    @State private var pointIndex = 0
    @State private var phase: Phase = .prep
    @State private var collected: [(Float, Float, Float, Float)] = []
    @State private var allSamples: [(Float, Float, Float, Float)] = []
    @State private var progress: Double = 0
    @State private var completionError: String?
    @State private var isShowingCameraPreview = false
    @State private var prepStartTime = Date()
    @State private var prepRemaining = Int(prepSeconds.rounded(.up))

    init(tracker: GazeTracker, appMode: AppMode, onBack: @escaping () -> Void, onComplete: @escaping (LinearCalibrationModel) -> Void) {
        self.tracker = tracker
        self.appMode = appMode
        self.onBack = onBack
        self.onComplete = onComplete
        _points = State(initialValue: calibrationPoints(for: appMode))
    }

    // Created exactly once (the initializer only runs the first time this
    // view's @State is set up, not on every body re-evaluation) -- unlike
    // creating Timer.publish(...) inline inside `body`, which would spawn a
    // brand-new timer on every redraw and cancel it before it ever fires,
    // since `tracker`'s @Published properties redraw this view ~30-60x/sec.
    @State private var tickTimer = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    private enum Phase { case prep, settling, collecting }

    var body: some View {
        CameraPreviewOverlay(tracker: tracker, isShowing: $isShowingCameraPreview) {
            calibrationContent
        }
    }

    private var calibrationContent: some View {
        GeometryReader { geo in
            ZStack {
                Color.black.ignoresSafeArea()

                if phase == .prep {
                    VStack(spacing: 12) {
                        Text("Get ready")
                            .font(.title2).bold()
                        Text("\(prepRemaining)")
                            .font(.system(size: 90, weight: .bold, design: .rounded))
                        Text("Calibration starts soon -- find a comfortable position and look at the screen")
                            .font(.subheadline)
                            .multilineTextAlignment(.center)
                            .foregroundColor(.white.opacity(0.8))
                            .padding(.horizontal, 40)
                    }
                    .foregroundColor(.white)
                    .position(x: geo.size.width / 2, y: geo.size.height / 2)
                } else if pointIndex < points.count {
                    let point = points[pointIndex]
                    let center = CGPoint(x: point.x * geo.size.width, y: point.y * geo.size.height)

                    Circle()
                        .stroke(Color.white, lineWidth: 2)
                        .frame(width: 34 - 22 * progress, height: 34 - 22 * progress)
                        .position(center)
                    Circle()
                        .fill(Color.white)
                        .frame(width: 8, height: 8)
                        .position(center)
                }

                if phase != .prep {
                    Text("Follow the dot with your eyes")
                        .foregroundColor(.white)
                        .position(x: geo.size.width / 2, y: geo.size.height * 0.06)
                }

                Text(diagnosticText)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundColor(.yellow)
                    .multilineTextAlignment(.center)
                    .padding(4)
                    .position(x: geo.size.width / 2, y: geo.size.height * 0.93)

                BackButton(action: onBack)
                    .position(x: 50, y: 24)
            }
            .onAppear { startPrepCountdown() }
            .onReceive(tickTimer) { _ in
                tick(in: geo.size)
            }
        }
    }

    private var diagnosticText: String {
        let hasReading = tracker.latest != nil ? "yes" : "no"
        let line0 = "point: \(pointIndex)/\(points.count), collected: \(collected.count), total: \(allSamples.count), phase: \(phase)"
        let line1 = "reading now: \(hasReading), blinking: \(tracker.latest?.blinking.description ?? "n/a"), ever: \(tracker.successfulReadingCount)"
        let line2 = "frames: \(tracker.frameCallbackCount), anchors: \(tracker.lastFrameAnchorCount), cam: \(tracker.cameraTrackingState)"
        var lines = [line0, line1, line2]
        if let completionError {
            lines.append("FIT FAILED: \(completionError)")
        }
        return lines.joined(separator: "\n")
    }

    private func startPrepCountdown() {
        phase = .prep
        prepStartTime = Date()
        prepRemaining = Int(prepSeconds.rounded(.up))
    }

    private func startPoint() {
        phase = .settling
        collected = []
        progress = 0
        DispatchQueue.main.asyncAfter(deadline: .now() + settleSeconds) {
            phase = .collecting
        }
    }

    private func tick(in size: CGSize) {
        guard !isShowingCameraPreview else { return }

        if phase == .prep {
            let elapsed = Date().timeIntervalSince(prepStartTime)
            let remaining = max(0, Int((prepSeconds - elapsed).rounded(.up)))
            if remaining != prepRemaining { prepRemaining = remaining }
            if elapsed >= prepSeconds { startPoint() }
            return
        }

        guard pointIndex < points.count else { return }
        guard phase == .collecting, let reading = tracker.latest, !reading.blinking else { return }

        let point = points[pointIndex]
        collected.append((reading.lookAtX, reading.lookAtY, Float(point.x), Float(point.y)))
        progress = min(1.0, Double(collected.count) / Double(targetValidSamples))

        if collected.count >= targetValidSamples {
            finishPoint()
        }
    }

    private func finishPoint() {
        allSamples.append(contentsOf: collected)
        pointIndex += 1
        if pointIndex >= points.count {
            if let model = LinearCalibrationModel(samples: allSamples) {
                onComplete(model)
            } else {
                completionError = "LinearCalibrationModel init returned nil with \(allSamples.count) samples"
            }
        } else {
            startPoint()
        }
    }
}
