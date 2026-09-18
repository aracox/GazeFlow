import SwiftUI

/// Fresh in-app calibration, same idea as gaze_ui_common.py's
/// cross_calibration_points()/run_calibration(): a 5-point cross (four
/// corners + center), calibrated against THIS window's own bounds each
/// run rather than reusing a stale session's fit.
struct CalibrationPoint: Identifiable {
    let id: String
    let x: CGFloat  // normalized [0,1] within the calibration area
    let y: CGFloat
}

let calibrationMargin: CGFloat = 0.15
let calibrationCrossPoints: [CalibrationPoint] = [
    CalibrationPoint(id: "center", x: 0.5, y: 0.5),
    CalibrationPoint(id: "tl", x: calibrationMargin, y: calibrationMargin),
    CalibrationPoint(id: "tr", x: 1 - calibrationMargin, y: calibrationMargin),
    CalibrationPoint(id: "bl", x: calibrationMargin, y: 1 - calibrationMargin),
    CalibrationPoint(id: "br", x: 1 - calibrationMargin, y: 1 - calibrationMargin),
].shuffled()

private let settleSeconds: Double = 0.4
private let targetValidSamples = 30
private let prepSeconds: Double = 5.0

struct CalibrationView: View {
    @ObservedObject var tracker: GazeTracker
    let onBack: () -> Void
    let onComplete: (LinearCalibrationModel) -> Void

    @State private var pointIndex = 0
    @State private var phase: Phase = .prep
    @State private var collected: [(Float, Float, Float, Float)] = []
    @State private var allSamples: [(Float, Float, Float, Float)] = []
    @State private var progress: Double = 0
    @State private var completionError: String?
    @State private var isShowingCameraPreview = false
    @State private var prepStartTime = Date()
    @State private var prepRemaining = Int(prepSeconds.rounded(.up))

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
                } else if pointIndex < calibrationCrossPoints.count {
                    let point = calibrationCrossPoints[pointIndex]
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
        let line0 = "point: \(pointIndex)/\(calibrationCrossPoints.count), collected: \(collected.count), total: \(allSamples.count), phase: \(phase)"
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

        guard pointIndex < calibrationCrossPoints.count else { return }
        guard phase == .collecting, let reading = tracker.latest, !reading.blinking else { return }

        let point = calibrationCrossPoints[pointIndex]
        collected.append((reading.lookAtX, reading.lookAtY, Float(point.x), Float(point.y)))
        progress = min(1.0, Double(collected.count) / Double(targetValidSamples))

        if collected.count >= targetValidSamples {
            finishPoint()
        }
    }

    private func finishPoint() {
        allSamples.append(contentsOf: collected)
        pointIndex += 1
        if pointIndex >= calibrationCrossPoints.count {
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
