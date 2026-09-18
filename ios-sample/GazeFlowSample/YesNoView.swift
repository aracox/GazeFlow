import SwiftUI

/// Port of sample_app.py's YES/NO picker, extended with a choice of
/// confirmation gesture (SelectionMethod): a single blink (the Python
/// app's final design), a double blink within a short window, or holding
/// gaze on a side for 3 seconds (dwell -- the Python app's original design,
/// removed there in favor of blink-only, offered here as a user choice
/// instead of picking one for everyone).
private let blinkMinConsecutiveFrames = 3
private let blinkSettleFrames = 3
private let doubleBlinkWindowSeconds: TimeInterval = 0.8
private let dwellSeconds: TimeInterval = 2.0
private let confirmationDisplaySeconds = 1.2

struct YesNoView: View {
    @ObservedObject var tracker: GazeTracker
    let model: LinearCalibrationModel
    let question: String
    let selectionMethod: SelectionMethod
    let showGazeDot: Bool
    let gazeSmoothing: GazeSmoothing
    let onBack: () -> Void

    @State private var smoothedX: Float = 0.5
    @State private var smoothedY: Float = 0.5
    @State private var zone: Zone?
    @State private var confirmedZone: Zone?
    @State private var answerHistory: [String] = []

    @State private var eyesClosedRun = 0
    @State private var openFrameStreak = 0
    @State private var firstBlinkTime: Date?
    @State private var armedZone: Zone?
    @State private var dwellStartTime: Date?
    @State private var isShowingCameraPreview = false

    // See CalibrationView's tickTimer for why this must be @State, not an
    // inline Timer.publish(...) in `body`.
    @State private var tickTimer = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    private enum Zone { case no, yes }

    var body: some View {
        CameraPreviewOverlay(tracker: tracker, isShowing: $isShowingCameraPreview) {
            yesNoContent
        }
    }

    private var yesNoContent: some View {
        GeometryReader { geo in
            ZStack {
                if let confirmed = confirmedZone {
                    (confirmed == .no ? Color.red : Color.green).opacity(0.85).ignoresSafeArea()
                    Text("Selected: \(confirmed == .no ? "NO" : "YES")")
                        .font(.system(size: 48, weight: .bold))
                        .foregroundColor(.white)
                } else {
                    HStack(spacing: 0) {
                        Rectangle().fill(armedSideColor(for: .no))
                        Rectangle().fill(armedSideColor(for: .yes))
                    }
                    .ignoresSafeArea()

                    Text(question)
                        .font(.title2).bold()
                        .foregroundColor(.white)
                        .position(x: geo.size.width / 2, y: geo.size.height * 0.10)
                    Text(selectionMethod.instructions)
                        .font(.subheadline)
                        .foregroundColor(.white.opacity(0.85))
                        .position(x: geo.size.width / 2, y: geo.size.height * 0.16)

                    Text("NO").font(.system(size: 60, weight: .bold)).foregroundColor(.white)
                        .position(x: geo.size.width * 0.25, y: geo.size.height * 0.5)
                    Text("YES").font(.system(size: 60, weight: .bold)).foregroundColor(.white)
                        .position(x: geo.size.width * 0.75, y: geo.size.height * 0.5)

                    if selectionMethod == .dwell, let z = zone {
                        dwellRing(for: z, geo: geo)
                    }

                    if showGazeDot {
                        Circle().stroke(Color.red, lineWidth: 3).frame(width: 20, height: 20)
                            .position(x: CGFloat(smoothedX) * geo.size.width, y: CGFloat(smoothedY) * geo.size.height)
                    }
                }

                Text("Output: \(answerHistory.isEmpty ? "-" : answerHistory.joined(separator: " "))")
                    .font(.system(size: 16, weight: .semibold, design: .monospaced))
                    .foregroundColor(.white)
                    .lineLimit(2)
                    .minimumScaleFactor(0.6)
                    .padding(.horizontal, 24)
                    .frame(width: geo.size.width * 0.8)
                    .multilineTextAlignment(.center)
                    .position(x: geo.size.width / 2, y: geo.size.height * 0.87)

                Text(diagnosticText)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundColor(.yellow)
                    .position(x: geo.size.width / 2, y: geo.size.height * 0.93)

                BackButton(action: onBack)
                    .position(x: 50, y: 24)
            }
            .onReceive(tickTimer) { _ in
                tick()
            }
        }
    }

    private func armedSideColor(for side: Zone) -> Color {
        let base = side == .no ? Color.red : Color.green
        if selectionMethod == .doubleBlink, armedZone == side { return base.opacity(1.0) }
        return base.opacity(zone == side ? 0.9 : 0.65)
    }

    private func dwellRing(for z: Zone, geo: GeometryProxy) -> some View {
        let progress = dwellStartTime.map { min(1.0, Date().timeIntervalSince($0) / dwellSeconds) } ?? 0
        let cx = z == .no ? geo.size.width * 0.25 : geo.size.width * 0.75
        return Circle()
            .trim(from: 0, to: progress)
            .stroke(Color.white, style: StrokeStyle(lineWidth: 6, lineCap: .round))
            .frame(width: 70, height: 70)
            .rotationEffect(.degrees(-90))
            .position(x: cx, y: geo.size.height * 0.72)
    }

    private var diagnosticText: String {
        String(format: "x: %.3f, y: %.3f, zone: %@", smoothedX, smoothedY, zone.map { $0 == .yes ? "YES" : "NO" } ?? "none")
    }

    private func tick() {
        guard !isShowingCameraPreview else { return }
        guard confirmedZone == nil else { return }
        guard let reading = tracker.latest else { return }

        var blinkEvent = false
        if reading.blinking {
            eyesClosedRun += 1
            openFrameStreak = 0
        } else {
            if eyesClosedRun >= blinkMinConsecutiveFrames { blinkEvent = true }
            eyesClosedRun = 0
            openFrameStreak += 1
        }

        // lookAtPoint is still recovering for a few frames right as the
        // eyes reopen (eyelid/cornea not fully clear yet), so trust
        // position again only once the eyes have been open for a short
        // settle window -- otherwise, on the very tick a blink ends,
        // resuming tracking immediately can slide the zone to the other
        // side in the SAME tick that confirms the blink, picking an
        // answer the user was never looking at.
        if openFrameStreak > blinkSettleFrames {
            let (predX, predY) = model.predict(lookAtX: reading.lookAtX, lookAtY: reading.lookAtY)
            let alpha = gazeSmoothing.alpha
            smoothedX = alpha * predX + (1 - alpha) * smoothedX
            smoothedY = alpha * predY + (1 - alpha) * smoothedY
            let newZone: Zone = smoothedX >= 0.5 ? .yes : .no

            if newZone != zone {
                zone = newZone
                dwellStartTime = Date()
                firstBlinkTime = nil
                armedZone = nil
            }
        }

        switch selectionMethod {
        case .singleBlink:
            if blinkEvent, let z = zone { confirm(z) }

        case .doubleBlink:
            guard blinkEvent, let z = zone else { break }
            if let armed = armedZone, armed == z,
               let first = firstBlinkTime, Date().timeIntervalSince(first) <= doubleBlinkWindowSeconds {
                confirm(z)
            } else {
                armedZone = z
                firstBlinkTime = Date()
            }

        case .dwell:
            if let z = zone, let start = dwellStartTime, Date().timeIntervalSince(start) >= dwellSeconds {
                confirm(z)
            }
        }
    }

    private func confirm(_ z: Zone) {
        confirmedZone = z
        answerHistory.append(z == .yes ? "yes" : "no")
        DispatchQueue.main.asyncAfter(deadline: .now() + confirmationDisplaySeconds) {
            confirmedZone = nil
            eyesClosedRun = 0
            firstBlinkTime = nil
            armedZone = nil
            dwellStartTime = Date()
        }
    }
}
