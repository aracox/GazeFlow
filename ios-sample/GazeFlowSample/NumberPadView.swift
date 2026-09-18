import SwiftUI

/// Single-stage digit picker for a small test range (0-5): one row of boxes
/// spanning the full width, so only left/right gaze position matters. Each
/// selection appends to the output and the view stays put -- it does not
/// reset or navigate back to the start menu, so you can keep picking digits
/// in a row.
private let blinkMinConsecutiveFrames = 3
private let blinkSettleFrames = 3
private let doubleBlinkWindowSeconds: TimeInterval = 0.8
private let dwellSeconds: TimeInterval = 2.0
private let flashDisplaySeconds: TimeInterval = 0.4
private let digitRowMargin: CGFloat = 0.10

private let digits: [String] = ["0", "1", "2", "3", "4", "5"]

private struct NumberBox {
    let label: String
    let cx: CGFloat
    let cy: CGFloat
    let w: CGFloat
    let h: CGFloat
}

private func digitBoxes() -> [NumberBox] {
    let n = digits.count
    let boxW = (1 - 2 * digitRowMargin) / CGFloat(n) * 0.85
    return digits.enumerated().map { i, d in
        let x = digitRowMargin + (1 - 2 * digitRowMargin) * CGFloat(i) / CGFloat(n - 1)
        return NumberBox(label: d, cx: x, cy: 0.5, w: boxW, h: 0.6)
    }
}

/// Exposes the digit boxes' exact x-positions so CalibrationView can
/// calibrate directly at these targets instead of a generic cross.
func numberPadTargetXPositions() -> [CGFloat] {
    digitBoxes().map { $0.cx }
}

struct NumberPadView: View {
    @ObservedObject var tracker: GazeTracker
    let model: LinearCalibrationModel
    let selectionMethod: SelectionMethod
    let showGazeDot: Bool
    let gazeSmoothing: GazeSmoothing
    let onBack: () -> Void

    @State private var smoothedX: Float = 0.5
    @State private var smoothedY: Float = 0.5
    @State private var boxes: [NumberBox] = digitBoxes()
    @State private var activeIndex: Int?
    @State private var armedIndex: Int?
    @State private var flashIndex: Int?
    @State private var flashUntil: Date?
    @State private var outputDigits: String = ""

    @State private var eyesClosedRun = 0
    @State private var openFrameStreak = 0
    @State private var firstBlinkTime: Date?
    @State private var dwellStartTime: Date?
    @State private var isShowingCameraPreview = false

    // See CalibrationView's tickTimer for why this must be @State, not an
    // inline Timer.publish(...) in `body`.
    @State private var tickTimer = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    var body: some View {
        CameraPreviewOverlay(tracker: tracker, isShowing: $isShowingCameraPreview) {
            content
        }
    }

    private var content: some View {
        GeometryReader { geo in
            ZStack {
                Color.black.ignoresSafeArea()

                ForEach(Array(boxes.enumerated()), id: \.offset) { index, box in
                    boxView(box, index: index, geo: geo)
                }

                if selectionMethod == .dwell, let active = activeIndex {
                    dwellRing(for: boxes[active], geo: geo)
                }

                if showGazeDot {
                    Circle().stroke(Color.red, lineWidth: 3).frame(width: 20, height: 20)
                        .position(x: CGFloat(smoothedX) * geo.size.width, y: CGFloat(smoothedY) * geo.size.height)
                }

                Text("Pick a digit: \(selectionMethod.instructions)")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.9))
                    .position(x: geo.size.width / 2, y: geo.size.height * 0.06)

                Text("Output: \(outputDigits.isEmpty ? "-" : outputDigits)")
                    .font(.system(size: 22, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                    .position(x: geo.size.width / 2, y: geo.size.height * 0.87)

                Text(String(format: "x: %.3f, y: %.3f", smoothedX, smoothedY))
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundColor(.yellow)
                    .position(x: geo.size.width / 2, y: geo.size.height * 0.93)

                BackButton(action: onBack)
                    .position(x: 50, y: 24)
            }
            .onReceive(tickTimer) { _ in tick() }
        }
    }

    private func boxView(_ box: NumberBox, index: Int, geo: GeometryProxy) -> some View {
        let w = box.w * geo.size.width
        let h = box.h * geo.size.height
        var fill = Color(white: 0.24)
        if flashIndex == index {
            fill = Color.green
        } else if selectionMethod == .doubleBlink, armedIndex == index {
            fill = Color.blue.opacity(0.85)
        } else if activeIndex == index {
            fill = Color(white: 0.40)
        }

        return ZStack {
            Rectangle().fill(fill)
            Rectangle().stroke(Color.white.opacity(0.8), lineWidth: 1)
            Text(box.label).font(.system(size: 50, weight: .bold)).foregroundColor(.white)
        }
        .frame(width: w, height: h)
        .position(x: box.cx * geo.size.width, y: box.cy * geo.size.height)
    }

    private func dwellRing(for box: NumberBox, geo: GeometryProxy) -> some View {
        let progress = dwellStartTime.map { min(1.0, Date().timeIntervalSince($0) / dwellSeconds) } ?? 0
        return Circle()
            .trim(from: 0, to: progress)
            .stroke(Color.white, style: StrokeStyle(lineWidth: 6, lineCap: .round))
            .frame(width: 50, height: 50)
            .rotationEffect(.degrees(-90))
            .position(x: box.cx * geo.size.width, y: (box.cy + box.h / 2 + 0.08) * geo.size.height)
    }

    private func tick() {
        guard !isShowingCameraPreview else { return }

        if let until = flashUntil, Date() >= until {
            flashIndex = nil
            flashUntil = nil
        }

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
        // resuming tracking immediately can slide the active box to a
        // neighbor in the SAME tick that confirms the blink, selecting a
        // digit the user was never looking at.
        if openFrameStreak > blinkSettleFrames {
            let (predX, predY) = model.predict(lookAtX: reading.lookAtX, lookAtY: reading.lookAtY)
            let alpha = gazeSmoothing.alpha
            smoothedX = alpha * predX + (1 - alpha) * smoothedX
            smoothedY = alpha * predY + (1 - alpha) * smoothedY

            let newActive = nearestBoxIndex(x: smoothedX, y: smoothedY)
            if newActive != activeIndex {
                activeIndex = newActive
                firstBlinkTime = nil
                armedIndex = nil
                dwellStartTime = Date()
            }
        }

        switch selectionMethod {
        case .singleBlink:
            if blinkEvent, let active = activeIndex { select(active) }

        case .doubleBlink:
            guard blinkEvent, let active = activeIndex else { return }
            if let armed = armedIndex, armed == active,
               let first = firstBlinkTime, Date().timeIntervalSince(first) <= doubleBlinkWindowSeconds {
                select(active)
            } else {
                armedIndex = active
                firstBlinkTime = Date()
            }

        case .dwell:
            if let active = activeIndex, let start = dwellStartTime, Date().timeIntervalSince(start) >= dwellSeconds {
                select(active)
            }
        }
    }

    private func nearestBoxIndex(x: Float, y: Float) -> Int? {
        guard !boxes.isEmpty else { return nil }
        var bestIndex = 0
        var bestDistance = Float.greatestFiniteMagnitude
        for (i, box) in boxes.enumerated() {
            let dx = x - Float(box.cx)
            let dy = y - Float(box.cy)
            let d = dx * dx + dy * dy
            if d < bestDistance {
                bestDistance = d
                bestIndex = i
            }
        }
        return bestIndex
    }

    private func select(_ index: Int) {
        outputDigits += boxes[index].label
        flashIndex = index
        flashUntil = Date().addingTimeInterval(flashDisplaySeconds)
        armedIndex = nil
        firstBlinkTime = nil
        dwellStartTime = Date()
    }
}
