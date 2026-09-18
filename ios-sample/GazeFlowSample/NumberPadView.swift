import SwiftUI
import Combine

/// Single-stage digit picker for a small test range (0-2): one row of boxes
/// spanning the full width, so only left/right gaze position matters. Each
/// selection appends to the output and the view stays put -- it does not
/// reset or navigate back to the start menu, so you can keep picking digits
/// in a row.
private let flashDisplaySeconds: TimeInterval = 0.4

private let digits: [String] = ["0", "1", "2"]

private struct NumberBox {
    let label: String
    let cx: CGFloat
    let cy: CGFloat
    let w: CGFloat
    let h: CGFloat
}

private func digitBoxes() -> [NumberBox] {
    let n = digits.count
    let boxW = 1 / CGFloat(n)
    return digits.enumerated().map { i, d in
        let x = (CGFloat(i) + 0.5) * boxW
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

    @State private var selection = NumberSelectionState(targets: numberPadTargetXPositions().map { Float($0) })
    @State private var boxes: [NumberBox] = digitBoxes()
    @State private var flashIndex: Int?
    @State private var flashUntil: Date?
    @State private var outputDigits: String = ""

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

                if selectionMethod == .dwell, let active = selection.activeIndex {
                    dwellRing(for: boxes[active], geo: geo)
                }

                if showGazeDot {
                    Circle().stroke(Color.red, lineWidth: 3).frame(width: 20, height: 20)
                        .position(x: CGFloat(min(1, max(0, selection.smoothedX))) * geo.size.width, y: CGFloat(min(1, max(0, selection.smoothedY))) * geo.size.height)
                }

                Text("Pick a digit: \(selectionMethod.instructions)")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.9))
                    .position(x: geo.size.width / 2, y: geo.size.height * 0.06)

                Text("Output: \(outputDigits.isEmpty ? "-" : outputDigits)")
                    .font(.system(size: 22, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                    .position(x: geo.size.width / 2, y: geo.size.height * 0.87)

                Text(String(format: "x: %.3f, y: %.3f", selection.smoothedX, selection.smoothedY))
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
        } else if selectionMethod == .doubleBlink, selection.armedIndex == index {
            fill = Color.blue.opacity(0.85)
        } else if selection.activeIndex == index {
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
        let progress = selection.dwellProgress
        return Circle()
            .trim(from: 0, to: progress)
            .stroke(Color.white, style: StrokeStyle(lineWidth: 6, lineCap: .round))
            .frame(width: 50, height: 50)
            .rotationEffect(.degrees(-90))
            .position(x: box.cx * geo.size.width, y: (box.cy + box.h / 2 + 0.08) * geo.size.height)
    }

    private func tick() {
        if let until = flashUntil, Date() >= until {
            flashIndex = nil
            flashUntil = nil
        }
        let reading = tracker.latest
        let position = reading.map {
            model.predict(lookAtX: $0.lookAtX, lookAtY: $0.lookAtY, clampToScreen: false)
        }
        if let index = selection.update(reading: reading, position: position,
                                        now: ProcessInfo.processInfo.systemUptime,
                                        enabled: !isShowingCameraPreview && !tracker.trackingLost,
                                        method: selectionMethod, smoothingAlpha: gazeSmoothing.alpha) {
            outputDigits += boxes[index].label
            flashIndex = index
            flashUntil = Date().addingTimeInterval(flashDisplaySeconds)
        }
    }
}
