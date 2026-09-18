import SwiftUI

/// Binary-elimination word picker for building a sentence, modeled on
/// Google's "Look to Speak": rather than a flat grid of word targets (which
/// NumberPadView showed doesn't hold up at 6+ narrow zones with this
/// tracker), the full word list is repeatedly split into a left half and a
/// right half -- the same big two-zone left/right split YesNoView already
/// handles reliably -- until only one word remains in the chosen half. That
/// word is appended to the sentence and the full list resets for the next
/// word.
private let coreWordList: [String] = ["I", "want", "need", "help", "more", "stop", "yes", "no"]

private let blinkMinConsecutiveFrames = 3
private let doubleBlinkWindowSeconds: TimeInterval = 0.8
private let dwellSeconds: TimeInterval = 2.0
private let flashDisplaySeconds: TimeInterval = 0.4

private func splitHalves(_ words: [String]) -> (left: [String], right: [String]) {
    let mid = words.count / 2
    return (Array(words[..<mid]), Array(words[mid...]))
}

struct WordPickerView: View {
    @ObservedObject var tracker: GazeTracker
    let model: LinearCalibrationModel
    let selectionMethod: SelectionMethod
    let showGazeDot: Bool
    let gazeSmoothing: GazeSmoothing
    let onBack: () -> Void

    @State private var smoothedX: Float = 0.5
    @State private var smoothedY: Float = 0.5
    @State private var remainingWords: [String] = coreWordList
    @State private var sentence: [String] = []
    @State private var zone: Zone?
    @State private var flashZone: Zone?
    @State private var flashUntil: Date?

    @State private var eyesClosedRun = 0
    @State private var firstBlinkTime: Date?
    @State private var armedZone: Zone?
    @State private var dwellStartTime: Date?
    @State private var isShowingCameraPreview = false

    // See CalibrationView's tickTimer for why this must be @State, not an
    // inline Timer.publish(...) in `body`.
    @State private var tickTimer = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    private enum Zone { case left, right }

    var body: some View {
        CameraPreviewOverlay(tracker: tracker, isShowing: $isShowingCameraPreview) {
            content
        }
    }

    private var content: some View {
        GeometryReader { geo in
            let halves = splitHalves(remainingWords)
            ZStack {
                HStack(spacing: 0) {
                    zoneView(.left, words: halves.left, geo: geo)
                    zoneView(.right, words: halves.right, geo: geo)
                }
                .ignoresSafeArea()

                Text(selectionMethod.instructions)
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.85))
                    .position(x: geo.size.width / 2, y: geo.size.height * 0.06)

                if selectionMethod == .dwell, let z = zone {
                    dwellRing(for: z, geo: geo)
                }

                if showGazeDot {
                    Circle().stroke(Color.red, lineWidth: 3).frame(width: 20, height: 20)
                        .position(x: CGFloat(smoothedX) * geo.size.width, y: CGFloat(smoothedY) * geo.size.height)
                }

                Text("Sentence: \(sentence.isEmpty ? "-" : sentence.joined(separator: " "))")
                    .font(.system(size: 20, weight: .semibold, design: .monospaced))
                    .foregroundColor(.white)
                    .lineLimit(2)
                    .minimumScaleFactor(0.6)
                    .padding(.horizontal, 24)
                    .frame(width: geo.size.width * 0.8)
                    .multilineTextAlignment(.center)
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

    private func zoneView(_ z: Zone, words: [String], geo: GeometryProxy) -> some View {
        var fill = Color(white: 0.16)
        if flashZone == z {
            fill = Color.green.opacity(0.85)
        } else if selectionMethod == .doubleBlink, armedZone == z {
            fill = Color.blue.opacity(0.75)
        } else if zone == z {
            fill = Color(white: 0.28)
        }

        return ZStack {
            fill
            VStack(spacing: 10) {
                ForEach(words, id: \.self) { word in
                    Text(word).font(.system(size: 34, weight: .bold)).foregroundColor(.white)
                }
            }
        }
        .frame(width: geo.size.width / 2, height: geo.size.height)
    }

    private func dwellRing(for z: Zone, geo: GeometryProxy) -> some View {
        let progress = dwellStartTime.map { min(1.0, Date().timeIntervalSince($0) / dwellSeconds) } ?? 0
        let cx = z == .left ? geo.size.width * 0.25 : geo.size.width * 0.75
        return Circle()
            .trim(from: 0, to: progress)
            .stroke(Color.white, style: StrokeStyle(lineWidth: 6, lineCap: .round))
            .frame(width: 70, height: 70)
            .rotationEffect(.degrees(-90))
            .position(x: cx, y: geo.size.height * 0.72)
    }

    private func tick() {
        guard !isShowingCameraPreview else { return }

        if let until = flashUntil, Date() >= until {
            flashZone = nil
            flashUntil = nil
        }

        guard let reading = tracker.latest else { return }

        // lookAtPoint drifts while the eyes are physically closing/opening,
        // so only update the tracked position and zone while eyes are open
        // -- otherwise a blink can slide the zone to the wrong side right
        // as it's detected, picking the half the user never looked at.
        if !reading.blinking {
            let (predX, predY) = model.predict(lookAtX: reading.lookAtX, lookAtY: reading.lookAtY)
            let alpha = gazeSmoothing.alpha
            smoothedX = alpha * predX + (1 - alpha) * smoothedX
            smoothedY = alpha * predY + (1 - alpha) * smoothedY
            let newZone: Zone = smoothedX >= 0.5 ? .right : .left

            if newZone != zone {
                zone = newZone
                dwellStartTime = Date()
                firstBlinkTime = nil
                armedZone = nil
            }
        }

        var blinkEvent = false
        if reading.blinking {
            eyesClosedRun += 1
        } else {
            if eyesClosedRun >= blinkMinConsecutiveFrames { blinkEvent = true }
            eyesClosedRun = 0
        }

        switch selectionMethod {
        case .singleBlink:
            if blinkEvent, let z = zone { select(z) }

        case .doubleBlink:
            guard blinkEvent, let z = zone else { return }
            if let armed = armedZone, armed == z,
               let first = firstBlinkTime, Date().timeIntervalSince(first) <= doubleBlinkWindowSeconds {
                select(z)
            } else {
                armedZone = z
                firstBlinkTime = Date()
            }

        case .dwell:
            if let z = zone, let start = dwellStartTime, Date().timeIntervalSince(start) >= dwellSeconds {
                select(z)
            }
        }
    }

    private func select(_ z: Zone) {
        let halves = splitHalves(remainingWords)
        let chosen = z == .left ? halves.left : halves.right

        flashZone = z
        flashUntil = Date().addingTimeInterval(flashDisplaySeconds)
        armedZone = nil
        firstBlinkTime = nil
        dwellStartTime = Date()

        if chosen.count <= 1 {
            if let word = chosen.first { sentence.append(word) }
            remainingWords = coreWordList
        } else {
            remainingWords = chosen
        }
    }
}
