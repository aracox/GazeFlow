import SwiftUI

/// A full-screen "preflight" style debug view showing exactly what the
/// front camera sees, with a dot over the tracked face position -- same
/// idea as the Mac/Python version's `a0.main preflight` overlay. Swipe in
/// from the right edge to reveal it, swipe right (or tap) to dismiss.
///
/// The wrapped `content()`'s own selection/collection logic should pause
/// while this is showing (check the `isShowing` binding) -- looking at your
/// own camera feed isn't a moment where a stray blink should register as
/// an answer.
struct CameraPreviewOverlay<Content: View>: View {
    @ObservedObject var tracker: GazeTracker
    @Binding var isShowing: Bool
    @ViewBuilder let content: () -> Content

    @State private var dragTranslation: CGFloat = 0

    private let edgeTriggerWidth: CGFloat = 40

    var body: some View {
        GeometryReader { geo in
            let restingOffset: CGFloat = isShowing ? 0 : geo.size.width
            let offset = min(geo.size.width, max(0, restingOffset + dragTranslation))

            ZStack(alignment: .trailing) {
                content()

                fullScreenPanel(geo: geo)
                    .frame(width: geo.size.width, height: geo.size.height)
                    .offset(x: offset)
            }
            .gesture(dragGesture(screenWidth: geo.size.width))
        }
    }

    private func dragGesture(screenWidth: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 8, coordinateSpace: .local)
            .onChanged { value in
                let startedNearRightEdge = value.startLocation.x > screenWidth - edgeTriggerWidth
                guard isShowing || startedNearRightEdge else { return }
                dragTranslation = value.translation.width
            }
            .onEnded { value in
                let startedNearRightEdge = value.startLocation.x > screenWidth - edgeTriggerWidth
                guard isShowing || startedNearRightEdge else { return }
                withAnimation(.interactiveSpring()) {
                    if isShowing {
                        // Dragging right closes it.
                        isShowing = value.translation.width < screenWidth / 4
                    } else {
                        // Dragging left from the edge opens it.
                        isShowing = value.translation.width < -screenWidth / 5
                    }
                    dragTranslation = 0
                }
            }
    }

    private func fullScreenPanel(geo: GeometryProxy) -> some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let cgImage = tracker.previewImage {
                GeometryReader { imgGeo in
                    let displayedImageSize = aspectFitSize(
                        image: CGSize(width: cgImage.width, height: cgImage.height),
                        in: imgGeo.size
                    )
                    let imageOrigin = CGPoint(
                        x: (imgGeo.size.width - displayedImageSize.width) / 2,
                        y: (imgGeo.size.height - displayedImageSize.height) / 2
                    )

                    Image(decorative: cgImage, scale: 1, orientation: .up)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: imgGeo.size.width, height: imgGeo.size.height)

                    if tracker.previewImageSize.width > 0 {
                        let toView: (CGPoint) -> CGPoint = { native in
                            CGPoint(
                                x: imageOrigin.x + (native.x / tracker.previewImageSize.width) * displayedImageSize.width,
                                y: imageOrigin.y + (native.y / tracker.previewImageSize.height) * displayedImageSize.height
                            )
                        }

                        // Sparse face mesh outline, small green dots.
                        ForEach(Array(tracker.faceOutlinePreviewPoints.enumerated()), id: \.offset) { _, native in
                            let p = toView(native)
                            Circle().fill(Color.green).frame(width: 5, height: 5).position(x: p.x, y: p.y)
                        }

                        // Eyes, larger red dots.
                        if let native = tracker.leftEyePreviewPoint {
                            let p = toView(native)
                            eyeDot().position(x: p.x, y: p.y)
                        }
                        if let native = tracker.rightEyePreviewPoint {
                            let p = toView(native)
                            eyeDot().position(x: p.x, y: p.y)
                        }
                    }
                }
            } else {
                Text("No camera frame yet").foregroundColor(.gray)
            }

            VStack {
                HStack {
                    Text("What the camera sees").font(.headline).foregroundColor(.white)
                    Spacer()
                    Text(tracker.facePreviewPoint != nil ? "face: tracked" : "face: not detected")
                        .font(.caption)
                        .foregroundColor(tracker.facePreviewPoint != nil ? .green : .orange)
                }
                .padding()
                Spacer()
                Text("Swipe right to go back")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.7))
                    .padding(.bottom, 20)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { withAnimation(.interactiveSpring()) { isShowing = false } }
    }

    private func eyeDot() -> some View {
        Circle()
            .fill(Color.red)
            .frame(width: 16, height: 16)
            .overlay(Circle().stroke(Color.white, lineWidth: 2))
    }

    private func aspectFitSize(image: CGSize, in container: CGSize) -> CGSize {
        guard image.width > 0, image.height > 0 else { return container }
        let scale = min(container.width / image.width, container.height / image.height)
        return CGSize(width: image.width * scale, height: image.height * scale)
    }
}
