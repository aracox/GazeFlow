import ARKit
import AVFoundation
import Combine
import CoreImage

/// Wraps an ARSession running ARFaceTrackingConfiguration and publishes the
/// latest gaze/blink reading each frame. No rendering involved -- we only
/// want the face-tracking data, not an AR view.
///
/// Two things ARKit gives us for free that a0/ had to build from scratch on
/// the Mac (MediaPipe landmarks + hand-tuned EAR thresholds):
///   - `lookAtPoint`: a face-relative estimate of where the user is looking
///     (x = left/right, y = up/down, z = forward distance, in meters).
///     Used directly as the calibration model's input features.
///   - `blendShapes[.eyeBlinkLeft/.eyeBlinkRight]`: 0...1 blink coefficients,
///     already calibrated by Apple across users -- no per-session baseline
///     EAR collection needed like the Python EAR-threshold approach.
final class GazeTracker: NSObject, ObservableObject, ARSessionDelegate {
    struct Reading {
        let lookAtX: Float
        let lookAtY: Float
        let blinking: Bool
    }

    @Published private(set) var latest: Reading?
    @Published private(set) var isSupported: Bool = ARFaceTrackingConfiguration.isSupported
    @Published private(set) var trackingLost: Bool = true

    // Diagnostics, since there's no easy way to tail the device console
    // remotely -- these get shown directly in the UI instead.
    @Published private(set) var errorMessage: String?
    @Published private(set) var frameCallbackCount: Int = 0
    @Published private(set) var cameraAuthStatus: String = GazeTracker.describe(AVCaptureDevice.authorizationStatus(for: .video))
    @Published private(set) var lastFrameAnchorCount: Int = 0
    @Published private(set) var cameraTrackingState: String = "unknown"
    @Published private(set) var successfulReadingCount: Int = 0       // never reset -- proves detection ever worked
    @Published private(set) var faceAnchorSeenButUntrackedCount: Int = 0

    // What the front camera actually sees, for the swipe-in preview panel.
    // Only updated a few times a second (not every frame) -- CIContext
    // rendering isn't free, and the preview doesn't need to be 60fps.
    @Published private(set) var previewImage: CGImage?
    // The point ARKit is tracking (face position), reprojected into the
    // *native* captured-image pixel space -- for the "preflight" style dot
    // overlay on the full-screen camera view. previewImageSize is that same
    // native pixel space's dimensions, so the overlay can scale the point
    // correctly no matter what size the (downscaled) previewImage is shown at.
    @Published private(set) var facePreviewPoint: CGPoint?
    @Published private(set) var leftEyePreviewPoint: CGPoint?
    @Published private(set) var rightEyePreviewPoint: CGPoint?
    @Published private(set) var faceOutlinePreviewPoints: [CGPoint] = []
    @Published private(set) var previewImageSize: CGSize = .zero
    private let ciContext = CIContext()
    private var framesSincePreviewUpdate = 0
    private let previewUpdateEveryNFrames = 6

    private let session = ARSession()
    private let blinkThreshold: Float = 0.5

    func start() {
        guard isSupported else {
            errorMessage = "ARFaceTrackingConfiguration.isSupported == false on this device"
            return
        }

        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            runSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.cameraAuthStatus = Self.describe(AVCaptureDevice.authorizationStatus(for: .video))
                    if granted {
                        self.runSession()
                    } else {
                        self.errorMessage = "Camera permission denied by user in the request dialog"
                    }
                }
            }
        case .denied:
            errorMessage = "Camera permission previously denied -- enable in Settings > Privacy & Security > Camera > GazeFlowSample"
        case .restricted:
            errorMessage = "Camera access restricted (parental controls or MDM policy)"
        @unknown default:
            errorMessage = "Unrecognized camera authorization status"
        }
    }

    private func runSession() {
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = false
        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    private static func describe(_ status: AVAuthorizationStatus) -> String {
        switch status {
        case .authorized: return "authorized"
        case .denied: return "denied"
        case .restricted: return "restricted"
        case .notDetermined: return "notDetermined"
        @unknown default: return "unknown(\(status.rawValue))"
        }
    }

    func stop() {
        session.pause()
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        errorMessage = error.localizedDescription
    }

    func sessionWasInterrupted(_ session: ARSession) {
        errorMessage = "Session interrupted"
    }

    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        guard let faceAnchor = anchors.compactMap({ $0 as? ARFaceAnchor }).first else { return }
        update(from: faceAnchor)
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let faceAnchor = anchors.compactMap({ $0 as? ARFaceAnchor }).first else { return }
        update(from: faceAnchor)
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        frameCallbackCount += 1
        lastFrameAnchorCount = frame.anchors.count
        cameraTrackingState = Self.describe(frame.camera.trackingState)

        let faceAnchor = frame.anchors.compactMap({ $0 as? ARFaceAnchor }).first

        framesSincePreviewUpdate += 1
        if framesSincePreviewUpdate >= previewUpdateEveryNFrames {
            framesSincePreviewUpdate = 0
            updatePreviewImage(from: frame.capturedImage)
            if let faceAnchor {
                updateFacePreviewOverlay(faceAnchor: faceAnchor, frame: frame)
            } else {
                facePreviewPoint = nil
                leftEyePreviewPoint = nil
                rightEyePreviewPoint = nil
                faceOutlinePreviewPoints = []
            }
        }

        guard let faceAnchor else {
            trackingLost = true
            return
        }
        update(from: faceAnchor)
    }

    /// Reprojects the tracked face -- its overall position, each eye, and a
    /// sparse sampling of the face mesh -- from ARKit's 3D face-local space
    /// into 2D native-image pixel coordinates, for the "preflight" style dot
    /// overlay (same idea as the Mac/Python version's landmark overlay,
    /// though ARKit gives us a full 3D face mesh rather than 2D landmarks).
    private func updateFacePreviewOverlay(faceAnchor: ARFaceAnchor, frame: ARFrame) {
        let w = CVPixelBufferGetWidth(frame.capturedImage)
        let h = CVPixelBufferGetHeight(frame.capturedImage)
        let viewport = CGSize(width: w, height: h)
        previewImageSize = viewport

        // .landscapeRight: our best guess for "no rotation from the native
        // buffer", matching how updatePreviewImage() displays the captured
        // image un-rotated. May need adjusting once actually seen on device.
        func project(_ worldPosition: SIMD3<Float>) -> CGPoint {
            frame.camera.projectPoint(worldPosition, orientation: .landscapeRight, viewportSize: viewport)
        }

        let faceTransform = faceAnchor.transform
        let facePosition = faceTransform.columns.3
        facePreviewPoint = project(SIMD3<Float>(facePosition.x, facePosition.y, facePosition.z))

        let leftEyeWorld = (faceTransform * faceAnchor.leftEyeTransform).columns.3
        let rightEyeWorld = (faceTransform * faceAnchor.rightEyeTransform).columns.3
        leftEyePreviewPoint = project(SIMD3<Float>(leftEyeWorld.x, leftEyeWorld.y, leftEyeWorld.z))
        rightEyePreviewPoint = project(SIMD3<Float>(rightEyeWorld.x, rightEyeWorld.y, rightEyeWorld.z))

        let vertices = faceAnchor.geometry.vertices
        let sampleStride = max(1, vertices.count / 40)  // ~40 sparse outline dots
        faceOutlinePreviewPoints = Swift.stride(from: 0, to: vertices.count, by: sampleStride).map { i in
            let world4 = faceTransform * SIMD4<Float>(vertices[i], 1)
            return project(SIMD3<Float>(world4.x, world4.y, world4.z))
        }
    }

    private func updatePreviewImage(from pixelBuffer: CVPixelBuffer) {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        // Downscale -- this is a debug preview panel, not a photo. Rendering
        // the full-resolution captured image every update would be wasteful.
        let scale: CGFloat = 360.0 / ciImage.extent.width
        let small = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        if let cgImage = ciContext.createCGImage(small, from: small.extent) {
            previewImage = cgImage
        }
    }

    private static func describe(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "normal"
        case .notAvailable: return "notAvailable"
        case .limited(.excessiveMotion): return "limited(excessiveMotion)"
        case .limited(.insufficientFeatures): return "limited(insufficientFeatures)"
        case .limited(.initializing): return "limited(initializing)"
        case .limited(.relocalizing): return "limited(relocalizing)"
        case .limited: return "limited(other)"
        }
    }

    private func update(from faceAnchor: ARFaceAnchor) {
        trackingLost = !faceAnchor.isTracked
        guard faceAnchor.isTracked else {
            faceAnchorSeenButUntrackedCount += 1
            return
        }

        let lookAt = faceAnchor.lookAtPoint
        let leftBlink = faceAnchor.blendShapes[.eyeBlinkLeft]?.floatValue ?? 0
        let rightBlink = faceAnchor.blendShapes[.eyeBlinkRight]?.floatValue ?? 0
        let blinking = ((leftBlink + rightBlink) / 2) >= blinkThreshold

        latest = Reading(lookAtX: lookAt.x, lookAtY: lookAt.y, blinking: blinking)
        successfulReadingCount += 1
    }
}
