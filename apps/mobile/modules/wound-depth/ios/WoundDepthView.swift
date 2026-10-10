import ARKit
import ExpoModulesCore

/// ARSession may be used from any thread, but is not marked Sendable; this lets the view share it with ARKit's
/// delegate queue and the capture call.
private final class SessionBox: @unchecked Sendable {
  let session = ARSession()
}

/// The camera preview for wound photos. ARKit runs the camera, so every photo comes with the distance to the wound.
final class WoundDepthView: ExpoView, ARSessionDelegate {
  private let sceneView = ARSCNView()
  private let box = SessionBox()
  // Read and written only on ARKit's delegate queue.
  private nonisolated(unsafe) var lastReported: TimeInterval = 0
  let onDistance = EventDispatcher()

  required init(appContext: AppContext? = nil) {
    super.init(appContext: appContext)
    clipsToBounds = true
    box.session.delegate = self
    sceneView.session = box.session
    sceneView.automaticallyUpdatesLighting = false
    addSubview(sceneView)
  }

  override func layoutSubviews() {
    super.layoutSubviews()
    sceneView.frame = bounds
  }

  // The camera runs only while the view is on screen.
  override func didMoveToWindow() {
    super.didMoveToWindow()
    if window == nil {
      box.session.pause()
      return
    }
    let configuration = ARWorldTrackingConfiguration()
    for depth in [ARConfiguration.FrameSemantics.sceneDepth, .smoothedSceneDepth]
    where ARWorldTrackingConfiguration.supportsFrameSemantics(depth) {
      configuration.frameSemantics.insert(depth)
    }
    // Phones without a depth sensor measure against the surfaces ARKit finds as the phone moves.
    configuration.planeDetection = [.horizontal, .vertical]
    box.session.run(configuration)
  }

  /// Tells the screen how far the wound is a few times a second, so the user sees the reading settle before capturing.
  nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
    guard frame.timestamp - lastReported > 0.25 else { return }
    lastReported = frame.timestamp
    let reading = DistanceCapture.reading(frame, session: session)
    Task { @MainActor [weak self] in
      self?.onDistance(Self.payload(reading))
    }
  }

  nonisolated func capture() throws -> [String: Any] {
    let session = box.session
    guard let frame = session.currentFrame else { throw CaptureError.noFrame }
    let reading = DistanceCapture.reading(frame, session: session)
    let folder = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0].appendingPathComponent("WoundCamera", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let url = folder.appendingPathComponent("\(UUID().uuidString).jpg")
    let saved = try DistanceCapture.writeJPEG(frame, reading: reading, to: url)
    var result = Self.payload(reading)
    result["uri"] = url.absoluteString
    result["width"] = saved.width
    result["height"] = saved.height
    result["hfovDeg"] = saved.hfovDeg
    return result
  }

  /// Empty while there is no reading; tiltDeg is there once the surface around the wound has been measured.
  private nonisolated static func payload(_ reading: DistanceReading?) -> [String: Any] {
    guard let reading else { return [:] }
    var result: [String: Any] = ["distanceMm": reading.distanceMm, "source": reading.source]
    if let surface = reading.surface { result["tiltDeg"] = surface.tiltDeg }
    return result
  }
}
