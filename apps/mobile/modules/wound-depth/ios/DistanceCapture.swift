import ARKit
import CoreImage
import ImageIO
import UniformTypeIdentifiers

/// How far the phone is from the surface at the centre of the camera's view, which way that surface faces, and
/// what measured it.
struct DistanceReading: Sendable {
  let distanceMm: Double
  /// "lidar": the depth sensor (iPhone Pro). "ar_raycast": ARKit's motion tracking, on phones without one.
  let source: String
  /// The surface the wound lies on, when enough of it was measured: lets the model correct for the phone's tilt.
  let surface: SurfacePlane?
}

struct SavedPhoto {
  let width: Int
  let height: Int
  /// The camera's field of view across the saved photo's width.
  let hfovDeg: Double
}

enum CaptureError: Error {
  case noFrame
  case encodingFailed
}

/// Measures the distance to the wound from an ARKit frame and saves the photo with that reading inside it.
/// No Expo imports here, so this file can be type-checked on its own against the iOS SDK.
enum DistanceCapture {
  /// The share of the view, around its centre, that the surface is measured over: the wound sits in the guide.
  static let centreShare = 0.4
  /// Where the surface is probed on phones without a depth sensor (the sensor image's 0 to 1 coordinates).
  static let probes: [CGPoint] = [
    CGPoint(x: 0.5, y: 0.5), CGPoint(x: 0.35, y: 0.5), CGPoint(x: 0.65, y: 0.5), CGPoint(x: 0.5, y: 0.35), CGPoint(x: 0.5, y: 0.65),
  ]

  static func reading(_ frame: ARFrame, session: ARSession) -> DistanceReading? {
    lidarReading(frame) ?? raycastReading(frame, session: session)
  }

  /// The depth sensor's points around the centre of the view, as a plane. The depth is the one ARKit has smoothed
  /// over recent frames where the phone offers it, which is steadier than a single frame's.
  static func lidarReading(_ frame: ARFrame) -> DistanceReading? {
    guard let depth = frame.smoothedSceneDepth ?? frame.sceneDepth else { return nil }
    let map = depth.depthMap
    guard CVPixelBufferGetPixelFormatType(map) == kCVPixelFormatType_DepthFloat32 else { return nil }
    CVPixelBufferLockBaseAddress(map, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(map, .readOnly) }
    let confidence = depth.confidenceMap
    if let confidence { CVPixelBufferLockBaseAddress(confidence, .readOnly) }
    defer { if let confidence { CVPixelBufferUnlockBaseAddress(confidence, .readOnly) } }
    guard let base = CVPixelBufferGetBaseAddress(map) else { return nil }

    let width = CVPixelBufferGetWidth(map), height = CVPixelBufferGetHeight(map)
    let rowBytes = CVPixelBufferGetBytesPerRow(map)
    let confidenceBase = confidence.flatMap { CVPixelBufferGetBaseAddress($0) }
    let confidenceRowBytes = confidence.map { CVPixelBufferGetBytesPerRow($0) } ?? 0
    // The depth map covers the same view as the camera image at a lower resolution, so the image's intrinsics
    // apply to it once scaled down.
    let scale = Double(width) / Double(CVPixelBufferGetWidth(frame.capturedImage))
    let k = frame.camera.intrinsics
    let fx = Double(k[0][0]) * scale, fy = Double(k[1][1]) * scale
    let cx = Double(k[2][0]) * scale, cy = Double(k[2][1]) * scale
    let halfW = max(2, Int(Double(width) * centreShare / 2)), halfH = max(2, Int(Double(height) * centreShare / 2))

    var points: [simd_double3] = []
    var depths: [Double] = []
    var looked = 0
    for y in stride(from: height / 2 - halfH, to: height / 2 + halfH, by: 2) {
      let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)
      let sure = confidenceBase?.advanced(by: y * confidenceRowBytes).assumingMemoryBound(to: UInt8.self)
      for x in stride(from: width / 2 - halfW, to: width / 2 + halfW, by: 2) {
        looked += 1
        if let sure, sure[x] < UInt8(ARConfidenceLevel.medium.rawValue) { continue }
        let metres = Double(row[x])
        guard metres.isFinite, metres > 0 else { continue }
        let mm = metres * 1000
        depths.append(mm)
        points.append(SurfaceFit.toPhotoAxes(simd_double3((Double(x) - cx) / fx * mm, (Double(y) - cy) / fy * mm, mm)))
      }
    }
    // Mostly unsure pixels (too close, a shiny wet surface): better no reading than a wrong one.
    guard depths.count * 2 >= looked else { return nil }
    if let surface = SurfaceFit.plane(through: points) {
      return DistanceReading(distanceMm: surface.distanceMm, source: "lidar", surface: surface)
    }
    depths.sort()
    return DistanceReading(distanceMm: depths[depths.count / 2], source: "lidar", surface: nil)
  }

  /// No depth sensor: where the view meets the surface ARKit has tracked, at the centre and four points around it.
  static func raycastReading(_ frame: ARFrame, session: ARSession) -> DistanceReading? {
    guard case .normal = frame.camera.trackingState else { return nil }
    let camera = frame.camera.transform
    let origin = simd_make_float3(camera.columns.3)
    // The camera's own axes, fixed to the sensor's landscape image: x to the image's right, y up, z back at the viewer.
    let right = simd_make_float3(camera.columns.0), up = simd_make_float3(camera.columns.1), back = simd_make_float3(camera.columns.2)

    var points: [simd_double3] = []
    var centreDepthMm: Double?
    for (index, probe) in probes.enumerated() {
      let query = frame.raycastQuery(from: probe, allowing: .estimatedPlane, alignment: .any)
      guard let hit = session.raycast(query).first else { continue }
      let toHit = simd_make_float3(hit.worldTransform.columns.3) - origin
      let depthMm = Double(-simd_dot(toHit, back)) * 1000
      guard depthMm > 0 else { continue }
      if index == 0 { centreDepthMm = depthMm }
      // In the sensor image's axes (x right, y down, z into the scene), then the photo's.
      points.append(SurfaceFit.toPhotoAxes(simd_double3(Double(simd_dot(toHit, right)) * 1000, Double(-simd_dot(toHit, up)) * 1000, depthMm)))
    }
    guard let centreDepthMm else { return nil }
    let surface = SurfaceFit.plane(through: points)
    return DistanceReading(distanceMm: surface?.distanceMm ?? centreDepthMm, source: "ar_raycast", surface: surface)
  }

  /// Saves the frame upright (the app is portrait; the sensor is landscape) as a JPEG. The reading goes in the
  /// EXIF user comment as {"woundScale": {...}}, which is where the model service looks for it, so it stays with
  /// the photo while it waits on the phone and when it is uploaded.
  static func writeJPEG(_ frame: ARFrame, reading: DistanceReading?, to url: URL) throws -> SavedPhoto {
    let sensorHeight = CVPixelBufferGetHeight(frame.capturedImage)
    let image = CIImage(cvPixelBuffer: frame.capturedImage).oriented(.right)
    guard let rendered = CIContext().createCGImage(image, from: image.extent) else { throw CaptureError.encodingFailed }
    // Turned upright, the photo's width is the sensor's height, so the view across it comes from fy.
    let fy = Double(frame.camera.intrinsics[1][1])
    let hfovDeg = 2 * atan(Double(sensorHeight) / (2 * fy)) * 180 / .pi

    var exif: [CFString: Any] = [:]
    if let reading {
      var scale: [String: Any] = [
        "distance_mm": (reading.distanceMm * 10).rounded() / 10,
        "hfov_deg": (hfovDeg * 100).rounded() / 100,
        "source": reading.source,
      ]
      if let surface = reading.surface {
        scale["normal"] = [surface.normal.x, surface.normal.y, surface.normal.z].map { ($0 * 10000).rounded() / 10000 }
        scale["surface_rms_mm"] = (surface.rmsMm * 10).rounded() / 10
      }
      let json = try JSONSerialization.data(withJSONObject: ["woundScale": scale], options: [.sortedKeys])
      exif[kCGImagePropertyExifUserComment] = String(decoding: json, as: UTF8.self)
    }
    let properties: [CFString: Any] = [
      kCGImageDestinationLossyCompressionQuality: 0.92,
      kCGImagePropertyOrientation: 1,
      kCGImagePropertyExifDictionary: exif,
    ]
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
      throw CaptureError.encodingFailed
    }
    CGImageDestinationAddImage(destination, rendered, properties as CFDictionary)
    guard CGImageDestinationFinalize(destination) else { throw CaptureError.encodingFailed }
    return SavedPhoto(width: rendered.width, height: rendered.height, hfovDeg: hfovDeg)
  }
}
