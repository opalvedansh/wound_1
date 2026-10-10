import Foundation
import simd

/// A flat surface fitted to what the phone measured around the centre of its view. Axes are the saved photo's:
/// x right, y down, z into the scene. Lengths are millimetres.
struct SurfacePlane: Sendable {
  let normal: simd_double3
  /// Depth of the surface along the photo's centre ray.
  let distanceMm: Double
  /// How far the measured points lie from the plane (root mean square): large on curved skin.
  let rmsMm: Double

  /// How far the surface faces away from the camera.
  var tiltDeg: Double { acos(min(1, abs(normal.z))) * 180 / .pi }
}

/// Plain geometry with no camera imports, so it can be compiled and run on a Mac to check it.
enum SurfaceFit {
  /// The sensor's image is landscape and the saved photo is that image turned a quarter-turn clockwise, so a
  /// direction (x right, y down) in the sensor's image is (-y, x) in the photo.
  static func toPhotoAxes(_ v: simd_double3) -> simd_double3 {
    simd_double3(-v.y, v.x, v.z)
  }

  /// The plane through the points, refitted once without the points furthest from it: a wound's crater or the edge
  /// of a limb should not tip the plane of the skin around it.
  static func plane(through points: [simd_double3]) -> SurfacePlane? {
    guard let first = leastSquares(points) else { return nil }
    let kept = points.count >= 8 ? points.filter { abs(first.offset($0)) <= max(2 * first.rms, 0.5) } : points
    guard kept.count < points.count, kept.count * 2 >= points.count, let second = leastSquares(kept) else {
      return first.plane
    }
    // Unevenness is judged on every point, so curved skin is not hidden by leaving its far parts out of the fit.
    let rms = (points.reduce(0.0) { $0 + second.offset($1) * second.offset($1) } / Double(points.count)).squareRoot()
    return SurfacePlane(normal: second.plane.normal, distanceMm: second.plane.distanceMm, rmsMm: rms)
  }

  private struct Fit {
    let a: Double, b: Double, c: Double  // z = a*x + b*y + c
    let rms: Double

    /// Distance of a point from the plane, along the plane's normal.
    func offset(_ p: simd_double3) -> Double {
      (p.z - (a * p.x + b * p.y + c)) / (a * a + b * b + 1).squareRoot()
    }

    var plane: SurfacePlane {
      // The plane meets the centre ray (x = y = 0) at depth c.
      SurfacePlane(normal: simd_normalize(simd_double3(a, b, -1)), distanceMm: c, rmsMm: rms)
    }
  }

  /// Least squares for z = a*x + b*y + c. The surface faces the camera, so z is the direction to solve for.
  private static func leastSquares(_ points: [simd_double3]) -> Fit? {
    guard points.count >= 3 else { return nil }
    let mean = points.reduce(simd_double3(), +) / Double(points.count)
    var sxx = 0.0, sxy = 0.0, syy = 0.0, sxz = 0.0, syz = 0.0
    for p in points {
      let d = p - mean
      sxx += d.x * d.x
      sxy += d.x * d.y
      syy += d.y * d.y
      sxz += d.x * d.z
      syz += d.y * d.z
    }
    let det = sxx * syy - sxy * sxy
    // Points in a line (or one spot) do not fix a plane.
    guard det > 1e-9 * max(sxx * syy, 1e-9) else { return nil }
    let a = (sxz * syy - syz * sxy) / det
    let b = (syz * sxx - sxz * sxy) / det
    let c = mean.z - a * mean.x - b * mean.y
    let unscored = Fit(a: a, b: b, c: c, rms: 0)
    let rms = (points.reduce(0.0) { $0 + unscored.offset($1) * unscored.offset($1) } / Double(points.count)).squareRoot()
    return c > 0 ? Fit(a: a, b: b, c: c, rms: rms) : nil
  }
}
