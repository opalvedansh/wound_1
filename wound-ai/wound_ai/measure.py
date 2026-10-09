"""Real-world wound measurement from a printed ArUco calibration sticker.

The sticker (see scripts/make_marker.py) is a square of known side length placed
next to the wound, on the same skin plane. Its four corners give a homography
from image pixels to millimetres on that plane, which also corrects for the phone
being held at an angle.

Limits you must state in the report and in validation:
* Accurate only where the wound lies roughly in the marker's plane. Heels, toes
  and other curved sites can be under-estimated; validate against manual tracing.
* Area only, no depth. Depth needs a probe or a depth sensor.
* If the marker is not detected, NO measurement is reported. Never guess a scale.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

ARUCO_DICT = "DICT_4X4_50"


@dataclass
class Calibration:
    homography: np.ndarray      # 3x3, image px -> mm on the marker plane
    marker_px_side: float       # mean marker side length in pixels (for sanity checks)
    marker_id: int


@dataclass
class WoundMeasurement:
    area_cm2: float
    length_cm: float            # longest dimension
    width_cm: float             # dimension perpendicular to length
    perimeter_cm: float
    n_regions: int

    def to_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def find_marker(img_rgb: np.ndarray, marker_mm: float = 20.0, marker_id: int | None = None) -> Calibration | None:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corners matter: integer corners on a ~100 px marker gave >20% area error under tilt in testing.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    ids = ids.flatten().tolist()
    k = ids.index(marker_id) if marker_id is not None and marker_id in ids else 0
    c = corners[k].reshape(4, 2).astype(np.float32)  # TL, TR, BR, BL (clockwise)
    dst = np.array([[0, 0], [marker_mm, 0], [marker_mm, marker_mm], [0, marker_mm]], np.float32)
    H = cv2.getPerspectiveTransform(c, dst)
    side_px = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
    return Calibration(homography=H, marker_px_side=side_px, marker_id=int(ids[k]))


def region_area_mm2(region: np.ndarray, H: np.ndarray) -> float:
    """Area in mm² that the region's pixels cover on the marker plane.

    For a homography the local area scale is |det(H)| / w(x, y)^3, with
    w = h20*x + h21*y + h22, so summing it over the region's pixels gives the exact
    projected area (a contour through pixel centres would lose half a pixel per edge).
    """
    Hn = H / H[2, 2]
    ys, xs = np.nonzero(region)
    w = Hn[2, 0] * xs + Hn[2, 1] * ys + Hn[2, 2]
    return float((np.abs(np.linalg.det(Hn)) / np.abs(w) ** 3).sum())


def measure_wound(mask: np.ndarray, calib: Calibration, min_area_mm2: float = 4.0) -> WoundMeasurement | None:
    """Area, length, width and perimeter of every wound region above min_area_mm2."""
    m = (mask > 0).astype(np.uint8)
    n_lab, labels = cv2.connectedComponents(m)
    total_area = total_perim = 0.0
    pts_all = []
    n = 0
    for lab in range(1, n_lab):
        region = (labels == lab).astype(np.uint8)
        area = region_area_mm2(region, calib.homography)
        if area < min_area_mm2:
            continue
        cnt = max(cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)
        mm = cv2.perspectiveTransform(cnt.astype(np.float32).reshape(-1, 1, 2), calib.homography)
        n += 1
        total_area += area
        total_perim += cv2.arcLength(mm, True)
        pts_all.append(mm.reshape(-1, 2))
    if n == 0:
        return None
    (_, _), (a, b), _ = cv2.minAreaRect(np.concatenate(pts_all).astype(np.float32))
    return WoundMeasurement(
        area_cm2=total_area / 100.0,
        length_cm=max(a, b) / 10.0,
        width_cm=min(a, b) / 10.0,
        perimeter_cm=total_perim / 10.0,
        n_regions=n,
    )


def area_change(current_cm2: float, previous_cm2: float, days: float | None = None) -> dict:
    """Percent area reduction since the last photo (positive = healing)."""
    if previous_cm2 <= 0:
        return {}
    pct = 100.0 * (previous_cm2 - current_cm2) / previous_cm2
    out = {"previous_area_cm2": round(previous_cm2, 2), "percent_area_reduction": round(pct, 1)}
    if days:
        out["days_between"] = days
    return out


# The printed sheet (scripts/make_marker.py) puts a neutral grey square of the marker's size this far to the
# right of each marker, on the same sticker.
GREY_GAP_MM = 4.0  # about one marker module of white, so detection still sees the marker's edge


def _plane_mean(img_rgb: np.ndarray, calib: Calibration, x0: float, x1: float, y0: float, y1: float):
    """Mean and spread of the pixels covering a rectangle (mm, marker plane); None if it leaves the photo."""
    corners_mm = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(corners_mm, np.linalg.inv(calib.homography)).reshape(-1, 2)
    h, w = img_rgb.shape[:2]
    if (corners < 0).any() or (corners[:, 0] >= w).any() or (corners[:, 1] >= h).any():
        return None
    region = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(region, corners.round().astype(np.int32), 1)
    px = img_rgb[region > 0].astype(np.float32)
    return (px.mean(0), px.std(0)) if len(px) >= 20 else None


def grey_patch_gains(img_rgb: np.ndarray, calib: Calibration, marker_mm: float = 20.0) -> list[float] | None:
    """Per-channel gains that make the sticker's grey patch neutral, i.e. undo the lighting's colour cast.

    Tissue is judged by colour (red granulation, yellow slough, black necrosis), so a warm lamp or a phone's white
    balance shifts the tissue mix. The white gap between marker and patch is lit by the same light, so a real patch
    is about half as bright as the gap in every channel whatever the cast; skin or bare paper is not. Returns None
    when the patch is missing (older sticker sheets), out of the photo, uneven, or too dark or bright to trust.
    """
    inset = 0.2 * marker_mm
    patch = _plane_mean(img_rgb, calib, marker_mm + GREY_GAP_MM + inset, 2 * marker_mm + GREY_GAP_MM - inset,
                        inset, marker_mm - inset)
    gap = _plane_mean(img_rgb, calib, marker_mm + 0.3 * GREY_GAP_MM, marker_mm + 0.7 * GREY_GAP_MM,
                      inset, marker_mm - inset)
    if patch is None or gap is None:
        return None
    (mean, std), (white, _) = patch, gap
    ratio = mean / np.maximum(white, 1)
    if std.max() > 12 or not 40 <= mean.mean() <= 220 or not ((0.3 <= ratio) & (ratio <= 0.7)).all() \
            or ratio.max() / ratio.min() > 1.3:  # not 1.0: bright paper clips in the strongest channel
        return None
    return [round(float(np.clip(mean.mean() / m, 0.6, 1.6)), 3) for m in mean]
