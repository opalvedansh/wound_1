"""Real-world wound measurement from a printed ArUco calibration sticker.

The sticker (see scripts/make_marker.py) is a square of known side length placed
next to the wound, on the same skin plane. Its four corners give a homography
from image pixels to millimetres on that plane, which also corrects for the phone
being held at an angle.

Limits you must state in the report and in validation:
* Accurate only where the wound lies roughly in the marker's plane. Heels, toes
  and other curved sites can be under-estimated; validate against manual tracing.
* Area only, no depth. Depth needs a probe or a depth sensor.
* Without the marker, a size is reported only from something that was measured: the distance the phone read when
  it took the photo (distance_calibration), or the wound's length the clinician measured with a ruler
  (length_calibration). Never guess a scale: a photo with none of these gets NO measurement.
"""
from __future__ import annotations

import io
import json
import math
import re
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

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


# What a phone can plausibly report for a wound photo: outside these the reading is refused, not used.
DISTANCE_MM = (80.0, 1000.0)
HFOV_DEG = (20.0, 120.0)
LENGTH_CM = (0.2, 60.0)  # a wound length a clinician could have measured with a ruler
MAX_TILT_DEG = 60.0  # a surface seen more edge-on than this is too foreshortened to outline or measure


def surface_tilt_deg(normal) -> float | None:
    """How far the surface faces away from the camera, from its normal (photo axes: x right, y down, z into the
    scene). None when the normal is not three finite numbers with a length."""
    try:
        n = np.array(normal, dtype=float)
    except (TypeError, ValueError):
        return None
    if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-9:
        return None
    return math.degrees(math.acos(min(1.0, abs(n[2]) / np.linalg.norm(n))))


def valid_reading(reading) -> bool:
    """A phone's distance reading that could be real. Anything else is refused rather than measured with."""
    def within(key: str, lo: float, hi: float) -> bool:
        v = reading.get(key)
        return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi

    if not (isinstance(reading, dict) and within("distance_mm", *DISTANCE_MM) and within("hfov_deg", *HFOV_DEG)):
        return False
    if reading.get("normal") is None:
        return True
    tilt = surface_tilt_deg(reading["normal"])
    return tilt is not None and tilt <= MAX_TILT_DEG


def phone_reading(photo: bytes) -> dict | None:
    """The distance reading the phone app wrote into the photo it took: JSON under "woundScale" in the EXIF user
    comment, so the reading travels with its photo through offline storage and upload.

    Only the app's own reading counts. A camera's generic EXIF subject distance is a focus estimate, not a
    measurement, and is never used.
    """
    try:
        comment = Image.open(io.BytesIO(photo)).getexif().get_ifd(0x8769).get(0x9286)
        text = comment.decode("utf-8", "ignore") if isinstance(comment, bytes) else str(comment or "")
        text = text.replace("\0", "")  # the comment's character-set header, and UTF-16 padding if a writer used it
        found = re.search(r"\{.*\}", text, re.S)
        reading = json.loads(found.group(0)).get("woundScale") if found else None
    except Exception:  # no EXIF, not JSON, a truncated file: the photo simply carries no reading
        return None
    return reading if valid_reading(reading) else None


def distance_calibration(distance_mm: float, hfov_deg: float, image_size: tuple[int, int],
                         normal: tuple[float, float, float] | None = None) -> Calibration:
    """The scale of a photo with no sticker, from what the phone measured when it took the photo (its depth sensor
    or AR tracking): the distance to the surface at the photo's centre, the camera's field of view across the
    photo's width, and the direction the surface faces (`normal`, in photo axes: x right, y down, z into the scene).

    The wound is taken to lie on that flat surface, so each pixel's ray is followed to the plane and the result is a
    homography from pixels to millimetres on it, as the sticker gives. That corrects for the phone being tilted.
    Without `normal` the surface is taken to face the camera squarely, which is right only if it did.
    """
    w, h = image_size
    focal = w / (2 * math.tan(math.radians(hfov_deg) / 2))
    n = np.array(normal if normal is not None else (0.0, 0.0, 1.0), dtype=float)
    n /= np.linalg.norm(n)
    centre = np.array([0.0, 0.0, distance_mm])
    # Millimetre axes on the surface: e1 along the photo's x as far as the surface allows, e2 across it.
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    px = np.array([[-100, -100], [100, -100], [100, 100], [-100, 100]], np.float32)
    on_surface = []
    for u, v in px:
        ray = np.array([u / focal, v / focal, 1.0])
        point = ray * (n @ centre) / (n @ ray)
        on_surface.append(((point - centre) @ e1, (point - centre) @ e2))
    H = cv2.getPerspectiveTransform(px + np.float32([w / 2, h / 2]), np.array(on_surface, np.float32))
    return Calibration(homography=H.astype(np.float64), marker_px_side=0.0, marker_id=-1)


def length_calibration(mask: np.ndarray, length_cm) -> Calibration | None:
    """The scale of a photo with no sticker and no phone reading, from the wound's longest length as the clinician
    measured it with a ruler: the largest outlined region is taken to be that long, which fixes how much each pixel
    covers. So the area and width are only as good as that one measurement and the outline, and, as with any
    single scale, the phone is taken to have been held square to the wound.
    """
    if isinstance(length_cm, bool) or not isinstance(length_cm, (int, float)) \
            or not LENGTH_CM[0] <= length_cm <= LENGTH_CM[1]:
        return None
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if n_lab < 2:
        return None
    largest = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnt = max(cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)
    (_, _), (a, b), _ = cv2.minAreaRect(cnt)  # the same longest dimension measure_wound reports
    if max(a, b) < 2:
        return None
    mm_per_px = 10.0 * length_cm / max(a, b)
    return Calibration(homography=np.diag([mm_per_px, mm_per_px, 1.0]), marker_px_side=0.0, marker_id=-1)


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
