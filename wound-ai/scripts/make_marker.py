"""Make a printable sheet of calibration stickers (ArUco markers) for wound photos.

    python scripts/make_marker.py --mm 20 --count 12 --out markers.pdf

Print at 100% ("actual size", never "fit to page"), then check the 50 mm bar with
a ruler. Each sticker is the marker plus a neutral grey square beside it (cut them out
together): the app corrects the photo's colour cast from the grey before judging tissue. Laminate or print on matte sticker paper (gloss causes glare). Use a new
sticker per patient, or disinfect per local infection-control policy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image

from wound_ai.measure import ARUCO_DICT, GREY_GAP_MM

GREY = 128  # neutral mid grey, printed with black ink only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", type=float, default=20.0, help="marker side length in mm (black square)")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="markers.pdf")
    a = ap.parse_args()

    px_per_mm = a.dpi / 25.4
    side = int(round(a.mm * px_per_mm))
    gap = int(round(12 * px_per_mm))
    page_w, page_h = int(210 * px_per_mm), int(297 * px_per_mm)  # A4
    page = np.full((page_h, page_w), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))

    patch_gap = int(round(GREY_GAP_MM * px_per_mm))
    unit = 2 * side + patch_gap  # marker + grey patch
    cols = max(1, (page_w - gap) // (unit + gap))
    y0, x0 = int(25 * px_per_mm), gap
    for i in range(a.count):
        r, c = divmod(i, cols)
        y, x = y0 + r * (side + gap), x0 + c * (unit + gap)
        if y + side > page_h - int(40 * px_per_mm):
            break
        page[y:y + side, x:x + side] = cv2.aruco.generateImageMarker(dictionary, i, side, borderBits=1)
        page[y:y + side, x + side + patch_gap:x + unit] = GREY
        cv2.putText(page, f"id {i}", (x, y + side + int(4 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)

    bar_y = page_h - int(25 * px_per_mm)
    bar_len = int(round(50 * px_per_mm))
    cv2.rectangle(page, (gap, bar_y), (gap + bar_len, bar_y + int(2 * px_per_mm)), 0, -1)
    cv2.putText(page, "50 mm - check with a ruler. Print at 100% / actual size.",
                (gap, bar_y - int(3 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.putText(page, f"Wound calibration markers, {a.mm:g} mm, {ARUCO_DICT}",
                (gap, int(12 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)

    out = Path(a.out)
    img = Image.fromarray(page)
    if out.suffix.lower() == ".pdf":
        img.save(out, "PDF", resolution=a.dpi)
    else:
        img.save(out, dpi=(a.dpi, a.dpi))
    print(f"wrote {out} ({a.mm:g} mm markers = {side}px at {a.dpi} dpi)")


if __name__ == "__main__":
    main()
