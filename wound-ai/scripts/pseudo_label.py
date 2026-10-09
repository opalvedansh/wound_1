"""Pseudo-labels: a trained tissue model (the teacher) labels unlabelled wound photos where it is confident, so a
second model (the student) can learn from thousands of photos instead of a couple of hundred.

For each photo: the outline model finds the wound (no wound: skipped), the teacher runs on the same crop the app
uses, and each pixel keeps the teacher's class only if its confidence is at least --threshold; everything else is
255 (not learned from). Photos where under --min-coverage of the wound is confident are skipped. Classes listed in
--drop (too few human labels for the teacher to be trusted on them) are never pseudo-labelled.

    python scripts/pseudo_label.py --teacher runs/tissue_cv/fold0/teacher/tissue.pt --boundary checkpoints/boundary.pt \
        --unlabeled data/tissue_unlabeled.csv --out runs/tissue_cv/fold0/pseudo

Writes OUT/masks/*.png, OUT/outlines/*.png and OUT/pseudo.csv (rows for train_seg.py --extra-train).
Pseudo-labels are only ever training data: never validated or tested on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from wound_ai.data import IGNORE_INDEX, TISSUE_CLASSES, read_rgb
from wound_ai.pipeline import WoundAnalyzer


def pseudo_label(an: WoundAnalyzer, img: np.ndarray, threshold: float, min_coverage: float, drop: set[int]):
    """(label mask, outline mask, coverage) or None if the photo is skipped."""
    outline = an._segment(img, "boundary")
    if not outline.any():
        return None
    tissue, conf = an._tissue_map(img, outline)
    label = np.where(conf >= threshold, tissue, IGNORE_INDEX).astype(np.uint8)
    for k in drop:
        label[label == k] = IGNORE_INDEX
    bed = outline > 0
    coverage = float((label[bed] != IGNORE_INDEX).mean())
    if coverage < min_coverage:
        return None
    return label, outline, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="tissue.pt")
    ap.add_argument("--boundary", required=True, help="boundary.pt (finds the wound and the crop)")
    ap.add_argument("--unlabeled", required=True, help="CSV with image_path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--min-coverage", type=float, default=0.3)
    ap.add_argument("--drop", default="", help="comma-separated classes never pseudo-labelled")
    ap.add_argument("--max", type=int, default=0, help="debug: label at most this many photos")
    a = ap.parse_args()

    an = WoundAnalyzer(Path(a.out) / "_none")  # no checkpoints from a folder: the two models are loaded below
    an.seg["boundary"] = an._load_seg(Path(a.boundary))
    an.seg["tissue"] = an._load_seg(Path(a.teacher))
    known = an.tissue_classes()
    drop = {known.index(c) for c in a.drop.split(",") if c and c in known}
    labelled = ";".join(c for i, c in enumerate(known) if i not in drop)
    out = Path(a.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "outlines").mkdir(parents=True, exist_ok=True)

    paths = pd.read_csv(a.unlabeled).image_path.astype(str).tolist()
    if a.max:
        paths = paths[:a.max]
    rows, skipped, pixels = [], 0, np.zeros(len(TISSUE_CLASSES), np.int64)
    for i, p in enumerate(paths):
        try:
            img = read_rgb(p)
        except FileNotFoundError:
            skipped += 1
            continue
        res = pseudo_label(an, img, a.threshold, a.min_coverage, drop)
        if res is None:
            skipped += 1
            continue
        label, outline, coverage = res
        name = f"{i:05d}_{Path(p).stem}.png"
        cv2.imwrite(str(out / "masks" / name), label)
        cv2.imwrite(str(out / "outlines" / name), outline * 255)
        pixels += np.bincount(label[label != IGNORE_INDEX].ravel(), minlength=len(TISSUE_CLASSES))[:len(TISSUE_CLASSES)]
        rows.append({"image_path": p, "tissue_path": str(out / "masks" / name), "mask_path": str(out / "outlines" / name),
                     "source": "pseudo", "split": "train", "patient_id": f"pseudo_{i}", "tissue_classes": labelled,
                     "coverage": round(coverage, 3)})
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(paths)} photos: {len(rows)} pseudo-labelled, {skipped} skipped", flush=True)
    cols = ["image_path", "tissue_path", "mask_path", "source", "split", "patient_id", "tissue_classes", "coverage"]
    pd.DataFrame(rows, columns=cols).to_csv(out / "pseudo.csv", index=False)  # header even if nothing qualified
    share = {c: round(float(pixels[k] / max(1, pixels.sum())), 3) for k, c in enumerate(TISSUE_CLASSES) if pixels[k]}
    stats = {"photos": len(paths), "pseudo_labelled": len(rows), "skipped": skipped, "threshold": a.threshold,
             "dropped_classes": sorted(TISSUE_CLASSES[k] for k in drop), "pixel_share": share}
    (out / "pseudo_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
