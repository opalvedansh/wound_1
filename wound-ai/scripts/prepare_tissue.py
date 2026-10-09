"""Rewrite a tissue dataset's masks into this project's tissue classes (wound_ai.data.TISSUE_CLASSES).

Public tissue sets (WoundTissue, DFUTissue, your annotation tool's export) each code tissue differently: a grey
value or an RGB colour per class, with classes of their own. A map file says what each code means here:

    {"source": "woundtissue",
     "values": {"0": "background", "1": "granulation", "2": "slough", "3": "necrosis", "4": "ignore"},
     "unlisted": "ignore"}

Keys are grey values ("3") or RGB colours ("255,0,0"); a value is one of TISSUE_CLASSES or "ignore" (written as
255, never learned from: use it for classes we don't have, such as callus, and for anything an annotator marked
unsure). Check every code against the dataset's own documentation; configs/tissue_map.example.json is a template.

    python scripts/prepare_tissue.py --images data/raw/woundtissue/images --masks data/raw/woundtissue/masks \
        --map configs/woundtissue_map.json --out-dir data/tissue/woundtissue --csv data/tissue/woundtissue.csv
    python scripts/prepare_data.py --csv data/tissue/woundtissue.csv ... --out data/manifest.csv
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

from wound_ai.data import IGNORE_INDEX, TISSUE_CLASSES

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def class_index(name: str) -> int:
    if name == "ignore":
        return IGNORE_INDEX
    if name not in TISSUE_CLASSES:
        raise SystemExit(f"unknown class {name!r}: use one of {TISSUE_CLASSES} or 'ignore'")
    return TISSUE_CLASSES.index(name)


def remap(mask: np.ndarray, values: dict[str, str], unlisted: int) -> tuple[np.ndarray, set[str]]:
    """The mask in class indices, and the codes found that the map does not list."""
    out = np.full(mask.shape[:2], unlisted, np.uint8)
    if mask.ndim == 2:
        codes = {str(v) for v in np.unique(mask)}
        for code, name in values.items():
            out[mask == int(code)] = class_index(name)
    else:
        flat = mask.reshape(-1, 3)
        codes = {",".join(map(str, c)) for c in np.unique(flat, axis=0)}
        for code, name in values.items():
            rgb = np.array([int(x) for x in code.split(",")], np.uint8)
            out[(mask == rgb).all(-1)] = class_index(name)
    return out, codes - set(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out-dir", required=True, help="where the rewritten masks go")
    ap.add_argument("--csv", required=True, help="rows for prepare_data.py --csv")
    a = ap.parse_args()

    spec = json.loads(Path(a.map).read_text())
    values, unlisted = spec["values"], class_index(spec.get("unlisted", "ignore"))
    rgb = any("," in k for k in values)
    masks = {p.stem: p for p in Path(a.masks).rglob("*") if p.suffix.lower() in IMG_EXT}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, unknown, missing = [], set(), 0
    for img in sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in IMG_EXT):
        m = masks.get(img.stem)
        if m is None:
            missing += 1
            continue
        raw = cv2.imread(str(m), cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)
        if rgb:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        mask, extra = remap(raw, values, unlisted)
        unknown |= extra
        dst = out_dir / f"{img.stem}.png"
        cv2.imwrite(str(dst), mask)
        rows.append({"image_path": str(img), "tissue_path": str(dst), "source": spec.get("source", out_dir.name)})
    if not rows:
        sys.exit("no image/mask pairs found - check the paths")
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.csv, index=False)
    print(f"{len(rows)} masks rewritten to {out_dir} ({missing} images without a mask skipped) -> {a.csv}")
    if unknown:
        # Anti-aliased edges or JPEG masks produce stray codes; a long list means the map is wrong.
        print(f"WARNING: {len(unknown)} code(s) not in the map were set to {spec.get('unlisted', 'ignore')}: "
              f"{sorted(unknown)[:20]}")


if __name__ == "__main__":
    main()
