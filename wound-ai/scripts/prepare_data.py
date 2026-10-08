"""Build one manifest CSV from many datasets, with leakage-safe splits.

Sources it understands:
  --pairs   NAME:IMAGES_DIR:MASKS_DIR             image + mask folders matched by file stem
                                                  (FUSeg, DFUC2022, Mendeley lower-limb set, your own exports)
  --classes NAME:ROOT:TARGET[:MAP.json]           ROOT/<class folder>/*.jpg  (AZH wound types, PIID stages, ...)
                                                  MAP.json renames folders to your label set, e.g. {"D": "diabetic"}
  --csv     FILE                                  rows already in manifest format (your clinical data export)

Public datasets rarely give patient IDs, and the same ulcer is often photographed
several times. Near-duplicate photos are therefore grouped (perceptual hash) into
pseudo-patients before splitting, so copies of one wound never straddle train/test.

    python scripts/prepare_data.py \
        --pairs fuseg:data/raw/fuseg/train/images:data/raw/fuseg/train/labels \
        --classes azh:data/raw/azh:wound_type:configs/azh_map.json \
        --csv data/clinic/annotations.csv --out data/manifest.csv
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

from wound_ai.data import patient_level_split

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def images_in(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def from_pairs(name: str, img_dir: str, mask_dir: str) -> list[dict]:
    masks = {p.stem: p for p in images_in(Path(mask_dir))}
    rows, missing = [], 0
    for p in images_in(Path(img_dir)):
        m = masks.get(p.stem)
        if m is None:
            missing += 1
            continue
        rows.append({"image_path": str(p), "mask_path": str(m), "source": name})
    print(f"[{name}] {len(rows)} image/mask pairs ({missing} images without a mask skipped)")
    return rows


def from_class_folders(name: str, root: str, target: str, map_file: str | None) -> list[dict]:
    mapping = json.loads(Path(map_file).read_text()) if map_file else {}
    rows = []
    for cls_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        label = mapping.get(cls_dir.name, cls_dir.name)
        if label is None:  # map a folder to null to drop it
            continue
        for p in images_in(cls_dir):
            rows.append({"image_path": str(p), target: label, "source": name})
    print(f"[{name}] {len(rows)} labelled images for '{target}'")
    return rows


def dhash(path: str, size: int = 8) -> int:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return 0
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def group_near_duplicates(paths: list[str], max_hamming: int = 6) -> list[int]:
    """Union-find over perceptual hashes; returns a group id per image."""
    h = np.array([dhash(p) for p in paths], dtype=np.uint64)
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(h) - 1):
        x = np.bitwise_xor(h[i + 1:], h[i])
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.nonzero(d <= max_hamming)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[b] = a
    return [find(i) for i in range(len(paths))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--classes", nargs="*", default=[])
    ap.add_argument("--csv", nargs="*", default=[])
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = []
    for spec in a.pairs:
        rows += from_pairs(*spec.split(":", 2))
    for spec in a.classes:
        parts = spec.split(":")
        rows += from_class_folders(parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else None)
    df = pd.DataFrame(rows)
    for c in a.csv:
        df = pd.concat([df, pd.read_csv(c)], ignore_index=True)
    if df.empty:
        sys.exit("no data found - check the paths")

    for col in ("mask_path", "tissue_path", "patient_id"):
        if col not in df.columns:
            df[col] = np.nan
    # Same image listed by two sources (e.g. a mask set and a class set): merge into one row.
    df = df.groupby("image_path", as_index=False).first()

    no_pid = df.patient_id.isna()
    # An all-empty column is read as float; pandas 3 won't put text into it, so make it text first.
    df["patient_id"] = df["patient_id"].astype(object)
    if no_pid.any():
        groups = group_near_duplicates(df.loc[no_pid, "image_path"].tolist())
        # Never reuse an ID already in the data (e.g. a merged older manifest also has auto_0, auto_1, ...).
        taken, prefix, n = set(df.patient_id.dropna().astype(str)), "auto", 1
        while any(f"{prefix}_{g}" in taken for g in set(groups)):
            n += 1
            prefix = f"auto{n}"
        df.loc[no_pid, "patient_id"] = [f"{prefix}_{g}" for g in groups]
        n_groups = len(set(groups))
        print(f"{no_pid.sum()} images without patient_id -> {n_groups} near-duplicate groups")

    # Rows that already carry a split (e.g. a locked external test set) keep it.
    has_split = df["split"].notna() if "split" in df.columns else pd.Series(False, index=df.index)
    fresh = patient_level_split(df[~has_split], a.val_frac, a.test_frac, a.seed)
    df = pd.concat([df[has_split], fresh]).sort_values("image_path").reset_index(drop=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby(["split", "source"]).size().unstack(fill_value=0))
    leak = df.groupby("patient_id").split.nunique()
    assert (leak <= 1).all(), "a patient appears in more than one split"
    print(f"wrote {a.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
