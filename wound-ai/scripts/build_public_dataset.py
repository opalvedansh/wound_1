"""Add the public wound datasets in data/raw/public/ to the training data, without duplicates or test leakage.

Reads each downloaded dataset with an explicit folder -> label mapping (below), removes copies of the same photo
across datasets (perceptual hash), drops every new photo that duplicates one already in the manifests (so nothing
from the locked test sets slips into training), and writes:

    data/public_cls.csv   image_path, wound_type, source        (wound type)
    data/public_seg.csv   image_path, mask_path, source         (wound outline)

Then build the manifests (existing rows keep their split, new rows are split by near-duplicate group):

    python scripts/build_public_dataset.py
    python scripts/prepare_data.py --csv data/manifest_cls.csv data/public_cls.csv --out data/manifest_cls_v2.csv
    python scripts/prepare_data.py --csv data/manifest.csv data/public_seg.csv --out data/manifest_seg_v2.csv

Labels: diabetic, pressure, venous, surgical, burn, other (acute/traumatic and rarer wounds), not_wound.
Sources are public research/Kaggle datasets; most have no stated licence, so use is for research/learning only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import IMG_EXT, dhash  # noqa: E402

RAW = Path("data/raw/public")

# (source name, folder under data/raw/public, {subfolder: label or None to drop}). Listed by trust: when the same
# photo is in several datasets, the first source's label wins. Expert/clinic sets first, web collections last.
CLS_SOURCES: list[tuple[str, str, dict[str, str | None]]] = [
    ("medetec", "workshops11_medetec-dataset/medetec-dataset", {
        "leg-ulcer-images": "venous", "pressure-ulcer-images-a": "pressure", "pressure-ulcer-images-b": "pressure",
        "foot-ulcers": "diabetic", "burns": "burn", "orthopaedic-wounds": "surgical", "abdominal-wounds": "surgical",
        "malignant-wound-images": "other", "extravasation-wound-images": "other",
        # mixed or not wounds of the kinds the app assesses
        "toes": None, "miscellaneous": None, "meningitis": None, "haemangioma": None,
        "epidermolysis-bullosa": None, "pilonidal-sinus": None,
    }),
    ("pu_stages", "sinemgokoz_pressure-ulcers-stages/Dataset", {
        "Stage_I": "pressure", "Stage_II": "pressure", "Stage_III": "pressure", "Stage_IV": "pressure",
        "Unstageable": "pressure", "SDTI": "pressure", "Invalid": None,
    }),
    ("dfu", "laithjj_diabetic-foot-ulcer-dfu/DFU", {
        "Original Images": "diabetic", "Patches/Normal(Healthy skin)": "not_wound",
        # crops of the originals (would leak), unlabelled test set, mixed web wounds
        "Patches/Abnormal(Ulcer)": None, "TestSet": None, "Transfer-Learning images": None,
    }),
    ("wound_cls", "ibrahimfateen_wound-classification/Wound_dataset copy", {
        "Pressure Wounds": "pressure", "Venous Wounds": "venous", "Diabetic Wounds": "diabetic",
        "Surgical Wounds": "surgical", "Burns": "burn", "Abrasions": "other", "Laseration": "other", "Cut": "other",
        "Normal": "not_wound", "Bruises": None,
    }),
    ("acute", "yasinpratomo_wound-dataset/Wound_dataset", {
        "Burns": "burn", "Abrasions": "other", "Laceration": "other", "Cut": "other", "Stab_wound": "other",
        "Bruises": None, "Ingrown_nails": None,
    }),
]
# Web-collected burn photos with YOLO boxes; a photo with no box is not kept.
BURNS_YOLO = ("burns_web", "shubhambaid_skin-burn-dataset")
# Outline masks: FUSeg (incl. its 200 challenge-test images), WSNet/WoundSeg (many wound types), Medetec.
SEG_SOURCE = "leoscode_wound-segmentation-images/data_wound_seg"


def images_under(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)


def readable(p: Path) -> bool:
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img is not None and min(img.shape[:2]) >= 32


def collect_cls(raw: Path) -> pd.DataFrame:
    rows = []
    for source, base, mapping in CLS_SOURCES:
        root = raw / base
        if not root.exists():
            print(f"[{source}] missing ({root}), skipped")
            continue
        unmapped = set()
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for p in images_under(sub):
                rel = p.parent.relative_to(root).as_posix()
                key = next((k for k in mapping if rel == k or rel.startswith(k + "/")), None)
                if key is None:
                    unmapped.add(rel)
                    continue
                if mapping[key]:
                    rows.append({"image_path": str(p), "wound_type": mapping[key], "source": source})
        if unmapped:
            print(f"[{source}] folders without a mapping, skipped: {sorted(unmapped)}")
    name, base = BURNS_YOLO
    for p in images_under(raw / base):
        label = p.with_suffix(".txt")
        if label.exists() and label.read_text().strip():
            rows.append({"image_path": str(p), "wound_type": "burn", "source": name})
    return pd.DataFrame(rows)


def collect_seg(raw: Path) -> pd.DataFrame:
    rows = []
    base = raw / SEG_SOURCE
    for img_dir, mask_dir in (("train_images", "train_masks"), ("test_images", "test_masks")):
        masks = {p.stem: p for p in images_under(base / mask_dir)}
        for p in images_under(base / img_dir):
            if p.stem in masks:
                prefix = p.stem.split("_")[0]
                rows.append({"image_path": str(p), "mask_path": str(masks[p.stem]),
                             "source": {"fusc": "fuseg", "wsnet": "wsnet", "medetec": "medetec_seg"}.get(prefix, prefix)})
    return pd.DataFrame(rows)


def dedupe(new: pd.DataFrame, existing: list[str], max_hamming: int, label_col: str | None) -> pd.DataFrame:
    """Drops new photos that copy an existing one, and keeps one photo (the most trusted source's) per duplicate group."""
    new = new[[readable(Path(p)) for p in new.image_path]].reset_index(drop=True)
    paths = existing + new.image_path.tolist()
    h = np.array([dhash(p) for p in paths], dtype=np.uint64)
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(h) - 1):
        d = np.unpackbits(np.bitwise_xor(h[i + 1:], h[i]).view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        for j in np.nonzero(d <= max_hamming)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[max(a, b)] = min(a, b)  # the earlier (existing / more trusted) image becomes the root
    n_old = len(existing)
    roots = [find(i) for i in range(len(paths))]
    keep, copies_of_existing, within, conflicts = [], 0, 0, 0
    first_in_group: dict[int, int] = {}
    for k in range(len(new)):
        r = roots[n_old + k]
        if r < n_old:
            copies_of_existing += 1
            continue
        if r in first_in_group:
            within += 1
            if label_col and new.at[first_in_group[r], label_col] != new.at[k, label_col]:
                conflicts += 1
            continue
        first_in_group[r] = k
        keep.append(k)
    print(f"  {len(new)} readable new photos: {copies_of_existing} copies of photos already in the manifests dropped, "
          f"{within} duplicates across the new sets dropped ({conflicts} had a different label; the more trusted "
          f"source's label kept) -> {len(keep)} added")
    return new.loc[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--cls-manifest", default="data/manifest_cls.csv")
    ap.add_argument("--seg-manifest", default="data/manifest.csv")
    ap.add_argument("--max-hamming", type=int, default=6)
    a = ap.parse_args()
    raw = Path(a.raw)

    print("wound type:")
    cls = dedupe(collect_cls(raw), pd.read_csv(a.cls_manifest).image_path.tolist(), a.max_hamming, "wound_type")
    cls.to_csv("data/public_cls.csv", index=False)
    old = pd.read_csv(a.cls_manifest).wound_type.value_counts()
    print(pd.DataFrame({"existing": old, "added": cls.wound_type.value_counts()}).fillna(0).astype(int)
          .assign(total=lambda t: t.existing + t.added).sort_values("total", ascending=False).to_string())
    print(cls.groupby(["wound_type", "source"]).size().unstack(fill_value=0).to_string())

    print("\nwound outline:")
    seg = dedupe(collect_seg(raw), pd.read_csv(a.seg_manifest).image_path.tolist(), a.max_hamming, None)
    seg.to_csv("data/public_seg.csv", index=False)
    print(seg.source.value_counts().to_string())
    print(f"\nwrote data/public_cls.csv ({len(cls)} rows) and data/public_seg.csv ({len(seg)} rows)")


if __name__ == "__main__":
    main()
