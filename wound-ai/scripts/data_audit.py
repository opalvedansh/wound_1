"""Account for every image under data/raw/: is it in the training data, and if not, why not.

Each file gets one status:

    used        in a labelled manifest (wound type, outline, tissue or severity), with its split
    mask        a label image (a manifest's mask, or a file in a masks/labels folder), not a photo
    pool        in the unlabelled pool (data/tissue_unlabeled.csv): no label to learn from yet
    excluded    in a folder that is kept out on purpose, with the reason (EXCLUDED below)
    unchecked   belongs to a manifest that is built on Kaggle and is not on this machine
    copy        the same photo as one already used (perceptual hash), so adding it would teach nothing and could
                put a test photo into training
    unreadable  not an image OpenCV can open
    left out    none of the above: the only status to look into

Writes reports/data_audit.md (counts per dataset) and reports/data_audit.csv (every file that is not `used` or
`mask`, with what it copies). Run after the manifests are built:

    python scripts/build_severity_dataset.py --out data        # optional: the severity manifest is built on Kaggle
    python scripts/data_audit.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_data import IMG_EXT, dhash  # noqa: E402

# Labelled manifests, most complete first: a photo's status names the first one it is in.
LABELLED = ("manifest_cls_v2.csv", "manifest_seg_v2.csv", "tissue_manifest.csv", "severity_manifest.csv",
            "manifest_cls.csv", "manifest.csv")
POOL = "tissue_unlabeled.csv"
MASK_COLUMNS = ("mask_path", "tissue_path")
# Folder names that hold label images, not photos (lower case).
MASK_FOLDERS = {"labels", "label", "masks", "mask", "annotations", "annotation", "gt", "groundtruth", "wound_masked"}
MAX_HAMMING = 6  # as prepare_data.group_near_duplicates
# Folders kept out of training on purpose (path under data/raw, any depth below it) and why. The dataset builders
# drop the same folders; a new exclusion goes in both places.
EXCLUDED = {
    "public/brendarangelolvera_human-skin-burns": "cut-out patches on a white background, not whole photos",
    "public/laithjj_diabetic-foot-ulcer-dfu/DFU/Patches/Abnormal(Ulcer)": "crops of photos already used (would leak into the test set)",
    "public/laithjj_diabetic-foot-ulcer-dfu/DFU/TestSet": "no labels",
    "public/laithjj_diabetic-foot-ulcer-dfu/DFU/Transfer-Learning images": "mixed web wounds with no wound-type label",
    "public/ibrahimfateen_wound-classification/Wound_dataset copy/Bruises": "not a wound type the app assesses",
    "public/yasinpratomo_wound-dataset/Wound_dataset/Bruises": "not a wound type the app assesses",
    "public/yasinpratomo_wound-dataset/Wound_dataset/Ingrown_nails": "not a wound type the app assesses",
    "public/workshops11_medetec-dataset/medetec-dataset/toes": "mixed conditions, no single wound type",
    "public/workshops11_medetec-dataset/medetec-dataset/miscellaneous": "mixed conditions, no single wound type",
    "public/workshops11_medetec-dataset/medetec-dataset/meningitis": "not a wound type the app assesses",
    "public/workshops11_medetec-dataset/medetec-dataset/haemangioma": "not a wound type the app assesses",
    "public/workshops11_medetec-dataset/medetec-dataset/epidermolysis-bullosa": "not a wound type the app assesses",
    "public/workshops11_medetec-dataset/medetec-dataset/pilonidal-sinus": "not a wound type the app assesses",
    "public/sinemgokoz_pressure-ulcers-stages/Dataset/Invalid": "marked invalid by the dataset",
    "azh/Train/BG": "background class: no skin in the photo",
    "azh/Test/BG": "background class: no skin in the photo",
    "tissue/dfutissue/DFUTissue/Labeled/Original/Palette": "colour renderings of the masks",
    "tissue/dfutissue/DFUTissue/Labeled/Padded": "padded copies of the labelled photos and masks",
}
# Manifests built on Kaggle only: without one here, its dataset folder cannot be checked.
KAGGLE_ONLY = {"severity_manifest.csv": "severity"}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="reports")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def norm(p: str) -> str:
    """Manifest paths as written (relative to wound-ai/, or absolute on this machine), comparable with files on disk."""
    path = Path(p)
    return str(path.resolve().relative_to(ROOT)) if path.is_absolute() and ROOT in path.resolve().parents else str(Path(*path.parts))


def looks_like_mask(rel: str) -> bool:
    parts = [x.lower() for x in Path(rel).parts]
    return any(x in MASK_FOLDERS or "mask" in x for x in parts[:-1]) or "mask" in parts[-1]


def nearest(hashes: np.ndarray, pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each hash, the index of the closest hash in `pool` and its Hamming distance."""
    idx, dist = np.zeros(len(hashes), np.int64), np.full(len(hashes), 64, np.int64)
    for start in range(0, len(hashes), 256):
        x = np.bitwise_xor(hashes[start:start + 256, None], pool[None, :])
        d = np.unpackbits(x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8), axis=2).sum(2)
        idx[start:start + 256], dist[start:start + 256] = d.argmin(1), d.min(1)
    return idx, dist


def main():
    a = parse()
    data, out = ROOT / a.data, ROOT / a.out
    used: dict[str, str] = {}
    masks: set[str] = set()
    missing = []
    for name in LABELLED:
        f = data / name
        if not f.exists():
            missing.append(name)
            continue
        df = pd.read_csv(f)
        split = df["split"] if "split" in df.columns else pd.Series(["-"] * len(df))
        for path, s in zip(df.image_path, split):
            used.setdefault(norm(path), f"{name.removesuffix('.csv')}:{s}")
        for col in MASK_COLUMNS:
            if col in df.columns:
                masks.update(norm(p) for p in df[col].dropna())
    pool = {norm(p) for p in pd.read_csv(data / POOL).image_path} if (data / POOL).exists() else set()

    files = sorted(str(p.relative_to(ROOT)) for p in (data / "raw").rglob("*") if p.suffix.lower() in IMG_EXT)
    status: dict[str, str] = {}
    for rel in files:
        if rel in used:
            status[rel] = "used"
        elif rel in masks or looks_like_mask(rel):
            status[rel] = "mask"
        elif rel in pool:
            status[rel] = "pool"
    reason: dict[str, str] = {}
    unchecked = tuple(f"{a.data}/raw/{folder}/" for name, folder in KAGGLE_ONLY.items() if name in missing)
    for rel in files:
        if rel in status:
            continue
        under = rel.removeprefix(f"{a.data}/raw/")
        why = next((w for folder, w in EXCLUDED.items() if under.startswith(folder + "/")), None)
        if why:
            status[rel], reason[rel] = "excluded", why
        elif rel.startswith(unchecked) and unchecked:
            status[rel] = "unchecked"
    rest = [rel for rel in files if rel not in status]
    used_on_disk = [rel for rel in files if status.get(rel) == "used"]
    print(f"{len(files)} image files; hashing {len(used_on_disk)} used and {len(rest)} unexplained photos...", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        used_hash = np.array(list(ex.map(lambda r: dhash(str(ROOT / r)), used_on_disk)), dtype=np.uint64)
        rest_hash = np.array(list(ex.map(lambda r: dhash(str(ROOT / r)), rest)), dtype=np.uint64)

    copy_of: dict[str, str] = {}
    if len(rest) and len(used_hash):
        idx, dist = nearest(rest_hash, used_hash)
        for rel, h, i, d in zip(rest, rest_hash, idx, dist):
            if h == 0:  # dhash's value for a file OpenCV cannot read
                status[rel] = "unreadable"
            elif d <= MAX_HAMMING:
                status[rel], copy_of[rel] = "copy", used_on_disk[i]
            else:
                status[rel] = "left out"
    for rel in rest:
        status.setdefault(rel, "left out")

    order = ("used", "mask", "pool", "excluded", "unchecked", "copy", "unreadable", "left out")
    source = lambda rel: "/".join(Path(rel).parts[2:4]) if Path(rel).parts[2] in ("public", "severity", "tissue") else Path(rel).parts[2]  # noqa: E731
    by_source: dict[str, Counter] = {}
    for rel, s in status.items():
        by_source.setdefault(source(rel), Counter())[s] += 1
    total = Counter(status.values())
    splits = Counter(used[rel] for rel in used_on_disk)

    md = ["# Training data audit", "",
          f"{len(files)} image files under `{a.data}/raw`. Every one has a status; `left out` is the only one to look into.", "",
          "| Dataset | Files | " + " | ".join(s.capitalize() for s in order) + " |", "| --- | --- | " + " | ".join("---" for _ in order) + " |"]
    for src in sorted(by_source):
        c = by_source[src]
        md.append(f"| {src} | {sum(c.values())} | " + " | ".join(str(c[s]) for s in order) + " |")
    md.append(f"| **Total** | **{len(files)}** | " + " | ".join(f"**{total[s]}**" for s in order) + " |")
    md += ["", "## Used photos by manifest and split", "", "| Manifest:split | Photos |", "| --- | --- |",
           *[f"| {k} | {v} |" for k, v in sorted(splits.items())], "",
           "## What the statuses mean", "",
           "- **Used**: in a labelled manifest. The split says whether it trains the model or tests it.",
           "- **Mask**: a label image that belongs to a photo, not a photo.",
           "- **Pool**: unlabelled photos kept for semi-supervised training (not used: it scored worse).",
           "- **Excluded**: in a folder kept out on purpose; the reasons are in the table below.",
           "- **Unchecked**: its manifest is built on Kaggle and is not on this machine, so the photo could not be checked.",
           f"- **Copy**: the same photo as a used one (perceptual hash within {MAX_HAMMING} bits). Datasets on Kaggle re-upload "
           "each other, and many add rotated or recoloured copies.",
           "- **Unreadable**: OpenCV cannot open the file.",
           "- **Left out**: in no manifest and not a copy. Either its folder has no label mapping or its dataset's "
           "builder dropped it; `data_audit.csv` lists each one."]
    why_counts = Counter(reason.values())
    md += ["", "## Why photos are excluded", "", "| Reason | Photos |", "| --- | --- |",
           *[f"| {w} | {n} |" for w, n in why_counts.most_common()]]
    left = Counter(str(Path(rel).parent.relative_to(Path(a.data) / "raw")) for rel, s in status.items() if s == "left out")
    if left:
        md += ["", "## Left out, by folder", "", "| Folder | Photos |", "| --- | --- |", *[f"| {f} | {n} |" for f, n in left.most_common()]]
    if missing:
        md += ["", "Manifests not found on this machine: " + ", ".join(f"`{m}`" for m in missing) + "."]
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_audit.md").write_text("\n".join(md) + "\n")
    rows = [{"image_path": rel, "status": s, "copy_of": copy_of.get(rel, ""), "reason": reason.get(rel, "")}
            for rel, s in status.items() if s not in ("used", "mask")]
    pd.DataFrame(rows).to_csv(out / "data_audit.csv", index=False)
    print("\n".join(md[4:5 + len(by_source) + 2]))
    print(f"\n-> {out / 'data_audit.md'}, {out / 'data_audit.csv'}")


if __name__ == "__main__":
    main()
