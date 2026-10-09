"""Build the tissue-segmentation data: three public tissue datasets mapped to TISSUE_CLASSES, a locked test set, an
unlabelled pool for pseudo-labelling, and how well clinicians agree with each other (the bar for the model).

Expects under --raw (see notebooks/kaggle_tissue_cv.ipynb for the downloads):
    woundtissue/{image,label}                         github.com/akabircs/WoundTissue (14 published images)
    dfutissue/DFUTissue/{Labeled/Original,Unlabeled}  Kaggle ibrahimshehada/dfutissue (110 labelled + 600 unlabelled)
    lutseg/{Images,Masks,Wound_Masks,gold_standard}   Hugging Face ksanchez84/LUTSeg (141 images, 39 patients)
and, for the unlabelled pool only, the photo folders in POOL_SOURCES under --pool-raw (link_kaggle_inputs.py --set pool).

    python scripts/build_tissue_dataset.py --raw data/raw/tissue \
        --pool data/manifest_cls_v2.csv data/manifest_seg_v2.csv --out data

Writes
    OUT/tissue_manifest.csv        labelled rows: split test (locked) or train (the cross-validation pool),
                                   patient_id, source, tissue_classes (the classes that row's dataset labels)
    OUT/tissue_unlabeled.csv       photos to pseudo-label, none a near-duplicate of a labelled or test photo
    OUT/tissue_rater_agreement.json  per-class Dice between LUTSeg's five clinicians on its gold-standard subset
    OUT/tissue/<source>/*.png      the rewritten masks

Locked test set: every photo of LUTSeg's nine gold-standard patients (their later photos too, so no patient straddles
train and test) and DFUTissue's own Test split. It is never trained on and never pseudo-labelled.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prepare_data import IMG_EXT, dhash, group_near_duplicates  # noqa: E402
from prepare_tissue import class_index, remap  # noqa: E402
from wound_ai.data import TISSUE_CLASSES  # noqa: E402

MAPS = ROOT / "configs" / "tissue_maps"
# Extra unlabelled photos: (source, photo folder under --pool-raw). Photo folders only, never their mask folders.
POOL_SOURCES = [
    ("dfuc2022", "pabodhamallawa_dfuc2022-train-release/DFUC2022_train_release/DFUC2022_train_images"),
    ("dfuc2022", "tushartalukder4_dfuc-c5/combination_5/testing/input"),
    ("dfuc2022", "tushartalukder4_dfuc-c5/combination_5/train/images"),  # copies of DFUC2022 train, if any: dropped
    ("dfuc2022", "tushartalukder4_dfuc-c5/combination_5/test/images"),
    ("postop", "abdulazizalghaili_postoperative-wound-infection/wound_focus_clean/images"),
    ("fuseg_kaggle", "mohamadtaher_wound-data/test/images"),
    ("fuseg_kaggle", "mohamadtaher_wound-data/train/images"),
    ("fuseg_kaggle", "mohamadtaher_wound-data/validation/images"),
]


def load_map(name: str) -> tuple[dict, int, str]:
    spec = json.loads((MAPS / f"{name}.json").read_text())
    labelled = sorted({v for v in spec["values"].values() if v != "ignore"}, key=TISSUE_CLASSES.index)
    return spec["values"], class_index(spec.get("unlisted", "ignore")), ";".join(labelled)


def write_mask(src: Path, dst: Path, values: dict, unlisted: int) -> set[str]:
    rgb = any("," in k for k in values)
    raw = cv2.imread(str(src), cv2.IMREAD_COLOR if rgb else cv2.IMREAD_UNCHANGED)
    if rgb:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    mask, unknown = remap(raw, values, unlisted)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), mask)
    return unknown


def readable(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT and cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8) is not None


def woundtissue(raw: Path, out: Path) -> list[dict]:
    values, unlisted, classes = load_map("woundtissue")
    labels = {p.stem: p for p in (raw / "woundtissue/label").iterdir()}
    rows = []
    for img in sorted((raw / "woundtissue/image").iterdir()):
        if img.stem in labels and readable(img):
            dst = out / "tissue/woundtissue" / f"{img.stem.replace(' ', '_')}.png"
            if unknown := write_mask(labels[img.stem], dst, values, unlisted):
                print(f"woundtissue {img.name}: unmapped colours {sorted(unknown)} ignored")
            rows.append({"image_path": str(img), "tissue_path": str(dst), "source": "woundtissue", "split": "train",
                         "tissue_classes": classes})
    return rows


def dfutissue(raw: Path, out: Path) -> list[dict]:
    values, unlisted, classes = load_map("dfutissue")
    base = raw / "dfutissue/DFUTissue/Labeled/Original"
    rows = []
    for part, split in (("TrainVal", "train"), ("Test", "test")):
        for img in sorted((base / "Images" / part).iterdir()):
            ann = base / "Annotations" / part / f"{img.stem}.png"
            if ann.exists() and readable(img):
                dst = out / "tissue/dfutissue" / f"{img.stem}.png"
                if unknown := write_mask(ann, dst, values, unlisted):
                    print(f"dfutissue {img.name}: unmapped values {sorted(unknown)} ignored")
                rows.append({"image_path": str(img), "tissue_path": str(dst), "source": "dfutissue", "split": split,
                             "tissue_classes": classes})
    return rows


def lutseg(raw: Path, out: Path) -> list[dict]:
    values, unlisted, classes = load_map("lutseg")
    base = raw / "lutseg"
    gold = {json.loads(line)["patient_id"] for line in (base / "gold_standard/manifest.jsonl").read_text().splitlines() if line}
    rows = []
    for line in (base / "metadata.jsonl").read_text().splitlines():
        if not line:
            continue
        m = json.loads(line)
        img = base / m["image_file_name"]
        if not readable(img):
            continue
        dst = out / "tissue/lutseg" / f"{m['image_id']}.png"
        if unknown := write_mask(base / m["mask_file_name"], dst, values, unlisted):
            print(f"lutseg {m['image_id']}: unmapped values {sorted(unknown)} ignored")
        rows.append({"image_path": str(img), "tissue_path": str(dst), "mask_path": str(base / m["wound_mask_file_name"]),
                     "patient_id": f"lutseg_{m['patient_id']}", "source": "lutseg",
                     "split": "test" if m["patient_id"] in gold else "train", "tissue_classes": classes})
    return rows


def rater_agreement(raw: Path) -> dict:
    """Pairwise per-class Dice between LUTSeg's five clinicians on the gold-standard images: how well experts agree,
    which is as well as a model can be shown to do."""
    base = raw / "lutseg"
    values, unlisted, _ = load_map("lutseg")
    per_class: dict[str, list[float]] = {}
    for line in (base / "gold_standard/manifest.jsonl").read_text().splitlines():
        if not line:
            continue
        m = json.loads(line)
        masks = []
        for key in sorted(k for k in m if k.endswith("_tissue_mask")):
            raw_mask = cv2.imread(str(base / m[key]), cv2.IMREAD_UNCHANGED)
            if raw_mask is not None:
                masks.append(remap(raw_mask, values, unlisted)[0])
        for a, b in itertools.combinations(masks, 2):
            for k, c in enumerate(TISSUE_CLASSES):
                pa, pb = a == k, b == k
                if k and (pa.any() or pb.any()):
                    per_class.setdefault(c, []).append(2 * float((pa & pb).sum()) / float(pa.sum() + pb.sum()))
    return {c: {"mean": round(float(np.mean(v)), 3), "sd": round(float(np.std(v)), 3), "pairs": len(v)}
            for c, v in per_class.items()}


def near_dupe_of(paths: list[str], reference: list[str], max_hamming: int) -> np.ndarray:
    """True for each path whose perceptual hash is within max_hamming of any reference photo."""
    ref = np.array([dhash(p) for p in reference], dtype=np.uint64)
    out = np.zeros(len(paths), bool)
    for i, p in enumerate(paths):
        x = np.bitwise_xor(ref, np.uint64(dhash(p)))
        out[i] = (np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1) <= max_hamming).any()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/tissue")
    ap.add_argument("--pool", nargs="*", default=[], help="manifests whose photos join the unlabelled pool")
    ap.add_argument("--pool-raw", default="data/raw/pool", help="POOL_SOURCES folders that exist here join the pool")
    ap.add_argument("--max-unlabeled", type=int, default=0, help="sample at most this many (0: all)")
    ap.add_argument("--max-hamming", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)

    df = pd.DataFrame(woundtissue(raw, out) + dfutissue(raw, out) + lutseg(raw, out))
    # Datasets without patient IDs: one pseudo-patient per near-duplicate group, so copies stay in one fold.
    no_pid = df["patient_id"].isna()
    groups = group_near_duplicates(df.loc[no_pid, "image_path"].tolist(), a.max_hamming)
    df.loc[no_pid, "patient_id"] = [f"{s}_g{g}" for s, g in zip(df.loc[no_pid, "source"], groups)]
    # A training photo that duplicates a test photo (any dataset) would leak: drop it.
    test = df[df.split == "test"].image_path.tolist()
    leak = (df.split == "train") & near_dupe_of(df.image_path.tolist(), test, a.max_hamming)
    if leak.any():
        print(f"dropped {int(leak.sum())} training photo(s) that duplicate a test photo")
    df = df[~leak].reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tissue_manifest.csv", index=False)

    # Pixels per class and photos per class, by dataset and split: what the model can learn and be tested on.
    counts = []
    for r in df.itertuples():
        m = cv2.imread(r.tissue_path, cv2.IMREAD_UNCHANGED)
        counts.append({c: int((m == k).any()) for k, c in enumerate(TISSUE_CLASSES) if k})
    present = pd.concat([df[["source", "split"]], pd.DataFrame(counts)], axis=1).groupby(["split", "source"]).sum()
    print(f"{len(df)} labelled photos ({(df.split == 'test').sum()} locked test). Photos containing each class:")
    print(present.loc[:, (present.sum() > 0)].to_string())

    # Unlabelled pool: DFUTissue's unlabelled photos, the given manifests and POOL_SOURCES, one photo per
    # near-duplicate group, without copies of any labelled photo.
    pool = [(str(p), "dfutissue_unlabeled") for p in sorted((raw / "dfutissue/DFUTissue/Unlabeled").glob("*"))]
    for m in a.pool:
        pool += [(p, "public") for p in pd.read_csv(m).image_path.dropna().astype(str)]
    for source, sub in POOL_SOURCES:
        if (d := Path(a.pool_raw) / sub).is_dir():
            pool += [(str(p), source) for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXT]
    first: dict[str, str] = {}
    for p, s in pool:
        first.setdefault(p, s)
    unl = pd.DataFrame([(p, s) for p, s in first.items() if readable(Path(p))], columns=["image_path", "source"])
    copies = pd.Series(group_near_duplicates(unl.image_path.tolist(), a.max_hamming)).duplicated().values
    unl = unl[~copies]
    labelled = near_dupe_of(unl.image_path.tolist(), df.image_path.tolist(), a.max_hamming)
    unl = unl[~labelled]
    if a.max_unlabeled and len(unl) > a.max_unlabeled:
        unl = unl.sample(a.max_unlabeled, random_state=a.seed)
    unl.to_csv(out / "tissue_unlabeled.csv", index=False)
    print(f"unlabelled pool: {len(unl)} photos ({int(copies.sum())} near-duplicates within the pool and "
          f"{int(labelled.sum())} of labelled photos dropped)")
    print(unl.source.value_counts().to_string())

    agree = rater_agreement(raw)
    (out / "tissue_rater_agreement.json").write_text(json.dumps(agree, indent=1))
    print("clinician-to-clinician Dice on LUTSeg's gold standard:",
          ", ".join(f"{c} {v['mean']:.2f}" for c, v in agree.items()))


if __name__ == "__main__":
    main()
