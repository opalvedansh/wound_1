"""Build the training data for the severity heads: pressure-injury stage, burn depth and diabetic-foot Wagner grade,
from every public dataset that labels them, without letting copies of a photo inflate the scores.

Several of these datasets are mostly augmented copies of each other (rotated, flipped, cropped, recoloured), which a
perceptual hash misses. So copies are found by image content: an ImageNet ConvNeXt embedding averaged over the 8
rotations/flips (cosine similarity), and:
  - photos at least --copy-sim alike are one group ("patient", so one fold); a group whose sources disagree on the
    label is dropped;
  - the locked test set is drawn only from groups with a photo from a source that looks original (TEST_SOURCES), and
    every training photo at least --guard-sim alike to a test photo is removed (a looser bar: losing a few training
    photos is cheap, a copy of a test photo in training is not).
The similarity distributions are printed so the two bars can be checked. Needs a GPU for speed (Kaggle: ~5 min).

Expects the datasets under --public (link_kaggle_inputs.py) and --raw (link_kaggle_inputs.py --set severity).

    python scripts/build_severity_dataset.py --out data

Writes OUT/severity_manifest.csv: image_path, source, patient_id (copy group), split (train | test) and one of
pu_stage / burn_depth / dfu_wagner per row, and prints raw photos, unique photos and labels per head.
Research use only: most of these datasets state no licence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prepare_data import IMG_EXT  # noqa: E402

PU = {"Stage_I": "stage_1", "Stage_II": "stage_2", "Stage_III": "stage_3", "Stage_IV": "stage_4",
      "Stage_1": "stage_1", "Stage_2": "stage_2", "Stage_3": "stage_3", "Stage_4": "stage_4",
      "Unstageable": "unstageable", "SDTI": "deep_tissue_injury", "Invalid": None}
BURN = {0: "superficial", 1: "partial_thickness", 2: "full_thickness"}  # YOLO class ids (nomi6677 data.yaml)

# (head, source, folder under its root, {subfolder: label or None to drop}, root: "public" | "raw")
FOLDER_SOURCES = [
    ("pu_stage", "pu_sinemgokoz", "sinemgokoz_pressure-ulcers-stages/Dataset", PU, "public"),
    ("pu_stage", "pu_stagewise", "divyanshmahajan2311_pressure-ulcer-stagewise/pressure_ulcers_stages_enhanced", PU, "raw"),
    ("pu_stage", "pu_corrected", "divyanshmahajan2311_corrected-ulcer-dataset/ulcers_self", PU, "raw"),
    ("pu_stage", "pu_final", "gursahiba_final-ulcer-dataset/ulcers_self", PU, "raw"),
    ("burn_depth", "burn_fares", "faresabbasai2022_burn-dataset/burn dataset", {
        "1st degree burn": "superficial", "2nd degree burn": "partial_thickness", "3rd degree burn": "full_thickness"}, "raw"),
    ("burn_depth", "burn_fares13", "faresabbasai2022_burn-dataset13/last_databurn", {
        "1st degree": "superficial", "2nd degree": "partial_thickness", "3nd degree": "full_thickness"}, "raw"),
    ("burn_depth", "burn_dimas", "mohammaddimasnoufal_skin-burn-dataset/Dataset Gambar Skin Burn", {
        "First Degree Burn Superficial Thickness": "superficial",
        "Second Degree Burn Partial of Intermediate Thickness": "partial_thickness",
        "Third Degree Burn Full hickness": "full_thickness", "No Sunburn": None}, "raw"),
    ("dfu_wagner", "wagner", "purushomohan_dfu-wagners-classification/Dataset/Training", {
        "Grade 0": "grade_0", "Grade 1": "grade_1", "Grade 2": "grade_2", "Grade 3": "grade_3",
        "Normal(Healthy skin)": None}, "raw"),
    ("dfu_wagner", "wagner", "purushomohan_dfu-wagners-classification/Dataset/Validation", {
        "Grade 0": "grade_0", "Grade 1": "grade_1", "Grade 2": "grade_2", "Grade 3": "grade_3",
        "Normal(Healthy skin)": None}, "raw"),
]
# YOLO-boxed burns: (source, image folder, label folder, root). A photo's depth is its deepest box.
YOLO_SOURCES = [
    ("burn_shubham", "shubhambaid_skin-burn-dataset", "shubhambaid_skin-burn-dataset", "public"),
    *[("burn_nomi", f"nomi6677_skin-burn-classification-with-degrees/dataset/{s}/images",
       f"nomi6677_skin-burn-classification-with-degrees/dataset/{s}/labels", "raw") for s in ("train", "valid", "test")],
]
# Sources that look like original photos (not augmented re-uploads): the only ones the locked test set comes from.
TEST_SOURCES = {"pu_sinemgokoz", "burn_shubham", "burn_fares", "wagner"}


def images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT) if d.is_dir() else []


MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def _views(path: str, size: int) -> list[np.ndarray] | None:
    im = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if im is None or min(im.shape[:2]) < 16:
        return None
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
    im = (im / np.float32(255) - MEAN) / STD
    return [np.ascontiguousarray(np.rot90(f, k)) for f in (im, im[:, ::-1]) for k in range(4)]


def embed(paths: list[str], size: int = 160, batch: int = 32) -> np.ndarray:
    """Unit-length ImageNet embedding per photo, averaged over its 8 rotations/flips (NaN rows: unreadable)."""
    import timm
    import torch

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=0).to(device).eval()
    out = np.full((len(paths), model.num_features), np.nan, np.float32)
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            views = [(j, _views(p, size)) for j, p in enumerate(paths[i:i + batch], start=i)]
            views = [(j, v) for j, v in views if v is not None]
            if not views:
                continue
            x = torch.from_numpy(np.stack([im for _, v in views for im in v])).permute(0, 3, 1, 2).to(device)
            f = torch.nn.functional.normalize(model(x).float().reshape(len(views), 8, -1).mean(1), dim=1)
            out[[j for j, _ in views]] = f.cpu().numpy()
            if (i // batch) % 50 == 0:
                print(f"  embedded {i + len(views)}/{len(paths)}", flush=True)
    return out


def group(emb: np.ndarray, threshold: float, block: int = 2048) -> np.ndarray:
    """Connected components of 'at least threshold alike' (cosine); returns a group id per photo."""
    parent = np.arange(len(emb))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, len(emb), block):
        sim = emb[s:s + block] @ emb.T
        for a, b in zip(*np.nonzero(sim >= threshold)):
            a, b = find(s + int(a)), find(int(b))
            if a != b:
                parent[b] = a
    return np.array([find(i) for i in range(len(emb))])


def nearest(emb: np.ndarray, ref: np.ndarray, block: int = 2048) -> np.ndarray:
    """Highest cosine similarity of each photo to any reference photo."""
    if not len(ref):
        return np.full(len(emb), -1.0)
    return np.concatenate([(emb[s:s + block] @ ref.T).max(1) for s in range(0, len(emb), block)])


def collect(public: Path, raw: Path) -> pd.DataFrame:
    roots = {"public": public, "raw": raw}
    rows = []
    for head, source, folder, labels, root in FOLDER_SOURCES:
        base = roots[root] / folder
        if not base.is_dir():
            print(f"missing {base}: skipped")
            continue
        for sub, label in labels.items():
            if label:
                rows += [{"image_path": str(p), "source": source, "head": head, "label": label} for p in images(base / sub)]
    for source, img_dir, lbl_dir, root in YOLO_SOURCES:
        img_dir, lbl_dir = roots[root] / img_dir, roots[root] / lbl_dir
        if not img_dir.is_dir():
            print(f"missing {img_dir}: skipped")
            continue
        for p in images(img_dir):
            txt = lbl_dir / f"{p.stem}.txt"
            ids = [int(float(line.split()[0])) for line in txt.read_text().splitlines() if line.strip()] if txt.exists() else []
            if ids and max(ids) in BURN:
                rows.append({"image_path": str(p), "source": source, "head": "burn_depth", "label": BURN[max(ids)]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--public", default="data/raw/public")
    ap.add_argument("--raw", default="data/raw/severity")
    ap.add_argument("--copy-sim", type=float, default=0.95, help="photos at least this alike are copies (one group)")
    ap.add_argument("--guard-sim", type=float, default=0.85, help="training photos this alike to a test photo are dropped")
    ap.add_argument("--max-per-source", type=int, default=0, help="debug: at most this many photos per source")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    df = collect(Path(a.public), Path(a.raw))
    if a.max_per_source:
        df = df.groupby("source", group_keys=False).head(a.max_per_source).reset_index(drop=True)
    print(f"embedding {len(df)} photos", flush=True)
    emb = embed(df.image_path.tolist())
    unreadable = np.isnan(emb).any(1)
    df, emb = df[~unreadable].reset_index(drop=True), emb[~unreadable]
    raw_counts = df.groupby(["head", "source"]).size()

    parts = []
    for head, d in df.groupby("head"):
        e = emb[d.index.to_numpy()]
        d = d.reset_index(drop=True)
        d["group"] = group(e, a.copy_sim)
        conflict = (d.groupby("group").label.transform("nunique") > 1).to_numpy()
        d, e = d[~conflict].reset_index(drop=True), e[~conflict]
        # Locked test: whole groups, stratified by label, from groups holding an original-looking photo.
        g = d.groupby("group").agg(label=("label", "first"), eligible=("source", lambda s: bool(set(s) & TEST_SOURCES)))
        rng = np.random.default_rng(a.seed)
        test_groups = []
        for _, lab in g[g.eligible].groupby("label"):
            n = max(1, round(a.test_frac * len(lab)))
            test_groups += rng.choice(lab.index.to_numpy(), size=min(n, len(lab)), replace=False).tolist()
        d["split"] = np.where(d.group.isin(test_groups), "test", "train")
        is_test = (d.split == "test").to_numpy()
        sim_to_test = nearest(e, e[is_test])
        near = ~is_test & (sim_to_test >= a.guard_sim)
        # How alike the photos are: copies sit near 1, different wounds lower. Check the two bars against this.
        orig = d.source.isin(TEST_SOURCES).to_numpy()
        if orig.any() and (~orig).any():
            q = np.percentile(nearest(e[~orig], e[orig]), [10, 25, 50, 75, 90]).round(3)
            print(f"\n{head}: other sources' photos, similarity to the nearest original-source photo (10/25/50/75/90%): {q}")
        q = np.percentile(sim_to_test[~is_test], [50, 90, 99]).round(3)
        print(f"{head}: training photos' similarity to the nearest test photo (50/90/99%): {q}; "
              f"largest copy group: {int(d.group.value_counts().max())} photos")
        d = d[~near].reset_index(drop=True)
        d["patient_id"] = [f"{head}_g{x}" for x in d.group]
        d[head] = d.label
        print(f"\n== {head}: {int(raw_counts[head].sum())} photos -> {d.group.nunique()} unique "
              f"({int(conflict.sum())} in groups with conflicting labels and {int(near.sum())} too close to a test "
              f"photo dropped)")
        print(pd.crosstab(d[head], d.split, margins=True).to_string())
        print("unique photos per source:", d.groupby("source").group.nunique().to_dict())
        parts.append(d)

    out = pd.concat(parts, ignore_index=True)
    cols = ["image_path", "source", "patient_id", "split", "pu_stage", "burn_depth", "dfu_wagner"]
    out = out.reindex(columns=cols)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(a.out) / "severity_manifest.csv", index=False)
    print(f"\n{len(out)} rows ({int(unreadable.sum())} unreadable photos skipped) -> {Path(a.out) / 'severity_manifest.csv'}")
    print("raw photos per source:\n" + raw_counts.to_string())


if __name__ == "__main__":
    main()
