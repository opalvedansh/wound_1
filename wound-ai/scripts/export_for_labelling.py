"""A batch of clinic photos for a clinician to outline tissue in CVAT, with the model's outlines filled in to correct.

Picks the visits most worth labelling first: where the nurse recorded necrosis, exposed bone/tendon, red or
macerated skin around the wound or callus (what the model is weakest on), at most --per-wound photos per wound.
Photos come from the clinic's private storage, so run this on a clinic computer with the clinic's keys:

    export SUPABASE_URL=https://<project>.supabase.co SUPABASE_SECRET_KEY=<service key>
    python scripts/export_for_labelling.py wound-validation-<date>-deid.csv --n 100 --out labelling/batch1

Writes OUT/images/<visit>.jpg (upload these to a CVAT task), OUT/annotations.zip (Segmentation mask 1.1: import it
into the task, then correct), OUT/labelling.csv (which photo is which) and OUT/README.txt with the steps.
Use a CVAT you run yourself (or one covered by your data agreement): these are patient photos.
Then: python scripts/import_labels.py <CVAT export>.zip --batch OUT
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from wound_ai.data import TISSUE_CLASSES, read_rgb

# One colour per class in the CVAT label map (and back again in import_labels.py).
COLORS = {"background": (0, 0, 0), "granulation": (220, 40, 60), "slough": (240, 210, 60), "necrosis": (40, 40, 40),
          "epithelial": (250, 160, 200), "periwound_erythema": (255, 110, 0), "maceration": (200, 200, 255),
          "exposed_structure": (255, 255, 255), "callus": (150, 120, 60)}
# What makes a visit worth labelling first: the nurse saw something the model is weak on.
PRIORITY = [("nurse_wound_bed_tissue", "Necrosis/Eschar", 3), ("nurse_wound_bed_tissue", "Exposed bone/tendon", 3),
            ("nurse_periwound", "Erythematous", 2), ("nurse_periwound", "Macerated", 2), ("nurse_edge", "Macerated", 1),
            ("nurse_edge", "Callused", 1), ("nurse_wound_bed_tissue", "Slough", 1)]

README = """Labelling batch: {n} photos

1. In CVAT, create a task with these labels (Raw):
{labels}
   and upload the photos in images/.
2. Task > Actions > Upload annotations > "Segmentation mask 1.1" > annotations.zip. The model's outlines appear.
3. Correct every photo: draw what you see, remove what isn't there. Label every tissue in the wound bed and the
   redness, maceration or callus in the skin around it. Leave everything else unlabelled (background).
4. Task > Actions > Export task dataset > "Segmentation mask 1.1" (without images). Then on this computer:
   python scripts/import_labels.py <the export>.zip --batch {out}
"""


def stable(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def pick(df: pd.DataFrame, n: int, per_wound: int, done: set[str]) -> pd.DataFrame:
    df = df[df.photo_path.notna() & ~df.visit_id.isin(done)].copy()
    df["priority"] = 0
    df["why"] = ""
    for col, value, weight in PRIORITY:
        hit = df[col].fillna("").astype(str).str.split(";").map(lambda xs, v=value: v in xs)
        df.loc[hit, "priority"] += weight
        df.loc[hit, "why"] += f"{value}; "
    df["tiebreak"] = df.visit_id.astype(str).map(stable)
    df = df.sort_values(["priority", "tiebreak"], ascending=[False, True])
    return df.groupby("case_id", sort=False).head(per_wound).head(n)


def download(path: str, dest: Path) -> None:
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SECRET_KEY")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SECRET_KEY (the clinic's), or pass --photos with the photos.")
    req = urllib.request.Request(f"{url.rstrip('/')}/storage/v1/object/images/{path}",
                                 headers={"Authorization": f"Bearer {key}", "apikey": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())


def colour_mask(tissue: np.ndarray, classes: list[str]) -> np.ndarray:
    rgb = np.zeros((*tissue.shape, 3), np.uint8)
    for k, c in enumerate(classes):
        rgb[tissue == k] = COLORS.get(c, (0, 0, 0))
    return rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="the validation export (GET /exports/validation)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per-wound", type=int, default=2)
    ap.add_argument("--ckpt-dir", default="checkpoints", help="outline + tissue models for the pre-filled outlines")
    ap.add_argument("--photos", default="", help="folder with <visit_id>.jpg already downloaded (instead of the keys)")
    ap.add_argument("--done", nargs="*", default=[], help="labelling.csv of earlier batches: those visits are skipped")
    ap.add_argument("--out", default="labelling/batch1")
    a = ap.parse_args()

    done = {v for f in a.done for v in pd.read_csv(f).visit_id.astype(str)}
    batch = pick(pd.read_csv(a.csv, dtype={"visit_id": str}), a.n, a.per_wound, done)
    if batch.empty:
        sys.exit("No photos to label (no photo paths in the export, or all already labelled).")
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    from wound_ai.pipeline import WoundAnalyzer

    an = WoundAnalyzer(a.ckpt_dir)
    classes = an.tissue_classes() if "tissue" in an.seg else TISSUE_CLASSES
    masks: dict[str, np.ndarray] = {}
    for r in batch.itertuples():
        dest = out / "images" / f"{r.visit_id}.jpg"
        if not dest.exists():
            src = Path(a.photos) / f"{r.visit_id}.jpg" if a.photos else None
            if src is not None:
                dest.write_bytes(src.read_bytes())
            else:
                download(r.photo_path, dest)
        img = read_rgb(str(dest))
        tissue = np.zeros(img.shape[:2], np.uint8)
        if "boundary" in an.seg and "tissue" in an.seg:
            outline = an._segment(img, "boundary")
            if outline.any():
                tissue, _ = an._tissue_map(img, outline)
        masks[r.visit_id] = colour_mask(tissue, classes)
        print(f"{len(masks)}/{len(batch)} {r.visit_id} ({r.why.strip('; ') or 'no priority finding'})", flush=True)

    with zipfile.ZipFile(out / "annotations.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("labelmap.txt", "# label:color_rgb:parts:actions\n"
                   + "".join(f"{c}:{','.join(map(str, COLORS[c]))}::\n" for c in TISSUE_CLASSES))
        z.writestr("ImageSets/Segmentation/default.txt", "\n".join(masks) + "\n")
        for name, rgb in masks.items():
            ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            z.writestr(f"SegmentationClass/{name}.png", png.tobytes())
    batch[["visit_id", "patient_code", "case_id", "photo_path", "priority", "why"]].to_csv(out / "labelling.csv", index=False)
    labels = "".join(f"     {c}  #{'%02x%02x%02x' % COLORS[c]}\n" for c in TISSUE_CLASSES)
    (out / "README.txt").write_text(README.format(n=len(masks), labels=labels, out=out))
    print(f"\n{len(masks)} photos -> {out}. Next: {out / 'README.txt'}")


if __name__ == "__main__":
    main()
