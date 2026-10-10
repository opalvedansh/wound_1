"""Clinician-corrected tissue outlines from CVAT -> training rows for the tissue model.

    python scripts/import_labels.py cvat-export.zip --batch labelling/batch1 --out data

Reads the CVAT "Segmentation mask 1.1" export (colours -> classes from its labelmap.txt), writes one class-index mask
per photo to OUT/tissue/clinic/ and its wound outline (the wound-bed classes) beside it, copies the photo to
OUT/raw/clinic/, and adds the rows to OUT/tissue_clinic.csv (re-importing a photo replaces its row).

About 30% of the clinic's patients form a locked clinic test set (by a stable hash of the patient code, so a patient
never moves between batches): never trained on, it shows how the model does on this clinic. To train on public and
clinic photos together, --merge writes OUT/tissue_manifest_all.csv from OUT/tissue_manifest.csv and the clinic rows;
pass that to tissue_cv.py --manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from wound_ai.data import TISSUE_CLASSES, WOUND_BED


def test_patient(code: str, frac: float) -> bool:
    return int(hashlib.sha256(f"clinic:{code}".encode()).hexdigest()[:8], 16) % 1000 < frac * 1000


def labelmap(z: zipfile.ZipFile) -> dict[tuple[int, int, int], str]:
    name = next((n for n in z.namelist() if n.endswith("labelmap.txt")), None)
    if name is None:
        raise SystemExit("No labelmap.txt in the export: export as 'Segmentation mask 1.1'.")
    colours = {}
    for line in z.read(name).decode().splitlines():
        if line.strip() and not line.startswith("#"):
            label, rgb = line.split(":")[:2]
            colours[tuple(int(x) for x in rgb.split(","))] = label.strip()
    return colours


def to_classes(rgb: np.ndarray, colours: dict) -> tuple[np.ndarray, set[str]]:
    out, unknown = np.zeros(rgb.shape[:2], np.uint8), set()
    for colour, label in colours.items():
        hit = (rgb == np.array(colour, np.uint8)).all(-1)
        if not hit.any():
            continue
        if label in TISSUE_CLASSES:
            out[hit] = TISSUE_CLASSES.index(label)
        else:
            unknown.add(label)
    return out, unknown


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="CVAT export (Segmentation mask 1.1) .zip")
    ap.add_argument("--batch", required=True, help="the folder export_for_labelling.py wrote")
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--merge", action="store_true", help="also write OUT/tissue_manifest_all.csv (public + clinic)")
    a = ap.parse_args()
    batch, out = Path(a.batch), Path(a.out)
    info = pd.read_csv(batch / "labelling.csv", dtype={"visit_id": str, "patient_code": str}).set_index("visit_id")
    (out / "tissue/clinic").mkdir(parents=True, exist_ok=True)
    (out / "raw/clinic").mkdir(parents=True, exist_ok=True)

    rows, empty = [], 0
    with zipfile.ZipFile(a.export) as z:
        colours = labelmap(z)
        for name in sorted(n for n in z.namelist() if "SegmentationClass/" in n and n.endswith(".png")):
            visit = Path(name).stem
            if visit not in info.index:
                print(f"{visit}: not in {batch / 'labelling.csv'}, skipped")
                continue
            bgr = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            mask, unknown = to_classes(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), colours)
            if unknown:
                print(f"{visit}: labels {sorted(unknown)} aren't tissue classes, left as background")
            if not mask.any():
                empty += 1  # nothing outlined: not used (an unlabelled photo would teach "no wound")
                continue
            photo = out / "raw/clinic" / f"{visit}.jpg"
            shutil.copy(batch / "images" / f"{visit}.jpg", photo)
            tissue_path, outline_path = out / "tissue/clinic" / f"{visit}.png", out / "tissue/clinic" / f"{visit}_outline.png"
            cv2.imwrite(str(tissue_path), mask)
            bed = np.isin(mask, [TISSUE_CLASSES.index(c) for c in WOUND_BED])
            cv2.imwrite(str(outline_path), bed.astype(np.uint8) * 255)
            code = str(info.loc[visit, "patient_code"])
            rows.append({"image_path": str(photo), "tissue_path": str(tissue_path), "mask_path": str(outline_path),
                         "patient_id": f"clinic_{code}", "source": "clinic",
                         "split": "test" if test_patient(code, a.test_frac) else "train", "tissue_classes": ""})

    new = pd.DataFrame(rows)
    path = out / "tissue_clinic.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=new.columns)
    clinic = pd.concat([old[~old.image_path.isin(new.get("image_path", []))], new], ignore_index=True)
    clinic.to_csv(path, index=False)
    print(f"{len(new)} photos imported ({empty} with nothing outlined skipped); clinic total {len(clinic)}: "
          f"{(clinic.split == 'train').sum()} train, {(clinic.split == 'test').sum()} locked test, "
          f"{clinic.patient_id.nunique()} patients -> {path}")
    for c in TISSUE_CLASSES[1:]:
        k = TISSUE_CLASSES.index(c)
        n = sum(int((cv2.imread(p, cv2.IMREAD_UNCHANGED) == k).any()) for p in clinic.tissue_path)
        if n:
            print(f"  {c}: {n} photos")
    if a.merge:
        public = pd.read_csv(out / "tissue_manifest.csv")
        pd.concat([public, clinic], ignore_index=True).to_csv(out / "tissue_manifest_all.csv", index=False)
        print(f"-> {out / 'tissue_manifest_all.csv'} (python scripts/tissue_cv.py --manifest {out / 'tissue_manifest_all.csv'} --teacher-only ...)")


if __name__ == "__main__":
    main()
