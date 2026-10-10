"""The CVAT round-trip: export_for_labelling.py picks and packages photos, import_labels.py reads corrections back."""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from export_for_labelling import COLORS, pick  # noqa: E402
from wound_ai.data import TISSUE_CLASSES  # noqa: E402


def rows(n: int) -> pd.DataFrame:
    bed = ["Granulation", "Granulation;Necrosis/Eschar", "Slough", "Granulation"]
    peri = ["Healthy", "Healthy", "Erythematous", "Healthy"]
    return pd.DataFrame({"visit_id": [f"v{i}" for i in range(n)], "patient_code": [f"P{i}" for i in range(n)],
                         "case_id": ["c0", "c0", "c0", "c1"][:n], "photo_path": [f"k/treatments/t{i}/pre.jpg" for i in range(n)],
                         "nurse_wound_bed_tissue": bed[:n], "nurse_periwound": peri[:n], "nurse_edge": [None] * n})


def test_pick_puts_necrosis_and_redness_first_and_spreads_over_wounds():
    picked = pick(rows(4), n=3, per_wound=2, done={"v3"})
    assert picked.visit_id.tolist()[:2] == ["v1", "v2"]  # necrosis (3), erythema (2) + slough (1)
    assert (picked.case_id == "c0").sum() <= 2 and "v3" not in picked.visit_id.tolist()


def test_round_trip(tmp_path):
    photos, batch, data = tmp_path / "photos", tmp_path / "batch", tmp_path / "data"
    photos.mkdir()
    df = rows(2)
    for v in df.visit_id:
        cv2.imwrite(str(photos / f"{v}.jpg"), np.full((64, 64, 3), 120, np.uint8))
    df.to_csv(tmp_path / "validation.csv", index=False)
    run = lambda *args: subprocess.run([sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True)  # noqa: E731
    run("scripts/export_for_labelling.py", str(tmp_path / "validation.csv"), "--photos", str(photos),
        "--ckpt-dir", str(tmp_path / "no_models"), "--out", str(batch))
    assert sorted(p.name for p in (batch / "images").iterdir()) == ["v0.jpg", "v1.jpg"]
    with zipfile.ZipFile(batch / "annotations.zip") as z:
        assert "labelmap.txt" in z.namelist() and "SegmentationClass/v1.png" in z.namelist()

    # The clinician outlines necrosis on v1 and leaves v0 empty; CVAT exports the same format.
    mask = np.zeros((64, 64, 3), np.uint8)
    mask[10:30, 10:30] = COLORS["necrosis"]
    with zipfile.ZipFile(batch / "annotations.zip") as z, zipfile.ZipFile(tmp_path / "cvat.zip", "w") as out:
        out.writestr("labelmap.txt", z.read("labelmap.txt"))
        for v, m in (("v0", np.zeros_like(mask)), ("v1", mask)):
            out.writestr(f"SegmentationClass/{v}.png", cv2.imencode(".png", cv2.cvtColor(m, cv2.COLOR_RGB2BGR))[1].tobytes())
    data.mkdir()
    pd.DataFrame({"image_path": ["x.jpg"], "tissue_path": ["x.png"], "split": ["train"]}).to_csv(data / "tissue_manifest.csv", index=False)
    run("scripts/import_labels.py", str(tmp_path / "cvat.zip"), "--batch", str(batch), "--out", str(data), "--merge")

    clinic = pd.read_csv(data / "tissue_clinic.csv")
    assert len(clinic) == 1 and clinic.source[0] == "clinic" and clinic.patient_id[0] == "clinic_P1"
    m = cv2.imread(clinic.tissue_path[0], cv2.IMREAD_UNCHANGED)
    assert (m == TISSUE_CLASSES.index("necrosis")).sum() == 400 and set(np.unique(m)) == {0, TISSUE_CLASSES.index("necrosis")}
    assert cv2.imread(clinic.mask_path[0], cv2.IMREAD_GRAYSCALE).max() == 255
    assert len(pd.read_csv(data / "tissue_manifest_all.csv")) == 2
