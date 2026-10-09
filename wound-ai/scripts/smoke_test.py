"""End-to-end smoke test on synthetic images (CPU, ~2-4 minutes).

Checks that every script runs and fits together: data prep -> training ->
evaluation -> pipeline -> report -> API, and that measurement is numerically
right (including under camera tilt). It says nothing about accuracy on real
wounds; synthetic blobs are only for plumbing.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from wound_ai.measure import ARUCO_DICT, GREY_GAP_MM, find_marker, grey_patch_gains, measure_wound

PX_PER_MM = 4.0
MARKER_MM = 20.0


def paste_marker(img: np.ndarray, x: int, y: int, marker_id: int = 0) -> None:
    side = int(MARKER_MM * PX_PER_MM)
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    m = cv2.aruco.generateImageMarker(d, marker_id, side, borderBits=1)
    q = 12  # white quiet zone around the marker, like the printed sticker
    img[y - q:y + side + q, x - q:x + side + q] = 255
    img[y:y + side, x:x + side] = m[..., None]


def test_measurement() -> None:
    img = np.full((600, 800, 3), (190, 150, 130), np.uint8)
    paste_marker(img, 60, 60)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[300:380, 400:520] = 1           # 120 x 80 px = 30 x 20 mm = 6.00 cm^2
    cal = find_marker(img, MARKER_MM)
    assert cal is not None, "marker not detected"
    m = measure_wound(mask, cal)
    print("flat:", m.to_dict())
    assert abs(m.area_cm2 - 6.0) / 6.0 < 0.02 and abs(m.length_cm - 3.0) < 0.06 and abs(m.width_cm - 2.0) < 0.06

    # Simulate holding the phone at an angle: warp image and mask with the same homography.
    src = np.float32([[0, 0], [800, 0], [800, 600], [0, 600]])
    dst = np.float32([[60, 30], [760, 0], [800, 600], [0, 560]])
    H = cv2.getPerspectiveTransform(src, dst)
    img_w = cv2.warpPerspective(img, H, (800, 600), borderValue=(190, 150, 130))
    mask_w = cv2.warpPerspective(mask, H, (800, 600), flags=cv2.INTER_NEAREST)
    m2 = measure_wound(mask_w, find_marker(img_w, MARKER_MM))
    print("tilted:", m2.to_dict())
    # The marker is only 80 px wide here and the wound sits ~4 marker-widths away, so small
    # corner errors are extrapolated. Real photos (marker 200+ px, placed right next to the
    # wound) do better; validate on your own data against clinician tracings.
    assert abs(m2.area_cm2 - 6.0) / 6.0 < 0.05, "perspective correction off by more than 5%"

    # The grey patch beside the marker, photographed under a warm light: the gains must undo the cast.
    warm = img.copy()
    side, gap = int(MARKER_MM * PX_PER_MM), int(GREY_GAP_MM * PX_PER_MM)
    warm[60:60 + side, 60 + side + gap:60 + 2 * side + gap] = 128
    warm = np.clip(warm * np.array([1.15, 1.0, 0.85]), 0, 255).astype(np.uint8)
    gains = grey_patch_gains(warm, find_marker(warm, MARKER_MM), MARKER_MM)
    print("white balance gains:", gains)
    assert gains and abs(gains[0] * 1.15 - gains[2] * 0.85) < 0.03, "grey patch should neutralise the cast"
    assert grey_patch_gains(img, cal, MARKER_MM) is None, "no patch (old sticker sheet): no correction"


def make_synthetic(root: Path, n_patients: int = 24, per_patient: int = 2) -> Path:
    rng = np.random.default_rng(0)
    (root / "img").mkdir(parents=True)
    (root / "mask").mkdir()
    (root / "tissue").mkdir()
    types = {"diabetic": ((60, 40, 200), "foot_plantar"), "pressure": ((40, 170, 210), "sacrum_buttock"),
             "venous": ((90, 70, 150), "lower_leg")}
    rows = []
    for p in range(n_patients):
        wtype = list(types)[p % 3]
        bgr, loc = types[wtype]
        for k in range(per_patient):
            img = np.clip(np.full((512, 512, 3), (150, 170, 205), np.float32)
                          + rng.normal(0, 8, (512, 512, 3)), 0, 255).astype(np.uint8)
            paste_marker(img, 40, 40)
            mask = np.zeros((512, 512), np.uint8)
            c = (int(rng.integers(260, 400)), int(rng.integers(260, 400)))
            ax = (int(rng.integers(40, 100)), int(rng.integers(30, 70)))
            cv2.ellipse(mask, c, ax, float(rng.uniform(0, 180)), 0, 360, 1, -1)
            tissue = mask.copy()                                    # 1 = granulation
            half = np.zeros_like(mask)
            cv2.ellipse(half, c, ax, 0, 0, 120, 1, -1)
            tissue[(half == 1) & (mask == 1)] = 2                   # 2 = slough
            img[mask == 1] = bgr
            img[tissue == 2] = (60, 200, 230)
            ring = cv2.dilate(mask, np.ones((15, 15), np.uint8)) - mask
            tissue[ring == 1] = 5                                   # periwound erythema
            name = f"p{p:02d}_{k}"
            cv2.imwrite(str(root / "img" / f"{name}.png"), img)
            cv2.imwrite(str(root / "mask" / f"{name}.png"), mask * 255)
            cv2.imwrite(str(root / "tissue" / f"{name}.png"), tissue)
            rows.append({"image_path": str(root / "img" / f"{name}.png"),
                         "tissue_path": str(root / "tissue" / f"{name}.png"),
                         "patient_id": f"P{p:02d}", "wound_type": wtype, "body_location": loc,
                         "fitzpatrick": "IV-VI" if p % 2 else "I-III",
                         "area_cm2_manual": round(float(mask.sum()) / PX_PER_MM ** 2 / 100, 3)})
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)
    return root


def run(*args: str) -> None:
    print("\n$", " ".join(args[1:]))
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main():
    test_measurement()
    work = Path(tempfile.mkdtemp(prefix="wound_smoke_"))
    try:
        d = make_synthetic(work / "synth")
        man, ck = str(work / "manifest.csv"), work / "checkpoints"
        ck.mkdir()
        run("scripts/prepare_data.py", "--pairs", f"synth:{d / 'img'}:{d / 'mask'}",
            "--csv", str(d / "labels.csv"), "--out", man, "--val-frac", "0.25", "--test-frac", "0.25")
        common = ["--manifest", man, "--size", "128", "--epochs", "2", "--batch-size", "4", "--workers", "0",
                  "--no-pretrained"]
        run("scripts/train_seg.py", *common, "--task", "boundary", "--arch", "unet", "--encoder", "resnet18",
            "--out", str(work / "runs/boundary"))
        run("scripts/train_seg.py", *common, "--task", "tissue", "--arch", "segformer", "--encoder", "mit_b0",
            "--crop", "--class-weights", "0.5,1,1,1,1,1,1,1", "--out", str(work / "runs/tissue"))
        run("scripts/train_cls.py", *common, "--target", "wound_type", "--backbone", "resnet18",
            "--meta-cols", "body_location", "--out", str(work / "runs/wound_type"))
        for src in ("runs/boundary/boundary.pt", "runs/tissue/tissue.pt", "runs/wound_type/wound_type.pt"):
            shutil.copy(work / src, ck)
        run("scripts/evaluate.py", "--manifest", man, "--ckpt-dir", str(ck), "--group-cols", "fitzpatrick",
            "--out", str(work / "eval.json"))

        # Tissue: teacher -> pseudo-labels -> student, cross-validated, as the Kaggle tissue notebook runs it.
        unl = work / "unlabeled.csv"
        pd.DataFrame({"image_path": sorted(str(p) for p in (d / "img").iterdir())[:12]}).to_csv(unl, index=False)
        run("scripts/tissue_cv.py", "--manifest", man, "--unlabeled", str(unl), "--boundary", str(ck / "boundary.pt"),
            "--arch", "unet", "--encoder", "resnet18", "--size", "128", "--batch", "4", "--workers", "0",
            "--folds", "2", "--smoke", "--min-test-photos", "1", "--out", str(work / "tissue_cv"))
        assert (work / "tissue_cv/best/tissue.pt").exists() and (work / "tissue_cv/summary.md").exists()

        from wound_ai.pipeline import WoundAnalyzer

        an = WoundAnalyzer(ck)
        img = str(d / "img" / "p00_0.png")
        f = an.analyze(img, {"body_location": "foot_plantar", "diabetes": "yes", "fever": "no"},
                       previous={"area_cm2": 12.0, "days_ago": 28}, overlay_path=str(work / "overlay.png"))
        print("\nstatus:", f["status"], "| flags:", [x["text"][:50] for x in f.get("flags", [])])
        print(f.get("report_markdown", "")[:900])
        assert f["status"] in ("ok", "no_wound_found", "retake")
        if f["status"] == "ok":
            assert f["outline"], "an ok result with a boundary model must carry the outline"
            assert all(0 <= v <= 1 for poly in f["outline"] for pt in poly for v in pt)

        from wound_ai.report import build_report, check_narrative

        demo = {"wound_type": {"label": "diabetic", "prob": 0.86, "top": [("diabetic", 0.86)]},
                "measurement": {"area_cm2": 4.2, "length_cm": 3.1, "width_cm": 1.8, "perimeter_cm": 8.4,
                                "n_regions": 1},
                "tissue_pct": {"granulation": 70, "slough": 30}, "intake": {"diabetes": "yes"}}
        assert check_narrative("Area is 4.2 cm² with mostly granulation (70%).", demo) == []
        assert check_narrative("Area is 9.9 cm²; prescribe antibiotic 500 mg.", demo), "guard should reject"
        _, probs = build_report(dict(demo), "The ulcer is 7.5 cm long.")
        assert probs, "invented number must be caught"

        from fastapi.testclient import TestClient
        import os

        os.environ["CKPT_DIR"] = str(ck)
        os.environ["WOUND_API_KEY"] = "smoke-test-key"
        sys.modules.pop("api.server", None)
        from api.server import app

        c = TestClient(app, headers={"X-API-Key": "smoke-test-key"})
        assert c.get("/health").status_code == 200
        assert len(c.get("/intake/questions").json()) >= 8
        with open(img, "rb") as fh:
            r = c.post("/analyze", files={"image": ("w.png", fh, "image/png")},
                       data={"intake": json.dumps({"diabetes": "yes", "cause": "started_on_its_own",
                                                   "body_location": "foot_plantar"})})
        assert r.status_code == 200, r.text
        print("API /analyze status:", r.json()["status"], "| follow-ups:",
              [q["id"] for q in r.json().get("follow_up_questions", [])])

        # Three visits of one wound, PRE and POST each, through the treatment report.
        from wound_ai.progress import observation

        treatments = []
        for seq, name in enumerate(("p01_0.png", "p01_1.png", "p02_0.png"), start=1):
            obs = {}
            for phase in ("pre", "post"):
                with open(d / "img" / name, "rb") as fh:
                    fa = c.post("/analyze", files={"image": ("w.png", fh, "image/png")},
                                data={"intake": json.dumps({"diabetes": "no", "cause": "started_on_its_own",
                                                            "body_location": "lower_leg"}), "phase": phase}).json()
                obs[phase] = observation(fa, f"2026-01-{1 + 14 * (seq - 1):02d}T10:00:00+00:00")
            treatments.append({"sequence": seq, **obs, "assessment": {"exudate_level": "Moderate"}, "dressing": "Foam"})
        r = c.post("/treatment-report", json={"wound_type": fa.get("wound_type"), "intake": {"body_location": "lower_leg"},
                                              "treatments": treatments})
        assert r.status_code == 200, r.text
        tr = r.json()
        print("treatment report: healing", tr["progress"]["healing"].get("trajectory"),
              "| suggestions", [x["action"] for x in tr["suggestions"]])
        assert "# Treatment 3 assessment" in tr["report_markdown"]
        print("\nSMOKE TEST PASSED")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
