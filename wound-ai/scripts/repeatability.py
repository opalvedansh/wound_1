"""Measurement noise from a repeat-photo study: the change in wound area (and tissue mix) that is too small to
tell apart from re-photographing the same wound. Protocol: docs/repeatability_study.md.

Photos are named <wound>_<photographer>.jpg, e.g. W017_nurseA.jpg and W017_nurseB.jpg: the same wound,
photographed independently by two (or more) staff in the same visit. Each photo goes through the app's own
pipeline (quality gate, outline, sticker), so the result is the noise of the whole system, not just the model.

    python scripts/repeatability.py --photos data/repeatability --ckpt-dir checkpoints --sites data/sites.csv
    python scripts/repeatability.py ... --write      # once the study is big enough: use it in the app

--sites (optional) is a CSV with columns wound, body_location, for the noise by site: curved sites (heel, toe)
are usually noisier, and a band that holds on flat skin may not hold there.

--write saves wound_ai/measurement_noise.json, which progress.py reads instead of its placeholder. It refuses with
fewer than --min-wounds wounds. The file ships with the code, so commit it, and record it in the clinical sign-off.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from wound_ai.metrics import repeatability
from wound_ai.pipeline import WoundAnalyzer
from wound_ai.progress import NOISE_FILE, nonviable

IMG_EXT = {".jpg", ".jpeg", ".png"}


def measure(an: WoundAnalyzer, photos: list[Path]) -> pd.DataFrame:
    rows = []
    for p in photos:
        wound, _, who = p.stem.rpartition("_")
        if not wound:
            print(f"skipped {p.name}: name it <wound>_<photographer>{p.suffix}")
            continue
        f = an.analyze(str(p), {})
        m = f.get("measurement") or {}
        reason = None if f["status"] == "ok" and m else (f["status"] if f["status"] != "ok" else
                                                           "no sticker found" if not f.get("marker_found") else "no outline")
        rows.append({"wound": wound, "photographer": who, "photo": p.name, "status": f["status"],
                     "area_cm2": m.get("area_cm2"), "nonviable_pct": nonviable(f.get("tissue_pct")), "excluded": reason})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    ok = df[df.excluded.isna()]
    areas = [g.area_cm2.tolist() for _, g in ok.groupby("wound") if len(g) >= 2]
    tissue = [g.nonviable_pct.dropna().tolist() for _, g in ok.groupby("wound")]
    return {"area": repeatability(areas, log=True),
            "tissue": repeatability([t for t in tissue if len(t) >= 2], log=False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--marker-mm", type=float, default=20.0)
    ap.add_argument("--sites", default="", help="CSV: wound, body_location")
    ap.add_argument("--min-wounds", type=int, default=30)
    ap.add_argument("--out", default="reports/repeatability.json")
    ap.add_argument("--write", action="store_true", help=f"save the result to {NOISE_FILE.name} for the app to use")
    a = ap.parse_args()

    an = WoundAnalyzer(a.ckpt_dir, marker_mm=a.marker_mm)
    if "boundary" not in an.seg:
        sys.exit("needs the outline model (boundary.pt): area cannot be measured without it")
    photos = sorted(p for p in Path(a.photos).iterdir() if p.suffix.lower() in IMG_EXT)
    df = measure(an, photos)
    if df.empty:
        sys.exit("no photos found")

    res = {"study": {"date": date.today().isoformat(), "n_photos": len(df), "n_wounds_photographed": int(df.wound.nunique()),
                     "photographers": sorted(df.photographer.unique().tolist()),
                     "excluded": df[df.excluded.notna()].groupby("excluded").size().to_dict(),
                     "model_versions": {k: v["version"] for k, v in an.seg.items()}},
           **summarise(df)}
    if a.sites:
        sites = pd.read_csv(a.sites).set_index("wound")["body_location"]
        df["body_location"] = df.wound.map(sites).fillna("unknown")
        res["by_site"] = {site: summarise(g)["area"] for site, g in df.groupby("body_location")}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out).with_suffix(".csv"), index=False)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))

    area = res["area"]
    if area is None:
        sys.exit("no wound has two usable photos: nothing to compute (see the excluded counts above)")
    print(f"\nArea noise: a wound must be more than {area['smaller_pct']}% smaller or {area['larger_pct']}% larger "
          f"to count as changed ({area['n_wounds']} wounds; RC 95% CI {area['rc_ci95']}).")
    if a.write:
        if area["n_wounds"] < a.min_wounds:
            sys.exit(f"not written: {area['n_wounds']} wounds with two usable photos, fewer than --min-wounds "
                     f"{a.min_wounds}; the band would be too uncertain")
        NOISE_FILE.write_text(json.dumps({"study": res["study"], "area": area, "tissue": res["tissue"],
                                          "by_site": res.get("by_site")}, indent=1, default=str) + "\n")
        print(f"wrote {NOISE_FILE}: healing now uses this band. Commit it and record it in docs/clinical_signoff.md (H1).")


if __name__ == "__main__":
    main()
