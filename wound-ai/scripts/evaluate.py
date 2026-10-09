"""Evaluate trained components on the held-out TEST split, the same way the app runs them.

    python scripts/evaluate.py --manifest data/manifest.csv --ckpt-dir checkpoints \
        --group-cols source,fitzpatrick,device --out reports/eval_test.json

Reports 95% bootstrap confidence intervals and breaks results down by subgroup
(skin tone, hospital, phone model). A model that is good on average but poor on
darker skin or on one hospital's phones is not ready.

Optional manifest column `area_cm2_manual` (clinician tracing) enables a
measurement-agreement check (bias and 95% limits of agreement).

Tissue (rows with `tissue_path`): Dice per class, the error in each tissue's share of the wound bed, and agreement
on the decision the care rules make from it (non-viable tissue at or above NONVIABLE_DEBRIDE_PCT). Compare these
with two clinicians' agreement on the same photos: the model is ready only if it is not clearly worse.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from wound_ai.data import IGNORE_INDEX, WOUND_BED, annotated_classes, crop_to_wound, read_mask, read_rgb
from wound_ai.measure import find_marker, measure_wound
from wound_ai.metrics import (bootstrap_ci, cohen_kappa, dice_iou, expected_calibration_error, macro_auroc, macro_f1,
                              per_class_sens_spec)
from wound_ai.pipeline import WoundAnalyzer


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--group-cols", default="source")
    p.add_argument("--use-gt-mask", action="store_true", help="crop with ground-truth masks instead of predictions")
    p.add_argument("--marker-mm", type=float, default=20.0)
    p.add_argument("--out", default="reports/eval.json")
    return p.parse_args()


def main():
    a = parse()
    df = pd.read_csv(a.manifest)
    df = df[df.split == a.split].reset_index(drop=True)
    an = WoundAnalyzer(a.ckpt_dir, marker_mm=a.marker_mm)
    groups = [g for g in a.group_cols.split(",") if g and g in df.columns]
    res = {"split": a.split, "n_images": len(df), "components": {}}

    preds_mask = {}
    if "boundary" in an.seg:
        rows = []
        for i, r in df.iterrows():
            mp = str(r.get("mask_path", "") or "")
            if not mp or mp == "nan":
                continue
            img = read_rgb(r.image_path)
            pm = an._segment(img, "boundary")
            preds_mask[i] = pm
            d, j = dice_iou(pm, read_mask(mp, True))
            row = {"dice": d, "iou": j, **{g: r[g] for g in groups}}
            if "area_cm2_manual" in df.columns and pd.notna(r.get("area_cm2_manual")):
                cal = find_marker(img, a.marker_mm)
                m = measure_wound(pm, cal) if cal else None
                row["area_pred"] = m.area_cm2 if m else np.nan
                row["area_manual"] = r["area_cm2_manual"]
            rows.append(row)
        s = pd.DataFrame(rows)
        out = {"n": len(s)}
        if len(s):
            out["dice"] = bootstrap_ci(np.mean, s.dice.values)
            out["iou"] = bootstrap_ci(np.mean, s.iou.values)
            out["by_group"] = {g: s.groupby(g).dice.agg(["mean", "count"]).round(3).to_dict("index") for g in groups}
        if "area_pred" in s and s.area_pred.notna().any():
            ok = s.dropna(subset=["area_pred", "area_manual"])
            diff = ok.area_pred - ok.area_manual
            out["area_agreement"] = {
                "n": len(ok), "bias_cm2": round(float(diff.mean()), 3),
                "loa_95_cm2": [round(float(diff.mean() - 1.96 * diff.std()), 3),
                               round(float(diff.mean() + 1.96 * diff.std()), 3)],
                "mean_abs_pct_error": round(float((diff.abs() / ok.area_manual).mean() * 100), 1),
            }
        res["components"]["boundary"] = out

    if "tissue" in an.seg:
        res["components"]["tissue"] = evaluate_tissue(an, df, preds_mask, groups, a.use_gt_mask)

    for name, c in an.cls.items():
        target = c["cfg"]["target"]
        if target not in df.columns:
            continue
        sub = df[df[target].notna() & df[target].astype(str).isin(c["classes"])]
        probs, ys, rows = [], [], []
        for i, r in sub.iterrows():
            img = read_rgb(r.image_path)
            mp = str(r.get("mask_path", "") or "")
            mask = read_mask(mp, True) if a.use_gt_mask and mp and mp != "nan" else preds_mask.get(i)
            _, p = an._classify(crop_to_wound(img, mask), name, r.to_dict(), return_probs=True)
            probs.append(p)
            ys.append(c["classes"].index(str(r[target])))
            rows.append({g: r[g] for g in groups})
        if not ys:
            continue
        P, Y = np.array(probs), np.array(ys)
        pred = P.argmax(1)
        g = pd.DataFrame(rows).assign(correct=(pred == Y))
        res["components"][name] = {
            "n": len(Y),
            "macro_f1": bootstrap_ci(macro_f1, Y, pred),
            "accuracy": bootstrap_ci(lambda y, p: float((y == p).mean()), Y, pred),
            "macro_auroc": macro_auroc(P, Y),
            "ece": expected_calibration_error(P, Y),
            "per_class": per_class_sens_spec(Y, pred, c["classes"]),
            "accuracy_by_group": {gc: g.groupby(gc).correct.agg(["mean", "count"]).round(3).to_dict("index")
                                  for gc in groups},
        }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def evaluate_tissue(an: WoundAnalyzer, df: pd.DataFrame, preds_mask: dict, groups: list[str], use_gt: bool) -> dict:
    from wound_ai.care import NONVIABLE_DEBRIDE_PCT

    classes = an.tissue_classes()
    bed_ids = [classes.index(c) for c in WOUND_BED if c in classes]
    nonviable_ids = [classes.index(c) for c in ("slough", "necrosis") if c in classes]
    per_class = {c: [] for c in classes[1:]}
    photos = {c: 0 for c in classes[1:]}  # test photos that contain each class
    rows, n = [], 0
    for i, r in df.iterrows():
        tp, mp = str(r.get("tissue_path", "") or ""), str(r.get("mask_path", "") or "")
        if not tp or tp == "nan":
            continue
        img, gt = read_rgb(r.image_path), read_mask(tp, binary=False)
        gt_bed = np.isin(gt, bed_ids)
        mask = read_mask(mp, True) if use_gt and mp and mp != "nan" else preds_mask.get(i)
        if mask is None:
            mask = gt_bed.astype(np.uint8)  # no boundary model or mask: crop with the annotated wound bed
        pred, _ = an._tissue_map(img, mask)
        n += 1
        valid = gt != IGNORE_INDEX
        # Each dataset labels some classes only: a class is scored only on photos whose dataset labels it.
        labelled = annotated_classes(r.get("tissue_classes")).tolist()
        for k, c in enumerate(classes[1:], start=1):
            if k >= len(labelled) or not labelled[k]:
                continue
            p, t = (pred == k) & valid, gt == k
            photos[c] += int(t.any())
            if p.any() or t.any():
                per_class[c].append(2 * float((p & t).sum()) / float(p.sum() + t.sum()))
        if not all(labelled[classes.index(c)] for c in ("slough", "necrosis") if c in classes):
            continue  # the non-viable share needs both labelled
        # Shares of the wound bed: annotated bed and annotated tissue against the app's outline and prediction.
        gt_nv = 100 * np.isin(gt[gt_bed], nonviable_ids).mean() if gt_bed.any() else np.nan
        pr_bed = pred[mask > 0]
        pr_bed = pr_bed[np.isin(pr_bed, bed_ids)]
        pr_nv = 100 * np.isin(pr_bed, nonviable_ids).mean() if len(pr_bed) else np.nan
        rows.append({"nonviable_gt": gt_nv, "nonviable_pred": pr_nv, **{g: r[g] for g in groups}})
    s = pd.DataFrame(rows).dropna(subset=["nonviable_gt", "nonviable_pred"]) if rows else pd.DataFrame()
    out = {"n": n, "dice_per_class": {c: bootstrap_ci(np.mean, np.array(v)) for c, v in per_class.items() if v},
           "photos_per_class": {c: n for c, n in photos.items() if n}}
    if len(s):
        s = s.assign(err=(s.nonviable_pred - s.nonviable_gt).abs(),
                     debride_gt=s.nonviable_gt >= NONVIABLE_DEBRIDE_PCT, debride_pred=s.nonviable_pred >= NONVIABLE_DEBRIDE_PCT)
        out["nonviable_abs_error_points"] = bootstrap_ci(np.mean, s.err.values)
        out["debridement_threshold"] = {"agreement": round(float((s.debride_gt == s.debride_pred).mean()), 3),
                                        "kappa": cohen_kappa(s.debride_gt.values.astype(int), s.debride_pred.values.astype(int))}
        out["nonviable_error_by_group"] = {g: s.groupby(g).err.agg(["mean", "count"]).round(1).to_dict("index") for g in groups}
    return out


if __name__ == "__main__":
    main()
