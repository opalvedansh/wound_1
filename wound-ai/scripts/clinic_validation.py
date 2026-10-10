"""How well the model agrees with the clinic's own nurses on the clinic's own photos.

Input: the de-identified validation export from the web portal (Admin: GET /exports/validation, one row per analysed
visit: the nurse's assessment next to what the model found in the same before-treatment photo).

    python scripts/clinic_validation.py wound-validation-2026-10-10-deid.csv --out reports/clinic_validation

For each finding the nurse also records, it reports agreement with a 95% bootstrap CI and Cohen's kappa:
wound type, pressure-injury stage, burn depth, Wagner grade, each wound-bed tissue present/absent, periwound redness
and maceration; plus how often clinicians approved, edited or rejected the drafts, and agreement by body site.
Agreement is with one nurse, not with a gold standard: two clinicians also disagree (see the LUTSeg rater figures).
Writes OUT/clinic_validation.json and OUT/clinic_validation.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from wound_ai.metrics import bootstrap_ci, cohen_kappa, macro_f1, per_class_sens_spec

# Nurse answers (packages/domain questions.ts) -> the model's labels. None: the model has no such class.
WOUND_TYPE = {"Pressure Ulcer": "pressure", "DFU": "diabetic", "VLU": "venous", "Surgical Wound": "surgical",
              "Burn": "burn", "Traumatic": "other", "Other": "other", "Arterial Ulcer": None}
PU_STAGE = {"Stage 1": "stage_1", "Stage 2": "stage_2", "Stage 3": "stage_3", "Stage 4": "stage_4",
            "Unstageable": "unstageable", "Deep tissue injury": "deep_tissue_injury"}
# The model can't tell superficial from deep partial thickness (its public training labels don't).
BURN_DEPTH = {"Superficial": "superficial", "Superficial partial": "partial_thickness",
              "Deep partial": "partial_thickness", "Full thickness": "full_thickness"}
WAGNER = {"Grade 0": "grade_0", "Grade 1": "grade_1", "Grade 2": "grade_2", "Grade 3": "grade_3"}
TISSUE = {"Granulation": "granulation", "Slough": "slough", "Necrosis/Eschar": "necrosis", "Epithelial": "epithelial",
          "Exposed bone/tendon": "exposed_structure"}
TISSUE_PRESENT_PCT = 5  # the model "sees" a tissue if it covers at least this share of the wound bed
# Same thresholds as the care rules (docs/clinical_signoff.md 1.3 and 1.4).
ERYTHEMA_FRAC, MACERATION_FRAC = 0.30, 0.20


def split(value) -> list[str]:
    return [x for x in str(value).split(";") if x and x != "nan"] if pd.notna(value) else []


def loads(value) -> dict:
    try:
        return json.loads(value) if isinstance(value, str) and value else {}
    except json.JSONDecodeError:
        return {}


def agreement(nurse: pd.Series, model: pd.Series, groups: pd.Series | None = None) -> dict | None:
    """Accuracy (95% CI), kappa, macro-F1 and per-class sensitivity where both have an answer."""
    ok = nurse.notna() & model.notna()
    if ok.sum() < 2:
        return {"n": int(ok.sum())} if ok.any() else None
    n, m = nurse[ok].astype(str), model[ok].astype(str)
    classes = sorted(set(n) | set(m))
    y, p = np.array([classes.index(x) for x in n]), np.array([classes.index(x) for x in m])
    out = {"n": int(ok.sum()),
           "accuracy": bootstrap_ci(lambda a, b: float((a == b).mean()), y, p),
           "kappa": round(cohen_kappa(y, p), 3) if len(set(y) | set(p)) > 1 else None,
           "macro_f1": round(macro_f1(y, p), 3),
           "per_class": {c: v for c, v in per_class_sens_spec(y, p, classes).items() if v["support"]},
           "confusion": pd.crosstab(n.rename("nurse"), m.rename("model")).to_dict()}
    if groups is not None:
        g = pd.DataFrame({"group": groups[ok].fillna("unknown"), "correct": (y == p)})
        out["accuracy_by_group"] = g.groupby("group").correct.agg(["mean", "count"]).round(3).to_dict("index")
    return out


def presence(nurse: pd.Series, model: pd.Series) -> dict | None:
    """Present/absent agreement: sensitivity, specificity (model vs nurse) and kappa."""
    if len(nurse) < 2:
        return None
    y, p = nurse.astype(int).to_numpy(), model.astype(int).to_numpy()
    tp, fn = int((y & p).sum()), int((y & (1 - p)).sum())
    tn, fp = int(((1 - y) & (1 - p)).sum()), int(((1 - y) & p).sum())
    return {"n": len(y), "nurse_present": int(y.sum()), "model_present": int(p.sum()),
            "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
            "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
            "agreement": bootstrap_ci(lambda a, b: float((a == b).mean()), y, p),
            "kappa": round(cohen_kappa(y, p), 3) if len(set(y) | set(p)) > 1 else None}


def evaluate(df: pd.DataFrame) -> dict:
    sev = df.model_severity.map(loads)
    tissue = df.model_tissue_pct.map(loads)
    untrusted = df.model_tissue_untrusted.map(split)
    res: dict = {"visits": len(df), "patients": int(df.patient_code.nunique()), "wounds": int(df.case_id.nunique())}

    nurse_type = df.nurse_wound_type.map(WOUND_TYPE)
    res["arterial_ulcers_not_scored"] = int((df.nurse_wound_type == "Arterial Ulcer").sum())
    res["wound_type"] = agreement(nurse_type, df.model_wound_type, df.wound_location)

    for key, nurse_col, mapping in (("pu_stage", "nurse_pressure_stage", PU_STAGE),
                                    ("burn_depth", "nurse_burn_depth", BURN_DEPTH),
                                    ("dfu_wagner", "nurse_wagner_grade", WAGNER)):
        model = sev.map(lambda s, k=key: (s.get(k) or {}).get("label"))
        res[key] = agreement(df[nurse_col].map(mapping), model)

    answered = df.nurse_wound_bed_tissue.map(split).map(bool)
    res["tissue_present"] = {}
    for option, cls in TISSUE.items():
        nurse = df.nurse_wound_bed_tissue[answered].map(lambda v, o=option: o in split(v))
        model = [(t.get(cls, 0) >= TISSUE_PRESENT_PCT) or (cls in u) for t, u in zip(tissue[answered], untrusted[answered])]
        res["tissue_present"][cls] = presence(nurse, pd.Series(model, index=nurse.index))

    peri = df.nurse_periwound.notna()
    erythema = df.model_periwound_erythema_frac[peri].notna()
    res["periwound_erythema"] = presence(
        (df.nurse_periwound[peri][erythema] == "Erythematous"),
        df.model_periwound_erythema_frac[peri][erythema] >= ERYTHEMA_FRAC) if erythema.any() else None
    macerated = df.model_periwound_maceration_frac[peri].notna()
    res["periwound_maceration"] = presence(
        (df.nurse_periwound[peri][macerated] == "Macerated") | (df.nurse_edge[peri][macerated] == "Macerated"),
        df.model_periwound_maceration_frac[peri][macerated] >= MACERATION_FRAC) if macerated.any() else None

    res["review"] = df.review.fillna("pending").value_counts().to_dict()
    fields = [k for c in df.corrections.map(loads) for k in c]
    res["corrected_fields"] = pd.Series(fields, dtype=str).value_counts().to_dict()
    res["model_versions"] = df.model_versions.dropna().value_counts().head(5).to_dict()
    return res


def ci(v) -> str:
    return "—" if not v else f"{v[0]:.0%} ({v[1]:.0%}–{v[2]:.0%})"


def markdown(res: dict) -> str:
    md = ["# Model vs nurse on the clinic's photos", "",
          f"{res['visits']} analysed visits, {res['wounds']} wounds, {res['patients']} patients. Agreement with one nurse's "
          "assessment of the same visit, 95% CI in brackets; kappa: 0 = chance, 1 = perfect.", "",
          "| Finding | Visits with both | Agreement | Kappa | Macro-F1 |", "| --- | --- | --- | --- | --- |"]
    for key, name in (("wound_type", "Wound type"), ("pu_stage", "Pressure-injury stage"), ("burn_depth", "Burn depth"),
                      ("dfu_wagner", "Wagner grade")):
        v = res.get(key) or {}
        md.append(f"| {name} | {v.get('n', 0)} | {ci(v.get('accuracy'))} | {v.get('kappa', '—')} | {v.get('macro_f1', '—')} |")
    md += ["", "| Present/absent | Visits | Nurse saw | Model saw | Sensitivity | Specificity | Agreement | Kappa |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    rows = [*((f"Tissue: {c}", v) for c, v in res["tissue_present"].items()),
            ("Periwound redness", res["periwound_erythema"]), ("Periwound maceration", res["periwound_maceration"])]
    for name, v in rows:
        if v:
            md.append(f"| {name} | {v['n']} | {v['nurse_present']} | {v['model_present']} | {v['sensitivity']} | "
                      f"{v['specificity']} | {ci(v['agreement'])} | {v['kappa']} |")
    md += ["", f"Clinician review of the drafts: {res['review']}. Fields corrected: {res['corrected_fields'] or 'none'}.",
           f"Arterial ulcers (no model class, not scored): {res['arterial_ulcers_not_scored']}."]
    by_site = (res.get("wound_type") or {}).get("accuracy_by_group")
    if by_site:
        md += ["", "Wound-type agreement by body site: " + ", ".join(f"{k} {v['mean']:.0%} (n={v['count']})"
                                                                     for k, v in by_site.items())]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="the validation export (GET /exports/validation)")
    ap.add_argument("--out", default="reports/clinic_validation")
    a = ap.parse_args()
    res = evaluate(pd.read_csv(a.csv))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "clinic_validation.json").write_text(json.dumps(res, indent=1, default=str))
    md = markdown(res)
    (out / "clinic_validation.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
