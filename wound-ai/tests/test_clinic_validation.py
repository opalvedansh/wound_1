"""clinic_validation.py on synthetic rows shaped like the API's validation export."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

from clinic_validation import evaluate, markdown  # noqa: E402


def row(i: int, nurse_type: str, model_type: str, **kw) -> dict:
    return {"patient_code": f"P{i % 3}", "case_id": f"c{i}", "visit": "T1", "visit_month": "2026-10",
            "wound_location": kw.get("site", "heel"), "nurse_wound_type": nurse_type,
            "nurse_wound_bed_tissue": kw.get("bed", "Granulation;Slough"), "nurse_pressure_stage": kw.get("stage"),
            "nurse_burn_depth": kw.get("depth"), "nurse_wagner_grade": None, "nurse_periwound": kw.get("peri", "Healthy"),
            "nurse_edge": "Well-defined", "nurse_infection_signs": "", "nurse_exudate_level": "Moderate",
            "model_wound_type": model_type, "model_wound_type_prob": 0.8,
            "model_severity": json.dumps(kw.get("sev", {})),
            "model_tissue_pct": json.dumps(kw.get("tissue", {"granulation": 70, "slough": 30})),
            "model_tissue_untrusted": kw.get("untrusted", ""), "model_periwound_erythema_frac": kw.get("ery", 0.05),
            "model_periwound_maceration_frac": 0.0, "model_periwound_callus_frac": 0.0, "area_cm2": 3.0,
            "review": kw.get("review", "approved"), "corrections": json.dumps(kw.get("corr", {})),
            "model_versions": '{"wound_type": "v1"}', "flags_version": "flags-0.1-unsigned"}


def frame() -> pd.DataFrame:
    rows = [row(0, "Pressure Ulcer", "pressure", stage="Stage 3", sev={"pu_stage": {"label": "stage_3"}}),
            row(1, "Pressure Ulcer", "pressure", stage="Stage 2", sev={"pu_stage": {"label": "stage_3"}}),
            row(2, "DFU", "diabetic", site="toe", bed="Granulation;Necrosis/Eschar", untrusted="necrosis"),
            row(3, "VLU", "pressure", site="lower_leg", peri="Erythematous", ery=0.4, review="edited",
                corr={"wound_type": "venous"}),
            row(4, "Burn", "burn", site="hand", depth="Deep partial", sev={"burn_depth": {"label": "partial_thickness"}}),
            row(5, "Arterial Ulcer", "venous", site="lower_leg")]
    # pandas reads the CSV's empty cells as NaN: the same here
    return pd.DataFrame(rows).replace({"": None})


def test_wound_type_agreement_maps_nurse_answers_and_skips_arterial():
    res = evaluate(frame())
    wt = res["wound_type"]
    assert wt["n"] == 5 and res["arterial_ulcers_not_scored"] == 1
    assert wt["accuracy"][0] == 0.8  # the venous ulcer called pressure is the one miss
    assert wt["accuracy_by_group"]["lower_leg"]["mean"] == 0.0


def test_severity_and_presence_are_scored_where_the_nurse_answered():
    res = evaluate(frame())
    assert res["pu_stage"]["n"] == 2 and res["pu_stage"]["accuracy"][0] == 0.5
    assert res["burn_depth"]["n"] == 1  # deep partial counts as partial thickness: agreement
    nec = res["tissue_present"]["necrosis"]
    assert nec["nurse_present"] == 1 and nec["model_present"] == 1 and nec["sensitivity"] == 1.0  # via "possibly also"
    assert res["periwound_erythema"]["sensitivity"] == 1.0 and res["periwound_erythema"]["specificity"] == 1.0
    assert res["review"] == {"approved": 5, "edited": 1} and res["corrected_fields"] == {"wound_type": 1}


def test_markdown_has_every_section():
    md = markdown(evaluate(frame()))
    for text in ("Wound type", "Pressure-injury stage", "Tissue: necrosis", "Periwound redness", "by body site"):
        assert text in md
