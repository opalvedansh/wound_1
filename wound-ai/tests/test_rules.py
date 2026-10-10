"""The flowchart rules (docs/wound_flowcharts.md), one test each. Thresholds are placeholders for clinical sign-off.

    .venv/bin/python -m pytest tests
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wound_ai.pipeline import apply_diabetic_foot_rule  # noqa: E402
from wound_ai.report import build_report, red_flags  # noqa: E402


def flags(intake: dict, **findings) -> list[dict]:
    return red_flags({"intake": intake, "measurement": {"area_cm2": 2.0}, **findings})


def texts(fl: list[dict], level: str | None = None) -> list[str]:
    return [f["text"] for f in fl if level is None or f["level"] == level]


# --- Chart 1: danger signs first ---------------------------------------------------------------

def test_fever_with_spreading_redness_is_a_danger_sign():
    urgent = texts(flags({"fever": "yes", "redness_spreading": "yes"}), "urgent")
    assert any(t.startswith("Danger sign") and "spreading infection" in t for t in urgent)


def test_spreading_redness_without_fever_needs_review_not_emergency():
    fl = flags({"fever": "no", "redness_spreading": "yes"})
    assert not texts(fl, "urgent")
    assert any("within 24 hours" in t for t in texts(fl, "review"))


def test_dark_or_cold_toes_are_a_danger_sign_with_or_without_diabetes():
    for diabetes in ("yes", "no"):
        urgent = texts(flags({"diabetes": diabetes, "foot_cold_or_dark": "yes"}), "urgent")
        assert any("gangrene" in t for t in urgent), diabetes


def test_chemical_or_electrical_burns_are_danger_signs():
    for agent in ("chemical", "electrical"):
        assert any("burns unit" in t for t in texts(flags({"cause": "burn", "burn_agent": agent}), "urgent"))


def test_burns_on_face_hands_or_feet_go_to_a_burns_unit():
    for site in ("head_neck", "hand", "heel"):
        assert any("face, neck, hands or feet" in t for t in texts(flags({"cause": "burn", "body_location": site}), "urgent"))
    assert not any("face, neck" in t for t in texts(flags({"cause": "burn", "body_location": "thigh"})))


def test_burns_on_several_areas_are_a_danger_sign():
    assert any("more than one body area" in t for t in texts(flags({"cause": "burn", "burn_other_sites": "yes"}), "urgent"))


def test_burn_depth_full_thickness_is_urgent_partial_thickness_needs_review():
    def depth(label):
        return flags({"cause": "burn"}, severity={"burn_depth": {"label": label, "prob": 0.9}})

    assert any("deep burn" in t for t in texts(depth("full_thickness"), "urgent"))
    partial = depth("partial_thickness")
    assert not any("deep burn" in t for t in texts(partial, "urgent"))
    assert any("superficial or deep partial" in t for t in texts(partial, "review"))
    assert not any("burn" in t.lower() and "depth" in t for t in texts(depth("superficial")))


def test_deep_pressure_injury_stages_and_wagner_3_need_review():
    for stage in ("stage_3", "stage_4", "unstageable", "deep_tissue_injury"):
        assert any("pressure injury" in t for t in texts(flags({}, severity={"pu_stage": {"label": stage}}), "review")), stage
    for stage in ("stage_1", "stage_2"):
        assert not any("pressure injury" in t for t in texts(flags({}, severity={"pu_stage": {"label": stage}}))), stage
    assert any("Wagner grade 3" in t for t in texts(flags({}, severity={"dfu_wagner": {"label": "grade_3"}}), "review"))
    assert not any("Wagner" in t for t in texts(flags({}, severity={"dfu_wagner": {"label": "grade_2"}})))


def test_report_leads_with_emergency_care_when_a_danger_sign_fires():
    report, _ = build_report({"intake": {"fever": "yes", "redness_spreading": "yes"}})
    assert report.splitlines()[3].startswith("**Emergency care now")
    calm, _ = build_report({"intake": {"fever": "no"}})
    assert "Emergency care now" not in calm


# --- Chart 1: any foot wound with diabetes is a diabetic foot ulcer -----------------------------

def test_diabetes_and_a_foot_site_make_it_a_diabetic_foot_ulcer_whatever_the_model_says():
    guess = {"label": "pressure", "prob": 0.81, "top": [("pressure", 0.81), ("diabetic", 0.12)]}
    wt = apply_diabetic_foot_rule(guess, {"diabetes": "yes", "body_location": "heel"})
    assert wt["label"] == "diabetic" and wt["rule"] and wt["model"] == guess


def test_the_rule_works_before_any_classifier_is_trained():
    wt = apply_diabetic_foot_rule(None, {"diabetes": "yes", "body_location": "toe"})
    assert wt == {"label": "diabetic", "prob": None, "top": [], "rule": "diabetes and a foot location", "model": None}


def test_no_rule_without_diabetes_or_off_the_foot():
    guess = {"label": "pressure", "prob": 0.81, "top": []}
    assert apply_diabetic_foot_rule(guess, {"diabetes": "no", "body_location": "heel"}) is guess
    assert apply_diabetic_foot_rule(guess, {"diabetes": "yes", "body_location": "sacrum_buttock"}) is guess
    assert apply_diabetic_foot_rule(None, {"diabetes": "not_sure", "body_location": "heel"}) is None


def test_a_rule_answer_is_not_reported_as_an_uncertain_model():
    fl = flags({"diabetes": "yes", "body_location": "heel"},
               wound_type=apply_diabetic_foot_rule(None, {"diabetes": "yes", "body_location": "heel"}))
    assert not any("uncertain" in t for t in texts(fl))
    report, _ = build_report({"intake": {"diabetes": "yes", "body_location": "heel"},
                              "wound_type": {"label": "diabetic", "prob": None, "top": [], "rule": "diabetes and a foot location"}})
    assert "Diabetic foot ulcer (by rule: diabetes and a foot location)" in report


# --- Charts 2 and 3: blood flow is never assumed ----------------------------------------------

def test_leg_and_foot_ulcers_say_blood_flow_was_not_assessed():
    for site in ("lower_leg", "ankle", "heel"):
        assert any("Blood flow not assessed" in t for t in texts(flags({"cause": "started_on_its_own", "body_location": site})))


def test_no_blood_flow_flag_away_from_the_legs_or_for_a_fresh_injury():
    assert not any("Blood flow" in t for t in texts(flags({"cause": "pressure_lying_or_sitting", "body_location": "sacrum_buttock"})))
    assert not any("Blood flow" in t for t in texts(flags({"cause": "injury_cut_or_fall", "body_location": "lower_leg"})))


def test_a_diabetic_foot_always_gets_the_blood_flow_check_even_after_an_injury():
    fl = flags({"cause": "injury_cut_or_fall", "diabetes": "yes", "body_location": "toe"}, wound_type={"label": "diabetic", "prob": None, "top": []})
    assert any("Blood flow not assessed" in t for t in texts(fl))


def test_an_entered_abpi_replaces_the_flag_and_low_or_high_values_are_flagged():
    base = {"cause": "started_on_its_own", "body_location": "lower_leg"}
    assert not any("ABPI" in t or "Blood flow" in t for t in texts(flags({**base, "abpi": 1.0})))
    assert any("Vascular review" in t for t in texts(flags({**base, "abpi": "0.6"}), "urgent"))
    assert any("calcified" in t for t in texts(flags({**base, "abpi": 1.5}), "review"))
