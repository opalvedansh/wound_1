"""Care suggestions (wound_ai/care.py). The rules are placeholders for clinical sign-off; these tests pin what
they do so a change to a rule is always a deliberate one.

    .venv/bin/python -m pytest tests
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wound_ai.care import treatment_report  # noqa: E402


def obs(tissue: dict | None = None, area: float | None = 4.0, day: int = 0, **extra):
    return {"taken_at": f"2026-01-{1 + day:02d}T10:00:00+00:00", "status": "ok", "area_cm2": area,
            "length_cm": 2.0, "width_cm": 2.0, "perimeter_cm": 7.0, "tissue_pct": tissue, "flags": [], **extra}


def report(wound_type: str = "venous", intake: dict | None = None, assessment: dict | None = None,
           pre: dict | None = None, post: dict | None = None, history: list | None = None, **cur):
    current = {"sequence": len(history or []) + 1, "pre": pre or obs({"granulation": 100}), "post": post,
               "assessment": {"exudate_level": "Scant"} if assessment is None else assessment, **cur}
    return treatment_report({"wound_type": {"label": wound_type, "prob": 0.9, "top": []},
                             "intake": intake or {}, "treatments": [*(history or []), current]})


def actions(r: dict) -> list[str]:
    return [s["action"] for s in r["suggestions"]]


def avoided(r: dict) -> dict[str, str]:
    return {c["action"]: c["reason"] for c in r["contraindications"]}


def test_venous_ulcer_without_abpi_gets_no_compression_and_a_check():
    r = report("venous", {"body_location": "lower_leg"})
    assert "Compression" not in actions(r)
    assert any(c["rule_id"] == "R0" for c in r["checks"])


def test_venous_ulcer_with_normal_abpi_gets_compression():
    assert "Compression" in actions(report("venous", {"body_location": "lower_leg", "abpi": 1.0}))


def test_compression_on_the_plan_with_low_abpi_is_listed_to_avoid():
    r = report("venous", {"body_location": "lower_leg", "abpi": 0.6}, therapy=["Compression"])
    assert "Compression" not in actions(r)
    assert "below" in avoided(r)["Compression"]


def test_heavy_exudate_suggests_alginate_or_foam_and_rules_out_hydrocolloid():
    r = report("surgical", assessment={"exudate_level": "Heavy"}, dressing="Hydrocolloid")
    s = next(s for s in r["suggestions"] if s["rule_id"] == "M1")
    assert s["action"] == "Alginate" and s["alternatives"] == ["Foam"]
    assert "Hydrocolloid" in avoided(r)


def test_slough_suggests_debridement_with_the_reason():
    r = report("pressure", pre=obs({"granulation": 50, "slough": 50}))
    s = next(s for s in r["suggestions"] if s["action"] == "Debridement")
    assert s["rule_id"] == "T1" and "50%" in s["because"][0]


def test_residual_slough_after_the_session_suggests_continuing_next_visit():
    r = report("pressure", pre=obs({"slough": 80, "granulation": 20}), post=obs({"slough": 40, "granulation": 60}))
    assert any(s["rule_id"] == "T2" for s in r["suggestions"])
    assert "## This visit" in r["report_markdown"]


def test_poor_blood_flow_blocks_debridement_hydrogel_and_npwt():
    r = report("diabetic", {"body_location": "toe", "diabetes": "yes", "foot_cold_or_dark": "yes"},
               pre=obs({"necrosis": 60, "granulation": 40}))
    assert not {"Debridement", "Hydrogel"} & set(actions(r))
    assert "blood flow" in avoided(r)["Debridement"]


def test_dry_stable_heel_eschar_is_left_alone():
    r = report("pressure", {"body_location": "heel"}, pre=obs({"necrosis": 80, "granulation": 20}))
    assert "Debridement" not in actions(r)
    assert "heel eschar" in avoided(r)["Debridement"]


def test_diabetic_foot_gets_offloading_and_pressure_injury_redistribution():
    assert "Offloading" in actions(report("diabetic", {"body_location": "foot_plantar", "abpi": 1.0}))
    assert "Pressure redistribution" in actions(report("pressure", {"body_location": "sacrum_buttock"}))


def test_infection_signs_suggest_silver_and_an_assessment():
    r = report("surgical", assessment={"exudate_level": "Moderate", "infection_signs": ["Erythema", "Malodor"]})
    assert "Silver" in actions(r) and any(c["rule_id"] == "I2" for c in r["checks"])


def test_stalled_wound_behind_benchmark_suggests_advanced_therapy_unless_contraindicated():
    hist = [{"sequence": 1, "pre": obs({"granulation": 100}, 10.0, 0)}, {"sequence": 2, "pre": obs({"granulation": 100}, 9.5, 14)}]
    r = report("venous", {"body_location": "lower_leg", "abpi": 1.0}, pre=obs({"granulation": 100}, 9.0, 28), history=hist)
    s = next(s for s in r["suggestions"] if s["rule_id"] == "E2")
    assert s["action"] == "Negative Pressure (NPWT)"
    r = report("venous", {"body_location": "lower_leg", "abpi": 1.0}, pre=obs({"granulation": 90, "necrosis": 10}, 9.0, 28),
               history=hist)
    s = next(s for s in r["suggestions"] if s["rule_id"] == "E2")
    assert s["action"] == "Skin Substitute"  # NPWT ruled out by the necrosis


def test_same_dressing_for_three_visits_without_improvement_asks_for_a_plan_review():
    hist = [{"pre": obs({"granulation": 100}, 10.0, d), "dressing": "Foam"} for d in (0, 7)]
    r = report("surgical", pre=obs({"granulation": 100}, 10.0, 14), history=hist, dressing="Foam")
    assert any(c["rule_id"] == "E3" for c in r["checks"])


def test_missing_tissue_and_exudate_become_checks_not_guesses():
    r = report("surgical", pre=obs(None), assessment={})
    ids = {c["rule_id"] for c in r["checks"]}
    assert {"T0", "M0"} <= ids and not r["suggestions"]


def test_danger_signs_put_urgent_care_first():
    pre = obs({"granulation": 100}, flags=[{"level": "urgent", "text": "Danger sign: test"}])
    r = report("surgical", pre=pre)
    assert r["checks"][0]["rule_id"] == "D0"
    assert r["flags"][0]["level"] == "urgent"
    assert "Emergency care now" in r["report_markdown"]


def test_report_states_depth_redness_and_a_later_after_photo():
    red = {"delta_a": 9.0, "level": "marked", "white_balanced": True}
    r = report(pre=obs({"granulation": 100}, 10.0, periwound_redness=red), assessment={"depth_cm": 0.8})
    md = r["report_markdown"]
    assert "Depth, probed by the clinician: 0.8 cm" in md and "Redness around the wound" in md and "marked" in md
    assert "Depth: not recorded" in report()["report_markdown"]
    later = report(pre=obs({"granulation": 100}, 10.0), post=obs({"granulation": 100}, 6.0, day=9))
    assert "## This treatment (before → after, days apart)" in later["report_markdown"]
    assert "After treatment: area 6.0 cm²" in later["report_markdown"]
    assert later["progress"]["response"]["trajectory"] == "improving"


def test_report_names_rules_version_and_never_prescribes():
    md = report("diabetic", {"body_location": "heel", "diabetes": "yes"}, pre=obs({"slough": 60, "granulation": 40}))["report_markdown"]
    assert "Rules version: care-" in md
    for word in ("prescribe", "antibiotic", "mg"):
        assert word not in md.lower()


def test_a_dressing_that_has_not_helped_goes_behind_its_alternative():
    hist = [{"pre": obs({"granulation": 100}, 10.0, d), "dressing": "Foam"} for d in (0, 7)]
    r = report("surgical", pre=obs({"granulation": 100}, 10.0, 14), history=hist, dressing="Foam",
               assessment={"exudate_level": "Moderate"})
    s = next(s for s in r["suggestions"] if s["rule_id"] == "M2")
    assert s["action"] == "Alginate" and s["alternatives"] == ["Foam"]
