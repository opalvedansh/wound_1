"""Healing and session effect (wound_ai/progress.py). Thresholds are placeholders for clinical sign-off.

    .venv/bin/python -m pytest tests
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wound_ai.progress import healing, observation, progress, push_score, session_effect  # noqa: E402
from wound_ai.report import red_flags  # noqa: E402


def obs(day: int, area: float | None = None, tissue: dict | None = None, status: str = "ok", perimeter: float = 6.0):
    o = {"taken_at": f"2026-01-{1 + day:02d}T10:00:00+00:00", "status": status, "area_cm2": area,
         "length_cm": 3.0 if area else None, "width_cm": 2.0 if area else None,
         "perimeter_cm": perimeter if area else None, "tissue_pct": tissue, "flags": []}
    return o


def test_debridement_that_enlarges_the_wound_is_expected_not_worse():
    pre = obs(0, 4.0, {"granulation": 40, "slough": 60})
    post = obs(0, 5.0, {"granulation": 90, "slough": 10})
    p = progress([{"pre": pre, "post": post}], "venous")
    assert p["session"]["nonviable_removed_points"] == 50
    assert "expected" in p["session"]["area_note"]
    assert p["healing"]["comparable"] is False  # one visit: nothing to compare yet
    assert not p["flags"]


def test_healing_compares_post_with_post_never_pre_with_post():
    t1 = {"pre": obs(0, 9.0), "post": obs(0, 10.0)}
    t2 = {"pre": obs(14, 8.0), "post": obs(14, 7.0)}
    h = healing([t1, t2])
    assert h["phase"] == "post"
    assert h["since_last"]["percent_area_reduction"] == 30.0  # 10 -> 7, not 9 -> 7 or 10 -> 8
    assert h["trajectory"] == "improving" and h["basis"] == "area"
    assert h["since_last"]["cm2_per_week"] == 1.5
    assert h["since_last"]["edge_advance_cm_per_week"] == 0.25


def test_post_photo_days_later_is_the_response_to_the_treatment():
    t1 = {"pre": obs(0, 10.0)}
    t2 = {"pre": obs(7, 9.0), "post": obs(16, 6.0)}  # the post photo 9 days after the pre photo
    p = progress([t1, t2], "venous")
    assert p["session"] is None  # not a cleaning effect
    r = p["response"]
    assert r["days"] == 9.0 and r["percent_area_reduction"] == 33.3 and r["trajectory"] == "improving"
    h = p["healing"]
    assert h["phase"] == "pre" and h["n_photos"] == 3  # the later photo joins the as-found series
    assert h["since_last"]["percent_area_reduction"] == 33.3 and h["since_first"]["percent_area_reduction"] == 40.0
    worse = progress([t1, {"pre": obs(7, 9.0), "post": obs(16, 12.0)}], "venous")
    assert worse["response"]["trajectory"] == "deteriorating"
    assert any("increased" in f["text"] for f in worse["flags"])


def test_same_visit_post_photo_is_never_a_response():
    p = progress([{"pre": obs(0, 9.0), "post": obs(0, 10.0)}], "venous")
    assert p["response"] is None and p["session"]["area_after_cm2"] == 10.0


def test_first_after_cleaning_photo_falls_back_to_the_as_found_photos():
    h = healing([{"pre": obs(0, 10.0)}, {"pre": obs(7, 6.0), "post": obs(7, 7.0)}])
    assert h["phase"] == "pre" and h["since_last"]["percent_area_reduction"] == 40.0  # 10 -> 6, never 10 -> 7


def test_depth_is_the_clinicians_and_shown_beside_the_last_one():
    ts = [{"pre": obs(0, 10.0), "assessment": {"depth_cm": 1.2}}, {"pre": obs(7, 9.0), "assessment": {}},
          {"pre": obs(14, 8.0), "assessment": {"depth_cm": 0.8}}]
    assert progress(ts)["depth"] == {"depth_cm": 0.8, "previous_cm": 1.2}
    assert progress(ts[:1])["depth"] == {"depth_cm": 1.2}
    assert progress(ts[:2])["depth"] is None  # not probed at this visit


def test_without_a_post_photo_the_pre_series_is_used():
    h = healing([{"pre": obs(0, 10.0)}, {"pre": obs(7, 6.0), "post": obs(7, status="retake")}])
    assert h["phase"] == "pre" and h["since_last"]["percent_area_reduction"] == 40.0


def test_no_sticker_means_no_area_comparison_but_tissue_still_compares():
    h = healing([{"pre": obs(0, None, {"granulation": 30, "slough": 70})},
                 {"pre": obs(7, None, {"granulation": 80, "slough": 20})}])
    assert "percent_area_reduction" not in h["since_last"]
    assert h["trajectory"] == "improving" and h["basis"] == "tissue"
    h = healing([{"pre": obs(0)}, {"pre": obs(7)}])
    assert h["comparable"] is False and "sticker" in h["reason"]


def test_change_inside_measurement_noise_is_static():
    for after in (9.0, 11.0):  # 10% smaller or larger: inside the 15% band
        h = healing([{"pre": obs(0, 10.0)}, {"pre": obs(7, after)}])
        assert h["trajectory"] == "static", after
    assert healing([{"pre": obs(0, 10.0)}, {"pre": obs(7, 12.0)}])["trajectory"] == "deteriorating"


def test_dfu_behind_the_4_week_benchmark_is_flagged():
    ts = [{"pre": obs(0, 10.0)}, {"pre": obs(14, 9.0)}, {"pre": obs(28, 7.0)}]
    p = progress(ts, "diabetic")
    fw = p["healing"]["four_week"]
    assert fw == {"days": 28.0, "percent_area_reduction": 30.0, "target": 50.0, "on_track": False}
    assert any("4 weeks" in f["text"] for f in p["flags"])
    assert progress(ts, "venous")["healing"]["four_week"]["target"] == 40.0
    assert progress(ts, "surgical")["healing"]["four_week"] is None  # no benchmark for this type


def test_push_score():
    p = push_score(obs(0, 6.0, {"granulation": 70, "slough": 30}), "Moderate")  # L x W = 6 cm²
    assert p["size"] == 7 and p["exudate"] == 2 and p["tissue"] == 3 and p["score"] == 12
    assert push_score(obs(0, 6.0, {"granulation": 100}), None) is None  # exudate not recorded


def test_observation_reduces_an_analyze_result():
    f = {"status": "ok", "measurement": {"area_cm2": 2.0, "length_cm": 2, "width_cm": 1, "perimeter_cm": 5},
         "tissue_pct": {"granulation": 100}, "flags": [{"level": "review", "text": "x"}]}
    o = observation(f, "2026-01-01T00:00:00+00:00")
    assert o["area_cm2"] == 2.0 and o["tissue_pct"] == {"granulation": 100} and o["flags"]
    assert session_effect(o, observation({"status": "retake"}, o["taken_at"])) is None


def test_analyze_previous_photo_uses_the_same_noise_band():
    base = {"intake": {}, "measurement": {"area_cm2": 2.0}}
    small = red_flags({**base, "change": {"percent_area_reduction": -5.0}})
    big = red_flags({**base, "change": {"percent_area_reduction": -40.0}})
    assert not any("increased" in f["text"] for f in small)
    assert any("increased" in f["text"] for f in big)


def test_repeatability_recovers_the_noise_of_simulated_repeat_photos():
    import numpy as np
    from wound_ai.metrics import repeatability

    rng = np.random.default_rng(1)
    sd = 0.06  # each photo's area off by about 6% (log scale)
    true = rng.uniform(1, 20, 400)
    r = repeatability([list(a * np.exp(rng.normal(0, sd, 2))) for a in true])
    assert abs(r["sw"] - sd) < 0.006
    # RC = 1.96 * sqrt(2) * 0.06 = 0.166 -> 15.3% smaller or 18.1% larger is beyond noise.
    assert abs(r["smaller_pct"] - 15.3) < 1.5 and abs(r["larger_pct"] - 18.1) < 2.0
    assert r["rc_ci95"][0] < r["rc"] < r["rc_ci95"][1]
    assert repeatability([[2.0, 2.1]]) is None  # one wound is not a study


def test_the_study_file_replaces_the_placeholder(tmp_path):
    import json
    from wound_ai.progress import PLACEHOLDER_NOISE, load_noise

    assert load_noise(tmp_path / "none.json") == PLACEHOLDER_NOISE
    f = tmp_path / "noise.json"
    f.write_text(json.dumps({"study": {"date": "2026-11-01"}, "tissue": None,
                             "area": {"smaller_pct": 9.5, "larger_pct": 10.5, "n_wounds": 40}}))
    n = load_noise(f)
    assert (n["smaller_pct"], n["larger_pct"]) == (9.5, 10.5)
    assert n["tissue_points"] == PLACEHOLDER_NOISE["tissue_points"]  # tissue not measured: keep the placeholder
    assert "40 wounds" in n["source"]


def test_noise_band_applies_each_way_and_is_reported():
    noise = {"smaller_pct": 10.0, "larger_pct": 12.0, "tissue_points": 15.0, "source": "study"}
    ts = lambda after: [{"pre": obs(0, 10.0)}, {"pre": obs(7, after)}]  # noqa: E731
    assert healing(ts(8.9), noise=noise)["trajectory"] == "improving"      # 11% smaller
    assert healing(ts(11.1), noise=noise)["trajectory"] == "static"        # 11% larger: inside 12%
    assert healing(ts(11.3), noise=noise)["trajectory"] == "deteriorating"
    assert healing(ts(9.0), noise=noise)["noise_band"]["source"] == "study"
