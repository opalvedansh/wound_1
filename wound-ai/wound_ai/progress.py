"""How a wound is changing: what one session achieved, and how it is healing from visit to visit.

Each treatment has two photos taken minutes apart in the same visit: PRE (as found) and POST (after cleaning or
debridement, before the dressing). So:

* PRE -> POST of one treatment is the SESSION EFFECT: what the procedure did. Debridement often makes the wound
  larger (the edges are opened up) while cleaning the bed, so a larger area here is expected, never "worse".
* HEALING is measured visit to visit, comparing like with like only: POST with POST (both after cleaning, the
  most reliable view of the wound bed), or PRE with PRE when there are no POST photos. Never PRE with POST.

Areas are compared only when both photos had the calibration sticker. Without it only the tissue mix, which
does not depend on scale, is compared.

THRESHOLDS ARE PLACEHOLDERS for clinical sign-off. A change smaller than the measurement noise is reported as
static. The noise band comes from a repeat-photo study (docs/repeatability_study.md: two staff photograph the same
wound in the same visit; scripts/repeatability.py writes wound_ai/measurement_noise.json). Until that file exists the
placeholder below is used, and every healing result says so.

Pure functions on plain dicts, no models: an "observation" is one analysed photo, reduced by `observation()`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

NOISE_FILE = Path(__file__).with_name("measurement_noise.json")
# Used until a repeat-photo study has measured the real noise: % area change either way, and percentage points of
# non-viable tissue, treated as noise.
PLACEHOLDER_NOISE = {"smaller_pct": 15.0, "larger_pct": 15.0, "tissue_points": 15.0,
                     "source": "placeholder (no repeat-photo study yet)"}


def load_noise(path: Path = NOISE_FILE) -> dict:
    """The noise band: from the repeat-photo study's file if there is one, else the placeholder. A study that
    measured area but not tissue (no tissue model yet) keeps the placeholder for tissue."""
    if not path.exists():
        return dict(PLACEHOLDER_NOISE)
    study = json.loads(path.read_text())
    area, tissue = study["area"], study.get("tissue")
    return {"smaller_pct": area["smaller_pct"], "larger_pct": area["larger_pct"],
            "tissue_points": tissue["rc"] if tissue else PLACEHOLDER_NOISE["tissue_points"],
            "source": f"repeat-photo study {study['study']['date']} ({area['n_wounds']} wounds)"}


NOISE = load_noise()
NONVIABLE = ("slough", "necrosis")
# Percent area reduction expected by about 4 weeks; below it, healing is unlikely on the current plan.
FOUR_WEEK_TARGET = {"diabetic": 50.0, "venous": 40.0}
FOUR_WEEK_WINDOW = (21, 35)  # days from the first photo in which the 4-week check is made
PUSH_AREA_BOUNDS = (0.6, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 24.0)  # L x W cm², upper bounds for PUSH scores 2-9
PUSH_EXUDATE = {"none": 0, "scant": 1, "light": 1, "moderate": 2, "heavy": 3}
PUSH_TISSUE = (("necrosis", 4), ("slough", 3), ("granulation", 2), ("epithelial", 1))


def observation(findings: dict, taken_at: str) -> dict:
    """The parts of an /analyze result that progress needs."""
    m = findings.get("measurement") or {}
    return {
        "taken_at": taken_at,
        "status": findings.get("status"),
        "area_cm2": m.get("area_cm2"),
        "length_cm": m.get("length_cm"),
        "width_cm": m.get("width_cm"),
        "perimeter_cm": m.get("perimeter_cm"),
        "tissue_pct": findings.get("tissue_pct"),
        "periwound_erythema_frac": findings.get("periwound_erythema_frac"),
        "periwound_maceration_frac": findings.get("periwound_maceration_frac"),
        "periwound_callus_frac": findings.get("periwound_callus_frac"),
        "flags": findings.get("flags") or [],
    }


def usable(obs: dict | None) -> bool:
    return bool(obs) and obs.get("status") == "ok"


def nonviable(tissue: dict | None) -> int | None:
    return sum(tissue.get(c, 0) for c in NONVIABLE) if tissue else None


def _days(a: dict, b: dict) -> float | None:
    try:
        d = datetime.fromisoformat(b["taken_at"]) - datetime.fromisoformat(a["taken_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return round(d.total_seconds() / 86400, 1)


def _pct_change(before: float, after: float) -> float:
    """Percent area reduction: positive = smaller."""
    return round(100.0 * (before - after) / before, 1)


# --------------------------------------------------------------------------- session effect

def session_effect(pre: dict | None, post: dict | None) -> dict | None:
    """What this visit's cleaning or debridement did, from its PRE and POST photos."""
    if not (usable(pre) and usable(post)):
        return None
    out: dict = {}
    if pre.get("area_cm2") and post.get("area_cm2"):
        out["area_before_cm2"], out["area_after_cm2"] = pre["area_cm2"], post["area_cm2"]
        # An enlarged area after debridement is the edges being opened up, which is expected.
        out["area_note"] = "larger after the procedure, as expected after debridement" \
            if _pct_change(pre["area_cm2"], post["area_cm2"]) < -NOISE["larger_pct"] else None
    nv_pre, nv_post = nonviable(pre.get("tissue_pct")), nonviable(post.get("tissue_pct"))
    if nv_pre is not None and nv_post is not None:
        out["nonviable_before_pct"], out["nonviable_after_pct"] = nv_pre, nv_post
        out["nonviable_removed_points"] = nv_pre - nv_post
    return out or None


# --------------------------------------------------------------------------- visit-to-visit healing

def healing_series(treatments: list[dict]) -> tuple[str, list[dict]]:
    """Like-for-like photos across treatments: POST photos if the current treatment has one, else PRE photos."""
    current = treatments[-1] if treatments else {}
    phase = "post" if usable(current.get("post")) else "pre"
    return phase, [t[phase] for t in treatments if usable(t.get(phase))]


def _compare(a: dict, b: dict) -> dict:
    out: dict = {"days": _days(a, b)}
    days = out["days"]
    if a.get("area_cm2") and b.get("area_cm2"):
        out["area_before_cm2"], out["area_after_cm2"] = a["area_cm2"], b["area_cm2"]
        out["percent_area_reduction"] = _pct_change(a["area_cm2"], b["area_cm2"])
        if days:
            out["cm2_per_week"] = round((a["area_cm2"] - b["area_cm2"]) / days * 7, 2)
            perimeter = ((a.get("perimeter_cm") or 0) + (b.get("perimeter_cm") or 0)) / 2
            if perimeter:
                # Gilman: how far the edge moves in, independent of the wound's size.
                out["edge_advance_cm_per_week"] = round((a["area_cm2"] - b["area_cm2"]) / perimeter / days * 7, 3)
    nv_a, nv_b = nonviable(a.get("tissue_pct")), nonviable(b.get("tissue_pct"))
    if nv_a is not None and nv_b is not None:
        out["nonviable_before_pct"], out["nonviable_after_pct"] = nv_a, nv_b
    return out


def within_noise(par: float, noise: dict | None = None) -> bool:
    """Is this percent area reduction (positive = smaller) inside the measurement noise?"""
    n = noise or NOISE
    return -n["larger_pct"] <= par <= n["smaller_pct"]


def trajectory(change: dict, noise: dict | None = None) -> tuple[str | None, str | None]:
    """improving | static | deteriorating, and what it was based on. Changes inside the noise band are static."""
    n = noise or NOISE
    par = change.get("percent_area_reduction")
    if par is not None:
        return ("static" if within_noise(par, n) else "improving" if par > 0 else "deteriorating"), "area"
    if "nonviable_before_pct" in change:
        d = change["nonviable_after_pct"] - change["nonviable_before_pct"]
        t = n["tissue_points"]
        return ("improving" if d < -t else "deteriorating" if d > t else "static"), "tissue"
    return None, None


def four_week_check(series: list[dict], wound_type: str | None) -> dict | None:
    """Percent area reduction around week 4 against the benchmark for the wound type (if it has one)."""
    target = FOUR_WEEK_TARGET.get(wound_type or "")
    if not target or len(series) < 2 or not series[0].get("area_cm2"):
        return None
    lo, hi = FOUR_WEEK_WINDOW
    timed = [(d, o) for o in series[1:] if (d := _days(series[0], o)) is not None and lo <= d <= hi and o.get("area_cm2")]
    if not timed:
        return None
    days, obs = min(timed, key=lambda x: abs(x[0] - 28))
    par = _pct_change(series[0]["area_cm2"], obs["area_cm2"])
    return {"days": days, "percent_area_reduction": par, "target": target, "on_track": par >= target}


def healing(treatments: list[dict], wound_type: str | None = None, noise: dict | None = None) -> dict:
    n = noise or NOISE
    phase, series = healing_series(treatments)
    out: dict = {"phase": phase, "n_photos": len(series),
                 "noise_band": {k: n[k] for k in ("smaller_pct", "larger_pct", "source")}}
    if len(series) < 2:
        out["comparable"] = False
        out["reason"] = "first analysed photo of this wound" if series else "no analysed photo"
        return out
    since_last, since_first = _compare(series[-2], series[-1]), _compare(series[0], series[-1])
    out["since_last"], out["since_first"] = since_last, since_first
    out["trajectory"], out["basis"] = trajectory(since_last, n)
    out["comparable"] = out["trajectory"] is not None
    if not out["comparable"]:
        out["reason"] = "no calibration sticker in both photos and no tissue estimate to compare"
    out["four_week"] = four_week_check(series, wound_type)
    return out


# --------------------------------------------------------------------------- PUSH score

def push_score(obs: dict | None, exudate_level: str | None) -> dict | None:
    """PUSH (Pressure Ulcer Scale for Healing, 0-17; lower is better). Validated for pressure injuries only.
    Needs length and width (the sticker), the clinician's exudate amount, and the tissue mix."""
    if not usable(obs) or not obs.get("length_cm") or not obs.get("width_cm") or not obs.get("tissue_pct"):
        return None
    ex = PUSH_EXUDATE.get((exudate_level or "").lower())
    if ex is None:
        return None
    lw = obs["length_cm"] * obs["width_cm"]
    size = 1 if lw < 0.3 else next((i + 2 for i, b in enumerate(PUSH_AREA_BOUNDS) if lw <= b), 10)
    tissue = next((s for c, s in PUSH_TISSUE if obs["tissue_pct"].get(c)), 0)
    return {"score": size + ex + tissue, "size": size, "exudate": ex, "tissue": tissue,
            "note": "validated for pressure injuries"}


# --------------------------------------------------------------------------- flags

def progress_flags(heal: dict, wound_type: str | None) -> list[dict]:
    """Review flags from healing. PLACEHOLDER RULES - clinical sign-off required."""
    flags = []
    if heal.get("trajectory") == "deteriorating":
        what = "Wound area has increased" if heal.get("basis") == "area" else "More non-viable tissue"
        flags.append({"level": "review", "text": f"{what} since the last visit, beyond measurement noise."})
    fw = heal.get("four_week")
    if fw and not fw["on_track"]:
        flags.append({"level": "review", "text": f"Less than {fw['target']:.0f}% of the area has closed by about 4 weeks: "
                                                 "review the care plan."})
    return flags


def progress(treatments: list[dict], wound_type: str | None = None, exudate_level: str | None = None) -> dict:
    """Everything above for the current (last) treatment."""
    current = treatments[-1] if treatments else {}
    heal = healing(treatments, wound_type)
    return {
        "session": session_effect(current.get("pre"), current.get("post")),
        "healing": heal,
        "push": push_score(current.get("post") if usable(current.get("post")) else current.get("pre"), exudate_level),
        "flags": progress_flags(heal, wound_type),
    }
