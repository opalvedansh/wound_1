"""Care suggestions: what the reviewing clinician may consider next, from the TIME framework
(Tissue, Infection/inflammation, Moisture, Edge) plus the cause of the wound (R).

Design rules:
* Deterministic. A rule table decides every suggestion; a language model may only word the summary.
* Every suggestion names the findings that triggered it, and says "consider": the clinician decides.
* Actions are the app's own therapy and dressing options, so a clinician can apply one with a tap.
* A photo cannot show blood flow, depth or infection, so rules also read the clinician's assessment and the
  intake answers. Where an input a rule needs is missing, it adds a CHECK instead of a suggestion.
* CONTRAINDICATIONS remove any suggestion they conflict with, and are listed when that therapy was suggested
  or is on the current plan.

ALL RULES AND THRESHOLDS ARE PLACEHOLDERS written from general wound-care practice. Your clinical partner must
review, edit and sign off every rule before any patient use; bump RULES_VERSION on every change, because each
review stores the rule ids it accepted or declined.
"""
from __future__ import annotations

from .intake import FOOT_SITES, LEG_AND_FOOT_SITES
from .progress import nonviable, progress, same_visit, usable
from .report import DANGER, render_treatment_report

# Stays "-unsigned" until a clinician signs docs/clinical_signoff.md; then "care-1.0", with the sign-off recorded there.
RULES_VERSION = "care-0.1-unsigned"

# The app's therapy and dressing options (packages/domain/src/lib/questions.ts), spelled the same way.
DEBRIDEMENT, CLEANSING, NPWT, SKIN_SUBSTITUTE = "Debridement", "Cleansing", "Negative Pressure (NPWT)", "Skin Substitute"
COMPRESSION, OFFLOADING, PRESSURE_REDISTRIBUTION = "Compression", "Offloading", "Pressure redistribution"
FOAM, ALGINATE, HYDROGEL, HYDROCOLLOID, SILVER = "Foam", "Alginate", "Hydrogel", "Hydrocolloid", "Silver"

NONVIABLE_DEBRIDE_PCT = 30  # slough + necrosis share of the bed above which debridement is suggested
ERYTHEMA_INFECTION_FRAC = 0.3  # share of the ring around the wound that is red
MACERATION_FRAC = 0.2  # share of the ring around the wound that is macerated
CALLUS_FRAC = 0.1  # share of the ring around the wound that is callus
INFECTION_SIGNS_MIN = 2
ABPI_LOW, ABPI_HIGH = 0.8, 1.3  # same as report.py
SAME_PLAN_VISITS = 3


def wound_type_label(req: dict) -> str | None:
    """`wound_type` is the PRE photo's class result ({"label", "prob", ...}) as /analyze returned it."""
    return (req.get("wound_type") or {}).get("label")


def _abpi(intake: dict) -> float | None:
    try:
        return float(intake["abpi"]) if intake.get("abpi") not in (None, "") else None
    except (TypeError, ValueError):
        return None


class _Ctx:
    """The facts the rules read, with the photo findings of the current treatment (POST if it was analysed)."""

    def __init__(self, req: dict, prog: dict):
        self.intake = req.get("intake") or {}
        self.wt = wound_type_label(req)
        treatments = req.get("treatments") or [{}]
        cur = treatments[-1]
        self.pre, self.post = cur.get("pre"), cur.get("post")
        self.latest = self.post if usable(self.post) else self.pre if usable(self.pre) else {}
        self.assessment = cur.get("assessment") or {}
        self.history = treatments
        self.prog = prog
        self.tissue = self.latest.get("tissue_pct")
        self.nonviable = nonviable(self.tissue)
        a = self.assessment
        self.exudate = (a.get("exudate_level") or "").lower() or None
        self.signs = a.get("infection_signs") or []
        site = self.intake.get("body_location")
        self.foot = site in FOOT_SITES
        self.leg_or_foot = site in LEG_AND_FOOT_SITES
        self.abpi = _abpi(self.intake)
        self.ischaemia = self.intake.get("foot_cold_or_dark") == "yes" or (self.abpi is not None and self.abpi < ABPI_LOW)
        ery = self.latest.get("periwound_erythema_frac")
        self.infection = (len([s for s in self.signs if s != "Fever/systemic signs"]) >= INFECTION_SIGNS_MIN
                          or (ery is not None and ery >= ERYTHEMA_INFECTION_FRAC)
                          or (a.get("exudate_type") or "").lower() == "purulent"
                          or self.intake.get("discharge") == "thick_yellow_or_green")
        self.macerated = "Macerated" in (a.get("edge_condition"), a.get("periwound_condition")) \
            or (self.latest.get("periwound_maceration_frac") or 0) >= MACERATION_FRAC
        self.tried = None  # set by rule E3: a dressing that has not helped
        self.current_plan = set(cur.get("therapy") or []) | ({cur["dressing"]} if cur.get("dressing") else set())


def _s(rule_id: str, domain: str, action: str, text: str, because: list[str], alternatives: list[str] | None = None):
    return {"rule_id": rule_id, "domain": domain, "action": action, "text": text, "because": because,
            "alternatives": alternatives or []}


# --------------------------------------------------------------------------- rules

def _tissue(c: _Ctx, out: list, checks: list):
    if c.tissue is None:
        checks.append({"rule_id": "T0", "text": "Tissue mix not assessed by the model: record it on examination."})
    if (c.latest.get("periwound_callus_frac") or 0) >= CALLUS_FRAC:
        checks.append({"rule_id": "T3", "text": "Callus around the ulcer: consider removing it and review offloading "
                                                "(callus raises pressure on the wound)."})
    if c.tissue is None:
        return
    if c.nonviable < NONVIABLE_DEBRIDE_PCT:
        return
    if usable(c.post) and same_visit(c.pre, c.post):
        out.append(_s("T2", "T", DEBRIDEMENT, "Consider continuing debridement at the next visit.",
                      [f"non-viable tissue still {c.nonviable}% after this session"]))
    else:
        out.append(_s("T1", "T", DEBRIDEMENT, "Consider debridement of non-viable tissue.",
                      [f"slough and necrosis {c.nonviable}% of the wound bed"]))


def _infection(c: _Ctx, out: list, checks: list):
    if c.infection:
        out.append(_s("I1", "I", SILVER, "Consider an antimicrobial (silver) dressing and assess for infection.",
                      [f"infection signs: {', '.join(c.signs) or 'redness or pus'}"]))
        checks.append({"rule_id": "I2", "text": "Assess for local or spreading infection per local protocol."})


def _moisture(c: _Ctx, out: list, checks: list):
    if c.exudate is None:
        checks.append({"rule_id": "M0", "text": "Record the exudate amount to guide the dressing choice."})
    elif c.exudate == "heavy":
        out.append(_s("M1", "M", ALGINATE, "Consider a highly absorbent dressing.", ["heavy exudate"], [FOAM]))
    elif c.exudate == "moderate":
        out.append(_s("M2", "M", FOAM, "Consider an absorbent foam dressing.", ["moderate exudate"], [ALGINATE]))
    elif c.exudate in ("none", "scant") and c.tissue and c.nonviable and not c.infection:
        out.append(_s("M3", "M", HYDROGEL, "Consider adding moisture to help soften non-viable tissue.",
                      [f"{c.exudate} exudate", f"non-viable tissue {c.nonviable}%"]))
    if c.macerated:
        checks.append({"rule_id": "M4", "text": "Macerated skin: protect the periwound skin and review absorbency."})


def _edge(c: _Ctx, out: list, checks: list):
    edge = c.assessment.get("edge_condition")
    if edge in ("Rolled", "Undermined"):
        checks.append({"rule_id": "E1", "text": f"{edge} edge: probe for undermining; the edge may need refreshing."})
    fw = (c.prog.get("healing") or {}).get("four_week")
    if fw and not fw["on_track"]:
        out.append(_s("E2", "E", NPWT, "Healing is behind the 4-week benchmark: reassess the cause and consider an "
                                       "advanced therapy.",
                      [f"{fw['percent_area_reduction']}% area reduction by day {fw['days']:.0f} "
                       f"(target {fw['target']:.0f}%)"], [SKIN_SUBSTITUTE]))
    # The same dressing for several visits without improvement.
    traj = (c.prog.get("healing") or {}).get("trajectory")
    dressings = [t.get("dressing") for t in c.history[-SAME_PLAN_VISITS:]]
    if traj in ("static", "deteriorating") and len(dressings) == SAME_PLAN_VISITS and dressings[0] \
            and len(set(dressings)) == 1:
        checks.append({"rule_id": "E3", "text": f"{dressings[0]} for {SAME_PLAN_VISITS} visits and the wound is {traj}: "
                                                "review the care plan."})
        c.tried = dressings[0]


def _cause(c: _Ctx, out: list, checks: list):
    if c.wt == "venous":
        if c.abpi is None:
            checks.append({"rule_id": "R0", "text": "Measure the ABPI: compression is the main treatment for venous "
                                                    "ulcers but needs adequate blood flow."})
        elif ABPI_LOW <= c.abpi <= ABPI_HIGH:
            out.append(_s("R1", "R", COMPRESSION, "Consider compression therapy.",
                          ["venous leg ulcer", f"ABPI {c.abpi}"]))
    if c.wt == "diabetic" and c.foot:
        out.append(_s("R2", "R", OFFLOADING, "Consider offloading the ulcer (footwear, cast or boot).",
                      ["diabetic foot ulcer"]))
    if c.wt == "pressure" or c.intake.get("cause") == "pressure_lying_or_sitting":
        out.append(_s("R3", "R", PRESSURE_REDISTRIBUTION, "Consider repositioning and a pressure-redistributing surface.",
                      ["pressure injury"]))


RULES = (_tissue, _infection, _moisture, _edge, _cause)


def contraindications(c: _Ctx) -> dict[str, str]:
    """{action: reason} that must never be suggested in this situation."""
    out = {}
    if c.ischaemia:
        out[DEBRIDEMENT] = "possible poor blood flow: vascular assessment before sharp debridement"
        out[HYDROGEL] = "possible poor blood flow: dry necrosis is usually kept dry until assessed"
    if c.intake.get("body_location") == "heel" and (c.tissue or {}).get("necrosis", 0) >= 50 \
            and c.exudate in ("none", "scant") and not c.infection:
        out[DEBRIDEMENT] = "dry, stable heel eschar is often left intact"
    if c.leg_or_foot or c.wt == "venous":
        if c.abpi is None:
            out[COMPRESSION] = "ABPI not recorded"
        elif c.abpi < ABPI_LOW:
            out[COMPRESSION] = f"ABPI {c.abpi} is below {ABPI_LOW}"
        elif c.abpi > ABPI_HIGH:
            out[COMPRESSION] = f"ABPI {c.abpi} is above {ABPI_HIGH} (arteries may be calcified): check toe pressures"
    if c.infection or c.exudate == "heavy":
        out[HYDROCOLLOID] = "infection signs or heavy exudate"
    npwt = [r for r, on in (("necrotic tissue in the bed", bool((c.tissue or {}).get("necrosis"))),
                            ("signs of infection", c.infection), ("possible poor blood flow", c.ischaemia)) if on]
    if npwt:
        out[NPWT] = "; ".join(npwt)
    return out


def suggest(req: dict, prog: dict) -> dict:
    """{suggestions, contraindications, checks, rules_version} for the current treatment."""
    c = _Ctx(req, prog)
    raw, checks = [], []
    if any(f["level"] == "urgent" and f["text"].startswith(DANGER) for o in (c.pre, c.post) if o
           for f in o.get("flags") or []):
        checks.append({"rule_id": "D0", "text": "Danger signs present: arrange urgent care before routine wound care."})
    for rule in RULES:
        rule(c, raw, checks)
    contra = contraindications(c)
    suggestions = []
    for s in raw:
        options = sorted((a for a in [s["action"], *s["alternatives"]] if a not in contra), key=lambda a: a == c.tried)
        if options:
            suggestions.append({**s, "action": options[0], "alternatives": options[1:]})
    blocked = {a for s in raw for a in [s["action"], *s["alternatives"]]} | c.current_plan
    return {
        "suggestions": suggestions,
        "contraindications": [{"action": a, "reason": r} for a, r in contra.items() if a in blocked],
        "checks": checks,
        "rules_version": RULES_VERSION,
    }


def treatment_report(req: dict) -> dict:
    """POST /treatment-report: progress, care suggestions, flags and the draft for the current (last) treatment.

    req = {"wound_type": <PRE class result>, "severity": <PRE severity results>, "intake": {...},
           "treatments": [{"sequence", "pre": obs, "post": obs, "assessment": {...}, "therapy": [...], "dressing"}]}
    in visit order, where obs is progress.observation() of an /analyze result. A single upload is one treatment
    with only "pre".
    """
    treatments = req.get("treatments") or []
    cur = treatments[-1] if treatments else {}
    wt = wound_type_label(req)
    prog = progress(treatments, wt, (cur.get("assessment") or {}).get("exudate_level"))
    care = suggest(req, prog)
    flags, seen = [], set()
    for fl in [*((cur.get("pre") or {}).get("flags") or []), *((cur.get("post") or {}).get("flags") or []),
               *prog["flags"]]:
        if fl["text"] not in seen:
            seen.add(fl["text"])
            flags.append(fl)
    flags.sort(key=lambda fl: fl["level"] != "urgent")
    return {"progress": prog, **care, "flags": flags,
            "report_markdown": render_treatment_report(req, prog, care, flags)}
