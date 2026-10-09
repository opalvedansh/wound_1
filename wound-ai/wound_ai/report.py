"""Turn model findings into a report draft a clinician can review and sign.

Design rule: everything safety-relevant is DETERMINISTIC. Numbers, red flags,
uncertainty notes and the disclaimer are rendered from the findings dict by this
module. A language model (MedGemma) is only allowed to write the short narrative
summary, and that summary is rejected (template fallback) if it contains any
number not present in the findings, or any banned phrase.

The red-flag rules below are PLACEHOLDERS written from general wound-care
practice. Your clinical partner must review, edit and sign off every rule and
threshold before any patient use.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .intake import FOOT_SITES, LEG_AND_FOOT_SITES, SPECIAL_BURN_SITES
from .progress import FOUR_WEEK_TARGET, FOUR_WEEK_WINDOW, progress_flags, trajectory, within_noise

DISCLAIMER = ("AI-generated draft for review by a qualified clinician. It is not a diagnosis and must not be used "
              "to start, stop or change treatment without clinical assessment.")

UNCERTAIN_BELOW = 0.70  # calibrated probability below which a classification is reported as uncertain
ABPI_LOW = 0.8  # below: arterial or mixed disease possible (guidelines differ; clinician sets the threshold)
ABPI_HIGH = 1.3  # above: arteries may be calcified, so the reading can be falsely reassuring
# Quality warnings (quality.py) as the flag names them.
QUALITY_WORDS = {"low_resolution": "low resolution", "blurry": "blurry", "dark": "dark", "glare": "glare"}
# Prefix of every danger-sign flag (Chart 1): the report then opens with "Emergency care now".
DANGER = "Danger sign"

LABELS = {
    "diabetic": "Diabetic foot ulcer", "pressure": "Pressure injury", "venous": "Venous leg ulcer",
    "surgical": "Surgical wound", "burn": "Burn", "other": "Other wound", "not_wound": "No wound detected",
}


# --------------------------------------------------------------------------- red flags

def red_flags(f: dict) -> list[dict]:
    """Return [{'level': 'urgent'|'review', 'text': ...}]. PLACEHOLDER RULES - clinical sign-off required."""
    a = f.get("intake", {})
    t = f.get("tissue_pct", {}) or {}
    wt = (f.get("wound_type") or {}).get("label")
    sev = f.get("severity", {}) or {}
    chg = f.get("change", {}) or {}
    flags = []

    def add(level, text):
        flags.append({"level": level, "text": text})

    # Chart 1: danger signs first. Each one means same-day care, whatever the wound type.
    if a.get("foot_cold_or_dark") == "yes":
        add("urgent", f"{DANGER}: a cold, pale or darkening foot or toes (possible gangrene or critical ischaemia): "
                      "same-day vascular or diabetic foot team.")
    elif a.get("diabetes") == "yes" and t.get("necrosis", 0) > 0:
        add("urgent", f"{DANGER}: dark/necrotic tissue in a person with diabetes: same-day assessment by a diabetic "
                      "foot or vascular team.")
    if a.get("fever") == "yes" and (a.get("redness_spreading") == "yes" or a.get("discharge") == "thick_yellow_or_green"):
        add("urgent", f"{DANGER}: fever with spreading redness or pus: possible spreading infection, needs urgent "
                      "medical review.")
    elif a.get("redness_spreading") == "yes":
        add("review", "Redness or swelling reported as spreading: clinician review within 24 hours.")
    if wt == "burn" or a.get("cause") == "burn":
        if a.get("burn_agent") in ("chemical", "electrical"):
            add("urgent", f"{DANGER}: chemical or electrical burn: refer to a burns unit; surface appearance can "
                          "underestimate damage.")
        if a.get("body_location") in SPECIAL_BURN_SITES:
            add("urgent", f"{DANGER}: burn on the face, neck, hands or feet: refer to a burns unit.")
        if a.get("burn_other_sites") == "yes":
            add("urgent", f"{DANGER}: burns on more than one body area: estimate the total area and refer to a burns unit.")
        if sev.get("burn_depth", {}).get("label") in ("deep_partial", "full_thickness"):
            add("urgent", "Possible deep burn: burns specialist assessment.")

    # Charts 2 and 3: a photo cannot show blood flow, so leg and foot ulcers say so until a clinician enters an ABPI.
    acute = a.get("cause") in ("burn", "surgery", "injury_cut_or_fall")
    leg_or_foot = a.get("body_location") in LEG_AND_FOOT_SITES
    if leg_or_foot and (not acute or wt == "diabetic"):
        abpi = _number(a.get("abpi"))
        if abpi is None:
            add("review", "Blood flow not assessed: check foot pulses and ABPI before any compression.")
        elif abpi < ABPI_LOW:
            add("urgent", f"ABPI {abpi} is below {ABPI_LOW}: arterial or mixed disease possible. Vascular review "
                          "before any compression.")
        elif abpi > ABPI_HIGH:
            add("review", f"ABPI {abpi} is above {ABPI_HIGH}: arteries may be calcified (common in diabetes and kidney "
                          "disease), so the reading can be falsely reassuring. Check toe pressures.")
    if a.get("wound_opening") == "yes":
        add("review", "Surgical wound reported as opening: contact the operating team.")
    pct = chg.get("percent_area_reduction")
    if pct is not None:  # the single-previous-photo comparison of /analyze; same rules as the treatment report
        heal = {"basis": "area", "trajectory": trajectory({"percent_area_reduction": pct})[0]}
        target = FOUR_WEEK_TARGET.get(wt or "")
        if target and chg.get("days_between", 0) >= FOUR_WEEK_WINDOW[0]:
            heal["four_week"] = {"target": target, "on_track": pct >= target}
        flags += progress_flags(heal, wt)
    if t.get("exposed_structure"):
        add("review", "Possible exposed bone or tendon in the wound bed: confirm on examination (risk of bone "
                      "infection).")
    conf = (f.get("wound_type") or {}).get("prob")
    if conf is not None and conf < UNCERTAIN_BELOW:
        add("review", "Model is uncertain about the wound type: clinician to classify.")
    if f.get("measurement") is None:
        add("review", "Size not measured (calibration sticker not detected or no wound region found).")
    warned = [QUALITY_WORDS[w] for w in (f.get("quality") or {}).get("warnings", []) if w in QUALITY_WORDS]
    if warned:
        add("review", f"Photo quality ({', '.join(warned)}): the outline, size and tissue estimates may be less "
                      "accurate. Check them against the photo.")
    return flags


def _number(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def has_danger_signs(flags: list[dict]) -> bool:
    return any(fl["level"] == "urgent" and fl["text"].startswith(DANGER) for fl in flags)


# --------------------------------------------------------------------------- template report

def _fmt_class(entry: dict | None) -> str:
    if not entry:
        return "not assessed"
    label = LABELS.get(entry["label"], entry["label"].replace("_", " "))
    if entry.get("rule"):
        model = entry.get("model")
        guess = f"; model estimate {_fmt_class(model)}" if model else ""
        return f"{label} (by rule: {entry['rule']}{guess})"
    p = entry.get("prob")
    if p is None:
        return label
    if p < UNCERTAIN_BELOW:
        alts = ", ".join(f"{LABELS.get(k, k)} {v:.0%}" for k, v in entry.get("top", [])[:3])
        return f"UNCERTAIN (top estimates: {alts})"
    return f"{label} (model confidence {p:.0%})"


def template_narrative(f: dict) -> str:
    parts = []
    wt = f.get("wound_type")
    if wt and wt.get("rule"):
        parts.append(f"Recorded as {LABELS.get(wt['label'], wt['label']).lower()} because of {wt['rule']}.")
    elif wt and (wt.get("prob") or 0) >= UNCERTAIN_BELOW:
        parts.append(f"Appearance is most consistent with {LABELS.get(wt['label'], wt['label']).lower()}.")
    elif wt:
        parts.append("The wound type could not be determined with confidence from the photo.")
    m = f.get("measurement")
    if m:
        parts.append(f"Measured area is {m['area_cm2']} cm² ({m['length_cm']} x {m['width_cm']} cm).")
    t = f.get("tissue_pct") or {}
    if t:
        main = max(t, key=t.get)
        parts.append(f"The wound bed is mostly {main} tissue ({t[main]}%).")
    c = f.get("change") or {}
    if "percent_area_reduction" in c:
        verb = "decreased" if c["percent_area_reduction"] >= 0 else "increased"
        noise = " (within measurement noise)" if within_noise(c["percent_area_reduction"]) else ""
        parts.append(f"Area has {verb} by {abs(c['percent_area_reduction'])}% since the previous photo{noise}.")
    return " ".join(parts)


def render_report(f: dict, narrative: str | None = None) -> str:
    flags = f.get("flags") or red_flags(f)
    m = f.get("measurement")
    t = f.get("tissue_pct") or {}
    a = f.get("intake", {})
    lines = [
        "# Wound assessment (AI-assisted draft)",
        f"_{DISCLAIMER}_",
        "",
    ]
    if has_danger_signs(flags):
        lines += ["**Emergency care now: do not wait for this report.** Danger signs are listed under Flags.", ""]
    lines += [
        f"Generated: {f.get('timestamp', datetime.now(timezone.utc).isoformat(timespec='minutes'))}",
        "",
        "## Flags",
    ]
    if flags:
        lines += [f"- **{fl['level'].upper()}**: {fl['text']}" for fl in flags]
    else:
        lines.append("- None raised by the automatic rules.")
    lines += ["", "## Summary", narrative or template_narrative(f), "", "## Findings",
              f"- Wound type: {_fmt_class(f.get('wound_type'))}"]
    for k, v in (f.get("severity") or {}).items():
        lines.append(f"- {k.replace('_', ' ').capitalize()}: {_fmt_class(v)}")
    if m:
        lines.append(f"- Size: area {m['area_cm2']} cm², length {m['length_cm']} cm, width {m['width_cm']} cm, "
                     f"perimeter {m['perimeter_cm']} cm ({m['n_regions']} region(s)). Depth not measurable from a photo.")
    else:
        lines.append("- Size: not measured.")
    if t:
        lines.append("- Wound bed tissue: " + ", ".join(f"{k} {v}%" for k, v in t.items()))
    if f.get("tissue_untrusted"):
        lines.append("- Possibly also: " + ", ".join(x.replace("_", " ") for x in f["tissue_untrusted"])
                     + " (the tissue model is not yet reliable for these: check on examination).")
    if not t and f.get("tissue_pct_uncertain"):
        lines.append(f"- Wound bed tissue: UNCERTAIN (model confidence {f['tissue_confidence']:.0%}; estimate "
                     + ", ".join(f"{k} {v}%" for k, v in f["tissue_pct_uncertain"].items()) + "). Assess on examination.")
    if f.get("change"):
        c = f["change"]
        lines.append(f"- Change: previous area {c.get('previous_area_cm2')} cm², "
                     f"area reduction {c.get('percent_area_reduction')}%"
                     + (f" over {c['days_between']} days" if c.get("days_between") else ""))
    if a:
        lines += ["", "## Reported by patient/carer"]
        lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in a.items() if v not in (None, "")]
    lines += ["", "## For the reviewing clinician",
              "- Confirm wound type, stage/depth and tissue estimates on direct examination.",
              "- Assess depth, undermining, pulses/perfusion and infection signs, which a photo cannot show.",
              "- Approve, edit or reject this draft; edits are logged and used to improve the model.",
              "", f"Model versions: {json.dumps(f.get('model_versions', {}))}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- LLM narrative

SYSTEM_PROMPT = (
    "You are drafting the short summary section of a wound assessment for a clinician to review. "
    "Use ONLY the facts in the FINDINGS JSON and what is visible in the image. Do not introduce any number that is "
    "not in the JSON. Do not name medicines, doses or procedures. Do not state a definitive diagnosis; use phrases "
    "like 'appearance is consistent with'. If a finding is marked uncertain, say it is uncertain. "
    "Write 3 to 5 plain sentences, no lists, no headings."
)


def llm_user_prompt(f: dict) -> str:
    keep = {k: f[k] for k in ("wound_type", "severity", "measurement", "tissue_pct", "change", "intake") if f.get(k)}
    return "FINDINGS JSON:\n" + json.dumps(keep, indent=1) + "\n\nWrite the summary."


BANNED = [r"\bdiagnos(ed|is)\b", r"\b\d+(\.\d+)?\s?(mg|ml|mcg|iu)\b", r"\bprescrib", r"\bamputat",
          r"\bantibiotic", r"\bdefinitely\b", r"\bcertainly\b"]


def _flatten_numbers(obj) -> set[float]:
    nums = set()
    if isinstance(obj, dict):
        for v in obj.values():
            nums |= _flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            nums |= _flatten_numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        nums.add(float(obj))
        if 0 <= obj <= 1:
            nums.add(round(float(obj) * 100))  # probabilities may be written as percentages
    elif isinstance(obj, str):
        nums |= {float(x) for x in re.findall(r"\d+(?:\.\d+)?", obj)}
    return nums


def check_narrative(text: str, f: dict) -> list[str]:
    """Return a list of problems. Empty list = narrative may be used."""
    problems = []
    allowed = _flatten_numbers(f)
    for s in re.findall(r"\d+(?:\.\d+)?", text):
        x = float(s)
        if not any(abs(x - y) <= 0.051 * max(1.0, abs(y)) for y in allowed):
            problems.append(f"number not in findings: {s}")
    for pat in BANNED:
        if re.search(pat, text, flags=re.I):
            problems.append(f"banned phrase: {pat}")
    if len(text.split()) > 160:
        problems.append("too long")
    return problems


def build_report(f: dict, llm_text: str | None = None) -> tuple[str, list[str]]:
    """Render the report, using the LLM narrative only if it passes the checks."""
    f.setdefault("flags", red_flags(f))
    problems = check_narrative(llm_text, f) if llm_text else []
    narrative = llm_text.strip() if llm_text and not problems else None
    return render_report(f, narrative), problems


# --------------------------------------------------------------------------- treatment report

TRAJECTORY_TEXT = {"improving": "Improving", "static": "Static, change within measurement noise",
                   "deteriorating": "Deteriorating"}


def _change_line(label: str, c: dict | None) -> str | None:
    if not c:
        return None
    parts = []
    if "percent_area_reduction" in c:
        verb = "smaller" if c["percent_area_reduction"] >= 0 else "larger"
        parts.append(f"area {c['area_before_cm2']} → {c['area_after_cm2']} cm² "
                     f"({abs(c['percent_area_reduction'])}% {verb})")
        if "cm2_per_week" in c:
            parts.append(f"{c['cm2_per_week']} cm²/week")
        if "edge_advance_cm_per_week" in c:
            parts.append(f"edge advance {c['edge_advance_cm_per_week']} cm/week")
    if "nonviable_before_pct" in c:
        parts.append(f"non-viable tissue {c['nonviable_before_pct']}% → {c['nonviable_after_pct']}%")
    if not parts:
        return None
    days = f" ({c['days']:g} days)" if c.get("days") else ""
    return f"- {label}{days}: " + "; ".join(parts)


def treatment_narrative(prog: dict, care: dict) -> str:
    parts = []
    h = prog.get("healing") or {}
    if h.get("trajectory"):
        basis = "wound area" if h["basis"] == "area" else "the tissue mix (no size: sticker missing)"
        parts.append(f"Since the last visit the wound is {h['trajectory']}, judged by {basis}.")
    elif h:
        parts.append(f"Healing not compared: {h.get('reason', 'not enough photos')}.")
    s = prog.get("session") or {}
    if "nonviable_removed_points" in s:
        parts.append(f"This session reduced non-viable tissue from {s['nonviable_before_pct']}% to "
                     f"{s['nonviable_after_pct']}%.")
    if care.get("suggestions"):
        parts.append("Suggested considerations: " + ", ".join(x["action"] for x in care["suggestions"]) + ".")
    return " ".join(parts)


def render_treatment_report(req: dict, prog: dict, care: dict, flags: list[dict], narrative: str | None = None) -> str:
    """The treatment's draft: the PRE photo's findings, what the session did, healing since earlier visits,
    and the care suggestions. Numbers and suggestions come from progress.py and care.py, never from the LLM."""
    treatments = req.get("treatments") or [{}]
    cur = treatments[-1]
    seq = cur.get("sequence") or len(treatments)
    lines = [f"# Treatment {seq} assessment (AI-assisted draft)", f"_{DISCLAIMER}_", ""]
    if has_danger_signs(flags):
        lines += ["**Emergency care now: do not wait for this report.** Danger signs are listed under Flags.", ""]
    lines.append("## Flags")
    lines += [f"- **{fl['level'].upper()}**: {fl['text']}" for fl in flags] or ["- None raised by the automatic rules."]
    lines += ["", "## Summary", narrative or treatment_narrative(prog, care), "",
              "## Findings", f"- Wound type: {_fmt_class(req.get('wound_type'))}"]
    for k, v in (req.get("severity") or {}).items():
        lines.append(f"- {k.replace('_', ' ').capitalize()}: {_fmt_class(v)}")
    for name, obs in (("Before treatment", cur.get("pre")), ("After cleaning", cur.get("post"))):
        if not obs:
            continue
        if obs.get("status") != "ok":
            lines.append(f"- {name}: photo not analysable ({obs.get('status')}).")
            continue
        size = f"area {obs['area_cm2']} cm², {obs['length_cm']} x {obs['width_cm']} cm" if obs.get("area_cm2") \
            else "size not measured (no sticker)"
        t = obs.get("tissue_pct")
        tissue = ("; tissue " + ", ".join(f"{k} {v}%" for k, v in t.items())) if t else ""
        lines.append(f"- {name}: {size}{tissue}")

    s = prog.get("session")
    if s:
        lines += ["", "## This visit (before → after cleaning)"]
        if "area_before_cm2" in s:
            note = f", {s['area_note']}" if s.get("area_note") else ""
            lines.append(f"- Area {s['area_before_cm2']} → {s['area_after_cm2']} cm²{note}")
        if "nonviable_removed_points" in s:
            lines.append(f"- Non-viable tissue {s['nonviable_before_pct']}% → {s['nonviable_after_pct']}%")

    h = prog.get("healing") or {}
    lines += ["", "## Healing"]
    if h.get("trajectory"):
        which = "after-cleaning" if h["phase"] == "post" else "before-treatment"
        lines.append(f"- Trajectory: {TRAJECTORY_TEXT[h['trajectory']]} (by {h['basis']}, comparing {which} photos)")
        nb = h.get("noise_band")
        if nb:
            lines.append(f"- Noise band: {nb['smaller_pct']:g}% smaller to {nb['larger_pct']:g}% larger counts as no "
                         f"change ({nb['source']})")
        lines += [x for x in (_change_line("Since last visit", h.get("since_last")),
                              _change_line("Since first visit", h.get("since_first"))) if x]
        fw = h.get("four_week")
        if fw:
            state = "on track" if fw["on_track"] else "NOT on track"
            lines.append(f"- 4-week check: {fw['percent_area_reduction']}% smaller at day {fw['days']:g} "
                         f"(target {fw['target']:.0f}%): {state}")
    else:
        lines.append(f"- Not compared: {h.get('reason', 'not enough photos')}.")
    p = prog.get("push")
    if p:
        lines.append(f"- PUSH score {p['score']}/17 (size {p['size']}, exudate {p['exudate']}, tissue {p['tissue']}; "
                     f"{p['note']})")

    lines += ["", "## For the clinician to consider"]
    for c in care.get("checks", []):
        lines.append(f"- **Check**: {c['text']}")
    for c in care.get("contraindications", []):
        lines.append(f"- **Avoid {c['action']}**: {c['reason']}.")
    for sg in care.get("suggestions", []):
        alt = f" (or {', '.join(sg['alternatives'])})" if sg["alternatives"] else ""
        lines.append(f"- **{sg['action']}**{alt}: {sg['text']} Because: {'; '.join(sg['because'])}.")
    if not any(care.get(k) for k in ("checks", "contraindications", "suggestions")):
        lines.append("- No rule-based suggestions.")
    lines += ["", "## For the reviewing clinician",
              "- Suggestions come from placeholder rules awaiting clinical sign-off; accept or decline each one.",
              "- Assess depth, undermining, pulses/perfusion and infection signs, which a photo cannot show.",
              "", f"Rules version: {care.get('rules_version')}"]
    return "\n".join(lines)
