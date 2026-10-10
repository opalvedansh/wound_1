"""Intake questions the app asks alongside the photo.

A photo alone cannot tell you about fever, diabetes, how the wound started or
how long it has been there, and those facts change the assessment. The app asks
the core questions every time, then follow-ups that depend on the answers (or on
the model's first guess of wound type).

Wording and options must be reviewed by your clinical partner and translated
(Hindi at minimum) before patients or ward staff use it.
"""
from __future__ import annotations

CORE_QUESTIONS = [
    {"id": "body_location", "text": "Where on the body is the wound?", "type": "choice",
     "options": ["foot_plantar", "foot_dorsal", "toe", "heel", "ankle", "lower_leg", "knee", "thigh", "sacrum_buttock",
                 "hip", "back", "abdomen", "chest", "arm", "hand", "head_neck", "other"],
     "used_by": ["wound_type classifier (metadata)", "report"]},
    {"id": "cause", "text": "How did the wound start?", "type": "choice",
     "options": ["pressure_lying_or_sitting", "burn", "surgery", "injury_cut_or_fall", "started_on_its_own", "unknown"],
     "used_by": ["wound_type classifier (metadata)", "follow-up selection"]},
    {"id": "duration_days", "text": "How long has the wound been there (days)?", "type": "number",
     "used_by": ["acute vs chronic", "report"]},
    {"id": "diabetes", "text": "Does the patient have diabetes?", "type": "choice", "options": ["yes", "no", "not_sure"],
     "used_by": ["red flags", "wound_type classifier (metadata)"]},
    {"id": "fever", "text": "Fever or chills in the last 48 hours?", "type": "choice", "options": ["yes", "no"],
     "used_by": ["red flags"]},
    {"id": "pain", "text": "Pain at the wound right now, 0 (none) to 10 (worst)?", "type": "number",
     "used_by": ["report", "infection screen"]},
    {"id": "odour", "text": "Is there a bad smell from the wound?", "type": "choice", "options": ["yes", "no"],
     "used_by": ["infection screen"]},
    {"id": "discharge", "text": "Discharge from the wound?", "type": "choice",
     "options": ["none", "clear_watery", "blood_stained", "thick_yellow_or_green"], "used_by": ["infection screen"]},
    {"id": "redness_spreading", "text": "Is redness or swelling around the wound spreading?", "type": "choice",
     "options": ["yes", "no", "not_sure"], "used_by": ["red flags"]},
    {"id": "current_treatment", "text": "Current dressing or treatment (free text)", "type": "text",
     "used_by": ["report"]},
    {"id": "abpi", "text": "ABPI (ankle-brachial pressure index), if measured. Leave blank if not.", "type": "number",
     "used_by": ["blood-flow flag for leg and foot ulcers"]},
    {"id": "measured_length_cm", "text": "Longest length of the wound measured with a ruler (cm), if measured. "
                                         "Leave blank if not.", "type": "number",
     "used_by": ["size, when the photo has no calibration sticker and no phone distance reading"]},
]

# Body sites where the flowcharts apply the diabetic-foot rule (Chart 1) and the blood-flow check (Charts 2 and 3).
FOOT_SITES = {"foot_plantar", "foot_dorsal", "toe", "heel"}
LEG_AND_FOOT_SITES = FOOT_SITES | {"ankle", "lower_leg"}
# Burns here go to a burns unit whatever their size (Chart 1 danger signs).
SPECIAL_BURN_SITES = {"head_neck", "hand"} | FOOT_SITES

FOLLOW_UPS = {
    "burn": [
        {"id": "burn_agent", "text": "What caused the burn?", "type": "choice",
         "options": ["flame", "hot_liquid", "hot_surface", "chemical", "electrical", "other"]},
        {"id": "hours_since_burn", "text": "How many hours ago did the burn happen?", "type": "number"},
        {"id": "burn_other_sites", "text": "Are other body areas burned too?", "type": "choice", "options": ["yes", "no"]},
    ],
    "surgery": [
        {"id": "days_since_surgery", "text": "How many days since the operation?", "type": "number"},
        {"id": "wound_opening", "text": "Has the cut opened up anywhere?", "type": "choice", "options": ["yes", "no"]},
    ],
    "pressure": [
        {"id": "mobility", "text": "Is the patient bed-bound or chair-bound?", "type": "choice",
         "options": ["bed_bound", "chair_bound", "walks_with_help", "walks"]},
        {"id": "repositioning", "text": "How often is the patient turned or repositioned?", "type": "choice",
         "options": ["every_2h_or_more", "few_times_a_day", "rarely", "unknown"]},
    ],
    "diabetic": [
        {"id": "foot_numbness", "text": "Numbness or loss of feeling in the feet?", "type": "choice",
         "options": ["yes", "no", "not_sure"]},
        {"id": "foot_cold_or_dark", "text": "Is the foot cold, pale, or are any toes turning dark?", "type": "choice",
         "options": ["yes", "no"]},
        {"id": "previous_ulcer_or_amputation", "text": "Any previous foot ulcer or amputation?", "type": "choice",
         "options": ["yes", "no"]},
    ],
}


def follow_up_questions(answers: dict, predicted_type: str | None = None) -> list[dict]:
    """Return the extra questions to ask, based on answers so far and the model's first guess."""
    keys = set()
    cause = answers.get("cause")
    if cause == "burn" or predicted_type == "burn":
        keys.add("burn")
    if cause == "surgery" or predicted_type == "surgical":
        keys.add("surgery")
    if cause == "pressure_lying_or_sitting" or predicted_type == "pressure":
        keys.add("pressure")
    if answers.get("diabetes") == "yes" or predicted_type == "diabetic":
        keys.add("diabetic")
    qs = []
    for k in sorted(keys):
        qs.extend(q for q in FOLLOW_UPS[k] if q["id"] not in answers)
    return qs
