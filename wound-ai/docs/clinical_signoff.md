# Clinical sign-off: care suggestions, healing thresholds and red flags

**Rules under review:** `care-0.1-unsigned` (wound_ai/care.py, wound_ai/progress.py, wound_ai/pipeline.py) and `flags-0.1-unsigned` (wound_ai/report.py)
**Reviewer:** a clinician responsible for wound care at the deploying clinic

## What you are signing off

After each visit, the app writes a **draft** for a clinician to review. It covers:

- **Healing:** how the wound is healing since earlier visits.
- **Session effect:** what this visit's cleaning or debridement did.
- **Considerations:** a short list of options to consider, for example "Consider debridement. Because: slough and necrosis 45% of the wound bed".

A fixed rule table produces these considerations, not an AI model. Each one names the finding that triggered it, and a clinician must accept or decline each one before the draft is approved. Nothing reaches the patient record or the nurse's phone until then.

The app is limited to the clinic's own therapy and dressing options. It never names medicines or doses, and never says "diagnosis" or "prescribe".

Your job is to confirm, change or remove every rule and number below. Until all of them are signed off, the drafts are labelled "placeholder rules, not yet clinically signed off".

**What a photo cannot show:** depth, undermining, blood flow, pain or infection. The rules therefore also use the clinician's assessment form (exudate, infection signs, edge, periwound skin) and the intake answers (diabetes, ABPI, foot cold or dark). If something a rule needs is missing, the draft asks for it ("Record the exudate amount…") rather than guessing.

For each item, tick one box and write the new value or wording if you change it.

---

## 1. Terms the rules use

| # | Term | Current definition | Decision |
|---|---|---|---|
| 1.1 | **Non-viable tissue** | Slough + necrosis, as a % of the wound bed, from the tissue model. Not used if the model's confidence is below the gate in H9. | ☐ Approve ☐ Change ☐ Remove |
| 1.2 | **Possible poor blood flow** (ischaemia) | "Foot cold, pale or toes darkening" answered yes, **or** ABPI below 0.8 | ☐ Approve ☐ Change ☐ Remove |
| 1.3 | **Signs of infection** | Any one of: (a) 2 or more infection signs ticked on the assessment form, not counting "Fever/systemic signs"; (b) redness covering 30% or more of the skin ring around the wound (from the photo); (c) exudate type Purulent; (d) discharge "thick yellow or green" | ☐ Approve ☐ Change ☐ Remove |
| 1.4 | **Macerated skin** | Edge or periwound recorded as Macerated, **or** maceration covering 20% or more of the skin ring around the wound (from the photo) | ☐ Approve ☐ Change ☐ Remove |
| 1.5 | **Normal ABPI range for compression** | 0.8 to 1.3 (also used by the existing blood-flow flag) | ☐ Approve ☐ Change ☐ Remove |
| 1.6 | **Leg or foot wound** | Body site foot (sole, top), toe, heel, ankle or lower leg | ☐ Approve ☐ Change ☐ Remove |

Notes: ____________________________________________

## 2. Considerations (what the draft may suggest)

| ID | When | Draft says | Decision |
|---|---|---|---|
| D0 | A danger sign is flagged (e.g. fever with spreading redness, darkening toes) | "Danger signs present: arrange urgent care before routine wound care." (listed first) | ☐ Approve ☐ Change ☐ Remove |
| T0 | Tissue mix not available from the model | Check: "Tissue mix not assessed by the model: record it on examination." | ☐ Approve ☐ Change ☐ Remove |
| T1 | Non-viable tissue **30% or more** before treatment (no after-cleaning photo yet) | **Debridement:** "Consider debridement of non-viable tissue." | ☐ Approve ☐ Change threshold to ___% ☐ Remove |
| T2 | Non-viable tissue still **30% or more** in the after-cleaning photo | **Debridement:** "Consider continuing debridement at the next visit." | ☐ Approve ☐ Change threshold to ___% ☐ Remove |
| I1 | Signs of infection (1.3) | **Silver:** "Consider an antimicrobial (silver) dressing and assess for infection." | ☐ Approve ☐ Change ☐ Remove |
| I2 | Signs of infection (1.3) | Check: "Assess for local or spreading infection per local protocol." | ☐ Approve ☐ Change ☐ Remove |
| M0 | Exudate amount not recorded | Check: "Record the exudate amount to guide the dressing choice." | ☐ Approve ☐ Change ☐ Remove |
| M1 | Exudate **Heavy** | **Alginate** (or Foam): "Consider a highly absorbent dressing." | ☐ Approve ☐ Change ☐ Remove |
| M2 | Exudate **Moderate** | **Foam** (or Alginate): "Consider an absorbent foam dressing." | ☐ Approve ☐ Change ☐ Remove |
| M3 | Exudate None or Scant, some non-viable tissue, no signs of infection | **Hydrogel:** "Consider adding moisture to help soften non-viable tissue." | ☐ Approve ☐ Change ☐ Remove |
| M4 | Macerated skin (1.4) | Check: "Macerated skin: protect the periwound skin and review absorbency." | ☐ Approve ☐ Change ☐ Remove |
| E1 | Edge recorded as Rolled or Undermined | Check: "Probe for undermining; the edge may need refreshing." | ☐ Approve ☐ Change ☐ Remove |
| E2 | Behind the 4-week benchmark (section 4) | **NPWT** (or Skin Substitute): "Healing is behind the 4-week benchmark: reassess the cause and consider an advanced therapy." | ☐ Approve ☐ Change ☐ Remove |
| E3 | Same dressing for the last **3** visits and the wound is static or deteriorating | Check: "Review the care plan." That dressing is also listed after its alternative in M1/M2. | ☐ Approve ☐ Change visits to ___ ☐ Remove |
| T3 | Callus covering 10% or more of the skin ring around the wound (tissue model) | Check: "Callus around the ulcer: consider removing it and review offloading (callus raises pressure on the wound)." | ☐ Approve ☐ Change ☐ Remove |
| R0 | Venous ulcer, ABPI not recorded | Check: "Measure the ABPI: compression is the main treatment for venous ulcers but needs adequate blood flow." | ☐ Approve ☐ Change ☐ Remove |
| R1 | Venous ulcer, ABPI 0.8–1.3 | **Compression:** "Consider compression therapy." | ☐ Approve ☐ Change ☐ Remove |
| R2 | Diabetic foot ulcer on the foot | **Offloading:** "Consider offloading the ulcer (footwear, cast or boot)." | ☐ Approve ☐ Change ☐ Remove |
| R3 | Pressure injury (by type or cause) | **Pressure redistribution:** "Consider repositioning and a pressure-redistributing surface." | ☐ Approve ☐ Change ☐ Remove |

Missing rules you want added: ____________________________________________

## 3. Contraindications (the draft never suggests these, and warns if they are on the current plan)

| # | Never suggest | When | Decision |
|---|---|---|---|
| C1 | Debridement | Possible poor blood flow (1.2): "vascular assessment before sharp debridement" | ☐ Approve ☐ Change ☐ Remove |
| C2 | Debridement | Heel, necrosis 50% or more, exudate None/Scant, no signs of infection: "dry, stable heel eschar is often left intact" | ☐ Approve ☐ Change ☐ Remove |
| C3 | Hydrogel | Possible poor blood flow (1.2): "dry necrosis is usually kept dry until assessed" | ☐ Approve ☐ Change ☐ Remove |
| C4 | Compression | Leg or foot wound (or venous ulcer) with ABPI not recorded, below 0.8, or above 1.3 (above: "arteries may be calcified: check toe pressures") | ☐ Approve ☐ Change ☐ Remove |
| C5 | Hydrocolloid | Signs of infection or heavy exudate | ☐ Approve ☐ Change ☐ Remove |
| C6 | NPWT | Any necrosis in the bed, signs of infection, or possible poor blood flow | ☐ Approve ☐ Change ☐ Remove |

Missing contraindications (e.g. malignancy, exposed vessels, untreated osteomyelitis; these are not captured by the forms today): ____________________________________________

## 4. Healing thresholds

Healing compares **like with like**: after-cleaning photo with after-cleaning photo, or photos of the wound as found with each other. Area is compared only when the calibration sticker is in both photos. Without it, only the tissue mix is compared. A larger area right after debridement is reported as expected, never as worse.

A treatment's post photo is one of two things, told apart by when it was taken (H11): taken in the same visit, it shows what the cleaning did; taken days later, it shows the wound as the treatment left it, and the change from the pre photo is reported as improving, static or deteriorating.

| # | Threshold | Current value | Decision |
|---|---|---|---|
| H1 | **Measurement noise (area):** change treated as "static" | Placeholder ±15% until the repeat-photo study ([repeatability_study.md](repeatability_study.md)) is done; then the measured band: ___% smaller / ___% larger, from ___ wounds (filled in from `wound_ai/measurement_noise.json`) | ☐ Approve the measured band ☐ Approve the placeholder for now ☐ Change |
| H2 | **Measurement noise (tissue):** change in non-viable tissue treated as "static" when no area is available | ±15 percentage points | ☐ Approve ☐ Change to ±___ |
| H3 | **4-week benchmark, diabetic foot ulcer** | At least 50% area reduction from the first photo | ☐ Approve ☐ Change to ___% |
| H4 | **4-week benchmark, venous leg ulcer** | At least 40% area reduction from the first photo | ☐ Approve ☐ Change to ___% |
| H5 | **4-week benchmark, other wound types** | None (area change reported, no benchmark) | ☐ Approve ☐ Add: ________ |
| H6 | **4-week window** | The visit between day 21 and day 35 closest to day 28 | ☐ Approve ☐ Change to ___–___ days |
| H7 | **Flag "Wound area has increased"** | Larger than the noise band (H1) since the last comparable photo | ☐ Approve ☐ Change ☐ Remove |
| H8 | **PUSH score** | Shown for all wounds, labelled "validated for pressure injuries". Exudate from the form: None 0, Scant 1, Moderate 2, Heavy 3 | ☐ Approve ☐ Pressure injuries only ☐ Remove |
| H10 | **Which tissue types the app may use** | Only classes whose cross-validated test Dice reaches 80% of the clinician-to-clinician Dice on the same photos (LUTSeg gold standard; 0.5 where no clinician figure exists), with at least 15 training and 10 test photos. Others are shown only as "possibly also: …, check on examination" | ☐ Approve ☐ Change ratio to ___ |
| H11 | **Same visit or later:** a post photo taken within this time of the pre photo is "after cleaning"; later than this it is compared with the pre photo for healing | 12 hours | ☐ Approve ☐ Change to ___ hours |
| H12 | **Redness around the wound, from the photo's colour:** how much redder the skin beside the wound is than skin further out (CIELAB a*). Shown as a hint with "shows less on darker skin: check on examination"; it raises no flag and feeds no suggestion | Mild from 4, marked from 8 | ☐ Approve ☐ Change to ___ / ___ ☐ Remove |
| H13 | **Depth** | Entered by the clinician (probe, cm) at each visit; shown beside the last recorded depth. No rule uses it yet | ☐ Approve ☐ Add a rule: ________ |
| H9 | **Tissue model confidence gate** | Tissue mix below 60% mean model confidence is shown as "uncertain" and not used by any rule | ☐ Approve ☐ Change to ___% |

## 5. Red flags and severity results (`wound_ai/report.py`, version `flags-0.1-unsigned`)

Red flags appear at the top of every draft. **Urgent** flags starting "Danger sign" make the report open with
"Emergency care now". Most flags come from the intake answers, not the photo.

| ID | When | Level | Draft says | Decision |
|---|---|---|---|---|
| F1 | Foot cold, pale or toes darkening (intake) | Urgent, danger sign | "A cold, pale or darkening foot or toes (possible gangrene or critical ischaemia): same-day vascular or diabetic foot team." | ☐ Approve ☐ Change ☐ Remove |
| F2 | Diabetes **and** necrosis in the tissue mix. *Note: the tissue model is not yet trusted on necrosis, so today this only fires from F1's intake answer.* | Urgent, danger sign | "Dark/necrotic tissue in a person with diabetes: same-day assessment by a diabetic foot or vascular team." | ☐ Approve ☐ Change ☐ Remove |
| F3 | Fever **and** (spreading redness **or** thick yellow/green discharge) | Urgent, danger sign | "Fever with spreading redness or pus: possible spreading infection, needs urgent medical review." | ☐ Approve ☐ Change ☐ Remove |
| F4 | Spreading redness without fever | Review | "Redness or swelling reported as spreading: clinician review within 24 hours." | ☐ Approve ☐ Change ☐ Remove |
| F5 | Burn, chemical or electrical | Urgent, danger sign | "Refer to a burns unit; surface appearance can underestimate damage." | ☐ Approve ☐ Change ☐ Remove |
| F6 | Burn on face, neck, hands or feet | Urgent, danger sign | "Refer to a burns unit." | ☐ Approve ☐ Change ☐ Remove |
| F7 | Burns on more than one body area | Urgent, danger sign | "Estimate the total area and refer to a burns unit." | ☐ Approve ☐ Change ☐ Remove |
| F8 | Burn-depth model says **full thickness** | Urgent | "Possible deep burn: burns specialist assessment." | ☐ Approve ☐ Change ☐ Remove |
| F9 | Burn-depth model says **partial thickness** (the public training data can't tell superficial from deep partial) | Review | "Possible partial-thickness burn: assess the depth (superficial or deep partial) on examination." | ☐ Approve ☐ Change ☐ Remove |
| F10 | Leg or foot ulcer (not an acute wound), ABPI not entered | Review | "Blood flow not assessed: check foot pulses and ABPI before any compression." | ☐ Approve ☐ Change ☐ Remove |
| F11 | ABPI below 0.8 (1.5) | Urgent | "Arterial or mixed disease possible. Vascular review before any compression." | ☐ Approve ☐ Change ☐ Remove |
| F12 | ABPI above 1.3 (1.5) | Review | "Arteries may be calcified … Check toe pressures." | ☐ Approve ☐ Change ☐ Remove |
| F13 | Pressure-stage model says **stage 3, stage 4, unstageable or deep tissue injury** | Review | "Possible … pressure injury: confirm the stage on examination and review pressure relief." | ☐ Approve ☐ Change level to ___ ☐ Remove |
| F14 | Wagner model says **grade 3** | Review | "Possible Wagner grade 3 ulcer (deep, abscess or bone involvement): diabetic foot team review this week; probe to bone on examination." | ☐ Approve ☐ Change level to ___ ☐ Remove |
| F15 | Surgical wound reported as opening | Review | "Contact the operating team." | ☐ Approve ☐ Change ☐ Remove |
| F16 | Possible exposed bone or tendon in the tissue mix | Review | "Confirm on examination (risk of bone infection)." | ☐ Approve ☐ Change ☐ Remove |
| F17 | Wound-type model below 70% confidence | Review | "Model is uncertain about the wound type: clinician to classify." | ☐ Approve ☐ Change threshold to ___% |
| F18 | No calibration sticker, or no wound found | Review | "Size not measured." | ☐ Approve ☐ Change ☐ Remove |
| F19 | Photo blurry, dark, glare or low resolution | Review | "Photo quality (…): the outline, size and tissue estimates may be less accurate." | ☐ Approve ☐ Change ☐ Remove |

**Severity results shown in the draft** (pressure-injury stage, burn depth, Wagner grade 0–3). Each is shown only for
its wound type and only if its cross-validated test macro-F1 reaches the bar below; under 70% confidence it is shown
as "uncertain" with the alternatives. Wagner grades 4–5 (gangrene) are not in the training data: F1 covers them.

| # | Item | Current value | Decision |
|---|---|---|---|
| S1 | Bar for showing a severity result at all | Mean test macro-F1 ≥ 0.60 over the folds | ☐ Approve ☐ Change to ___ |
| S2 | Burn depth classes | Superficial / partial thickness / full thickness | ☐ Approve ☐ Change |
| S3 | Pressure-injury classes | Stage 1–4, unstageable (deep tissue injury: too few public photos, not predicted) | ☐ Approve ☐ Change |

Missing red flags (e.g. signs of sepsis, uncontrolled pain, bleeding): ____________________________________________

## 6. Sign-off

When every box above is ticked, the developer:

1. applies the changes;
2. sets the rules versions to `care-1.0` and `flags-1.0`;
3. adds a test for each changed rule;
4. records the sign-off below.

Any later change to a rule or number needs a new sign-off and a new version (`care-1.1`, …). Each review in the app stores the version and which considerations the clinician accepted or declined, so the effect of a change can be checked.

| Name | Registration no. | Role | Signature | Date |
|---|---|---|---|---|
| | | | | |

**Signed-off version:** ________ **Code commit:** ________ (filled in by the developer)
