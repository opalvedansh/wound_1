# Clinical validation study: does the model agree with your clinicians on your patients?

**For:** the clinic lead, the clinician who signs off the rules, and whoever runs the analysis.
**Time:** no extra work per visit beyond the normal assessment form; about 3 months of clinic visits.

## Why

The model's accuracy so far comes from public photos (wound type ~90% on public test photos, 83% on an earlier
held-out set). Your patients, phones, lighting and wound mix are different, and only your own visits can show how well it
works for you. This study measures that, so you can decide which results the app should show and which to hide.

## How it works

The app already records two independent opinions on every visit:

1. **The nurse's assessment** on the phone: wound type, wound-bed tissue, pressure-injury stage, burn depth, Wagner
   grade, periwound skin, edges. The app only analyses the photo **after** the assessment is saved, so the nurse
   hasn't seen this visit's AI result. The phone does show the AI's wound type and area from **earlier** visits: during
   the study, assess each visit afresh from the wound in front of you, and nobody should show the nurse an AI draft
   before the form is saved.
2. **The model's findings** on the before-treatment photo of the same visit.

The study compares the two. Agreement is measured with one nurse, not a perfect answer: two clinicians also disagree,
so also do step 4 below.

## What to do

1. **Approval.** Get approval from your ethics committee or clinical governance lead, and follow your consent process for
   using patients' photos and assessments to evaluate the system. Record the approval reference here: __________.
2. **Who.** Every patient seen at the clinic during the study period (consecutive patients, not selected ones), every
   wound, every visit. Don't skip difficult photos.
3. **Answer every question** on the assessment form, including **wound bed tissue**, and the **stage, depth or grade**
   question for pressure injuries, burns and diabetic foot ulcers ("Not a …" for other wounds). These are the reference
   the model is scored against.
4. **Second opinion (inter-rater).** For at least **50 wounds**, a second clinician fills in the same questions from the
   same photo without seeing the first answers (paper form or a second assessment by a colleague). This shows how much
   two clinicians agree, which is the most the model can be expected to match.
5. **How many.** At least **200 visits** from at least **100 different wounds** (wound-type agreement to about ±5%), with
   at least **50 pressure injuries**, **50 diabetic foot ulcers** and **30 burns** for the stage, grade and depth results.
   Count wounds, not visits, for these minimums.
6. **Skin tone (strongly recommended).** Add a question for the Fitzpatrick skin type (I–VI) to the assessment form in the
   admin settings, so results can be checked for darker skin, where redness and tissue colour are harder to judge.
7. **Repeat-photo study** at the same time: [repeatability_study.md](repeatability_study.md). It sets how much a size
   change must be before the app calls it real.

## Targets (agree these with the clinician before the study starts)

| Finding | Shown in the app if agreement with the nurse is at least | And kappa at least |
|---|---|---|
| Wound type | ___% (suggested 80%) | ___ (suggested 0.7) |
| Pressure-injury stage, burn depth, Wagner grade | ___% each | ___ |
| Each wound-bed tissue present/absent | sensitivity ___%, specificity ___% | ___ |
| Periwound redness, maceration | sensitivity ___%, specificity ___% | ___ |

A finding that misses its target is hidden or shown only as "possibly, check on examination", the same way the tissue
model already handles classes it is not trusted on.

## Running the analysis

Monthly, and at the end:

1. Web portal, as an admin: download `GET /exports/validation` (de-identified: no names, months instead of dates).
2. Run:

   ```bash
   cd wound-ai
   python scripts/clinic_validation.py ~/Downloads/wound-validation-<date>-deid.csv --out reports/clinic_validation
   ```

3. `reports/clinic_validation/clinic_validation.md` has the agreement table with 95% confidence intervals, agreement by
   body site, and how often clinicians approved, edited or rejected the drafts and which fields they corrected.

Compare each figure with its target. Look at the confusion tables in the `.json` for patterns (e.g. venous ulcers called
pressure injuries), and at agreement by body site and skin tone.

## After the study

- Results that meet their targets: keep. Results that don't: hide, and add the clinic's labelled visits to the training
  data (the visits' assessments are training labels; corrected tissue outlines come from the CVAT round-trip).
- Retrain with the clinic's photos in the training data, keeping the **first 30% of patients as a locked clinic test
  set** that is never trained on, and run this analysis again on that locked set.
- Record the results, the model versions (in the export) and the sign-off in [clinical_signoff.md](clinical_signoff.md).

**Privacy:** the export is de-identified, but the photos are not part of it and stay in the clinic's storage. Keep the
CSV on clinic storage and delete it when the analysis is done.
