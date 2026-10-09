# wound-ai — clinician-assist wound assessment from a phone photo

Photo + a few intake questions → wound outline, size in cm², wound type, stage/depth,
tissue mix, red flags → a **draft** report a clinician approves, edits or rejects.

This is a starter kit for the roadmap document. It is a set of separately trained,
separately validated modules, not one black-box model, because a clinical tool has to
show *why* it said something and has to be validated piece by piece.

```
photo ─► quality gate ─► wound segmentation ─► crop ─► wound-type classifier ─► stage/depth head
           (retake?)          │                                 ▲
                              │                    intake answers (body site, diabetes…)
                              ├─► tissue segmentation (granulation / slough / necrosis / epithelial / redness)
                              └─► ArUco sticker ─► size in cm² (tilt-corrected)
                                                        │
                    findings JSON ─► red-flag rules ─► report (template, or MedGemma summary behind a guard)
                                                        │
                                              clinician review ─► logged corrections ─► retraining data
```

## Layout

| Path | What it does |
|---|---|
| `wound_ai/data.py` | Manifest-driven datasets, colour-safe augmentations, **patient-level** splits |
| `wound_ai/models.py` | SegFormer/U-Net builders, image+metadata classifier, temperature scaling |
| `wound_ai/quality.py` | Warns about low-resolution / blurry / dark / glare photos (analysed, flagged for review); refuses only photos with nothing to analyse |
| `wound_ai/measure.py` | ArUco sticker → homography → area, length, width, perimeter, % change |
| `wound_ai/intake.py` | Questions the app asks, with adaptive follow-ups (burn, surgical, pressure, diabetic) |
| `wound_ai/report.py` | Red-flag rules (placeholders for clinical sign-off), template report, LLM guard |
| `wound_ai/progress.py` | Healing visit to visit (like with like: after-cleaning photos), what a session's debridement did, 4-week check, PUSH |
| `wound_ai/care.py` | Care suggestions from a rule table (TIME framework, contraindications; placeholders for sign-off), the treatment report |
| `wound_ai/llm.py` | MedGemma loading and summary generation (optional) |
| `wound_ai/pipeline.py` | `WoundAnalyzer`: everything above, end to end |
| `scripts/prepare_data.py` | Merge datasets into one manifest; near-duplicate grouping to stop leakage |
| `scripts/repeatability.py` | Measurement noise from a repeat-photo study (docs/repeatability_study.md); `--write` makes healing use it |
| `scripts/build_tissue_dataset.py` | The three public tissue datasets mapped to our classes, locked test set, unlabelled pool, clinician agreement |
| `scripts/pseudo_label.py` | A trained tissue model labels unlabelled photos where it is confident (semi-supervised training) |
| `scripts/tissue_cv.py` | Tissue model: per fold teacher → pseudo-labels → student, scored on the locked test set against clinician agreement |
| `wound_ai/losses.py` | Partial-label losses: each image only teaches the tissue classes its dataset labels |
| `scripts/prepare_tissue.py` | Rewrite a tissue dataset's masks into this project's tissue classes (map file per dataset) |
| `scripts/train_seg.py` | Wound boundary or tissue segmentation (resumable for Kaggle time limits) |
| `scripts/train_cls.py` | Any label column (wound type, PU stage, burn depth, DFU infection) + calibration |
| `scripts/evaluate.py` | Test-set metrics with 95% CIs, subgroup breakdown, area agreement |
| `scripts/cv.py` | K-fold cross-validation (grouped by patient, stratified), timed first: 5 folds if they fit the budget, else 3; folds in parallel across GPUs; mean ± sd on the locked test set |
| `scripts/finetune_medgemma.py` | QLoRA fine-tune of the summary writer on clinician-approved text |
| `scripts/make_marker.py` | Printable A4 sheet of 20 mm calibration stickers |
| `scripts/smoke_test.py` | Synthetic end-to-end test of every piece (CPU, under a minute) |
| `docs/roadmap.md` | The full roadmap and build guide: architecture, data, training, evaluation, regulation |
| `docs/30_day_build_plan.md` | Day-by-day plan for the first month, from setup to the app integration |
| `docs/wound_flowcharts.md` | Wound classification flowcharts (Mermaid) = the labelling scheme; PNG copies in `docs/img/` |
| `api/server.py` | FastAPI backend for your Next.js front end |

## Quick start

```bash
pip install -r requirements.txt
python scripts/smoke_test.py          # should end with SMOKE TEST PASSED
python -m pytest tests                # API contract: key check, required answers, retake, outline
python scripts/make_marker.py         # markers.pdf, print at 100%
```

## Training on free Kaggle GPUs

1. Upload each dataset you're licensed to use as a **private** Kaggle dataset.
2. Notebook settings: GPU on, internet on (first run, for pretrained weights), persistence on.
3. Build the manifest, then train one component per session:

```bash
python scripts/prepare_data.py \
  --pairs fuseg:/kaggle/input/fuseg/train/images:/kaggle/input/fuseg/train/labels \
          mendeley:/kaggle/input/lower-limb/Wound_Main:/kaggle/input/lower-limb/Wound_Masked \
  --classes azh:/kaggle/input/azh:wound_type:configs/azh_map.json \
            piid:/kaggle/input/piid:pu_stage \
  --out /kaggle/working/manifest.csv

python scripts/train_seg.py --manifest /kaggle/working/manifest.csv --task boundary \
  --arch segformer --encoder mit_b2 --size 512 --batch-size 8 --epochs 60 --out /kaggle/working/runs/boundary
# session ended? same command + --resume

python scripts/train_cls.py --manifest /kaggle/working/manifest.csv --target wound_type \
  --backbone convnext_tiny.fb_in22k --meta-cols body_location --size 384 --out /kaggle/working/runs/wound_type

python scripts/evaluate.py --manifest /kaggle/working/manifest.csv --ckpt-dir checkpoints \
  --group-cols source,fitzpatrick --out reports/eval_test.json
```

Folder names above are examples; check each dataset's real layout after download.

Cross-validation instead of one split: `notebooks/kaggle_cv.ipynb` (runs `scripts/cv.py`, outline at 768 px).

Tissue model: `notebooks/kaggle_tissue_cv.ipynb` (runs `scripts/tissue_cv.py`). Only ~265 tissue-labelled photos exist
publicly (DFUTissue, LUTSeg, WoundTissue), so it is semi-supervised: a teacher labels a pool of ~11,400 other wound
photos (the public datasets + DFUC2022 + post-operative wounds, deduplicated; `link_kaggle_inputs.py --set pool`) where
it is confident and a student learns from both. Classes are scored against how well five clinicians agree with
each other (LUTSeg's gold standard), and the app uses only the classes that come close (`trusted_classes`).

More data: `notebooks/kaggle_public_cv.ipynb` adds seven public Kaggle wound datasets (attach them as inputs),
deduplicated against each other and the locked test sets by `scripts/build_public_dataset.py`: about 4,820 photos
for 7 wound types (burn, pressure, diabetic, venous, surgical, other, no wound) and 2,582 traced outlines.

## Serving

```bash
WOUND_API_KEY=change-me CKPT_DIR=checkpoints uvicorn api.server:app --port 8000
curl -H "X-API-Key: change-me" -F image=@photo.jpg \
  -F 'intake={"body_location":"heel","diabetes":"yes","cause":"started_on_its_own"}' localhost:8000/analyze
```

`POST /analyze` takes `phase=pre|post` (post: after cleaning, before the dressing, same visit). `POST /treatment-report`
takes a wound's analysed photos so far (JSON, no images) and returns healing, care suggestions and the treatment's draft;
see `wound_ai.care.treatment_report`.

Every route except `/health` needs the `X-API-Key` header, and the API refuses all requests if
`WOUND_API_KEY` is unset. `diabetes` and `cause` are required answers (HTTP 400 without them).
In this monorepo the NestJS API (`apps/api`) is the only caller: it holds the key, stores photos and
results, and serves the web portal.

## What has and hasn't been tested

- **Run here:** every module and script except `finetune_medgemma.py`, via `smoke_test.py`
  on synthetic images, plus a longer synthetic run (test Dice 0.97, area error 2.6%)
  confirming that training targets and inference-time resizing line up.
  Synthetic numbers say nothing about real wounds.
- **Not run here:** `finetune_medgemma.py` and `llm.py` (need a GPU and the gated model).
  Run the fine-tune with `--max-steps 5` first.

## Safety rules baked into the code

- Every output is labelled a draft for clinician review.
- Numbers, flags and disclaimers are rendered deterministically; the language model only
  writes the summary, which is discarded if it contains a number not in the findings or a
  banned phrase (doses, "diagnosed", "prescribe", …).
- No size is reported without the calibration sticker.
- Low-confidence classifications are reported as **uncertain** with the top alternatives.
- The red-flag rules in `report.py`, the care rules in `care.py` and the thresholds in `progress.py` are placeholders.
  Your clinical partner signs them off; bump `RULES_VERSION` on every change.
- Care suggestions come from the rule table, never from the language model; each names the finding behind it, and
  contraindications (e.g. compression without an ABPI) remove any suggestion they conflict with.
- Healing compares like with like (after-cleaning photos), only with the sticker in both, and calls changes inside
  the measurement noise static: a placeholder ±15% until `scripts/repeatability.py --write` saves the band measured
  by a repeat-photo study to `wound_ai/measurement_noise.json`. Every healing result names which one it used.
- The API stores nothing unless `STORE_CASES=1` (only inside an ethics-approved, consented study).
