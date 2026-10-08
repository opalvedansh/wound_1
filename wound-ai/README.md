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
| `wound_ai/quality.py` | Rejects blurry / dark / glare photos with a retake instruction |
| `wound_ai/measure.py` | ArUco sticker → homography → area, length, width, perimeter, % change |
| `wound_ai/intake.py` | Questions the app asks, with adaptive follow-ups (burn, surgical, pressure, diabetic) |
| `wound_ai/report.py` | Red-flag rules (placeholders for clinical sign-off), template report, LLM guard |
| `wound_ai/llm.py` | MedGemma loading and summary generation (optional) |
| `wound_ai/pipeline.py` | `WoundAnalyzer`: everything above, end to end |
| `scripts/prepare_data.py` | Merge datasets into one manifest; near-duplicate grouping to stop leakage |
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

## Serving

```bash
WOUND_API_KEY=change-me CKPT_DIR=checkpoints uvicorn api.server:app --port 8000
curl -H "X-API-Key: change-me" -F image=@photo.jpg \
  -F 'intake={"body_location":"heel","diabetes":"yes","cause":"started_on_its_own"}' localhost:8000/analyze
```

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
- The red-flag rules in `report.py` are placeholders. Your clinical partner signs them off.
- The API stores nothing unless `STORE_CASES=1` (only inside an ethics-approved, consented study).
