"""HTTP API for the wound analyzer (FastAPI). Your Next.js app calls this.

    WOUND_API_KEY=... CKPT_DIR=checkpoints uvicorn api.server:app --host 0.0.0.0 --port 8000

Endpoints (all but /health need the X-API-Key header to match WOUND_API_KEY)
  GET  /health                    model versions
  GET  /intake/questions          core intake questions for the form
  POST /intake/follow-ups         extra questions given answers so far (+ model's first guess)
  POST /analyze                   photo + intake answers -> findings + report draft
  POST /analyze/overlay           photo -> the photo with the wound outline drawn on it (JPEG), for checking by eye
  POST /treatment-report          a wound's analysed photos so far -> healing, care suggestions, draft
  POST /review                    clinician approves / edits / rejects a draft

Only the app's server holds the key; phones and browsers never call this API directly.
If WOUND_API_KEY is unset the API refuses every request rather than running open.

Privacy: by default nothing is written to disk. Set STORE_CASES=1 only inside an
ethics-approved study where patients consented to their data being kept; cases
are then saved under CASE_DIR for annotation and for fine-tuning the summary model.
Put this behind authentication (clinician accounts) and HTTPS before real use.
"""
from __future__ import annotations

import hmac
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from pydantic import BaseModel

from wound_ai.care import treatment_report
from wound_ai.intake import CORE_QUESTIONS, follow_up_questions
from wound_ai.pipeline import WoundAnalyzer, draw_overlay

CKPT_DIR = os.environ.get("CKPT_DIR", "checkpoints")
STORE = os.environ.get("STORE_CASES") == "1"
CASE_DIR = Path(os.environ.get("CASE_DIR", "data/cases"))
MAX_BYTES = 15 * 1024 * 1024
# The assessment changes with these answers (diabetic-foot rule, follow-up selection), so they are never optional.
REQUIRED_INTAKE = ("diabetes", "cause")

llm = None
if os.environ.get("USE_LLM") == "1":
    from wound_ai import llm as llm_mod

    llm = llm_mod.load(adapter=os.environ.get("LLM_ADAPTER") or None)

analyzer = WoundAnalyzer(CKPT_DIR, marker_mm=float(os.environ.get("MARKER_MM", 20)), llm=llm)
app = FastAPI(title="Wound assessment API", version="0.1.0")


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("WOUND_API_KEY", "")
    if not expected:
        raise HTTPException(503, "WOUND_API_KEY is not set on the server, so no request is accepted")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(401, "missing or wrong X-API-Key")


keyed = [Depends(require_key)]


class FollowUpRequest(BaseModel):
    answers: dict
    predicted_type: str | None = None


class TreatmentReportRequest(BaseModel):
    wound_type: dict | None = None   # the PRE photo's class result from /analyze
    severity: dict | None = None     # and its severity results (pu_stage, burn_depth, dfu_infection)
    intake: dict = {}
    treatments: list[dict]           # visit order, the current treatment last (see wound_ai.care.treatment_report)


class Review(BaseModel):
    case_id: str
    reviewer_id: str
    decision: str              # approved | edited | rejected
    final_summary: str | None = None
    corrections: dict | None = None   # e.g. {"wound_type": "venous", "pu_stage": null}


@app.get("/health")
def health():
    return {"ok": True, "models": {**{k: v["version"] for k, v in analyzer.seg.items()},
                                   **{k: v["version"] for k, v in analyzer.cls.items()}}}


@app.get("/intake/questions", dependencies=keyed)
def questions():
    return CORE_QUESTIONS


@app.post("/intake/follow-ups", dependencies=keyed)
def follow_ups(req: FollowUpRequest):
    return follow_up_questions(req.answers, req.predicted_type)


# A plain `def`: inference is synchronous, so FastAPI runs it in a worker thread instead of blocking the server.
@app.post("/analyze", dependencies=keyed)
def analyze(image: UploadFile = File(...), intake: str = Form("{}"), previous: str = Form(""),
            phase: str = Form("pre")):
    try:
        answers = json.loads(intake or "{}")
        prev = json.loads(previous) if previous else None
    except json.JSONDecodeError:
        raise HTTPException(400, "intake/previous must be JSON")
    if not isinstance(answers, dict):
        raise HTTPException(400, "intake must be a JSON object")
    if phase not in ("pre", "post"):
        raise HTTPException(400, "phase must be pre or post")
    missing = [k for k in REQUIRED_INTAKE if answers.get(k) in (None, "")]
    if missing:
        raise HTTPException(400, {"error": "required intake answers missing", "missing": missing})
    data, rgb = read_upload(image)
    case_id = uuid.uuid4().hex
    f = analyzer.analyze(rgb, answers, prev)
    f["case_id"] = case_id
    f["phase"] = phase
    # After cleaning, in the same visit: the PRE photo's answers already chose the follow-ups.
    if f.get("wound_type") and phase == "pre":
        f["follow_up_questions"] = follow_up_questions(answers, f["wound_type"]["label"])
    if STORE:
        d = CASE_DIR / case_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "image.jpg").write_bytes(data)
        (d / "findings.json").write_text(json.dumps(f, indent=1, default=str))
    return f


def read_upload(image: UploadFile) -> tuple[bytes, np.ndarray]:
    """The uploaded photo's bytes and RGB pixels (EXIF rotation applied, as phones save portrait photos)."""
    data = image.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not read image")
    return data, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@app.post("/analyze/overlay", dependencies=keyed, response_class=Response,
          responses={200: {"content": {"image/jpeg": {}}, "description": "The photo with the outline drawn on it"},
                     422: {"description": "No outline to draw: the photo needs a retake, or no wound was found"}})
def analyze_overlay(image: UploadFile = File(...), tissue: bool = Form(False)):
    """The photo with the wound outline the model found drawn on it (and the tissue layers if `tissue` is true and a
    tissue model is installed), as a JPEG to look at or download. The outline does not depend on the intake answers,
    so none are needed. Nothing is stored."""
    _, rgb = read_upload(image)
    f = analyzer.analyze(rgb, {})
    if not f.get("outline"):
        raise HTTPException(422, {"status": f.get("status"), "issues": (f.get("quality") or {}).get("issues") or [],
                                  "detail": "no outline to draw" if f.get("status") == "ok" else "photo not analysed"})
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(draw_overlay(rgb, f, tissue), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(jpg.tobytes(), media_type="image/jpeg",
                    headers={"Content-Disposition": 'inline; filename="wound-outlined.jpg"'})


@app.post("/treatment-report", dependencies=keyed)
def report(req: TreatmentReportRequest):
    if not req.treatments:
        raise HTTPException(400, "treatments must list at least the current treatment")
    return treatment_report(req.model_dump())


@app.post("/review", dependencies=keyed)
def review(r: Review):
    if r.decision not in ("approved", "edited", "rejected"):
        raise HTTPException(400, "decision must be approved, edited or rejected")
    if STORE:
        d = CASE_DIR / r.case_id
        if not d.exists():
            raise HTTPException(404, "unknown case")
        (d / "review.json").write_text(r.model_dump_json(indent=1))
    return {"ok": True, "stored": STORE}
