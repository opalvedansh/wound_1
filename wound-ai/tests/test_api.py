"""API contract the app relies on: key check, required answers, retake, and the outline.

Runs with an empty checkpoint folder, i.e. before any model is trained, which is how the app is wired up.

    .venv/bin/python -m pytest tests
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["CKPT_DIR"] = tempfile.mkdtemp(prefix="wound_ckpt_empty_")

from fastapi.testclient import TestClient  # noqa: E402

from api.server import app  # noqa: E402
from wound_ai.pipeline import mask_outline  # noqa: E402

KEY = "test-key"
INTAKE = {"diabetes": "no", "cause": "pressure_lying_or_sitting", "body_location": "heel"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("WOUND_API_KEY", KEY)
    return TestClient(app)


def jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def textured_photo() -> bytes:
    """Sharp, evenly lit 640x480 image that passes the quality gate."""
    rng = np.random.default_rng(0)
    return jpeg(rng.integers(40, 210, (480, 640, 3), dtype=np.uint8))


def analyze(client: TestClient, photo: bytes, intake: dict | str = INTAKE, key: str | None = KEY):
    headers = {"X-API-Key": key} if key is not None else {}
    body = intake if isinstance(intake, str) else json.dumps(intake)
    return client.post("/analyze", files={"image": ("w.jpg", photo, "image/jpeg")}, data={"intake": body}, headers=headers)


def test_health_is_open(client):
    assert client.get("/health").status_code == 200


def test_every_other_route_needs_the_key(client):
    assert client.get("/intake/questions").status_code == 401
    assert client.get("/intake/questions", headers={"X-API-Key": "wrong"}).status_code == 401
    assert analyze(client, textured_photo(), key=None).status_code == 401
    assert client.post("/review", json={"case_id": "x", "reviewer_id": "y", "decision": "approved"}).status_code == 401
    assert client.get("/intake/questions", headers={"X-API-Key": KEY}).status_code == 200


def test_refuses_everything_when_no_key_is_configured(monkeypatch):
    monkeypatch.delenv("WOUND_API_KEY", raising=False)
    c = TestClient(app)
    assert c.get("/intake/questions", headers={"X-API-Key": ""}).status_code == 503
    assert analyze(c, textured_photo(), key="anything").status_code == 503


def test_diabetes_and_cause_are_required(client):
    r = analyze(client, textured_photo(), {"body_location": "heel"})
    assert r.status_code == 400
    assert r.json()["detail"]["missing"] == ["diabetes", "cause"]
    assert analyze(client, textured_photo(), {"diabetes": "yes", "cause": ""}).json()["detail"]["missing"] == ["cause"]
    assert analyze(client, textured_photo(), "[1, 2]").status_code == 400


def test_works_before_any_model_is_trained(client):
    r = analyze(client, textured_photo())
    assert r.status_code == 200, r.text
    f = r.json()
    assert f["status"] == "ok"
    assert f["measurement"] is None
    assert f["outline"] is None
    assert any("Size not measured" in flag["text"] for flag in f["flags"])
    assert "AI-assisted draft" in f["report_markdown"]
    assert f["case_id"]


def test_blank_photo_asks_for_a_retake(client):
    flat = jpeg(np.full((480, 640, 3), 128, np.uint8))
    f = analyze(client, flat).json()
    assert f["status"] == "retake"
    assert any("blurry" in issue for issue in f["quality"]["issues"])
    assert "report_markdown" not in f


def test_outline_is_scaled_to_the_photo():
    mask = np.zeros((200, 400), np.uint8)
    mask[50:150, 100:300] = 1  # one 200x100 px wound
    mask[0:2, 0:2] = 1  # a speck, dropped
    polygons = mask_outline(mask)
    assert polygons is not None and len(polygons) == 1
    xs = [x for x, _ in polygons[0]]
    ys = [y for _, y in polygons[0]]
    assert min(xs) == pytest.approx(0.25, abs=0.01) and max(xs) == pytest.approx(0.75, abs=0.01)
    assert min(ys) == pytest.approx(0.25, abs=0.01) and max(ys) == pytest.approx(0.75, abs=0.01)


def test_no_outline_without_a_wound():
    assert mask_outline(None) is None
    assert mask_outline(np.zeros((10, 10), np.uint8)) is None


def test_black_letterbox_borders_are_not_glare():
    from wound_ai.quality import check_quality
    rng = np.random.default_rng(1)
    photo = rng.integers(40, 210, (600, 800, 3), dtype=np.uint8)
    padded = np.zeros((900, 900, 3), np.uint8)
    padded[150:750, 50:850] = photo  # a quarter of the frame is black padding
    q = check_quality(padded)
    assert q.ok, q.issues
    assert q.metrics["clipped_frac"] < 0.01


def test_post_photo_skips_follow_ups_and_rejects_unknown_phases(client):
    h = {"X-API-Key": KEY}
    files = {"image": ("w.jpg", textured_photo(), "image/jpeg")}
    r = client.post("/analyze", files=files, data={"intake": json.dumps(INTAKE), "phase": "post"}, headers=h)
    assert r.status_code == 200 and r.json()["phase"] == "post"
    assert "follow_up_questions" not in r.json()
    files = {"image": ("w.jpg", textured_photo(), "image/jpeg")}
    assert client.post("/analyze", files=files, data={"intake": json.dumps(INTAKE), "phase": "x"}, headers=h).status_code == 400


def test_treatment_report_from_one_analysed_photo(client):
    f = analyze(client, textured_photo()).json()
    from wound_ai.progress import observation

    body = {"wound_type": f.get("wound_type"), "intake": INTAKE,
            "treatments": [{"sequence": 1, "pre": observation(f, "2026-01-01T10:00:00+00:00"),
                            "assessment": {"exudate_level": "Heavy"}}]}
    assert client.post("/treatment-report", json=body).status_code == 401
    r = client.post("/treatment-report", json=body, headers={"X-API-Key": KEY})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["progress"]["healing"]["comparable"] is False
    assert any(s["action"] == "Alginate" for s in out["suggestions"])
    assert out["rules_version"] and "# Treatment 1 assessment" in out["report_markdown"]
    assert client.post("/treatment-report", json={"treatments": []}, headers={"X-API-Key": KEY}).status_code == 400


def test_overlay_needs_the_key_and_explains_when_there_is_no_outline(client):
    files = lambda: {"image": ("w.jpg", textured_photo(), "image/jpeg")}  # noqa: E731
    assert client.post("/analyze/overlay", files=files()).status_code == 401
    r = client.post("/analyze/overlay", files=files(), headers={"X-API-Key": KEY})
    assert r.status_code == 422  # no outline model in this test setup
    assert r.json()["detail"]["detail"] == "no outline to draw"


def test_overlay_draws_the_outline_where_the_model_put_it():
    from wound_ai.pipeline import draw_overlay

    img = np.full((200, 400, 3), 120, np.uint8)
    out = draw_overlay(img, {"outline": [[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]],
                             "tissue_outline": {"slough": [[[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]]}},
                       tissue=True)
    assert tuple(out[100, 100]) == (57, 255, 20)        # on the outline (x = 0.25 of 400)
    assert out[10, 10].tolist() == [120, 120, 120]      # outside: untouched
    assert out[100, 200, 0] > out[100, 200, 2] + 40     # inside the slough patch: yellow tint
    plain = draw_overlay(img, {"outline": [[[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]]})
    assert abs(int(plain[100, 200, 0]) - int(plain[100, 200, 2])) < 10  # no tissue layer: only the faint green fill


def test_small_or_soft_photos_are_analysed_with_a_quality_warning(client):
    rng = np.random.default_rng(2)
    small = cv2.GaussianBlur(rng.integers(40, 210, (300, 400, 3), dtype=np.uint8), (7, 7), 0)
    f = analyze(client, jpeg(small)).json()
    assert f["status"] == "ok", f.get("quality")
    assert "low_resolution" in f["quality"]["warnings"]
    assert any(fl["text"].startswith("Photo quality (low resolution") for fl in f["flags"])
    assert "report_markdown" in f


def test_only_photos_with_nothing_to_analyse_are_refused():
    from wound_ai.quality import check_quality
    rng = np.random.default_rng(3)
    assert not check_quality(np.full((600, 800, 3), 3, np.uint8)).usable                       # black
    assert not check_quality(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).usable          # thumbnail
    small = rng.integers(40, 210, (320, 320, 3), dtype=np.uint8)
    q = check_quality(small)
    assert q.usable and "blurry" not in q.warnings  # a sharp small photo is not called blurry (no upscaling)
