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
from wound_ai.pipeline import mask_outline, periwound_redness  # noqa: E402

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


def with_reading(reading: dict | None, subject_distance_m: float | None = None) -> bytes:
    """The textured photo as the phone app saves it: its distance reading in the EXIF user comment."""
    import io

    from PIL import Image

    rng = np.random.default_rng(0)
    exif = Image.Exif()
    if reading is not None:
        exif.get_ifd(0x8769)[0x9286] = b"ASCII\0\0\0" + json.dumps({"woundScale": reading}).encode()
    if subject_distance_m is not None:
        exif.get_ifd(0x8769)[0x9206] = subject_distance_m
    out = io.BytesIO()
    Image.fromarray(rng.integers(40, 210, (480, 640, 3), dtype=np.uint8)).save(out, "JPEG", exif=exif, quality=95)
    return out.getvalue()


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


def test_try_page_is_served_without_the_key(client):
    r = client.get("/try")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")


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


def test_check_reports_quality_and_the_sticker_without_running_a_model(client):
    def check(photo: bytes, key: str | None = KEY):
        return client.post("/check", files={"image": ("w.jpg", photo, "image/jpeg")},
                           headers={"X-API-Key": key} if key else {})

    assert check(textured_photo(), key=None).status_code == 401
    plain = check(textured_photo()).json()
    assert plain["quality"]["usable"] and plain["marker_found"] is False
    assert set(plain) == {"quality", "marker_found", "phone_reading"} and plain["phone_reading"] is None

    from wound_ai.measure import ARUCO_DICT
    with_sticker = np.full((480, 640, 3), 255, np.uint8)
    with_sticker[:, 320:] = np.random.default_rng(1).integers(40, 210, (480, 320, 3), dtype=np.uint8)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT)), 0, 120)
    with_sticker[100:220, 100:220] = marker[..., None]
    assert check(jpeg(with_sticker)).json()["marker_found"] is True

    flat = check(jpeg(np.full((480, 640, 3), 128, np.uint8))).json()
    assert flat["quality"]["usable"] is False and flat["quality"]["issues"]


def test_phone_distance_reading_gives_the_size_when_there_is_no_sticker(client, monkeypatch):
    from api.server import analyzer
    mask = np.zeros((480, 640), np.uint8)
    mask[100:280, 200:440] = 1  # 240 x 180 px
    monkeypatch.setattr(analyzer, "seg", {"boundary": {"version": "test"}})
    monkeypatch.setattr(analyzer, "_segment", lambda img, name: mask)

    def send(scale):
        return client.post("/analyze", files={"image": ("w.jpg", textured_photo(), "image/jpeg")},
                           data={"intake": json.dumps(INTAKE), "scale": json.dumps(scale)}, headers={"X-API-Key": KEY})

    # 300 mm away with a 90 degree view: the 640 px photo is 600 mm wide, so the wound is 22.5 x 16.9 cm.
    f = send({"distance_mm": 300, "hfov_deg": 90}).json()
    assert f["marker_found"] is False
    m = f["measurement"]
    assert m["method"] == "phone_distance"
    assert m["length_cm"] == pytest.approx(22.5, abs=0.15) and m["width_cm"] == pytest.approx(16.9, abs=0.15)
    assert m["area_cm2"] == pytest.approx(22.5 * 16.875, rel=0.01)
    assert any("phone's distance reading" in flag["text"] for flag in f["flags"])
    assert not any("Size not measured" in flag["text"] for flag in f["flags"])

    for bad in ({"distance_mm": 5, "hfov_deg": 70}, {"distance_mm": 300}, {"distance_mm": "300", "hfov_deg": 70}, [300, 70]):
        assert send(bad).status_code == 400, bad
    assert analyze(client, textured_photo()).json()["measurement"] is None  # no reading, no sticker: never guessed

    # The phone app writes its reading into the photo, so the photo alone is enough.
    carried = analyze(client, with_reading({"distance_mm": 300, "hfov_deg": 90, "source": "lidar"})).json()
    assert carried["phone_reading"]["source"] == "lidar"
    assert carried["measurement"]["length_cm"] == pytest.approx(22.5, abs=0.15)
    check = client.post("/check", files={"image": ("w.jpg", with_reading({"distance_mm": 300, "hfov_deg": 90}), "image/jpeg")},
                        headers={"X-API-Key": KEY}).json()
    assert check["phone_reading"] == {"distance_mm": 300, "hfov_deg": 90}
    facing = analyze(client, with_reading({"distance_mm": 300, "hfov_deg": 90, "normal": [0, 0, -1], "surface_rms_mm": 7.5})).json()
    assert facing["measurement"]["length_cm"] == pytest.approx(22.5, abs=0.15)
    assert any("tilt was corrected" in flag["text"] for flag in facing["flags"])
    assert any("curved (about 8 mm from flat)" in flag["text"] for flag in facing["flags"])
    # A reading that could not be real, or a camera's own subject distance, is ignored.
    assert analyze(client, with_reading({"distance_mm": 3, "hfov_deg": 90})).json()["measurement"] is None
    assert analyze(client, with_reading(None, subject_distance_m=0.3)).json()["measurement"] is None


def test_phone_reading_of_the_surface_corrects_for_tilt():
    import math

    from wound_ai.measure import distance_calibration, measure_wound, valid_reading

    w, h, hfov, d, tilt = 1440, 1920, 50.0, 250.0, math.radians(30)
    focal = w / (2 * math.tan(math.radians(hfov) / 2))
    # A 40 x 30 mm shape on a surface turned 30 degrees about the photo's vertical axis.
    corners = [(w / 2 + focal * x * math.cos(tilt) / (d + x * math.sin(tilt)), h / 2 + focal * y / (d + x * math.sin(tilt)))
               for x, y in ((-20, -15), (20, -15), (20, 15), (-20, 15))]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(np.array(corners)).astype(np.int32)], 1)
    normal = (-math.sin(tilt), 0.0, math.cos(tilt))

    square_on = measure_wound(mask, distance_calibration(d, hfov, (w, h)))
    corrected = measure_wound(mask, distance_calibration(d, hfov, (w, h), normal))
    assert square_on.area_cm2 == pytest.approx(12 * math.cos(tilt), rel=0.03)  # foreshortened: 13% too small
    assert corrected.area_cm2 == pytest.approx(12.0, rel=0.02)
    assert corrected.length_cm == pytest.approx(4.0, abs=0.05) and corrected.width_cm == pytest.approx(3.0, abs=0.05)

    reading = {"distance_mm": d, "hfov_deg": hfov}
    assert valid_reading({**reading, "normal": list(normal)})
    for bad in ([0, 0, 0], [1, 0], "up", [1, 0, 0.1], [float("nan"), 0, 1]):  # no direction, or seen nearly edge-on
        assert not valid_reading({**reading, "normal": bad}), bad


def test_sticker_and_phone_reading_in_one_photo_are_compared(client, monkeypatch):
    from api.server import analyzer
    from wound_ai.measure import ARUCO_DICT

    # 6 px per mm: the 640 px photo is 106.7 mm wide, which a 40 degree view sees from 146.5 mm.
    photo = np.full((480, 640, 3), 255, np.uint8)
    photo[:, 320:] = np.random.default_rng(1).integers(40, 210, (480, 320, 3), dtype=np.uint8)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT)), 0, 120)
    photo[100:220, 100:220] = marker[..., None]
    mask = np.zeros((480, 640), np.uint8)
    mask[260:440, 360:600] = 1  # 40 x 30 mm
    monkeypatch.setattr(analyzer, "seg", {"boundary": {"version": "test"}})
    monkeypatch.setattr(analyzer, "_segment", lambda img, name: mask)

    def send(distance_mm):
        return client.post("/analyze", files={"image": ("w.png", cv2.imencode(".png", photo)[1].tobytes(), "image/png")},
                           data={"intake": json.dumps(INTAKE), "scale": json.dumps({"distance_mm": distance_mm, "hfov_deg": 40})},
                           headers={"X-API-Key": KEY}).json()

    agree = send(146.5)
    assert agree["measurement"]["method"] == "sticker" and agree["measurement"]["area_cm2"] == pytest.approx(12.0, rel=0.02)
    assert abs(agree["measurement_check"]["phone_vs_sticker_pct"]) < 2
    assert not any("than the sticker" in flag["text"] for flag in agree["flags"])
    off = send(146.5 * 1.1)  # a reading 10% too long makes the area about 21% too large
    assert off["measurement"]["area_cm2"] == agree["measurement"]["area_cm2"]
    assert off["measurement_check"]["phone_vs_sticker_pct"] == pytest.approx(21, abs=1.5)
    assert any("21% larger than the sticker" in flag["text"] for flag in off["flags"])


def test_entered_wound_length_gives_the_size_when_nothing_else_does(client, monkeypatch):
    from api.server import analyzer
    mask = np.zeros((480, 640), np.uint8)
    mask[100:281, 200:441] = 1  # 240 x 180 px between its edge pixels' centres
    mask[20:24, 20:24] = 1      # a speck elsewhere must not change which region the length belongs to
    monkeypatch.setattr(analyzer, "seg", {"boundary": {"version": "test"}})
    monkeypatch.setattr(analyzer, "_segment", lambda img, name: mask)

    f = analyze(client, textured_photo(), {**INTAKE, "measured_length_cm": 4.0}).json()
    m = f["measurement"]
    assert m["method"] == "entered_length"
    assert m["length_cm"] == pytest.approx(4.0, abs=0.02) and m["width_cm"] == pytest.approx(3.0, abs=0.02)
    assert m["area_cm2"] == pytest.approx(12.0, rel=0.02)
    assert any("scaled from the wound length entered with the photo (4.0 cm)" in flag["text"] for flag in f["flags"])
    assert "scale from the wound length entered" in f["report_markdown"]

    for bad in (0, 0.05, 500, "4", True):  # not a length a ruler could give: no size rather than a wrong one
        assert analyze(client, textured_photo(), {**INTAKE, "measured_length_cm": bad}).json()["measurement"] is None, bad

    # With a phone reading too, the reading measures and the ruler length checks it: 22.5 cm against 4 cm is flagged.
    both = client.post("/analyze", files={"image": ("w.jpg", textured_photo(), "image/jpeg")},
                       data={"intake": json.dumps({**INTAKE, "measured_length_cm": 4.0}),
                             "scale": json.dumps({"distance_mm": 300, "hfov_deg": 90})}, headers={"X-API-Key": KEY}).json()
    assert both["measurement"]["method"] == "phone_distance"
    assert both["measurement_check"]["entered_length_cm"] == 4.0
    assert any("longer than the length entered with the photo" in flag["text"] for flag in both["flags"])
    ids = [q["id"] for q in client.get("/intake/questions", headers={"X-API-Key": KEY}).json()]
    assert "measured_length_cm" in ids


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


def test_redness_is_the_skin_beside_the_wound_against_skin_further_out():
    skin = np.full((400, 400, 3), (200, 160, 140), np.uint8)  # RGB
    mask = np.zeros((400, 400), np.uint8)
    cv2.circle(mask, (200, 200), 50, 1, -1)
    assert periwound_redness(skin, mask)["level"] == "none"
    red = skin.copy()
    cv2.circle(red, (200, 200), 62, (215, 110, 105), -1)  # a red halo just outside the wound
    r = periwound_redness(red, mask, white_balanced=True)
    assert r["level"] == "marked" and r["delta_a"] > 8 and r["white_balanced"] is True
    assert periwound_redness(skin, np.zeros((400, 400), np.uint8)) is None  # no wound, nothing to compare


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
