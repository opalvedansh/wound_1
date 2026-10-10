"""End-to-end inference: photo + intake answers -> findings -> report draft.

Checkpoints are read from one folder. Any missing component is skipped and
reported as 'not assessed', so you can ship modules as they become validated:

    checkpoints/
      boundary.pt        wound boundary segmentation (train_seg.py --task boundary)
      tissue.pt          tissue segmentation          (train_seg.py --task tissue)
      wound_type.pt      wound type classifier        (train_cls.py --target wound_type)
      pu_stage.pt        pressure injury stage        (only applied to pressure injuries)
      burn_depth.pt      burn depth                   (only applied to burns)
      dfu_wagner.pt      DFU Wagner grade 0-3         (only applied to diabetic foot ulcers)
      dfu_infection.pt   DFU infection/ischaemia      (only applied to diabetic foot ulcers)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import (IMAGENET_MEAN, IMAGENET_STD, PERIWOUND, TISSUE_CLASSES, WOUND_BED, MetaEncoder, crop_to_wound,
                   read_rgb, wound_box)
from .intake import FOOT_SITES
from .measure import (Calibration, area_change, distance_calibration, find_marker, grey_patch_gains, length_calibration,
                      measure_wound)
from .models import WoundClassifier, build_seg_model
from .quality import check_quality
from .report import FLAGS_VERSION, build_report, red_flags

SEVERITY_HEADS = {"pu_stage": "pressure", "burn_depth": "burn", "dfu_wagner": "diabetic", "dfu_infection": "diabetic"}
# Mean tissue-model confidence over the wound bed below which the tissue mix is reported as uncertain and not used
# by the healing or care rules. PLACEHOLDER: set from validation (confidence against accuracy on held-out photos).
TISSUE_MIN_CONFIDENCE = 0.6
# A class the tissue model is not trusted on is mentioned ("possible maceration: check") when it covers this share of
# the wound and the skin around it.
UNTRUSTED_MENTION_FRAC = 0.05
# Outline regions smaller than this share of the photo are specks, not wound (works without a sticker).
MIN_OUTLINE_FRAC = 0.0005
# How much redder (CIELAB a*) the skin beside the wound is than skin further out, to be called mild or marked.
# PLACEHOLDER: set from clinic photos the clinician has graded for redness.
REDNESS_MILD, REDNESS_MARKED = 4.0, 8.0


def apply_diabetic_foot_rule(wound_type: dict | None, intake: dict) -> dict | None:
    """Chart 1: any foot wound in a person with diabetes is a diabetic foot ulcer, whatever the photo looks like.
    The classifier's own guess, if there is one, is kept beside the rule's answer."""
    if intake.get("diabetes") != "yes" or intake.get("body_location") not in FOOT_SITES:
        return wound_type
    if wound_type and wound_type.get("label") == "diabetic":
        return {**wound_type, "rule": "diabetes and a foot location"}
    return {"label": "diabetic", "prob": None, "top": [], "rule": "diabetes and a foot location", "model": wound_type}


def mask_outline(mask: np.ndarray | None) -> list[list[list[float]]] | None:
    """The wound outline for drawing over the photo: one polygon per region, largest first, with points
    as [x, y] scaled to 0-1 by the image width and height, so they fit the photo at any display size."""
    if mask is None or not mask.any():
        return None
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < MIN_OUTLINE_FRAC * h * w:
            continue
        points = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(points) >= 3:
            polygons.append([[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in points])
    return polygons or None


def periwound_redness(img: np.ndarray, mask: np.ndarray, white_balanced: bool = False) -> dict | None:
    """Redness of the skin around the wound from colour alone: the skin beside the wound against skin further out
    (CIELAB a*, the red-green axis), so the patient's own skin is the reference. A hint for the clinician, used by
    no flag or care rule: redness shows less on darker skin, so "none" never rules it out."""
    bed = mask > 0
    if not bed.any():
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # Shadow, glare and the sticker's black and white are not skin.
    skin = (lab[..., 0] > 30) & (lab[..., 0] < 235)
    dist = cv2.distanceTransform((~bed).astype(np.uint8), cv2.DIST_L2, 3)
    w = max(4.0, 0.15 * float(np.sqrt(bed.sum())))  # ring width follows the wound's size, not the photo's
    near, far = skin & (dist > 0) & (dist <= w), skin & (dist > 2 * w) & (dist <= 3 * w)
    if near.sum() < 50 or far.sum() < 50:
        return None
    a = lab[..., 1].astype(np.float32)
    delta = round(float(np.median(a[near]) - np.median(a[far])), 1)
    level = "marked" if delta >= REDNESS_MARKED else "mild" if delta >= REDNESS_MILD else "none"
    return {"delta_a": delta, "level": level, "white_balanced": white_balanced}


# Same colours as the portal (apps/web VisitResult.tsx TISSUE_COLOR), as RGB.
OUTLINE_RGB = (57, 255, 20)
TISSUE_RGB = {"granulation": (225, 29, 72), "slough": (234, 179, 8), "necrosis": (17, 24, 39),
              "epithelial": (249, 168, 212), "exposed_structure": (248, 250, 252),
              "periwound_erythema": (249, 115, 22), "maceration": (147, 197, 253), "callus": (163, 230, 53)}


def draw_overlay(img: np.ndarray, findings: dict, tissue: bool = False) -> np.ndarray:
    """The photo (RGB) with the wound outline, and optionally the tissue layers, drawn as the portal draws them."""
    h, w = img.shape[:2]

    def pts(polygon):
        return np.array([[round(x * w), round(y * h)] for x, y in polygon], np.int32)

    out = img.copy()
    layers = [(TISSUE_RGB.get(c, (168, 85, 247)), polys, 0.45, max(1, w // 600))
              for c, polys in ((findings.get("tissue_outline") or {}).items() if tissue else [])]
    layers.append((OUTLINE_RGB, findings.get("outline") or [], 0.12, max(2, w // 300)))
    for colour, polygons, alpha, width in layers:
        if not polygons:
            continue
        fill = out.copy()
        cv2.fillPoly(fill, [pts(p) for p in polygons], colour)
        out = cv2.addWeighted(fill, alpha, out, 1 - alpha, 0)
        cv2.polylines(out, [pts(p) for p in polygons], True, colour, width, cv2.LINE_AA)
    return out


def _letterbox(img: np.ndarray, size: int):
    """Resize longest side to `size` and centre-pad (matches LongestMaxSize + PadIfNeeded in training)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, (top, left, nh, nw)


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    x = (img.astype(np.float32) / 255.0 - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


class WoundAnalyzer:
    def __init__(self, ckpt_dir: str | Path, device: str | None = None, marker_mm: float = 20.0, llm=None):
        self.dir = Path(ckpt_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.marker_mm = marker_mm
        self.llm = llm  # optional (model, processor) tuple from wound_ai.llm.load
        self.seg, self.cls = {}, {}
        for name in ("boundary", "tissue"):
            p = self.dir / f"{name}.pt"
            if p.exists():
                self.seg[name] = self._load_seg(p)
        for name in ["wound_type", *SEVERITY_HEADS]:
            p = self.dir / f"{name}.pt"
            if p.exists():
                self.cls[name] = self._load_cls(p)

    # ------------------------------------------------------------------ loading
    def _load_seg(self, path: Path) -> dict:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        c = ck["config"]
        m = build_seg_model(c["arch"], c["encoder"], c["num_classes"], pretrained=False)
        m.load_state_dict(ck["state_dict"])
        return {"model": m.to(self.device).eval(), "cfg": c, "version": ck.get("version", path.stem)}

    def _load_cls(self, path: Path) -> dict:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        c = ck["config"]
        meta = MetaEncoder(ck["meta"]["columns"], ck["meta"]["vocab"])
        m = WoundClassifier(c["backbone"], len(ck["classes"]), meta.dim, pretrained=False)
        m.load_state_dict(ck["state_dict"])
        return {"model": m.to(self.device).eval(), "cfg": c, "classes": ck["classes"], "meta": meta,
                "T": float(ck.get("temperature", 1.0)), "version": ck.get("version", path.stem)}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _segment(self, img: np.ndarray, name: str, with_confidence: bool = False):
        s = self.seg[name]
        size = s["cfg"]["size"]
        lb, (top, left, nh, nw) = _letterbox(img, size)
        logits = s["model"](_to_tensor(lb).to(self.device))[0]
        if s["cfg"]["num_classes"] == 1:
            prob = torch.sigmoid(logits[0])
            pred, conf = (prob > 0.5), torch.maximum(prob, 1 - prob)
        else:
            conf, pred = torch.softmax(logits, 0).max(0)
        size_back = (img.shape[1], img.shape[0])
        pred = cv2.resize(pred.cpu().numpy().astype(np.uint8)[top:top + nh, left:left + nw], size_back,
                          interpolation=cv2.INTER_NEAREST)
        if not with_confidence:
            return pred
        conf = cv2.resize(conf.float().cpu().numpy()[top:top + nh, left:left + nw], size_back,
                          interpolation=cv2.INTER_LINEAR)
        return pred, conf

    def tissue_classes(self) -> list[str]:
        cfg = self.seg["tissue"]["cfg"]
        return cfg.get("classes") or TISSUE_CLASSES[:cfg["num_classes"]]  # older checkpoints had 6 classes

    def _tissue_map(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tissue class and confidence per pixel, on the crop the tissue model was trained on (if it was)."""
        cfg = self.seg["tissue"]["cfg"]
        box = wound_box(mask, cfg["crop_margin"]) if cfg.get("crop_margin") else None
        if box is None:
            return self._segment(img, "tissue", with_confidence=True)
        tissue, conf = np.zeros(mask.shape, np.uint8), np.zeros(mask.shape, np.float32)
        tissue[box], conf[box] = self._segment(img[box], "tissue", with_confidence=True)
        return tissue, conf

    def _tissue(self, img: np.ndarray, mask: np.ndarray) -> dict:
        """Tissue mix of the wound bed and of the skin around it.

        A cross-validated checkpoint lists the classes it may be trusted on (scripts/tissue_cv.py). Others are left
        out of every share, layer and rule, and only named for the clinician to check if they cover a fair area."""
        classes = self.tissue_classes()
        tissue, conf = self._tissue_map(img, mask)
        bed, out = mask > 0, {}
        trusted = self.seg["tissue"]["cfg"].get("trusted_classes")
        if trusted is not None:
            ring_or_bed = (cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0)
            seen = [c for k, c in enumerate(classes) if k and c not in trusted
                    and (tissue[ring_or_bed] == k).mean() >= UNTRUSTED_MENTION_FRAC]
            if seen:
                out["tissue_untrusted"] = seen
        idx = {c: i for i, c in enumerate(classes) if trusted is None or c in trusted}
        counts = {c: int((tissue[bed] == idx[c]).sum()) for c in WOUND_BED if c in idx}
        total = sum(counts.values())
        out["tissue_confidence"] = round(float(conf[bed].mean()), 2)
        if total:
            pct = {c: round(100 * n / total) for c, n in counts.items() if n}
            # An unsure estimate is shown to the clinician but never feeds the healing or care rules.
            out["tissue_pct" if out["tissue_confidence"] >= TISSUE_MIN_CONFIDENCE else "tissue_pct_uncertain"] = pct
        ring = (cv2.dilate(mask, np.ones((25, 25), np.uint8)) - mask) > 0
        if ring.any():
            for c, key in (("periwound_erythema", "periwound_erythema_frac"), ("maceration", "periwound_maceration_frac"),
                           ("callus", "periwound_callus_frac")):
                if c in idx:
                    out[key] = round(float((tissue[ring] == idx[c]).mean()), 2)
        outline = {}
        for c in WOUND_BED + PERIWOUND:
            if c in idx and (poly := mask_outline(((tissue == idx[c]) & (bed if c in WOUND_BED else ring)).astype(np.uint8))):
                outline[c] = poly
        out["tissue_outline"] = outline or None
        return out

    @torch.no_grad()
    def _classify(self, crop: np.ndarray, name: str, intake: dict, return_probs: bool = False):
        c = self.cls[name]
        lb, _ = _letterbox(crop, c["cfg"]["size"])
        meta = torch.from_numpy(c["meta"].encode(intake)).unsqueeze(0).to(self.device)
        logits = c["model"](_to_tensor(lb).to(self.device), meta if c["meta"].dim else None)[0]
        probs = torch.softmax(logits / c["T"], dim=0).cpu().numpy()
        order = probs.argsort()[::-1]
        out = {"label": c["classes"][order[0]], "prob": round(float(probs[order[0]]), 3),
               "top": [(c["classes"][i], round(float(probs[i]), 3)) for i in order[:3]]}
        return (out, probs) if return_probs else out

    def capture_check(self, img: np.ndarray, scale: dict | None = None) -> tuple[dict, Calibration | None]:
        """What can be said about a photo before any model runs: its quality, and what gives it a scale (the
        calibration sticker in it, or the phone's distance reading sent with it). The app asks for this alone when
        a photo is taken, so it can be retaken while the patient is there."""
        calib = find_marker(img, self.marker_mm)
        q = check_quality(img)
        return {"quality": {"ok": q.ok, "usable": q.usable, "issues": q.issues, "warnings": q.warnings, **q.metrics},
                "marker_found": calib is not None, "phone_reading": scale}, calib

    def analyze(self, image: str | np.ndarray, intake: dict | None = None, previous: dict | None = None,
                overlay_path: str | None = None, scale: dict | None = None) -> dict:
        """`scale`: the phone's own reading when it took the photo, {"distance_mm", "hfov_deg"} (see
        measure.distance_calibration). It gives the size when the photo has no sticker."""
        intake = dict(intake or {})
        img = read_rgb(image) if isinstance(image, (str, Path)) else image
        checks, calib = self.capture_check(img, scale)
        f = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="minutes"), "intake": intake, **checks,
             "model_versions": {**{k: v["version"] for k, v in self.seg.items()},
                                **{k: v["version"] for k, v in self.cls.items()}}}
        if not f["quality"]["usable"]:  # blank, tiny, all black or white: nothing to analyse, ask for a retake
            f["status"] = "retake"
            return f

        mask = self._segment(img, "boundary") if "boundary" in self.seg else None
        if mask is not None and mask.sum() == 0:
            f["status"] = "no_wound_found"
            f["flags"] = [{"level": "review", "text": "No wound region detected. Check framing or review manually."}]
            return f

        crop = crop_to_wound(img, mask)
        # Classifiers see the whole photo, as in training (the class manifests have no masks to crop with);
        # on the outline crop the wound-type model fell from 83% to 50% on held-out photos.
        if "wound_type" in self.cls:
            f["wound_type"] = self._classify(img, "wound_type", intake)
        f["wound_type"] = apply_diabetic_foot_rule(f.get("wound_type"), intake)
        wt = (f.get("wound_type") or {}).get("label")
        f["severity"] = {}
        for head, applies_to in SEVERITY_HEADS.items():
            # A burn the intake reports is a burn, whatever the photo looks like (deep burns can pass for ulcers).
            if head in self.cls and (wt == applies_to or (applies_to == "burn" and intake.get("cause") == "burn")):
                f["severity"][head] = self._classify(img, head, intake)

        if mask is not None:
            # Tissue and redness are judged by colour, so correct the lighting's colour cast from the sticker's grey patch.
            gains = grey_patch_gains(img, calib, self.marker_mm) if calib is not None else None
            f["white_balance"] = gains
            balanced = img if gains is None else np.clip(img * np.array(gains, np.float32), 0, 255).astype(np.uint8)
            if "tissue" in self.seg:
                f.update(self._tissue(balanced, mask))
            f["periwound_redness"] = periwound_redness(balanced, mask, gains is not None)

        f["outline"] = mask_outline(mask)
        f["measurement"] = None
        # The sticker wins when the photo has both: it lies on the skin itself, so it needs no reading of the surface.
        by_phone = distance_calibration(scale["distance_mm"], scale["hfov_deg"], (img.shape[1], img.shape[0]),
                                        scale.get("normal")) if scale else None
        entered = intake.get("measured_length_cm")
        by_length = length_calibration(mask, entered) if mask is not None else None
        # What was measured most directly comes first; the clinician's ruler length is the last resort.
        method, ruler = next(((name, c) for name, c in (("sticker", calib), ("phone_distance", by_phone),
                                                        ("entered_length", by_length)) if c is not None), (None, None))
        if ruler is not None and mask is not None:
            m = measure_wound(mask, ruler)
            f["measurement"] = {**m.to_dict(), "method": method} if m else None
            check = {}
            other = measure_wound(mask, by_phone) if m and calib is not None and by_phone is not None else None
            if other:
                # Both in one photo: how far the phone's reading is from the sticker, which is how its accuracy is checked.
                check.update({
                    "sticker_area_cm2": round(m.area_cm2, 2), "phone_area_cm2": round(other.area_cm2, 2),
                    "phone_vs_sticker_pct": round(100 * (other.area_cm2 - m.area_cm2) / m.area_cm2, 1)})
            if m and method != "entered_length" and by_length is not None:
                # A ruler length alongside a measured size: the same check against the clinician's own measurement.
                check.update({"entered_length_cm": entered, "measured_length_cm": round(m.length_cm, 2),
                              "length_vs_entered_pct": round(100 * (m.length_cm - entered) / entered, 1)})
            if check:
                f["measurement_check"] = check
        if previous and f["measurement"] and previous.get("area_cm2"):
            f["change"] = area_change(f["measurement"]["area_cm2"], previous["area_cm2"], previous.get("days_ago"))

        f["flags"] = red_flags(f)
        f["flags_version"] = FLAGS_VERSION
        llm_text = None
        if self.llm is not None:
            from PIL import Image

            from .llm import generate_summary

            llm_text = generate_summary(*self.llm, Image.fromarray(crop), f)
        f["report_markdown"], f["llm_problems"] = build_report(f, llm_text)
        f["status"] = "ok"

        if overlay_path and f["outline"]:
            cv2.imwrite(overlay_path, cv2.cvtColor(draw_overlay(img, f, tissue=True), cv2.COLOR_RGB2BGR))
        return f
