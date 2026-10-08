"""End-to-end inference: photo + intake answers -> findings -> report draft.

Checkpoints are read from one folder. Any missing component is skipped and
reported as 'not assessed', so you can ship modules as they become validated:

    checkpoints/
      boundary.pt        wound boundary segmentation (train_seg.py --task boundary)
      tissue.pt          tissue segmentation          (train_seg.py --task tissue)
      wound_type.pt      wound type classifier        (train_cls.py --target wound_type)
      pu_stage.pt        pressure injury stage        (only applied to pressure injuries)
      burn_depth.pt      burn depth                   (only applied to burns)
      dfu_infection.pt   DFU infection/ischaemia      (only applied to diabetic foot ulcers)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD, TISSUE_CLASSES, MetaEncoder, crop_to_wound, read_rgb
from .intake import FOOT_SITES
from .measure import area_change, find_marker, measure_wound
from .models import WoundClassifier, build_seg_model
from .quality import check_quality
from .report import build_report, red_flags

SEVERITY_HEADS = {"pu_stage": "pressure", "burn_depth": "burn", "dfu_infection": "diabetic"}
WOUND_BED = ["granulation", "slough", "necrosis", "epithelial"]
# Outline regions smaller than this share of the photo are specks, not wound (works without a sticker).
MIN_OUTLINE_FRAC = 0.0005


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
    def _segment(self, img: np.ndarray, name: str) -> np.ndarray:
        s = self.seg[name]
        size = s["cfg"]["size"]
        lb, (top, left, nh, nw) = _letterbox(img, size)
        logits = s["model"](_to_tensor(lb).to(self.device))[0]
        if s["cfg"]["num_classes"] == 1:
            pred = (torch.sigmoid(logits[0]) > 0.5).cpu().numpy().astype(np.uint8)
        else:
            pred = logits.argmax(0).cpu().numpy().astype(np.uint8)
        pred = pred[top:top + nh, left:left + nw]
        return cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

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

    def analyze(self, image: str | np.ndarray, intake: dict | None = None, previous: dict | None = None,
                overlay_path: str | None = None) -> dict:
        intake = dict(intake or {})
        img = read_rgb(image) if isinstance(image, (str, Path)) else image
        calib = find_marker(img, self.marker_mm)
        q = check_quality(img)
        f = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="minutes"), "intake": intake,
             "quality": {"ok": q.ok, "issues": q.issues, **q.metrics}, "marker_found": calib is not None,
             "model_versions": {**{k: v["version"] for k, v in self.seg.items()},
                                **{k: v["version"] for k, v in self.cls.items()}}}
        if not q.ok:  # blurry / dark / glare: do not analyse, ask for a retake
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
            if head in self.cls and wt == applies_to:
                f["severity"][head] = self._classify(img, head, intake)

        if "tissue" in self.seg and mask is not None:
            tissue = self._segment(img, "tissue")
            bed = tissue[mask > 0]
            counts = {c: int((bed == TISSUE_CLASSES.index(c)).sum()) for c in WOUND_BED}
            total = sum(counts.values())
            if total:
                f["tissue_pct"] = {c: round(100 * n / total) for c, n in counts.items() if n}
            ring = cv2.dilate(mask, np.ones((25, 25), np.uint8)) - mask
            if ring.sum():
                ery = (tissue[ring > 0] == TISSUE_CLASSES.index("periwound_erythema")).mean()
                f["periwound_erythema_frac"] = round(float(ery), 2)

        f["outline"] = mask_outline(mask)
        f["measurement"] = None
        if calib is not None and mask is not None:
            m = measure_wound(mask, calib)
            f["measurement"] = m.to_dict() if m else None
        if previous and f["measurement"] and previous.get("area_cm2"):
            f["change"] = area_change(f["measurement"]["area_cm2"], previous["area_cm2"], previous.get("days_ago"))

        f["flags"] = red_flags(f)
        llm_text = None
        if self.llm is not None:
            from PIL import Image

            from .llm import generate_summary

            llm_text = generate_summary(*self.llm, Image.fromarray(crop), f)
        f["report_markdown"], f["llm_problems"] = build_report(f, llm_text)
        f["status"] = "ok"

        if overlay_path and mask is not None:
            vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, (0, 255, 0), max(2, img.shape[1] // 300))
            cv2.imwrite(overlay_path, vis)
        return f
