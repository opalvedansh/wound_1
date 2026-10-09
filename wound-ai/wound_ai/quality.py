"""Capture-quality check.

Every readable photo is analysed. Problems that make the result less reliable (low resolution, blur, glare,
darkness) are reported with it as warnings, so the clinician checks the outline and size against the photo. Only a
photo with nothing to analyse (blank, tiny, almost all black or white) is refused with a retake instruction.

The models are trained on public photos that are mostly small (median shortest side ~320 px) and often soft, so a
strict gate would refuse photos the models handle well: a gate that refused anything below phone quality turned
away 91% of them. Thresholds are starting points: tune them on your own phones, and log what is warned and refused.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Refused: nothing to analyse.
MIN_SIDE_REFUSE = 96       # px, shortest side
MIN_CONTRAST = 4.0         # grey-level standard deviation: below this the photo is blank or uniform
MEAN_REFUSE = (15, 245)    # mean brightness outside this: almost all black or all white
# Warned: analysed, but check the result.
MIN_SIDE_WARN = 480
BLUR_WARN = 25.0           # Laplacian variance at <= 640 px wide
DARK_WARN, BRIGHT_WARN, CLIPPED_WARN = 50.0, 215.0, 0.08


@dataclass
class QualityResult:
    usable: bool                                          # analyse it (False: ask for a retake)
    issues: list[str] = field(default_factory=list)       # everything found, refusal reasons first
    warnings: list[str] = field(default_factory=list)     # short codes of what was found on a usable photo
    metrics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """No issue at all."""
        return not self.issues


def content_region(gray: np.ndarray, floor: int = 5, min_share: float = 0.02) -> np.ndarray:
    """The image without uniform black borders: rows and columns where almost every pixel is black are dropped."""
    rows = np.where((gray > floor).mean(axis=1) > min_share)[0]
    cols = np.where((gray > floor).mean(axis=0) > min_share)[0]
    if len(rows) < 32 or len(cols) < 32:  # (almost) all black: judge the whole frame
        return gray
    return gray[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def check_quality(img_rgb: np.ndarray, marker_found: bool | None = None) -> QualityResult:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # Judge the photo, not black letterbox borders (padded or forwarded images): they would count as clipped
    # pixels and darken the mean.
    gray = content_region(gray)
    h, w = gray.shape[:2]
    # Large photos are scaled down to 640 px wide so blur scores compare across phones. Small ones are never scaled
    # up: enlarging smooths the image and would make every small photo look blurry.
    g = cv2.resize(gray, (640, max(1, int(640 * h / w))), interpolation=cv2.INTER_AREA) if w > 640 else gray
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    mean, contrast = float(g.mean()), float(g.std())
    clipped = float(((g >= 250) | (g <= 5)).mean())

    refuse = []
    if min(h, w) < MIN_SIDE_REFUSE:
        refuse.append(f"Photo is too small to analyse ({w}x{h}). Retake with the phone's main camera.")
    if contrast < MIN_CONTRAST:
        refuse.append("Photo is blank or out of focus throughout: nothing to analyse. Retake with the wound in view.")
    if not MEAN_REFUSE[0] <= mean <= MEAN_REFUSE[1]:
        refuse.append("Photo is almost completely dark or white: nothing to analyse. Retake in room light or daylight.")

    warn = []
    if min(h, w) < MIN_SIDE_WARN:
        warn.append(("low_resolution", f"Low resolution ({w}x{h}): next time use the phone's main camera at full "
                                       "resolution."))
    if blur < BLUR_WARN:
        warn.append(("blurry", "Photo looks blurry: next time hold the phone steady and tap to focus on the wound."))
    if DARK_WARN > mean >= MEAN_REFUSE[0]:
        warn.append(("dark", "Photo is dark: next time use room light or daylight (avoid flash)."))
    if BRIGHT_WARN < mean <= MEAN_REFUSE[1] or clipped > CLIPPED_WARN:
        warn.append(("glare", "Over-exposed or glare: next time avoid flash and direct light on wet wound surfaces."))
    if marker_found is False:
        warn.append(("no_sticker", "Calibration sticker not found: the size cannot be measured without it."))
    return QualityResult(usable=not refuse, issues=refuse + [text for _, text in warn],
                         warnings=[] if refuse else [code for code, _ in warn],
                         metrics={"blur_var": round(blur, 1), "mean_brightness": round(mean, 1),
                                  "contrast": round(contrast, 1), "clipped_frac": round(clipped, 3),
                                  "width": w, "height": h})
