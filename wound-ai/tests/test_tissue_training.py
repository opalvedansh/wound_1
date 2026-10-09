"""Training the tissue model on datasets that each label different classes, and pseudo-labelling.

    .venv/bin/python -m pytest tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from wound_ai.data import IGNORE_INDEX, TISSUE_CLASSES, annotated_classes  # noqa: E402
from wound_ai.losses import partial_label_ce, partial_label_dice  # noqa: E402

K = len(TISSUE_CLASSES)
EPI, GRAN = TISSUE_CLASSES.index("epithelial"), TISSUE_CLASSES.index("granulation")


def confident(cls: int, shape=(1, 4, 4)) -> torch.Tensor:
    """Logits that put almost all probability on one class everywhere."""
    logits = torch.full((shape[0], K, *shape[1:]), -10.0)
    logits[:, cls] = 10.0
    return logits


def test_unlabelled_class_on_a_background_pixel_is_not_penalised():
    target = torch.zeros(1, 4, 4, dtype=torch.long)                      # all "background"
    only_gran = annotated_classes("background;granulation")[None]       # a dataset without an epithelium label
    every = annotated_classes("")[None]
    epi = confident(EPI)
    assert partial_label_ce(epi, target, only_gran) < 0.01               # could be epithelium the dataset never labels
    assert partial_label_ce(epi, target, every) > 5                      # a dataset that labels epithelium: wrong
    assert partial_label_ce(confident(GRAN), target, only_gran) > 5     # a labelled class is still penalised


def test_ignored_pixels_and_class_weights():
    target = torch.full((1, 4, 4), IGNORE_INDEX, dtype=torch.long)
    target[0, 0, 0] = GRAN
    every = annotated_classes("")[None]
    assert partial_label_ce(confident(GRAN), target, every) < 0.01       # only the one labelled pixel counts
    w = torch.ones(K)
    w[GRAN] = 3.0
    wrong = confident(EPI)
    assert torch.isclose(partial_label_ce(wrong, target, every, w), partial_label_ce(wrong, target, every), rtol=1e-4)


def test_dice_only_on_classes_the_image_labels_and_contains():
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, :2] = GRAN
    only_gran = annotated_classes("background;granulation")[None]
    perfect = confident(0)
    perfect[0, :, :2] = -10.0
    perfect[0, GRAN, :2] = 10.0
    assert partial_label_dice(perfect, target, only_gran) < 0.01
    assert partial_label_dice(confident(EPI), target, only_gran) > 0.8  # 1 - 1/9 with the smoothing term on 8 pixels


def test_annotated_classes_rejects_unknown_names():
    assert annotated_classes(float("nan")).all()
    with pytest.raises(ValueError):
        annotated_classes("background;scab")


def test_class_weights_favour_rare_classes():
    from tissue_cv import class_weights

    pixels = np.zeros(K, np.int64)
    pixels[0], pixels[GRAN], pixels[TISSUE_CLASSES.index("necrosis")] = 10**7, 10**6, 10**4
    w = class_weights(pixels)
    assert w[0] == 0.5 and w[TISSUE_CLASSES.index("necrosis")] > w[GRAN]
    assert w[EPI] == 1.0  # absent class: neutral
    assert all(0.5 <= x <= 4.0 for x in w)


def test_pseudo_label_keeps_only_confident_pixels_inside_found_wounds():
    from pseudo_label import pseudo_label

    class Fake:
        def __init__(self, outline):
            self.outline = outline

        def _segment(self, img, name):
            return self.outline

        def _tissue_map(self, img, mask):
            tissue = np.full(mask.shape, GRAN, np.uint8)
            conf = np.full(mask.shape, 0.95, np.float32)
            conf[:, :5] = 0.5                       # the left half: not confident
            tissue[0, 5:] = EPI                     # a class to drop
            return tissue, conf

    img = np.zeros((10, 10, 3), np.uint8)
    assert pseudo_label(Fake(np.zeros((10, 10), np.uint8)), img, 0.9, 0.3, set()) is None  # no wound found
    wound = np.ones((10, 10), np.uint8)
    label, outline, coverage = pseudo_label(Fake(wound), img, 0.9, 0.3, {EPI})
    assert (label[:, :5] == IGNORE_INDEX).all()   # unsure pixels are not learned from
    assert (label[0, 5:] == IGNORE_INDEX).all()   # a dropped class is never pseudo-labelled
    assert (label[1:, 5:] == GRAN).all() and coverage == pytest.approx(0.45)
    assert pseudo_label(Fake(wound), img, 0.9, 0.6, set()) is None  # too little of the wound is confident
