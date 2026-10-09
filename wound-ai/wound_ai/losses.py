"""Losses for training on datasets that each label a different subset of the tissue classes.

WoundTissue labels maceration but not epithelium, DFUTissue labels callus but not necrosis, and so on. In each, a
pixel marked background may really be a class that dataset never labels, so plain cross-entropy would teach the
model that epithelium (say) is background. The marginal loss instead treats "background" as "background or any
class this image's dataset does not label": it maximises the summed probability of those classes.
"""
from __future__ import annotations

import torch

from .data import IGNORE_INDEX


def partial_label_ce(logits: torch.Tensor, target: torch.Tensor, annotated: torch.Tensor,
                     weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy with background merged with each image's unlabelled classes.

    logits B,K,H,W; target B,H,W (IGNORE_INDEX = not learned from); annotated B,K bool (True: the image's dataset
    labels that class; background is always True). With every class annotated this is ordinary weighted CE.
    """
    logp = logits.float().log_softmax(1)
    merge = ~annotated.bool()
    merge[:, 0] = True  # background, plus every class this image's dataset does not label
    bg = torch.logsumexp(logp.masked_fill(~merge[:, :, None, None], float("-inf")), dim=1)
    valid = target != IGNORE_INDEX
    t = torch.where(valid, target, torch.zeros_like(target))
    lp = torch.where(t == 0, bg, logp.gather(1, t[:, None]).squeeze(1))
    w = weight[t] if weight is not None else torch.ones_like(lp)
    w = w * valid
    return -(lp * w).sum() / w.sum().clamp(min=1e-6)


def partial_label_dice(logits: torch.Tensor, target: torch.Tensor, annotated: torch.Tensor,
                       eps: float = 1.0) -> torch.Tensor:
    """Soft Dice, per image, over the tissue classes that image's dataset labels and that appear in its mask
    (like smp's DiceLoss, absent classes are left to the cross-entropy). Ignored pixels do not count."""
    prob = logits.float().softmax(1)
    valid = (target != IGNORE_INDEX)[:, None]
    k = logits.shape[1]
    onehot = torch.zeros_like(prob).scatter_(1, torch.where(valid[:, 0], target, 0)[:, None], 1.0) * valid
    prob = prob * valid
    inter = (prob * onehot).sum((2, 3))
    denom = prob.sum((2, 3)) + onehot.sum((2, 3))
    dice = 1 - (2 * inter + eps) / (denom + eps)                       # B,K
    use = annotated.bool() & (onehot.sum((2, 3)) > 0)
    use[:, 0] = False                                                   # tissue classes only
    if k == 1 or not use.any():
        return logits.sum() * 0
    return dice[use].mean()
