"""Metrics that matter clinically, with confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> tuple[float, float]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:  # both empty -> perfect agreement
        return 1.0, 1.0
    dice = (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
    return float(dice), float((inter + eps) / (union + eps))


def per_class_sens_spec(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    out = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        out[c] = {
            "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
            "ppv": float(tp / (tp + fp)) if tp + fp else float("nan"),
            "support": int(tp + fn),
        }
    return out


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def macro_auroc(probs: np.ndarray, y_true: np.ndarray) -> float:
    try:
        if probs.shape[1] == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                   labels=list(range(probs.shape[1]))))
    except ValueError:
        return float("nan")  # a class is missing from this subset


def bootstrap_ci(fn, *arrays, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate + 95% CI by resampling cases. Report the CI, not just the number."""
    rng = np.random.default_rng(seed)
    point = fn(*arrays)
    size = len(arrays[0])
    stats = []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        v = fn(*(a[idx] for a in arrays))
        if not np.isnan(v):
            stats.append(v)
    if not stats:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement between two clinicians. This is the ceiling your model can be judged against."""
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b))


def repeatability(groups: list[list[float]], log: bool = True, n_boot: int = 1000, seed: int = 0) -> dict | None:
    """Test-retest repeatability from repeated measurements of the same wound (one list per wound, 2+ values,
    e.g. the area each photographer's photo gave in the same visit).

    Within-wound SD (Sw, pooled over wounds) -> repeatability coefficient RC = 1.96 * sqrt(2) * Sw: two measurements
    of an unchanged wound differ by less than RC 95% of the time, so a smaller change is noise. Areas are compared
    on the log scale (error grows with wound size), giving percentages; tissue shares on the raw scale (points).
    Returns {"n_wounds", "n_measurements", "sw", "rc", ...} with a bootstrap 95% CI on RC (resampling wounds);
    for log=True also the change, in percent, that is beyond noise for a wound getting smaller or larger.
    """
    vals = [np.log(np.asarray(g, float)) if log else np.asarray(g, float) for g in groups if len(g) >= 2]
    if log:
        vals = [v for v in vals if np.isfinite(v).all()]
    if len(vals) < 2:
        return None

    def rc_of(sample: list[np.ndarray]) -> float:
        ss = sum(((v - v.mean()) ** 2).sum() for v in sample)
        df = sum(len(v) - 1 for v in sample)
        return 1.96 * np.sqrt(2) * np.sqrt(ss / df)

    rc = rc_of(vals)
    rng = np.random.default_rng(seed)
    boot = [rc_of([vals[i] for i in rng.integers(0, len(vals), len(vals))]) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = {"n_wounds": len(vals), "n_measurements": int(sum(len(v) for v in vals)),
           "sw": round(float(rc / (1.96 * np.sqrt(2))), 4), "rc": round(float(rc), 4),
           "rc_ci95": [round(float(lo), 4), round(float(hi), 4)]}
    if log:
        # A change is beyond noise when the new/old area ratio leaves [exp(-RC), exp(RC)].
        out["smaller_pct"] = round(float(100 * (1 - np.exp(-rc))), 1)
        out["larger_pct"] = round(float(100 * (np.exp(rc) - 1)), 1)
        out["larger_pct_ci95_upper"] = round(float(100 * (np.exp(hi) - 1)), 1)
    return out
