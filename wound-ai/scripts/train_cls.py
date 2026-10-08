"""Train a classifier for any label column in the manifest, then calibrate it.

Examples (free T4/P100):
    # wound type, with body location + intake answers fused in
    python scripts/train_cls.py --manifest data/manifest.csv --target wound_type \
        --meta-cols body_location,cause,diabetes --out runs/wound_type

    # pressure injury stage (rows without a pu_stage label are ignored)
    python scripts/train_cls.py --manifest data/manifest.csv --target pu_stage --out runs/pu_stage

The saved checkpoint carries its class list, metadata vocabulary and a fitted
temperature, which the pipeline uses to decide when to say 'uncertain'.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wound_ai.data import ClassificationDataset, MetaEncoder, cv_train_val
from wound_ai.metrics import expected_calibration_error, macro_f1, per_class_sens_spec
from wound_ai.models import TemperatureScaler, WoundClassifier, load_matching


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--backbone", default="convnext_tiny.fb_in22k")
    p.add_argument("--meta-cols", default="", help="comma-separated categorical columns to fuse, e.g. body_location")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--out", default="runs/cls")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--init", default="", help="start from this checkpoint (e.g. the public-data model)")
    p.add_argument("--folds", type=int, default=0, help="cross-validation: number of folds (0 = use the manifest's train/val split)")
    p.add_argument("--fold", type=int, default=0, help="cross-validation: which fold is validation (0-based)")
    p.add_argument("--max-batches", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def collect_logits(model, dl, device, max_batches=0):
    model.eval()
    L, Y = [], []
    for i, (x, m, y) in enumerate(dl):
        if max_batches and i >= max_batches:
            break
        L.append(model(x.to(device), m.to(device) if m.numel() else None).float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # NVIDIA GPU (Kaggle), then Apple GPU (a Mac), then CPU. Mixed precision stays CUDA-only.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    df = pd.read_csv(a.manifest)
    df = df[df[a.target].notna()]
    df[a.target] = df[a.target].astype(str)
    if a.folds:
        # Folds balanced by wound type, so every fold sees every type in the same proportion.
        train_df, val_df = cv_train_val(df, a.fold, a.folds, df[a.target])
    else:
        train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    classes = sorted(train_df[a.target].unique().tolist())
    val_df = val_df[val_df[a.target].isin(classes)]
    meta_cols = [c for c in a.meta_cols.split(",") if c]
    meta = MetaEncoder(meta_cols).fit(train_df) if meta_cols else None

    tr = ClassificationDataset(train_df, a.target, classes, a.size, True, meta)
    va = ClassificationDataset(val_df, a.target, classes, a.size, False, meta)
    counts = train_df[a.target].value_counts().reindex(classes).fillna(0).values
    fold = f" | fold {a.fold + 1}/{a.folds}" if a.folds else ""
    print(f"classes {dict(zip(classes, counts.astype(int).tolist()))} | val {len(va)} | device {device}{fold}")

    dl_tr = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.batch_size)
    dl_va = DataLoader(va, a.batch_size, shuffle=False, num_workers=a.workers)

    model = WoundClassifier(a.backbone, len(classes), meta.dim if meta else 0, pretrained=not a.no_pretrained).to(device)
    if a.init:
        load_matching(model, a.init)
    # Inverse-sqrt-frequency weights: rarer classes (e.g. stage 4, full-thickness burns) count more.
    w = torch.tensor(1.0 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=(w / w.mean()).to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best, bad = -1.0, 0
    cfg = {"backbone": a.backbone, "size": a.size, "target": a.target}
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        for i, (x, m, y) in enumerate(dl_tr):
            if a.max_batches and i >= a.max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                loss = crit(model(x, m.to(device) if m.numel() else None), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()

        logits, ys = collect_logits(model, dl_va, device, a.max_batches)
        f1 = macro_f1(ys.numpy(), logits.argmax(1).numpy()) if len(ys) else 0.0
        print(f"epoch {ep:3d} loss {loss.item():.4f} val_macroF1 {f1:.4f} ({time.time() - t0:.0f}s)")
        if f1 > best:
            best, bad = f1, 0
            torch.save({"state_dict": model.state_dict()}, out / "_best_weights.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break

    # Reload best weights, calibrate on the validation set, write the final checkpoint.
    model.load_state_dict(torch.load(out / "_best_weights.pt", map_location=device)["state_dict"])
    logits, ys = collect_logits(model, dl_va, device, a.max_batches)
    ts = TemperatureScaler()
    T = ts.fit(logits, ys) if len(ys) else 1.0
    if len(ys) < 100:
        print(f"WARNING: only {len(ys)} validation cases; the fitted temperature ({T:.3f}) is unreliable. "
              "Calibrate again on a larger clinical validation set before trusting 'uncertain' thresholds.")
    T = float(min(max(T, 0.2), 10.0))  # guard against degenerate fits on tiny or perfectly separable sets
    probs = torch.softmax(logits / T, 1).numpy()
    y = ys.numpy()
    report = {
        "macro_f1": macro_f1(y, probs.argmax(1)),
        "ece_before": expected_calibration_error(torch.softmax(logits, 1).numpy(), y),
        "ece_after": expected_calibration_error(probs, y),
        "temperature": T,
        "per_class": per_class_sens_spec(y, probs.argmax(1), classes),
    }
    print(json.dumps(report, indent=1))
    torch.save({"state_dict": model.state_dict(), "config": cfg, "classes": classes,
                "meta": {"columns": meta_cols, "vocab": meta.vocab if meta else {}},
                "temperature": T, "val_report": report,
                "version": f"{a.target}-{a.backbone}-f1{report['macro_f1']:.3f}"}, out / f"{a.target}.pt")
    (out / "val_report.json").write_text(json.dumps(report, indent=1))
    print(f"saved {out / (a.target + '.pt')}")


if __name__ == "__main__":
    main()
