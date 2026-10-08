"""Train wound-boundary (binary) or tissue (multi-class) segmentation.

Free Kaggle/Colab GPU (T4/P100, 16 GB):
    python scripts/train_seg.py --manifest data/manifest.csv --task boundary \
        --arch segformer --encoder mit_b2 --size 512 --batch-size 8 --epochs 60 --out runs/boundary

Sessions time out, so a checkpoint is written every epoch. Restart with --resume.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
from torch.utils.data import DataLoader

from wound_ai.data import TISSUE_CLASSES, SegmentationDataset, cv_train_val, wound_size_bins
from wound_ai.models import build_seg_model, load_matching


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--task", choices=["boundary", "tissue"], default="boundary")
    p.add_argument("--arch", default="segformer")
    p.add_argument("--encoder", default="mit_b2")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=12, help="early stop after N epochs without val improvement")
    p.add_argument("--out", default="runs/seg")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--init", default="", help="start from this checkpoint (e.g. the public-data model)")
    p.add_argument("--folds", type=int, default=0, help="cross-validation: number of folds (0 = use the manifest's train/val split)")
    p.add_argument("--fold", type=int, default=0, help="cross-validation: which fold is validation (0-based)")
    p.add_argument("--max-batches", type=int, default=0, help="debug: limit batches per epoch")
    return p.parse_args()


def dice_scores(logits: torch.Tensor, target: torch.Tensor, binary: bool, k: int) -> list[float]:
    """Per-image Dice for binary; per-class (pooled over the batch) for multi-class."""
    if binary:
        pred = (torch.sigmoid(logits) > 0.5).float()
        inter = (pred * target).sum((1, 2, 3))
        denom = pred.sum((1, 2, 3)) + target.sum((1, 2, 3))
        d = torch.where(denom > 0, 2 * inter / denom.clamp(min=1e-7), torch.ones_like(denom))
        return d.tolist()
    pred = logits.argmax(1)
    out = []
    for c in range(1, k):
        p, t = pred == c, target == c
        denom = p.sum() + t.sum()
        if denom > 0:
            out.append(float(2 * (p & t).sum() / denom))
    return out


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # NVIDIA GPU (Kaggle), then Apple GPU (a Mac), then CPU. Mixed precision stays CUDA-only.
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    binary = a.task == "boundary"
    k = 1 if binary else len(TISSUE_CLASSES)
    mask_col = "mask_path" if binary else "tissue_path"

    df = pd.read_csv(a.manifest)
    if a.folds:
        # Folds balanced by wound size, so no fold gets all the tiny wounds.
        train_df, val_df = cv_train_val(df, a.fold, a.folds, wound_size_bins(df, mask_col))
    else:
        train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    tr = SegmentationDataset(train_df, a.size, True, mask_col, binary)
    va = SegmentationDataset(val_df, a.size, False, mask_col, binary)
    fold = f" | fold {a.fold + 1}/{a.folds}" if a.folds else ""
    print(f"train {len(tr)} | val {len(va)} | size {a.size} | batch {a.batch_size} | device {device}{fold}")
    dl_tr = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.batch_size,
                       pin_memory=device == "cuda")
    dl_va = DataLoader(va, a.batch_size, shuffle=False, num_workers=a.workers)

    model = build_seg_model(a.arch, a.encoder, k, pretrained=not a.no_pretrained).to(device)
    if a.init:
        load_matching(model, a.init)
    if binary:
        dice_loss, px_loss = DiceLoss("binary", from_logits=True), nn.BCEWithLogitsLoss()
    else:
        dice_loss, px_loss = DiceLoss("multiclass", from_logits=True), nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = max(1, len(dl_tr) if not a.max_batches else min(a.max_batches, len(dl_tr)))
    total, warm = a.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    start, best, bad = 0, -1.0, 0
    last = out / "last.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        print(f"resumed at epoch {start}, best {best:.4f}")

    cfg = {"arch": a.arch, "encoder": a.encoder, "num_classes": k, "size": a.size, "task": a.task}
    log = out / "history.csv"
    for ep in range(start, a.epochs):
        t0 = time.time()
        model.train()
        losses = []
        for i, (x, y) in enumerate(dl_tr):
            if a.max_batches and i >= a.max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(x)
                loss = dice_loss(logits, y) + px_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())

        model.eval()
        scores = []
        with torch.no_grad():
            for i, (x, y) in enumerate(dl_va):
                if a.max_batches and i >= a.max_batches:
                    break
                with torch.amp.autocast(device_type=device, enabled=device == "cuda"):
                    logits = model(x.to(device))
                scores += dice_scores(logits.float(), y.to(device), binary, k)
        val = float(np.mean(scores)) if scores else 0.0
        print(f"epoch {ep:3d} loss {np.mean(losses):.4f} val_dice {val:.4f} ({time.time() - t0:.0f}s)")

        improved = val > best
        if improved:
            best, bad = val, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg, "val_dice": val,
                        "version": f"{a.task}-{a.arch}-{a.encoder}-ep{ep}-dice{val:.3f}"}, out / f"{a.task}.pt")
        else:
            bad += 1
        torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": ep, "best": best, "bad": bad}, last)
        with open(log, "a", newline="") as fh:
            csv.writer(fh).writerow([ep, float(np.mean(losses)), val, int(improved)])
        if bad >= a.patience:
            print("early stopping")
            break
    print(f"best val dice {best:.4f} -> {out / (a.task + '.pt')}")


if __name__ == "__main__":
    main()
