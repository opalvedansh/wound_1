"""Cross-validation for the wound outline (SegFormer) and wound type (ConvNeXt) models.

Folds are grouped by patient and stratified (wound size for the outline, wound type for the classifier); the locked
test split is never trained or tuned on, and every fold's model is scored on it, so the result is a mean ± sd
instead of one lucky (or unlucky) split.

Before training, a short timed trial on this machine projects the run time. K folds (default 5) are used if they
fit the time budget, otherwise the fallback (default 3). With several GPUs, folds train in parallel, one per GPU.
Re-running the same command continues where it stopped (finished folds are skipped, the outline resumes per epoch).

    python scripts/cv.py --seg-manifest data/manifest.csv --cls-manifest data/manifest_cls.csv \
        --seg-size 768 --seg-batch 4 --out /kaggle/working/cv

Writes OUT/summary.json and OUT/summary.md, and copies the fold with the best validation score of each model to
OUT/best/ (ready for checkpoints/).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg-manifest", default="data/manifest.csv")
    p.add_argument("--cls-manifest", default="data/manifest_cls.csv")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fallback-folds", type=int, default=3)
    p.add_argument("--budget-hours", type=float, default=10.0, help="training time allowed (Kaggle stops at 12 h)")
    p.add_argument("--seg-arch", default="segformer")
    p.add_argument("--seg-encoder", default="mit_b2")
    p.add_argument("--seg-size", type=int, default=768)
    p.add_argument("--seg-batch", type=int, default=4)
    p.add_argument("--seg-epochs", type=int, default=60)
    p.add_argument("--seg-patience", type=int, default=12)
    p.add_argument("--cls-backbone", default="convnext_tiny.fb_in22k")
    p.add_argument("--cls-size", type=int, default=384)
    p.add_argument("--cls-batch", type=int, default=32)
    p.add_argument("--cls-epochs", type=int, default=40)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--skip-seg", action="store_true")
    p.add_argument("--skip-cls", action="store_true")
    p.add_argument("--out", default="runs/cv")
    p.add_argument("--smoke", action="store_true", help="debug: tiny run (3 batches, 1 epoch, no pretrained weights)")
    return p.parse_args()


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def run(cmd: list[str], gpu: int | None, tag: str, log: Path, quiet: bool = False) -> int:
    """Runs one script, streaming its lines (prefixed with the tag) to stdout and to a log file."""
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                text=True, bufsize=1) as proc:
        for line in proc.stdout:  # type: ignore[union-attr]
            fh.write(line)
            if not quiet and not re.search(r"MallocStackLogging|Warning", line):
                print(f"[{tag}] {line}", end="", flush=True)
        return proc.wait()


def smoke(a) -> list[str]:
    return ["--max-batches", "3", "--no-pretrained"] if a.smoke else []


def seg_cmd(a, fold: int, folds: int, out: Path, extra: list[str] | None = None) -> list[str]:
    return [PY, "scripts/train_seg.py", "--manifest", a.seg_manifest, "--task", "boundary", "--arch", a.seg_arch,
            "--encoder", a.seg_encoder, "--size", str(a.seg_size), "--batch-size", str(a.seg_batch),
            "--epochs", str(a.seg_epochs), "--patience", str(a.seg_patience), "--workers", str(a.workers),
            "--folds", str(folds), "--fold", str(fold), "--out", str(out), *smoke(a), *(extra or [])]


def cls_cmd(a, fold: int, folds: int, out: Path) -> list[str]:
    return [PY, "scripts/train_cls.py", "--manifest", a.cls_manifest, "--target", "wound_type",
            "--backbone", a.cls_backbone, "--size", str(a.cls_size), "--batch-size", str(a.cls_batch),
            "--epochs", str(a.cls_epochs), "--workers", str(a.workers), "--folds", str(folds), "--fold", str(fold),
            "--out", str(out), *smoke(a)]


def projected_hours(a, folds: int, gpus: int, out: Path) -> float:
    """Times 20 training + 20 validation batches of the outline model at the real size, and scales up."""
    import pandas as pd

    trial = out / "_trial"
    shutil.rmtree(trial, ignore_errors=True)
    lines: list[str] = []
    code = run(seg_cmd(a, 0, folds, trial, ["--epochs", "1", "--max-batches", "20"]), 0 if gpus else None,
               f"trial {folds}-fold", trial / "log.txt")
    lines = (trial / "log.txt").read_text().splitlines()
    shutil.rmtree(trial, ignore_errors=True)
    secs = [int(m.group(1)) for l in lines for m in [re.search(r"val_dice [\d.]+ \((\d+)s\)", l)] if m]
    if code != 0 or not secs:
        sys.exit("The timing trial failed; see the lines above (out of GPU memory? try --seg-batch 2).")
    n_pool = int((pd.read_csv(ROOT / a.seg_manifest if not Path(a.seg_manifest).is_absolute() else a.seg_manifest)
                  .split != "test").sum())
    n_train, n_val = n_pool * (folds - 1) / folds, n_pool / folds
    b_train, b_val = math.ceil(n_train / a.seg_batch), math.ceil(n_val / a.seg_batch)
    # A validation batch costs about a third of a training batch (no backward pass).
    per_epoch = secs[0] * (b_train + b_val / 3) / (min(20, b_train) + min(20, b_val) / 3)
    rounds = math.ceil(folds / max(1, gpus))
    seg_h = rounds * per_epoch * a.seg_epochs / 3600
    # Classifier: measured ~6 min per fold for ~700 photos at 384 px on a T4; scales with the photo count.
    n_cls = int((pd.read_csv(ROOT / a.cls_manifest if not Path(a.cls_manifest).is_absolute() else a.cls_manifest)
                 .split != "test").sum()) if not a.skip_cls else 0
    cls_h = rounds * 6 * max(1.0, n_cls / 700) / 60 + folds * 3 / 60  # + evaluation ~3 min per fold
    print(f"timing: {per_epoch:.0f} s per outline epoch at {a.seg_size}px; {folds} folds on {max(1, gpus)} GPU(s) "
          f"-> about {seg_h + cls_h:.1f} h (worst case: all {a.seg_epochs} epochs, no early stop)", flush=True)
    return seg_h + cls_h


def in_parallel(jobs: list[tuple[str, list[str], Path, Path]], gpus: int, quiet: bool = False) -> list[int]:
    """Runs (tag, cmd, log, done_marker) jobs, at most one per GPU at a time. Finished jobs are skipped."""
    codes: dict[int, int] = {}
    queue = list(enumerate(jobs))
    lock = threading.Lock()

    def worker(gpu: int | None):
        while True:
            with lock:
                if not queue:
                    return
                i, (tag, cmd, log, done) = queue.pop(0)
            if done.exists():
                print(f"[{tag}] already finished, skipped", flush=True)
                codes[i] = 0
                continue
            codes[i] = run(cmd, gpu, tag, log, quiet)
            if quiet:
                print(f"[{tag}] {'done' if codes[i] == 0 else f'FAILED, see {log}'}", flush=True)
            if codes[i] == 0:
                done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

    threads = [threading.Thread(target=worker, args=(g,)) for g in (range(gpus) if gpus else [None])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [codes.get(i, 1) for i in range(len(jobs))]


def mean_sd(values: list[float]) -> dict:
    vals = [v for v in values if v is not None and not math.isnan(v)]  # NaN: e.g. AUROC of a one-class test set
    if not vals:
        return {"mean": None, "sd": None}
    return {"mean": round(statistics.mean(vals), 4), "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0}


def main():
    a = parse()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpus = gpu_count()
    print(f"GPUs: {gpus or 'none (CPU/Apple GPU, folds run one at a time)'}", flush=True)

    plan_file = out / "plan.json"
    if plan_file.exists():
        folds = json.loads(plan_file.read_text())["folds"]
        print(f"continuing the earlier run: {folds} folds", flush=True)
    else:
        folds = a.folds
        if not a.skip_seg:
            hours = projected_hours(a, a.folds, gpus, out)
            if hours > a.budget_hours and a.fallback_folds < a.folds:
                print(f"{a.folds} folds would take about {hours:.1f} h, over the {a.budget_hours} h budget: "
                      f"switching to {a.fallback_folds} folds.", flush=True)
                folds = a.fallback_folds
                hours = projected_hours(a, folds, gpus, out)
                if hours > a.budget_hours:
                    print(f"WARNING: {folds} folds may still take {hours:.1f} h. If the session stops, run the same "
                          "command again: finished folds are kept and the outline resumes per epoch.", flush=True)
            else:
                print(f"{folds} folds fit the {a.budget_hours} h budget.", flush=True)
        plan_file.write_text(json.dumps({"folds": folds, "seg_size": a.seg_size, "started": time.strftime("%Y-%m-%d %H:%M")}))

    t0 = time.time()
    if not a.skip_seg:
        jobs = [(f"outline {f + 1}/{folds}", seg_cmd(a, f, folds, out / f"seg/fold{f}", ["--resume"]),
                 out / f"seg/fold{f}/log.txt", out / f"seg/fold{f}/.done") for f in range(folds)]
        for f in range(folds):
            (out / f"seg/fold{f}").mkdir(parents=True, exist_ok=True)
        codes = in_parallel(jobs, gpus)
        if any(codes):
            sys.exit(f"outline folds failed: {[i for i, c in enumerate(codes) if c]}. Run the same command again.")
    if not a.skip_cls:
        jobs = [(f"type {f + 1}/{folds}", cls_cmd(a, f, folds, out / f"cls/fold{f}"), out / f"cls/fold{f}/log.txt",
                 out / f"cls/fold{f}/.done") for f in range(folds)]
        for f in range(folds):
            (out / f"cls/fold{f}").mkdir(parents=True, exist_ok=True)
        codes = in_parallel(jobs, gpus)
        if any(codes):
            sys.exit(f"wound type folds failed: {[i for i, c in enumerate(codes) if c]}. Run the same command again.")
    print(f"training done in {(time.time() - t0) / 3600:.1f} h; scoring every fold on the locked test set", flush=True)

    # Every fold's model on the same locked test set.
    evals = []
    for kind, manifest, extra in (("seg", a.seg_manifest, []), ("cls", a.cls_manifest, ["--use-gt-mask"])):
        if (kind == "seg" and a.skip_seg) or (kind == "cls" and a.skip_cls):
            continue
        for f in range(folds):
            d = out / f"{kind}/fold{f}"
            report = d / "eval_test.json"
            if not report.exists():
                evals.append((f"test {kind} {f + 1}", [PY, "scripts/evaluate.py", "--manifest", manifest, "--ckpt-dir", str(d),
                                                       "--group-cols", "source", "--out", str(report), *extra],
                              d / "eval_log.txt", d / ".evaluated"))
    in_parallel(evals, gpus, quiet=True)  # the full reports are saved; the summary below has the numbers

    import torch

    summary: dict = {"folds": folds, "seg_size": a.seg_size, "seg": [], "cls": []}
    for f in range(folds):
        d = out / f"seg/fold{f}"
        if (d / "boundary.pt").exists():
            ck = torch.load(d / "boundary.pt", map_location="cpu", weights_only=False)
            test = json.loads((d / "eval_test.json").read_text())["components"]["boundary"] if (d / "eval_test.json").exists() else {}
            summary["seg"].append({"fold": f, "val_dice": ck.get("val_dice"), "version": ck.get("version"),
                                   "test_dice": (test.get("dice") or [None])[0], "test_iou": (test.get("iou") or [None])[0]})
        d = out / f"cls/fold{f}"
        if (d / "wound_type.pt").exists():
            val = json.loads((d / "val_report.json").read_text())
            test = json.loads((d / "eval_test.json").read_text())["components"]["wound_type"] if (d / "eval_test.json").exists() else {}
            summary["cls"].append({"fold": f, "val_macro_f1": val["macro_f1"], "test_macro_f1": (test.get("macro_f1") or [None])[0],
                                   "test_accuracy": (test.get("accuracy") or [None])[0], "test_auroc": test.get("macro_auroc"),
                                   "test_per_class_sensitivity": {k: v["sensitivity"] for k, v in (test.get("per_class") or {}).items()}})
    summary["seg_cv"] = {k: mean_sd([r[k] for r in summary["seg"]]) for k in ("val_dice", "test_dice", "test_iou")}
    summary["cls_cv"] = {k: mean_sd([r[k] for r in summary["cls"]]) for k in ("val_macro_f1", "test_macro_f1", "test_accuracy", "test_auroc")}

    # The fold with the best validation score (never chosen on test) is the one to install.
    best = out / "best"
    best.mkdir(exist_ok=True)
    if summary["seg"]:
        b = max(summary["seg"], key=lambda r: r["val_dice"] or 0)
        shutil.copy(out / f"seg/fold{b['fold']}/boundary.pt", best / "boundary.pt")
        summary["seg_best_fold"] = b
    if summary["cls"]:
        b = max(summary["cls"], key=lambda r: r["val_macro_f1"] or 0)
        shutil.copy(out / f"cls/fold{b['fold']}/wound_type.pt", best / "wound_type.pt")
        summary["cls_best_fold"] = b
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    def fmt(ms):
        return "—" if ms["mean"] is None else f"{ms['mean']:.3f} ± {ms['sd']:.3f}"

    md = [f"# Cross-validation: {folds} folds (outline at {a.seg_size}px)", "",
          "| Model | Validation (mean ± sd) | Locked test (mean ± sd) |", "| --- | --- | --- |"]
    if summary["seg"]:
        md.append(f"| Outline, Dice | {fmt(summary['seg_cv']['val_dice'])} | {fmt(summary['seg_cv']['test_dice'])} |")
        md.append(f"| Outline, IoU | | {fmt(summary['seg_cv']['test_iou'])} |")
    if summary["cls"]:
        md.append(f"| Wound type, macro-F1 | {fmt(summary['cls_cv']['val_macro_f1'])} | {fmt(summary['cls_cv']['test_macro_f1'])} |")
        md.append(f"| Wound type, accuracy | | {fmt(summary['cls_cv']['test_accuracy'])} |")
        md.append(f"| Wound type, AUROC | | {fmt(summary['cls_cv']['test_auroc'])} |")
    md += ["", "Per fold:", ""]
    md += [f"- outline fold {r['fold'] + 1}: val Dice {r['val_dice']:.3f}, test Dice {r['test_dice'] or 0:.3f}" for r in summary["seg"]]
    md += [f"- type fold {r['fold'] + 1}: val F1 {r['val_macro_f1']:.3f}, test F1 {r['test_macro_f1'] or 0:.3f}" for r in summary["cls"]]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
