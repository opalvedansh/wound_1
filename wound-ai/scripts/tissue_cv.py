"""Cross-validated, semi-supervised training of the tissue model.

Per fold (grouped by patient, balanced by dataset; the locked test set is never touched):
  1. teacher: trained on the fold's human-labelled training photos (init from the outline model)
  2. pseudo-labels: the teacher labels the unlabelled pool where it is confident (scripts/pseudo_label.py)
  3. student: trained on the same human labels plus the pseudo-labels (init from the teacher)
  4. the better of the two on the fold's human-labelled validation photos is kept (never chosen on test)
Every kept model is scored on the locked test set (scripts/evaluate.py), and each class is compared with how well
LUTSeg's five clinicians agree with each other on the same kind of photos.

--teacher-only skips steps 2-3 (each fold keeps its teacher): on the first Kaggle run the students were worse than
their teachers in 2 of 3 folds, so the pseudo-labels cost ~8 GPU hours for nothing.

A short timed trial first projects the run time: --folds if they fit --budget-hours, else --fallback-folds. With
several GPUs, folds run in parallel. Re-running the same command continues where it stopped.

    python scripts/tissue_cv.py --manifest data/tissue_manifest.csv --unlabeled data/tissue_unlabeled.csv \
        --boundary checkpoints/boundary.pt --rater data/tissue_rater_agreement.json --out /kaggle/working/tissue_cv

Writes OUT/summary.json, OUT/summary.md and OUT/best/tissue.pt (the best fold by validation, with the classes it may
be trusted on recorded in its config: the app uses only those).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cv import PY, gpu_count, in_parallel, mean_sd, run  # noqa: E402
from wound_ai.data import IGNORE_INDEX, TISSUE_CLASSES  # noqa: E402

PSEUDO_SECONDS_PER_PHOTO = 0.25  # outline + tissue model on a T4, measured order of magnitude
# The final fit holds out one fold of 10: with about 200 labelled photos that leaves about 20 to stop on.
FINAL_FOLDS = 10


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/tissue_manifest.csv")
    p.add_argument("--unlabeled", default="data/tissue_unlabeled.csv")
    p.add_argument("--boundary", default="checkpoints/boundary.pt")
    p.add_argument("--rater", default="data/tissue_rater_agreement.json")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fallback-folds", type=int, default=3)
    p.add_argument("--budget-hours", type=float, default=10.0)
    p.add_argument("--arch", default="segformer")
    p.add_argument("--encoder", default="mit_b2")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--teacher-epochs", type=int, default=80)
    p.add_argument("--teacher-patience", type=int, default=15)
    p.add_argument("--student-epochs", type=int, default=40)
    p.add_argument("--student-patience", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.9, help="teacher confidence needed for a pseudo-label")
    p.add_argument("--final-fit", action="store_true",
                   help="train a teacher once more on all but 1/10 of the labelled photos and install it if its "
                        "locked-test Dice over the trusted classes is at least the folds' mean")
    p.add_argument("--min-train-photos", type=int, default=15,
                   help="classes in fewer training photos are never pseudo-labelled nor trusted")
    p.add_argument("--min-test-photos", type=int, default=10, help="classes in fewer test photos are not trusted")
    p.add_argument("--trust-ratio", type=float, default=0.8,
                   help="trusted if test Dice >= this x clinician-to-clinician Dice (or --trust-dice without one)")
    p.add_argument("--trust-dice", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="runs/tissue_cv")
    p.add_argument("--teacher-only", action="store_true", help="no pseudo-labels, no student: each fold keeps its teacher")
    p.add_argument("--smoke", action="store_true", help="debug: tiny run (3 batches, 1 epoch, 20 pseudo photos)")
    return p.parse_args()


def class_stats(df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """Photos containing each class, and pixels per class, in the given rows."""
    photos, pixels = {c: 0 for c in TISSUE_CLASSES}, np.zeros(len(TISSUE_CLASSES), np.int64)
    for path in df.tissue_path:
        m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        m = m[m != IGNORE_INDEX]
        counts = np.bincount(m.ravel(), minlength=len(TISSUE_CLASSES))[:len(TISSUE_CLASSES)]
        pixels += counts
        for k, c in enumerate(TISSUE_CLASSES):
            photos[c] += int(counts[k] > 0)
    return photos, pixels


def class_weights(pixels: np.ndarray) -> list[float]:
    """Inverse square-root pixel frequency, median tissue class = 1, clipped to [0.5, 4]; background 0.5."""
    tissue = pixels[1:].astype(float)
    w = np.ones(len(pixels))
    present = tissue > 0
    if present.any():
        inv = np.zeros_like(tissue)
        inv[present] = 1 / np.sqrt(tissue[present])
        inv[present] /= np.median(inv[present])
        w[1:] = np.where(present, np.clip(inv, 0.5, 4.0), 1.0)
    w[0] = 0.5
    return [round(float(x), 2) for x in w]


def train_cmd(a, fold: int, folds: int, out: Path, weights: list[float], init: str, epochs: int, patience: int,
              extra: list[str] | None = None) -> list[str]:
    cmd = [PY, "scripts/train_seg.py", "--manifest", a.manifest, "--task", "tissue", "--crop", "--arch", a.arch,
           "--encoder", a.encoder, "--size", str(a.size), "--batch-size", str(a.batch), "--epochs", str(epochs),
           "--patience", str(patience), "--workers", str(a.workers), "--folds", str(folds), "--fold", str(fold),
           "--class-weights", ",".join(map(str, weights)), "--init", init, "--out", str(out), "--resume"]
    if a.smoke:
        cmd += ["--max-batches", "3", "--no-pretrained"]
    return cmd + (extra or [])


def projected_hours(a, folds: int, gpus: int, weights: list[float], n_pool: int, n_unlabeled: int, out: Path) -> float:
    """Times 20 teacher batches at the real size and scales up: teachers, pseudo-labelling, students."""
    trial = out / "_trial"
    shutil.rmtree(trial, ignore_errors=True)
    code = run(train_cmd(a, 0, folds, trial, weights, a.boundary, 1, 1, ["--max-batches", "20"]),
               0 if gpus else None, f"trial {folds}-fold", trial / "log.txt")
    lines = (trial / "log.txt").read_text().splitlines() if (trial / "log.txt").exists() else []
    shutil.rmtree(trial, ignore_errors=True)
    secs = [int(m.group(1)) for line in lines for m in [re.search(r"val_dice [\d.]+ \((\d+)s\)", line)] if m]
    if code != 0 or not secs:
        sys.exit("The timing trial failed; see the lines above (out of GPU memory? try --batch 4).")
    per_batch = secs[0] / 20
    n_train = n_pool * (folds - 1) / folds
    pseudo = 0.7 * n_unlabeled  # photos that usually get pseudo-labels
    teacher = a.teacher_epochs * math.ceil(n_train / a.batch) * per_batch * 1.15  # + validation
    student = a.student_epochs * math.ceil((n_train + pseudo) / a.batch) * per_batch * 1.05
    label = n_unlabeled * PSEUDO_SECONDS_PER_PHOTO
    if a.teacher_only:
        student = label = 0
    hours = math.ceil(folds / max(1, gpus)) * (teacher + label + student) / 3600
    print(f"timing: {per_batch:.2f} s per batch at {a.size}px; {folds} folds on {max(1, gpus)} GPU(s) -> about "
          f"{hours:.1f} h (worst case: all epochs, no early stop)", flush=True)
    return hours


def val_dice(ckpt: Path) -> float | None:
    if not ckpt.exists():
        return None
    import torch

    return torch.load(ckpt, map_location="cpu", weights_only=False).get("val_dice")


def main():
    a = parse()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpus = gpu_count()
    df = pd.read_csv(a.manifest)
    pool, test = df[df.split != "test"], df[df.split == "test"]
    n_unlabeled = 0 if a.teacher_only else len(pd.read_csv(a.unlabeled))
    if a.smoke:
        n_unlabeled = min(n_unlabeled, 20)
    train_photos, train_pixels = class_stats(pool)
    test_photos, _ = class_stats(test)
    weights = class_weights(train_pixels)
    rare = [c for c in TISSUE_CLASSES[1:] if train_photos[c] < a.min_train_photos]
    print(f"GPUs: {gpus or 'none'} | labelled pool {len(pool)}, locked test {len(test)}, unlabelled {n_unlabeled}", flush=True)
    print(f"training photos per class: {train_photos}\nclass weights: {weights}\n"
          f"too few training photos (never pseudo-labelled, never trusted): {rare}", flush=True)

    plan_file = out / "plan.json"
    if plan_file.exists():
        folds = json.loads(plan_file.read_text())["folds"]
        print(f"continuing the earlier run: {folds} folds", flush=True)
    else:
        folds = a.folds
        if not a.smoke:
            hours = projected_hours(a, folds, gpus, weights, len(pool), n_unlabeled, out)
            if hours > a.budget_hours and a.fallback_folds < folds:
                folds = a.fallback_folds
                print(f"over the {a.budget_hours} h budget: switching to {folds} folds", flush=True)
                if projected_hours(a, folds, gpus, weights, len(pool), n_unlabeled, out) > a.budget_hours:
                    print("WARNING: may still overrun. If the session stops, run the same command again: finished "
                          "steps are kept and training resumes per epoch.", flush=True)
        plan_file.write_text(json.dumps({"folds": folds, "weights": weights, "rare": rare,
                                         "started": time.strftime("%Y-%m-%d %H:%M")}))

    t0 = time.time()
    fd = [out / f"fold{f}" for f in range(folds)]
    for d in fd:
        d.mkdir(parents=True, exist_ok=True)
    stage = lambda name, jobs: (in_parallel(jobs, gpus), name)  # noqa: E731

    codes, _ = stage("teacher", [(f"teacher {f + 1}/{folds}", train_cmd(a, f, folds, fd[f] / "teacher", weights, a.boundary,
                                  1 if a.smoke else a.teacher_epochs, a.teacher_patience), fd[f] / "teacher/log.txt",
                                  fd[f] / "teacher/.done") for f in range(folds)])
    if any(codes):
        sys.exit(f"teacher folds failed: {[i for i, c in enumerate(codes) if c]}. Run the same command again.")
    if not a.teacher_only:
        codes, _ = stage("pseudo", [(f"pseudo-labels {f + 1}/{folds}",
                                     [PY, "scripts/pseudo_label.py", "--teacher", str(fd[f] / "teacher/tissue.pt"),
                                      "--boundary", a.boundary, "--unlabeled", a.unlabeled, "--out", str(fd[f] / "pseudo"),
                                      "--threshold", str(a.threshold), "--drop", ",".join(rare),
                                      *(["--max", "20"] if a.smoke else [])],
                                     fd[f] / "pseudo/log.txt", fd[f] / "pseudo/.done") for f in range(folds)])
        if any(codes):
            sys.exit(f"pseudo-labelling failed: {[i for i, c in enumerate(codes) if c]}. Run the same command again.")
        codes, _ = stage("student", [(f"student {f + 1}/{folds}", train_cmd(a, f, folds, fd[f] / "student", weights,
                                      str(fd[f] / "teacher/tissue.pt"), 1 if a.smoke else a.student_epochs, a.student_patience,
                                      ["--extra-train", str(fd[f] / "pseudo/pseudo.csv")]), fd[f] / "student/log.txt",
                                      fd[f] / "student/.done") for f in range(folds)])
        if any(codes):
            sys.exit(f"student folds failed: {[i for i, c in enumerate(codes) if c]}. Run the same command again.")
    print(f"training done in {(time.time() - t0) / 3600:.1f} h; scoring the kept model of each fold on the locked test set",
          flush=True)

    # Keep teacher or student per fold by validation (never by test), then score it on the locked test set.
    folds_out = []
    for f, d in enumerate(fd):
        tv, sv = val_dice(d / "teacher/tissue.pt"), val_dice(d / "student/tissue.pt")
        kept = "student" if (sv or 0) > (tv or 0) else "teacher"
        (d / "kept").mkdir(exist_ok=True)
        shutil.copy(d / kept / "tissue.pt", d / "kept/tissue.pt")
        stats = json.loads((d / "pseudo/pseudo_stats.json").read_text()) if (d / "pseudo/pseudo_stats.json").exists() else {}
        folds_out.append({"fold": f, "teacher_val_dice": tv, "student_val_dice": sv, "kept": kept,
                          "pseudo_labelled": stats.get("pseudo_labelled")})
    in_parallel([(f"test {f + 1}", [PY, "scripts/evaluate.py", "--manifest", a.manifest, "--ckpt-dir", str(d / "kept"),
                                    "--use-gt-mask", "--group-cols", "source", "--out", str(d / "eval_test.json")],
                  d / "eval_log.txt", d / ".evaluated") for f, d in enumerate(fd)], gpus, quiet=True)
    for r, d in zip(folds_out, fd):
        t = json.loads((d / "eval_test.json").read_text())["components"].get("tissue", {}) if (d / "eval_test.json").exists() else {}
        r["test_dice"] = {c: v[0] for c, v in (t.get("dice_per_class") or {}).items()}
        r["test_nonviable_error"] = (t.get("nonviable_abs_error_points") or [None])[0]
        r["test_debridement_kappa"] = (t.get("debridement_threshold") or {}).get("kappa")

    raters = json.loads(Path(a.rater).read_text()) if Path(a.rater).exists() else {}
    per_class = {}
    for c in TISSUE_CLASSES[1:]:
        ms = mean_sd([r["test_dice"].get(c) for r in folds_out])
        rater = (raters.get(c) or {}).get("mean")
        bar = round(a.trust_ratio * rater, 3) if rater else a.trust_dice
        trusted = (c not in rare and test_photos[c] >= a.min_test_photos and ms["mean"] is not None and ms["mean"] >= bar)
        per_class[c] = {"train_photos": train_photos[c], "test_photos": test_photos[c], "test_dice": ms,
                        "clinicians_dice": rater, "bar": bar, "trusted": trusted}
    trusted = [c for c, v in per_class.items() if v["trusted"]]

    import torch

    best = max(folds_out, key=lambda r: max(r["teacher_val_dice"] or 0, r["student_val_dice"] or 0))
    source, final = fd[best["fold"]] / "kept/tissue.pt", None
    if a.final_fit:
        # A fold's model never sees its validation fold. This one sees nearly every labelled photo, and is judged on
        # the classes the folds earned trust on (all scored classes if none did).
        d = out / "final"
        d.mkdir(exist_ok=True)
        n = 2 if a.smoke else FINAL_FOLDS
        if in_parallel([("final teacher", train_cmd(a, 0, n, d / "teacher", weights, a.boundary,
                                                    1 if a.smoke else a.teacher_epochs, a.teacher_patience),
                         d / "teacher/log.txt", d / "teacher/.done")], gpus)[0]:
            sys.exit("the final fit failed. Run the same command again.")
        in_parallel([("test final", [PY, "scripts/evaluate.py", "--manifest", a.manifest, "--ckpt-dir", str(d / "teacher"),
                                     "--use-gt-mask", "--group-cols", "source", "--out", str(d / "eval_test.json")],
                      d / "eval_log.txt", d / ".evaluated")], gpus, quiet=True)
        t = json.loads((d / "eval_test.json").read_text())["components"].get("tissue", {})
        dice = {c: v[0] for c, v in (t.get("dice_per_class") or {}).items()}
        judged = [c for c in (trusted or per_class) if dice.get(c) is not None and per_class[c]["test_dice"]["mean"] is not None]
        score = round(float(np.mean([dice[c] for c in judged])), 4) if judged else None
        folds_mean = round(float(np.mean([per_class[c]["test_dice"]["mean"] for c in judged])), 4) if judged else None
        final = {"test_dice": dice, "judged_on": judged, "mean_dice": score, "folds_mean": folds_mean,
                 "installed": score is not None and score >= folds_mean}
        if final["installed"]:
            source = d / "teacher/tissue.pt"
    (out / "best").mkdir(exist_ok=True)
    ck = torch.load(source, map_location="cpu", weights_only=False)
    ck["config"]["trusted_classes"] = trusted
    ck["cv"] = {"folds": folds, "per_class": per_class, "fold": best["fold"], "kept": best["kept"],
                **({"final_fit": final} if final else {})}
    torch.save(ck, out / "best/tissue.pt")
    summary = {"folds": folds, "weights": weights, "rare": rare, "per_fold": folds_out, "per_class": per_class,
               "trusted_classes": trusted, "best_fold": best, **({"final_fit": final} if final else {})}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    def fmt(ms):
        return "—" if ms["mean"] is None else f"{ms['mean']:.3f} ± {ms['sd']:.3f}"

    md = [f"# Tissue model: {folds}-fold cross-validation, {'teachers only' if a.teacher_only else 'semi-supervised'}", "",
          "Locked test: LUTSeg gold-standard patients + DFUTissue Test. Each class is scored only on photos whose "
          "dataset labels it. Clinicians = Dice between LUTSeg's five clinicians on the same kind of photos.", "",
          "| Class | Train photos | Test photos | Test Dice (mean ± sd over folds) | Clinicians | Bar | Trusted |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for c, v in per_class.items():
        md.append(f"| {c} | {v['train_photos']} | {v['test_photos']} | {fmt(v['test_dice'])} | "
                  f"{v['clinicians_dice'] if v['clinicians_dice'] is not None else '—'} | {v['bar']} | "
                  f"{'yes' if v['trusted'] else 'no'} |")
    md += ["", "| Fold | Teacher val Dice | Student val Dice | Kept | Pseudo-labelled photos | Non-viable error (points) |",
           "| --- | --- | --- | --- | --- | --- |"]
    dash = lambda x, f="{}": "—" if x is None else f.format(x)  # noqa: E731
    md += [f"| {r['fold'] + 1} | {dash(r['teacher_val_dice'], '{:.3f}')} | {dash(r['student_val_dice'], '{:.3f}')} | "
           f"{r['kept']} | {dash(r['pseudo_labelled'])} | {dash(r['test_nonviable_error'], '{:.1f}')} |"
           for r in folds_out]
    which = f"fold {best['fold'] + 1} ({best['kept']})"
    if final:
        md += ["", f"Final fit on all but 1/{FINAL_FOLDS} of the labelled photos: mean test Dice {dash(final['mean_dice'], '{:.3f}')} "
                   f"over {', '.join(final['judged_on']) or 'no class'} (folds {dash(final['folds_mean'], '{:.3f}')})."]
        which = "the final fit" if final["installed"] else f"{which}; the final fit scored below the folds"
    md += ["", f"Installed model: {which}, trusted on: {', '.join(trusted) or 'none'}."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
