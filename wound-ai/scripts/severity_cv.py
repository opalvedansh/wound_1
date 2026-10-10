"""Cross-validated severity heads: pressure-injury stage, burn depth and diabetic-foot Wagner grade.

Per head and fold (grouped by copy group so a photo and its augmented copies stay together, stratified by label; the
locked test set is never touched), train_cls.py starts from the installed wound-type model's backbone, which has
already learnt wounds from ~4,800 photos. Every fold's model is scored on the head's locked test photos.

A head is trusted only if it has at least --min-test test photos and its mean test macro-F1 over the folds reaches
--trust-f1: then its best fold (by validation,
never by test) goes to OUT/best/<head>.pt, ready for checkpoints/. Otherwise it goes to OUT/untrusted/ and the app
keeps saying nothing about that head. With several GPUs, folds run in parallel. Re-running continues where it stopped.

    python scripts/severity_cv.py --manifest data/severity_manifest.csv --init checkpoints/wound_type.pt \
        --out /kaggle/working/severity_cv

Writes OUT/summary.json and OUT/summary.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cv import FINAL_FOLDS, PY, gpu_count, in_parallel, mean_sd  # noqa: E402

HEADS = ("pu_stage", "burn_depth", "dfu_wagner")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="data/severity_manifest.csv")
    p.add_argument("--heads", default=",".join(HEADS))
    p.add_argument("--init", default="checkpoints/wound_type.pt", help="backbone to start from")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--backbone", default="convnext_tiny.fb_in22k")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--trust-f1", type=float, default=0.6, help="mean locked-test macro-F1 a head needs to be installed")
    p.add_argument("--min-test", type=int, default=30, help="test photos a head needs before it can be trusted at all")
    p.add_argument("--out", default="runs/severity_cv")
    p.add_argument("--final-fit", action="store_true",
                   help="train each trusted head once more on all but 1/20 of its non-test photos and install that "
                        "model if its locked-test macro-F1 is at least the folds' mean")
    p.add_argument("--smoke", action="store_true", help="debug: tiny run (3 batches, 1 epoch, no pretrained weights)")
    return p.parse_args()


def main():
    a = parse()
    out = Path(a.out).resolve()
    gpus = gpu_count()
    heads = [h for h in a.heads.split(",") if h]
    smoke = ["--max-batches", "3", "--no-pretrained", "--epochs", "1"] if a.smoke else []
    init = ["--init", a.init] if a.init and Path(a.init).exists() else []
    if a.init and not init:
        print(f"WARNING: {a.init} not found: starting from ImageNet weights", flush=True)
    print(f"GPUs: {gpus or 'none'} | heads {heads} | {a.folds} folds", flush=True)

    t0 = time.time()
    dirs = {(h, f): out / h / f"fold{f}" for h in heads for f in range(a.folds)}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    def train_cmd(h: str, fold: int, folds: int, d: Path) -> list[str]:
        return [PY, "scripts/train_cls.py", "--manifest", a.manifest, "--target", h, "--backbone", a.backbone,
                "--size", str(a.size), "--batch-size", str(a.batch), "--epochs", str(a.epochs), "--patience", str(a.patience),
                "--workers", str(a.workers), "--folds", str(folds), "--fold", str(fold), "--out", str(d), *init, *smoke]

    def test_cmd(d: Path) -> list[str]:
        return [PY, "scripts/evaluate.py", "--manifest", a.manifest, "--ckpt-dir", str(d), "--group-cols", "source",
                "--out", str(d / "eval_test.json")]

    jobs = [(f"{h} {f + 1}/{a.folds}", train_cmd(h, f, a.folds, d), d / "log.txt", d / ".done") for (h, f), d in dirs.items()]
    codes = in_parallel(jobs, gpus)
    if any(codes):
        sys.exit(f"failed: {[jobs[i][0] for i, c in enumerate(codes) if c]}. Run the same command again.")
    print(f"training done in {(time.time() - t0) / 3600:.1f} h; scoring every fold on its locked test photos", flush=True)
    in_parallel([(f"test {h} {f + 1}", test_cmd(d), d / "eval_log.txt", d / ".evaluated") for (h, f), d in dirs.items()],
                gpus, quiet=True)

    import torch

    summary: dict = {"folds": a.folds, "trust_f1": a.trust_f1, "heads": {}}
    for h in heads:
        folds = []
        for f in range(a.folds):
            d = dirs[(h, f)]
            val = json.loads((d / "val_report.json").read_text())
            ev = json.loads((d / "eval_test.json").read_text())["components"].get(h, {}) if (d / "eval_test.json").exists() else {}
            folds.append({"fold": f, "val_macro_f1": val["macro_f1"], "test_n": ev.get("n"),
                          "test_macro_f1": (ev.get("macro_f1") or [None])[0], "test_accuracy": (ev.get("accuracy") or [None])[0],
                          "test_auroc": ev.get("macro_auroc"),
                          "test_sensitivity": {k: v["sensitivity"] for k, v in (ev.get("per_class") or {}).items()}})
        cv = {k: mean_sd([r[k] for r in folds]) for k in ("val_macro_f1", "test_macro_f1", "test_accuracy", "test_auroc")}
        classes = sorted({c for r in folds for c in r["test_sensitivity"]})
        sens = {c: mean_sd([r["test_sensitivity"].get(c) for r in folds]) for c in classes}
        n_test = folds[0]["test_n"] or 0
        trusted = n_test >= a.min_test and cv["test_macro_f1"]["mean"] is not None and cv["test_macro_f1"]["mean"] >= a.trust_f1
        best = max(folds, key=lambda r: r["val_macro_f1"] or 0)
        source, final = dirs[(h, best["fold"])], None
        if a.final_fit and trusted:
            # The same settings on nearly all of the head's non-test photos, scored on the same locked test photos.
            d = out / h / "final"
            d.mkdir(parents=True, exist_ok=True)
            n = 2 if a.smoke else FINAL_FOLDS  # the smoke test's few patients cannot fill 20 folds
            if in_parallel([(f"final {h}", train_cmd(h, 0, n, d), d / "log.txt", d / ".done")], gpus)[0]:
                sys.exit(f"the final fit of {h} failed. Run the same command again.")
            in_parallel([(f"test final {h}", test_cmd(d), d / "eval_log.txt", d / ".evaluated")], gpus, quiet=True)
            score = (json.loads((d / "eval_test.json").read_text())["components"].get(h, {}).get("macro_f1") or [None])[0]
            final = {"test_macro_f1": score, "installed": score is not None and score >= cv["test_macro_f1"]["mean"]}
            if final["installed"]:
                source = d
        ck = torch.load(source / f"{h}.pt", map_location="cpu", weights_only=False)
        ck["cv"] = {"folds": a.folds, **cv, "sensitivity": sens, "fold": best["fold"], "trusted": trusted,
                    **({"final_fit": final} if final else {})}
        dest = out / ("best" if trusted else "untrusted")
        dest.mkdir(exist_ok=True)
        torch.save(ck, dest / f"{h}.pt")
        summary["heads"][h] = {"per_fold": folds, "cv": cv, "sensitivity": sens, "trusted": trusted, "best_fold": best["fold"],
                               **({"final_fit": final} if final else {})}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    def fmt(ms):
        return "—" if ms["mean"] is None else f"{ms['mean']:.3f} ± {ms['sd']:.3f}"

    md = [f"# Severity heads: {a.folds}-fold cross-validation", "",
          f"Locked test photos come only from datasets that look original; trusted = at least {a.min_test} test photos and "
          f"mean test macro-F1 ≥ {a.trust_f1}.", "",
          "| Head | Test photos | Val macro-F1 | Test macro-F1 | Test accuracy | Test AUROC | Trusted |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for h, v in summary["heads"].items():
        md.append(f"| {h} | {v['per_fold'][0]['test_n']} | {fmt(v['cv']['val_macro_f1'])} | {fmt(v['cv']['test_macro_f1'])} | "
                  f"{fmt(v['cv']['test_accuracy'])} | {fmt(v['cv']['test_auroc'])} | {'yes' if v['trusted'] else 'no'} |")
    for h, v in summary["heads"].items():
        if v.get("final_fit"):
            f = v["final_fit"]
            md += ["", f"{h}, final fit on all but 1/{FINAL_FOLDS} of its photos: test macro-F1 {f['test_macro_f1']:.3f} "
                       f"({'installed' if f['installed'] else 'below the folds, best fold kept'})"]
    for h, v in summary["heads"].items():
        md += ["", f"{h}, test sensitivity per class: " + ", ".join(f"{c} {fmt(s)}" for c, s in v["sensitivity"].items())]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
