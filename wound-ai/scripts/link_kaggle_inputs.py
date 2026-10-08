"""Makes the public wound datasets available in data/raw/public/, where build_public_dataset.py expects them
(same folder names as a local `kaggle datasets download`).

Datasets attached to the Kaggle notebook as inputs are linked; any that aren't are downloaded with kagglehub
(preinstalled on Kaggle; needs Internet on), so attaching inputs is optional.

    python scripts/link_kaggle_inputs.py                 # on Kaggle
    python scripts/link_kaggle_inputs.py --input-root X  # anywhere else

The datasets:
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Kaggle dataset -> folder name used locally (owner_slug).
DATASETS = {
    "ibrahimfateen/wound-classification": "ibrahimfateen_wound-classification",
    "yasinpratomo/wound-dataset": "yasinpratomo_wound-dataset",
    "sinemgokoz/pressure-ulcers-stages": "sinemgokoz_pressure-ulcers-stages",
    "laithjj/diabetic-foot-ulcer-dfu": "laithjj_diabetic-foot-ulcer-dfu",
    "workshops11/medetec-dataset": "workshops11_medetec-dataset",
    "shubhambaid/skin-burn-dataset": "shubhambaid_skin-burn-dataset",
    "leoscode/wound-segmentation-images": "leoscode_wound-segmentation-images",
}
__doc__ += "".join(f"\n    {ref}" for ref in DATASETS)


def find(root: Path, ref: str) -> Path | None:
    owner, slug = ref.split("/")
    # Kaggle has mounted inputs as /kaggle/input/<slug> and as /kaggle/input/datasets/<owner>/<slug>.
    for candidate in (root / slug, root / "datasets" / owner / slug, *root.glob(f"*/{slug}"), *root.glob(f"*/*/{slug}")):
        if candidate.is_dir():
            return candidate
    return None


def download(ref: str) -> Path | None:
    """Downloads a public Kaggle dataset with kagglehub (no sign-in needed for public datasets on Kaggle)."""
    try:
        import kagglehub

        print(f"downloading {ref} ...", flush=True)
        return Path(kagglehub.dataset_download(ref))
    except Exception as e:  # not installed, offline, renamed
        print(f"  could not download {ref}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--out", default="data/raw/public")
    ap.add_argument("--no-download", action="store_true", help="only link attached inputs, never download")
    a = ap.parse_args()
    root, out = Path(a.input_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for ref, local in DATASETS.items():
        src, dst = find(root, ref), out / local
        if src is None and not a.no_download:
            src = download(ref)
        if src is None:
            missing.append(ref)
            continue
        if dst.is_symlink() or dst.exists():
            dst.unlink() if dst.is_symlink() else None
        if not dst.exists():
            dst.symlink_to(src.resolve(), target_is_directory=True)
        print(f"linked {ref} -> {dst}")
    if missing:
        sys.exit("Couldn't get (is Internet on in Session options?):\n  " + "\n  ".join(missing))
    print(f"all {len(DATASETS)} datasets linked")


if __name__ == "__main__":
    main()
