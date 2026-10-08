"""Links the public wound datasets attached to a Kaggle notebook into data/raw/public/, where
build_public_dataset.py expects them (same folder names as a local `kaggle datasets download`).

    python scripts/link_kaggle_inputs.py                 # on Kaggle (inputs under /kaggle/input)
    python scripts/link_kaggle_inputs.py --input-root X  # anywhere else

Attach these as notebook inputs (Add Input -> search the name):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--out", default="data/raw/public")
    a = ap.parse_args()
    root, out = Path(a.input_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for ref, local in DATASETS.items():
        src, dst = find(root, ref), out / local
        if src is None:
            missing.append(ref)
            continue
        if dst.is_symlink() or dst.exists():
            dst.unlink() if dst.is_symlink() else None
        if not dst.exists():
            dst.symlink_to(src.resolve(), target_is_directory=True)
        print(f"linked {ref} -> {dst}")
    if missing:
        sys.exit("Not attached (Add Input -> search the name, then run again):\n  " + "\n  ".join(missing))
    print(f"all {len(DATASETS)} datasets linked")


if __name__ == "__main__":
    main()
