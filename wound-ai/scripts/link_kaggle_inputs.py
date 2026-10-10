"""Makes the public wound datasets available in data/raw/public/, where build_public_dataset.py expects them
(same folder names as a local `kaggle datasets download`).

Datasets attached to the Kaggle notebook as inputs are linked; any that aren't are downloaded with kagglehub
(preinstalled on Kaggle; needs Internet on), so attaching inputs is optional.

    python scripts/link_kaggle_inputs.py                 # on Kaggle
    python scripts/link_kaggle_inputs.py --input-root X  # anywhere else
    python scripts/link_kaggle_inputs.py --set tissue    # the tissue dataset, into data/raw/tissue/
    python scripts/link_kaggle_inputs.py --set pool      # extra unlabelled wound photos, into data/raw/pool/
    python scripts/link_kaggle_inputs.py --set severity  # stage/depth/grade-labelled photos, into data/raw/severity/

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
# Tissue-labelled data (build_tissue_dataset.py); WoundTissue and LUTSeg come from GitHub and Hugging Face instead.
TISSUE_DATASETS = {"ibrahimshehada/dfutissue": "dfutissue"}
# Extra photos for the tissue model's unlabelled pool only (pseudo-labels; never trained on with their own labels, never
# tested on). Not CO2Wounds: LUTSeg's photos come from it, so it would hold the locked test patients' other visits.
POOL_DATASETS = {
    "pabodhamallawa/dfuc2022-train-release": "pabodhamallawa_dfuc2022-train-release",  # DFUC2022 train, 2000 DFU
    "tushartalukder4/dfuc-c5": "tushartalukder4_dfuc-c5",  # DFUC2022 test images, 2000 DFU
    "abdulazizalghaili/postoperative-wound-infection": "abdulazizalghaili_postoperative-wound-infection",  # 380 surgical
    "mohamadtaher/wound-data": "mohamadtaher_wound-data",  # FUSeg incl. its 200 challenge-test photos
}
# Pressure-injury stage, burn depth and Wagner grade labels (build_severity_dataset.py; it also reads two public sets).
SEVERITY_DATASETS = {ref: ref.replace("/", "_") for ref in (
    "faresabbasai2022/burn-dataset", "faresabbasai2022/burn-dataset13", "mohammaddimasnoufal/skin-burn-dataset",
    "nomi6677/skin-burn-classification-with-degrees", "divyanshmahajan2311/pressure-ulcer-stagewise",
    "divyanshmahajan2311/corrected-ulcer-dataset", "gursahiba/final-ulcer-dataset",
    "purushomohan/dfu-wagners-classification")}
SETS = {"public": (DATASETS, "data/raw/public"), "tissue": (TISSUE_DATASETS, "data/raw/tissue"),
        "pool": (POOL_DATASETS, "data/raw/pool"), "severity": (SEVERITY_DATASETS, "data/raw/severity")}
__doc__ += "".join(f"\n    {ref}" for ref in [*DATASETS, *TISSUE_DATASETS, *POOL_DATASETS, *SEVERITY_DATASETS])


def find(root: Path, ref: str) -> Path | None:
    owner, slug = ref.split("/")
    # Kaggle has mounted inputs as /kaggle/input/<slug> and as /kaggle/input/datasets/<owner>/<slug>.
    for candidate in (root / slug, root / "datasets" / owner / slug, *root.glob(f"*/{slug}"), *root.glob(f"*/*/{slug}")):
        if candidate.is_dir():
            return candidate
    return None


NON_INTERACTIVE = False


def download(ref: str) -> Path | None:
    """Downloads a public Kaggle dataset with kagglehub (no sign-in needed for public datasets on Kaggle)."""
    try:
        import kagglehub

        print(f"downloading {ref} ...", flush=True)
        return Path(kagglehub.dataset_download(ref))
    except Exception as e:  # not installed, offline, renamed, or a background run
        global NON_INTERACTIVE
        NON_INTERACTIVE = NON_INTERACTIVE or "non-interactive" in str(e)
        print(f"  could not download {ref}: {str(e)[:160]}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--set", choices=list(SETS), default="public")
    ap.add_argument("--out", default="", help="default: data/raw/<set>")
    ap.add_argument("--no-download", action="store_true", help="only link attached inputs, never download")
    a = ap.parse_args()
    datasets, default_out = SETS[a.set]
    root, out = Path(a.input_root), Path(a.out or default_out)
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for ref, local in datasets.items():
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
    if missing and NON_INTERACTIVE:
        sys.exit("Kaggle only adds new datasets to a notebook in an interactive session. Open the notebook editor, "
                 "turn the session on and run the cells up to this one once (this attaches the datasets for good), "
                 "then Save Version -> Save & Run All again. Missing:\n  " + "\n  ".join(missing))
    if missing:
        sys.exit("Couldn't get (is Internet on in Session options?):\n  " + "\n  ".join(missing))
    print(f"all {len(datasets)} datasets linked")


if __name__ == "__main__":
    main()
