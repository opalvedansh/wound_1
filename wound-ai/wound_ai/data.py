"""Datasets, transforms and patient-level splitting.

Everything is driven by a single manifest CSV with (at least) these columns:

    image_path   path to the RGB photo
    mask_path    path to the wound mask PNG (optional; empty if none)
    tissue_path  path to the tissue-class mask PNG (optional)
    patient_id   who the photo belongs to  <- splits are done on THIS, never per image
    source       which dataset / hospital / device it came from
    split        train | val | test

plus any label columns you want to train on (wound_type, pu_stage, burn_depth,
dfu_infection, ...) and metadata columns (body_location, fitzpatrick, ...).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Tissue classes used in the tissue-segmentation masks (pixel value = class index). Wound bed: granulation to
# epithelial, plus exposed bone or tendon (a review flag). Around the wound: redness and macerated (soggy) skin.
# Callus (thick hard skin around a diabetic foot ulcer) is last so earlier indices never move.
TISSUE_CLASSES = ["background", "granulation", "slough", "necrosis", "epithelial", "periwound_erythema",
                  "maceration", "exposed_structure", "callus"]
WOUND_BED = ["granulation", "slough", "necrosis", "epithelial", "exposed_structure"]
PERIWOUND = ["periwound_erythema", "maceration", "callus"]
# Tissue mask pixels with this value are not learned from: an annotator was unsure, or a public dataset's class
# (such as dressing) has no place in TISSUE_CLASSES.
IGNORE_INDEX = 255
# Manifest column naming the tissue classes a row's dataset annotates, ";"-separated (all classes if empty). Public
# datasets label different subsets: a pixel one of them calls background may be a class it never labels.
# Tissue is fine detail, so the tissue model sees the wound's box plus a margin of skin, not the whole photo.
# Training varies the margin, as the predicted outline the app crops with will be a little off.
TISSUE_CROP_MARGIN = (0.10, 0.25)


# --------------------------------------------------------------------------- transforms

def build_transforms(size: int, train: bool) -> A.Compose:
    """Augmentations for wound photos.

    Colour carries clinical meaning (red granulation, yellow slough, black necrosis,
    redness of the surrounding skin), so hue shifts are kept very small. Geometry
    and lighting are varied more, because phone photos vary a lot there.
    """
    resize = [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
    ]
    if not train:
        return A.Compose(resize + [A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()])
    return A.Compose(
        resize
        + [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.Affine(scale=(0.8, 1.2), rotate=(-25, 25), translate_percent=(-0.05, 0.05), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=4, sat_shift_limit=12, val_shift_limit=12, p=0.3),
            A.RandomGamma(gamma_limit=(85, 115), p=0.2),
            A.OneOf([A.MotionBlur(blur_limit=5), A.GaussianBlur(blur_limit=(3, 5))], p=0.15),
            A.ImageCompression(quality_range=(60, 100), p=0.3),  # WhatsApp-style recompression
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str, binary: bool) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8) if binary else m.astype(np.uint8)


# --------------------------------------------------------------------------- datasets

class SegmentationDataset(Dataset):
    """Wound boundary (binary) or tissue (multi-class) segmentation.

    crop=True (tissue): photo and mask are cut to the wound's box plus a margin, as the app does with its predicted
    outline. The box comes from the boundary mask if the row has one, else from the tissue mask's wound bed."""

    def __init__(self, df: pd.DataFrame, size: int, train: bool, mask_col: str = "mask_path", binary: bool = True,
                 crop: bool = False):
        self.df = df[df[mask_col].fillna("").astype(str).str.len() > 0].reset_index(drop=True)
        self.tf = build_transforms(size, train)
        self.mask_col = mask_col
        self.binary = binary
        self.crop, self.train = crop, train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = read_rgb(row["image_path"])
        mask = read_mask(row[self.mask_col], self.binary)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.crop:
            lo, hi = TISSUE_CROP_MARGIN
            margin = float(np.random.uniform(lo, hi)) if self.train else (lo + hi) / 2
            bp = str(row.get("mask_path", "") or "")
            wound = read_mask(bp, binary=True) if bp and bp != "nan" else np.isin(
                mask, [TISSUE_CLASSES.index(c) for c in WOUND_BED]).astype(np.uint8)
            if wound.shape[:2] != img.shape[:2]:
                wound = cv2.resize(wound, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            box = wound_box(wound, margin)
            if box is not None:
                img, mask = img[box], mask[box]
        out = self.tf(image=img, mask=mask)
        mask_t = out["mask"].long()
        if self.binary:
            return out["image"], mask_t.float().unsqueeze(0)
        return out["image"], mask_t, annotated_classes(row.get("tissue_classes"))


def annotated_classes(value) -> torch.Tensor:
    """Which TISSUE_CLASSES a row's dataset labels (bool per class; background always). Empty: all of them."""
    names = [n.strip() for n in str(value or "").split(";") if n.strip() and str(value) != "nan"]
    if not names:
        return torch.ones(len(TISSUE_CLASSES), dtype=torch.bool)
    unknown = set(names) - set(TISSUE_CLASSES)
    if unknown:
        raise ValueError(f"unknown tissue classes in tissue_classes: {sorted(unknown)}")
    return torch.tensor([c == "background" or c in names for c in TISSUE_CLASSES])


def wound_box(mask: np.ndarray | None, margin: float) -> tuple[slice, slice] | None:
    """The wound's bounding box plus `margin` of its size on each side, clipped to the photo; None if no wound."""
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    py, px = int((y1 - y0 + 1) * margin), int((x1 - x0 + 1) * margin)
    H, W = mask.shape[:2]
    return slice(max(0, y0 - py), min(H, y1 + py + 1)), slice(max(0, x0 - px), min(W, x1 + px + 1))


def crop_to_wound(img: np.ndarray, mask: np.ndarray | None, margin: float = 0.35) -> np.ndarray:
    """Crop around the wound plus a margin of surrounding skin (periwound matters clinically)."""
    box = wound_box(mask, margin)
    return img if box is None else img[box]


class MetaEncoder:
    """One-hot encodes categorical metadata (e.g. body_location) with a fixed vocabulary."""

    def __init__(self, columns: list[str], vocab: dict[str, list[str]] | None = None):
        self.columns = columns
        self.vocab = vocab or {}

    def fit(self, df: pd.DataFrame) -> "MetaEncoder":
        for c in self.columns:
            self.vocab[c] = sorted(df[c].fillna("unknown").astype(str).unique().tolist())
            if "unknown" not in self.vocab[c]:
                self.vocab[c].append("unknown")
        return self

    @property
    def dim(self) -> int:
        return sum(len(v) for v in self.vocab.values())

    def encode(self, values: dict) -> np.ndarray:
        parts = []
        for c in self.columns:
            vocab = self.vocab[c]
            v = str(values.get(c, "unknown") or "unknown")
            vec = np.zeros(len(vocab), np.float32)
            vec[vocab.index(v) if v in vocab else vocab.index("unknown")] = 1.0
            parts.append(vec)
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)


class ClassificationDataset(Dataset):
    """Wound type / stage / depth classification on a wound-centred crop."""

    def __init__(self, df: pd.DataFrame, target: str, classes: list[str], size: int, train: bool,
                 meta: MetaEncoder | None = None):
        self.df = df[df[target].notna()].reset_index(drop=True)
        self.target, self.classes, self.meta = target, classes, meta
        self.tf = build_transforms(size, train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = read_rgb(row["image_path"])
        mp = str(row.get("mask_path", "") or "")
        if mp and mp != "nan":
            img = crop_to_wound(img, read_mask(mp, binary=True))
        x = self.tf(image=img)["image"]
        y = self.classes.index(str(row[self.target]))
        m = torch.from_numpy(self.meta.encode(row.to_dict())) if self.meta and self.meta.columns else torch.zeros(0)
        return x, m, y


# --------------------------------------------------------------------------- splitting

def patient_level_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15,
                        seed: int = 42) -> pd.DataFrame:
    """Assign train/val/test so that no patient appears in more than one split.

    Splitting per image leaks the same ulcer (photographed on different days) into
    both train and test and inflates every metric. Always split per patient.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["patient_id"] = df["patient_id"].fillna(df["image_path"])  # unknown -> treat each image as its own patient
    patients = rng.permutation(np.asarray(df["patient_id"].astype(str).unique(), dtype=object))
    n = len(patients)
    n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
    test = set(patients[:n_test])
    val = set(patients[n_test:n_test + n_val])
    df["split"] = [
        "test" if p in test else "val" if p in val else "train" for p in df["patient_id"].astype(str)
    ]
    return df


# --------------------------------------------------------------------------- cross-validation

def wound_size_bins(df: pd.DataFrame, mask_col: str = "mask_path") -> pd.Series:
    """Tiny / small / large wound (share of the photo), to balance segmentation folds. Rows without a mask: 'none'."""
    def size_bin(path) -> str:
        p = str(path or "")
        if not p or p == "nan":
            return "none"
        share = float(read_mask(p, binary=True).mean())
        return "empty" if share == 0 else "tiny" if share < 0.01 else "small" if share < 0.05 else "large"
    return df[mask_col].map(size_bin)


def cv_train_val(df: pd.DataFrame, fold: int, folds: int, strat: pd.Series | None = None,
                 seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and validation rows for one cross-validation fold.

    The locked test split is never touched: folds are made from the remaining (train + val) rows, grouped by
    patient so one patient's photos stay in one fold, and stratified by `strat` (wound type, or wound size) so
    every fold sees the same mix. Deterministic for a given seed, so every script gets the same folds.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    if not 0 <= fold < folds:
        raise ValueError(f"fold must be in 0..{folds - 1}")
    pool = df[df.split != "test"].reset_index(drop=True)
    y = (strat.loc[df.split != "test"].astype(str).to_numpy() if strat is not None else np.zeros(len(pool), dtype=str))
    groups = pool["patient_id"].fillna(pool["image_path"]).astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    tr_idx, va_idx = list(splitter.split(np.zeros(len(pool)), y, groups))[fold]
    return pool.iloc[tr_idx], pool.iloc[va_idx]
