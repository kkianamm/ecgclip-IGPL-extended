"""
Data plumbing for IPL on your rendered PTB-XL images.

Reads `work/labels.csv` + `work/images/<ecg_id>.png` produced by your
`prepare_data.py`. Robust to two common layouts:

  (a) one 0/1 column per superclass (NORM, MI, STTC, CD, HYP)
  (b) a single list-like column ("labels"/"superclass"/"diagnostic_superclass")

Split resolution order: explicit `split` column -> `strat_fold`
(1-8 train, 9 val, 10 test, the official PTB-XL split).
"""

from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resolve_targets(df: pd.DataFrame, classnames):
    if all(c in df.columns for c in classnames):
        return df[classnames].astype(float).values

    list_col = next((c for c in ["labels", "superclass", "superclasses",
                                 "diagnostic_superclass", "scp_superclass"]
                     if c in df.columns), None)
    if list_col is None:
        raise ValueError(
            f"labels.csv has neither per-class columns {classnames} nor a "
            f"list column. Columns present: {list(df.columns)}"
        )
    idx = {c: i for i, c in enumerate(classnames)}
    Y = np.zeros((len(df), len(classnames)), dtype=np.float32)
    for r, val in enumerate(df[list_col].tolist()):
        items = val if isinstance(val, (list, tuple)) else _parse_list(val)
        for it in items:
            if it in idx:
                Y[r, idx[it]] = 1.0
    return Y


def _parse_list(val):
    if not isinstance(val, str):
        return []
    try:
        out = ast.literal_eval(val)
        return list(out) if isinstance(out, (list, tuple)) else [str(out)]
    except (ValueError, SyntaxError):
        return [t.strip(" '\"[]") for t in val.split(",") if t.strip(" '\"[]")]


def _resolve_split(df: pd.DataFrame):
    if "split" in df.columns:
        s = df["split"].astype(str).str.lower()
        return s.map(lambda x: {"training": "train", "validation": "val",
                                "testing": "test"}.get(x, x))
    fold_col = next((c for c in ["strat_fold", "fold"] if c in df.columns), None)
    if fold_col is None:
        raise ValueError("labels.csv needs a `split` or `strat_fold` column.")
    f = df[fold_col].astype(int)
    return np.where(f <= 8, "train", np.where(f == 9, "val", "test"))


class ECGImageDataset(Dataset):
    def __init__(self, df, image_dir, classnames, preprocess, id_col="ecg_id"):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.ids = self.df[id_col].tolist()
        self.targets = torch.tensor(_resolve_targets(self.df, classnames), dtype=torch.float32)

    def __len__(self):
        return len(self.ids)

    def _path(self, ecg_id):
        for name in (f"{ecg_id}.png", f"{int(ecg_id):05d}.png", f"{ecg_id}.jpg"):
            p = os.path.join(self.image_dir, name)
            if os.path.exists(p):
                return p
        return os.path.join(self.image_dir, f"{ecg_id}.png")

    def __getitem__(self, i):
        img = Image.open(self._path(self.ids[i])).convert("RGB")
        return self.preprocess(img), self.targets[i], i


def build_splits(labels_csv, image_dir, classnames, preprocess, id_col="ecg_id"):
    df = pd.read_csv(labels_csv)
    df["_split"] = _resolve_split(df)
    out = {}
    for name in ["train", "val", "test"]:
        sub = df[df["_split"] == name]
        out[name] = ECGImageDataset(sub, image_dir, classnames, preprocess, id_col)
    return out


def few_shot_indices(dataset: ECGImageDataset, shots: int, seed: int = 1):
    """
    K-shot multi-label sampling: for each class collect `shots` positive
    examples. Because labels are multi-label, one record can satisfy several
    classes; we de-duplicate the final index set.
    """
    rng = np.random.default_rng(seed)
    Y = dataset.targets.numpy()
    chosen = set()
    for c in range(Y.shape[1]):
        pos = np.where(Y[:, c] == 1)[0]
        rng.shuffle(pos)
        chosen.update(pos[:shots].tolist())
    return sorted(chosen)


# --------------------------------------------------------------------------- #
# Frozen-feature cache. BiomedCLIP is frozen, so encode every image ONCE and
# train the prompt learner on cached features. This makes full-data training
# fast and mirrors your extract_features.py.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cache_features(model, dataset, batch_size=128, num_workers=8, indices=None):
    from torch.utils.data import DataLoader, Subset

    ds = dataset if indices is None else Subset(dataset, indices)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    feats, tgts = [], []
    for pixel_values, targets, _ in loader:
        feats.append(model.encode_image(pixel_values).cpu())
        tgts.append(targets)
    return torch.cat(feats), torch.cat(tgts)
