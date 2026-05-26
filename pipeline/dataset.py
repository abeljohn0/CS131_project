"""
SEN12MS-CR SAR/optical dataset loader (real-data interface).
"""
from __future__ import annotations
import os
import glob
import numpy as np
import tifffile


SAR_MAX = 2 ** 12


def load_sar(path: str) -> np.ndarray:
    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] <= 4:
            arr = arr[0]
        else:
            arr = arr[..., 0]
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max() * SAR_MAX
    return arr


def load_gt(path: str) -> np.ndarray:

    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def sar_to_unit(sar: np.ndarray) -> np.ndarray:
    return np.clip(sar / SAR_MAX, 0, 1).astype(np.float32)


def rgb_to_unit(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb / SAR_MAX, 0, 1).astype(np.float32)


def unit_to_rgb12(rgb_unit: np.ndarray) -> np.ndarray:
    return np.clip(rgb_unit * SAR_MAX, 0, SAR_MAX - 1).astype(np.uint16)


def discover_pairs(root: str, split: str = "test") -> list[tuple[str, str]]:
    candidates = [
        (f"sar_{split}", f"gt_{split}"),
        ("s1_123", "gt_123"),
    ]
    for sar_dir, gt_dir in candidates:
        sar_path = os.path.join(root, sar_dir)
        gt_path  = os.path.join(root, gt_dir)
        if not (os.path.isdir(sar_path) and os.path.isdir(gt_path)):
            continue
        sar_files = sorted(glob.glob(os.path.join(sar_path, "*.tif")))
        gt_files  = sorted(glob.glob(os.path.join(gt_path,  "*.tif")))
        gt_by_patch = {_patch_id(p): p for p in gt_files}
        pairs = []
        for s in sar_files:
            pid = _patch_id(s)
            if pid in gt_by_patch:
                pairs.append((s, gt_by_patch[pid]))
        if pairs:
            return pairs
    return []


def _patch_id(path: str) -> str:
    base = os.path.basename(path).replace(".tif", "")
    for tok in reversed(base.split("_")):
        if tok.startswith("p") and tok[1:].isdigit():
            return tok
    return base


def output_filename_from_sar(sar_path: str, method_name: str) -> str:
    base = os.path.basename(sar_path).replace(".tif", "")
    parts = base.split("_")
    if len(parts) >= 5 and parts[2] == "s1":
        return "_".join([parts[0], parts[1], "sarcolor", parts[3], parts[4]]) + ".tif"
    return base.replace("_s1_", "_sarcolor_") + ".tif"



PRIOR_PATCH_CATEGORIES = ("Mountain", "Urban", "Water",
                          "Water-Urban", "Water-Mountain")


def load_prior_categories(csv_path: str) -> dict[int, str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    return dict(zip(df["label"].astype(int), df["Category"].astype(str)))
