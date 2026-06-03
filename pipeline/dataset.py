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


# ---------------------------------------------------------------------------
# Raw SEN12MS-CR loaders (native 2-band Sentinel-1 / 13-band Sentinel-2).
#
# Step-1 preprocessing for the Bay Area patches pulled from the official
# SEN12MS-CR mirror into data/sen12mscr/ (ROIs*/s1_*/*.tif, s2_*/*.tif):
#   * Sentinel-1  -> single-channel SAR in [0,1]  (VV band, dB-clipped)
#   * Sentinel-2  -> (H,W,3) RGB ground truth in [0,1] (true-color stretch)
# We target raw optical RGB (not the prior project's IHS fusion): it is the
# standard SAR->optical colorization target and does not leak SAR structure
# into the ground truth. To reproduce the old IHS-based numbers instead,
# swap load_s2_rgb for an IHS variant.
# ---------------------------------------------------------------------------

S1_DB_LO = -25.0          # Sentinel-1 VV dB range mapped onto [0,1]
S1_DB_HI = 0.0
S2_RGB_MAX = 4000.0       # ~2.5x-reflectance true-color stretch (DN/10000*2.5)
S2_RGB_BANDS = (3, 2, 1)  # B4=red, B3=green, B2=blue within the 13-band S2 stack


def _bands_first(arr: np.ndarray) -> np.ndarray:
    """Return a multiband array as (bands, H, W) regardless of stored order."""
    if arr.ndim == 2:
        return arr[None]
    if arr.shape[0] <= 13 and arr.shape[0] < arr.shape[-1]:
        return arr                      # already (bands, H, W)
    return np.moveaxis(arr, -1, 0)      # (H, W, bands) -> (bands, H, W)


def load_s1_vv(path: str) -> np.ndarray:
    """Sentinel-1 patch -> single-channel SAR in [0,1] (VV, dB-clipped)."""
    bands = _bands_first(tifffile.imread(path).astype(np.float32))
    vv = np.clip(bands[0], S1_DB_LO, S1_DB_HI)
    return ((vv - S1_DB_LO) / (S1_DB_HI - S1_DB_LO)).astype(np.float32)


def load_s2_rgb(path: str) -> np.ndarray:
    """Sentinel-2 patch -> (H,W,3) RGB ground truth in [0,1]."""
    bands = _bands_first(tifffile.imread(path).astype(np.float32))
    rgb = np.stack([bands[i] for i in S2_RGB_BANDS], axis=-1)
    return np.clip(rgb / S2_RGB_MAX, 0, 1).astype(np.float32)


def _scene_patch_key(path: str) -> str:
    """ROIs1868_summer_s1_27_p100.tif -> '27_p100' (matches s1<->s2)."""
    parts = os.path.basename(path).replace(".tif", "").split("_")
    return "_".join(parts[-2:])


def discover_sen12mscr_pairs(root: str) -> list[tuple[str, str]]:
    """Match raw SEN12MS-CR Sentinel-1<->Sentinel-2 patches under `root`."""
    s1 = sorted(glob.glob(os.path.join(root, "**", "*_s1_*.tif"), recursive=True))
    s2 = sorted(glob.glob(os.path.join(root, "**", "*_s2_*.tif"), recursive=True))
    s2_by_key = {_scene_patch_key(p): p for p in s2}
    return [(sp, s2_by_key[k]) for sp in s1
            if (k := _scene_patch_key(sp)) in s2_by_key]
