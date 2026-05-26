"""
Stage 2a: Mask categorization.
"""
from __future__ import annotations
import numpy as np
from skimage.measure import regionprops
from skimage.filters import sobel


CATEGORIES = ["water", "road", "building", "vegetation", "other"]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}


def mask_features(sar: np.ndarray, mask: dict) -> dict:
    seg = mask["segmentation"]
    pix = sar[seg]
    if pix.size == 0:
        return None
    grad = sobel(sar)
    grad_pix = grad[seg]

    region = regionprops((seg.astype(np.uint8)))[0]
    minor = max(region.minor_axis_length, 1e-6)
    elongation = region.major_axis_length / minor
    solidity = region.solidity

    return {
        "sar_mean": float(pix.mean()),
        "sar_std": float(pix.std()),
        "gradient_mean": float(grad_pix.mean()),
        "elongation": float(elongation),
        "solidity": float(solidity),
        "area": int(seg.sum()),
    }


def categorize_mask(feats: dict,
                    water_thresh: float = 0.15,
                    road_low: float = 0.12,
                    road_high: float = 0.28,
                    building_thresh: float = 0.55,
                    veg_low: float = 0.32,
                    veg_high: float = 0.55,
                    veg_grad: float = 0.05) -> str:
    if feats is None:
        return "other"
    sar_mu = feats["sar_mean"]
    grad = feats["gradient_mean"]

    if sar_mu < water_thresh and grad < 0.02:
        return "water"
    if road_low <= sar_mu < road_high and grad < 0.05:
        return "road"
    if sar_mu > building_thresh:
        return "building"
    if veg_low <= sar_mu <= veg_high and grad > veg_grad:
        return "vegetation"
    return "other"


def categorize_all(sar: np.ndarray, masks: list[dict]) -> list[dict]:
    out = []
    for m in masks:
        f = mask_features(sar, m)
        cat = categorize_mask(f)
        m2 = dict(m)
        m2["features"] = f
        m2["category"] = cat
        out.append(m2)
    return out
