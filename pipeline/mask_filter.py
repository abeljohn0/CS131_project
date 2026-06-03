"""
Stage 2a: Mask categorization.

Categories are assigned from each mask's SAR signature. The discriminative cues
are physical: water is dark + smooth (low backscatter, specular), built-up is
bright + rough (corner-reflector returns), vegetation is mid-brightness +
textured (volume scattering), roads are dark + smooth + elongated.

The thresholds operate on a *per-image canonical contrast stretch* of the SAR
(robust 2nd-98th percentile -> [0,1]) and on texture *relative to the image's
own median*, so the same rules transfer across SAR sources whose absolute
scaling differs (synthetic [0,1], dB-clipped GeoTIFF, 8-bit JPEG). Fixed
absolute thresholds previously collapsed every real mask into one category.
"""
from __future__ import annotations
import numpy as np
from skimage.measure import regionprops
from skimage.filters import sobel


CATEGORIES = ["water", "road", "building", "vegetation", "other"]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}


def canonical_sar(sar: np.ndarray) -> np.ndarray:
    """Robust per-image contrast stretch to [0,1] (2nd-98th percentile)."""
    lo, hi = np.percentile(sar, 2), np.percentile(sar, 98)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((sar - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def mask_features(sar_canon: np.ndarray, grad: np.ndarray, mask: dict) -> dict:
    """Per-mask features on the canonical SAR; `grad` is its precomputed Sobel."""
    seg = mask["segmentation"]
    pix = sar_canon[seg]
    if pix.size == 0:
        return None
    region = regionprops(seg.astype(np.uint8))[0]
    # Floor the minor axis at 1px and cap the ratio: thin slivers otherwise
    # send elongation to ~1e8, which overflows the downstream regressor.
    minor = max(region.minor_axis_length, 1.0)
    elongation = min(region.major_axis_length / minor, 50.0)
    return {
        "sar_mean": float(pix.mean()),       # canonical brightness in [0,1]
        "sar_std": float(pix.std()),
        "gradient_mean": float(grad[seg].mean()),
        "elongation": float(elongation),
        "solidity": float(region.solidity),
        "area": int(seg.sum()),
    }


def categorize_mask(feats: dict, tex_ref: float,
                    water_b: float = 0.25, road_b: float = 0.45,
                    building_b: float = 0.60, smooth_rel: float = 0.75,
                    rough_rel: float = 1.10, road_elong: float = 3.0) -> str:
    """Classify one mask from canonical brightness `b`, texture relative to the
    image median `tex_ref`, and shape (`elongation`)."""
    if feats is None:
        return "other"
    b = feats["sar_mean"]
    rel_t = feats["gradient_mean"] / (tex_ref + 1e-6)
    elong = feats["elongation"]

    if b < water_b and rel_t < smooth_rel:                 # dark + smooth
        return "water"
    if b < road_b and rel_t < (smooth_rel + 0.15) and elong >= road_elong:
        return "road"                                      # dark + smooth + long
    if b > building_b and rel_t > rough_rel:               # bright + rough
        return "building"
    if water_b <= b <= 0.72 and rel_t >= smooth_rel:       # mid + textured
        return "vegetation"
    return "other"


def categorize_all(sar: np.ndarray, masks: list[dict]) -> list[dict]:
    """Canonical-normalize the SAR once, then feature-extract and categorize
    every mask relative to this image's own texture distribution."""
    canon = canonical_sar(sar)
    grad = sobel(canon)
    feats = [mask_features(canon, grad, m) for m in masks]
    tex = np.array([f["gradient_mean"] for f in feats if f is not None])
    tex_ref = float(np.median(tex)) if tex.size else 1e-6
    out = []
    for m, f in zip(masks, feats):
        m2 = dict(m)
        m2["features"] = f
        m2["category"] = categorize_mask(f, tex_ref)
        out.append(m2)
    return out
