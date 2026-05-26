"""
Stage 2c: Compositing.

"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .mask_filter import CATEGORIES, CATEGORY_TO_IDX


def composite(masks: list[dict],
              mask_colors: dict[int, np.ndarray],
              shape: tuple[int, int],
              category_smooth_sigma: float = 3.0) -> np.ndarray:
    H, W = shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)
    cat_map = np.full((H, W), -1, dtype=np.int8)

    order = sorted(range(len(masks)), key=lambda i: -masks[i]["area"])
    for i in order:
        seg = masks[i]["segmentation"]
        out[seg] = mask_colors[i]
        covered |= seg
        cat_map[seg] = CATEGORY_TO_IDX[masks[i]["category"]]

    if not covered.all():
        _, (ii, jj) = distance_transform_edt(~covered, return_indices=True)
        out[~covered] = out[ii[~covered], jj[~covered]]
        cat_map[~covered] = cat_map[ii[~covered], jj[~covered]]

    if category_smooth_sigma > 0:
        smoothed = np.zeros_like(out)
        for cat in CATEGORIES:
            cidx = CATEGORY_TO_IDX[cat]
            m = (cat_map == cidx).astype(np.float32)
            if m.sum() < 1:
                continue
            for c in range(3):
                num = gaussian_filter(out[..., c] * m, sigma=category_smooth_sigma)
                den = gaussian_filter(m, sigma=category_smooth_sigma) + 1e-6
                smoothed[..., c] += (num / den) * m
        out = smoothed

    return np.clip(out, 0, 1)
