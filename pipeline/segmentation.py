"""
Stage 1: Segmentation.
"""
from __future__ import annotations
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops, label
from skimage.color import gray2rgb


def _props_to_masks(label_img: np.ndarray, min_area: int = 25) -> list[dict]:
    masks = []
    for region in regionprops(label_img):
        if region.area < min_area:
            continue
        seg = (label_img == region.label)
        minr, minc, maxr, maxc = region.bbox
        masks.append({
            "segmentation": seg,
            "area": int(region.area),
            "bbox": (int(minc), int(minr), int(maxc - minc), int(maxr - minr)),
            "label": int(region.label),
        })
    return masks


def segment_with_slic(
    sar: np.ndarray,
    n_segments: int = 250,
    compactness: float = 6.0,
    min_area: int = 25,
) -> list[dict]:
# in case gpu not availavle
    if sar.ndim == 2:
        sar_rgb = gray2rgb(sar)
    else:
        sar_rgb = sar
    seg = slic(
        sar_rgb,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        channel_axis=-1,
    )
    return _props_to_masks(seg, min_area=min_area)


def segment_with_sam(sar: np.ndarray, sam_checkpoint: str, model_type: str = "vit_b",
                     device: str = "cuda", points_per_side: int = 32) -> list[dict]:
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sar_uint8 = np.clip(sar * 255.0, 0, 255).astype(np.uint8)
    if sar_uint8.ndim == 2:
        sar_uint8 = np.stack([sar_uint8] * 3, axis=-1)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=25,
    )
    return generator.generate(sar_uint8)
