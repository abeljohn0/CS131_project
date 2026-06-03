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


_SAM_GENERATORS: dict = {}


def _pick_device(device):
    import torch
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_sam_generator(sam_checkpoint: str, model_type: str = "vit_h",
                        device: str = None, points_per_side: int = 32,
                        min_mask_region_area: int = 50):
    """Load a SAM model once and wrap it in an automatic mask generator.
    Cached by config so repeated calls don't reload the 2.4 GB checkpoint."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    dev = _pick_device(device)
    key = (sam_checkpoint, model_type, dev, points_per_side, min_mask_region_area)
    if key not in _SAM_GENERATORS:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(dev)
        _SAM_GENERATORS[key] = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=min_mask_region_area,
        )
    return _SAM_GENERATORS[key]


def segment_with_sam(sar: np.ndarray, generator=None, sam_checkpoint: str = None,
                     model_type: str = "vit_h", device: str = None,
                     points_per_side: int = 32) -> list[dict]:
    """Run SAM automatic mask generation on a SAR image and return masks in the
    same dict format as segment_with_slic. Pass a prebuilt `generator` to avoid
    reloading the model per image. The SAR is contrast-stretched before SAM so
    its (natural-image-trained) encoder sees usable contrast."""
    from .mask_filter import canonical_sar
    if generator is None:
        generator = build_sam_generator(sam_checkpoint, model_type, device, points_per_side)

    vis = (canonical_sar(sar) * 255.0).astype(np.uint8)
    if vis.ndim == 2:
        vis = np.stack([vis] * 3, axis=-1)

    raw = generator.generate(vis)
    masks = []
    for i, m in enumerate(sorted(raw, key=lambda d: -d["area"])):
        minc, minr, w, h = m["bbox"]
        masks.append({
            "segmentation": m["segmentation"].astype(bool),
            "area": int(m["area"]),
            "bbox": (int(minc), int(minr), int(w), int(h)),
            "label": i + 1,
        })
    return masks
