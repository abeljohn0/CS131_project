"""
SAM segmentation validation: run SAM (ViT-H) automatic mask generation on a few
urban patches and compare against SLIC superpixels, with the adaptive categorizer
on top. The hypothesis: SAM yields coherent *object* masks (one river, one block)
that categorize far more cleanly than SLIC's arbitrary uniform superpixels.

Caches SAM masks to sam_masks/ (slow to recompute). Run with the MPS fallback
enabled for ops SAM uses that aren't yet implemented on Apple GPUs:

    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m demo.run_sam_demo
"""
from __future__ import annotations
import os, sys, time, pickle
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.segmentation import segment_with_slic, build_sam_generator, segment_with_sam
from pipeline.mask_filter import categorize_all, CATEGORIES

CKPT = "models/sam_vit_h_4b8939.pth"
IDS = [94807, 44095]                 # confirmed-urban HF patches (start small)
POINTS_PER_SIDE = 16                 # coarser grid -> faster on CPU; bump up later
# MPS rejects SAM's float64 point grids, so ViT-H runs on CPU here.
DEVICE = "cpu"
CAT_COLORS = {
    "water": [.10, .30, .70], "road": [.95, .85, .10], "building": [.85, .20, .20],
    "vegetation": [.20, .75, .30], "other": [.55, .55, .55],
}


def load_hf(idx):
    s1 = np.asarray(Image.open(f"data/hf_urban/idx{idx}_s1.png").convert("L"), np.float32) / 255.0
    s2 = np.asarray(Image.open(f"data/hf_urban/idx{idx}_s2.png").convert("RGB"), np.float32) / 255.0
    return s1, s2


def overlay(sar, masks, a=0.6):
    H, W = sar.shape
    ov = np.stack([sar] * 3, -1).copy()
    cm = np.zeros((H, W, 3), np.float32); cov = np.zeros((H, W), bool)
    for m in sorted(masks, key=lambda d: -d["area"]):
        s = m["segmentation"]; cm[s] = CAT_COLORS[m["category"]]; cov |= s
    ov[cov] = (1 - a) * ov[cov] + a * cm[cov]
    return ov


def main():
    os.makedirs("sam_masks", exist_ok=True); os.makedirs("figures", exist_ok=True)
    print("Building SAM ViT-H generator (loads 2.4 GB checkpoint once)...")
    gen = build_sam_generator(CKPT, model_type="vit_h", device=DEVICE,
                              points_per_side=POINTS_PER_SIDE)

    rows = []
    for idx in IDS:
        sar, rgb = load_hf(idx)
        cache = f"sam_masks/hf_{idx}.pkl"
        if os.path.exists(cache):
            sam_masks = pickle.load(open(cache, "rb")); dt = 0.0
        else:
            t = time.time(); sam_masks = segment_with_sam(sar, generator=gen); dt = time.time() - t
            pickle.dump(sam_masks, open(cache, "wb"))
        slic_masks = segment_with_slic(sar, n_segments=140, compactness=6.0)
        sam_cat = categorize_all(sar, sam_masks)
        slic_cat = categorize_all(sar, slic_masks)
        print(f"idx {idx}: SAM {len(sam_masks)} masks ({dt:.1f}s) | SLIC {len(slic_masks)}")
        print("   SAM  cats:", dict(Counter(m['category'] for m in sam_cat)))
        print("   SLIC cats:", dict(Counter(m['category'] for m in slic_cat)))
        rows.append((idx, sar, rgb, slic_cat, sam_cat))

    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(12, 3.0 * n)); ax = np.atleast_2d(ax)
    titles = ["SAR", "SLIC + category", "SAM + category", "optical GT"]
    for r, (idx, sar, rgb, sc, smc) in enumerate(rows):
        ax[r, 0].imshow(sar, cmap="gray")
        ax[r, 1].imshow(overlay(sar, sc))
        ax[r, 2].imshow(overlay(sar, smc))
        ax[r, 3].imshow(rgb)
        ax[r, 0].set_ylabel(f"idx {idx}", fontsize=9)
        for c in range(4):
            if r == 0: ax[r, c].set_title(titles[c], fontsize=10)
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in
               ["building", "vegetation", "road", "water", "other"]]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("SAM object masks vs SLIC superpixels (adaptive categorizer on both)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig("figures/val_sam_vs_slic.png", dpi=150, bbox_inches="tight")
    print("saved figures/val_sam_vs_slic.png")


if __name__ == "__main__":
    main()
