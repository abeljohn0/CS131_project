"""
Per-category PIXEL-level recoloring vs flat per-mask, on the 22 urban patches.
Four numbers on the same train/test split:
    baseline               : single global per-pixel regressor
    flat per-mask (SAM)    : one mean colour per SAM object (previous approach)
    pixel per-category(SLIC): per-category per-pixel regressors, SLIC masks
    pixel per-category(SAM) : per-category per-pixel regressors, SAM masks  <- test

    .venv/bin/python -m demo.run_sam_pixel_urban
"""
from __future__ import annotations
import os, sys, json, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.segmentation import segment_with_slic, build_sam_generator, segment_with_sam
from pipeline.mask_filter import categorize_all
from pipeline.recolor import PerCategoryRegressor, PerCategoryPixelRegressor
from pipeline.composite import composite
from demo.run_hf_urban import (load_pair, fit_baseline, apply_baseline,
                               all_metrics, agg, CAT_COLORS, render_overlay)

CKPT = "models/sam_vit_h_4b8939.pth"
POINTS_PER_SIDE = 16
DEVICE = "cpu"


def get_sam(idx, sar, gen):
    cache = f"sam_masks/hf_{idx}.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    m = segment_with_sam(sar, generator=gen)
    pickle.dump(m, open(cache, "wb"))
    return m


def main():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    os.makedirs("figures", exist_ok=True); os.makedirs("sam_masks", exist_ok=True)
    ids = sorted(json.load(open("data/hf_urban/urban_ids.json")))
    k = len(ids) // 2
    train, test = ids[:k], ids[k:]
    print(f"{len(ids)} urban patches -> {len(train)} train / {len(test)} test")

    gen = build_sam_generator(CKPT, model_type="vit_h", device=DEVICE,
                              points_per_side=POINTS_PER_SIDE)
    data = {i: load_pair(i) for i in ids}
    print("Segmenting (SLIC + cached SAM)...")
    slic_m = {i: categorize_all(data[i][0],
              segment_with_slic(data[i][0], n_segments=140, compactness=6.0)) for i in ids}
    sam_m = {i: categorize_all(data[i][0], get_sam(i, data[i][0], gen)) for i in ids}

    baseline = fit_baseline(train)
    flat_sam = PerCategoryRegressor(1.0).fit_pooled(
        [sam_m[i] for i in train], [data[i][1] for i in train])
    pix_slic = PerCategoryPixelRegressor(1.0).fit(
        [(data[i][0], data[i][1], slic_m[i]) for i in train])
    pix_sam = PerCategoryPixelRegressor(1.0).fit(
        [(data[i][0], data[i][1], sam_m[i]) for i in train])

    def ev_base():
        return agg([all_metrics(apply_baseline(baseline, data[i][0]), data[i][1]) for i in test])

    def ev_flat(reg, masks):
        return agg([all_metrics(composite(masks[i], reg.predict(masks[i]),
                    data[i][0].shape[:2]), data[i][1]) for i in test])

    def ev_pix(reg, masks):
        return agg([all_metrics(reg.predict(data[i][0], masks[i]), data[i][1]) for i in test])

    res = {
        "baseline (single global pixel)": ev_base(),
        "flat per-mask (SAM)": ev_flat(flat_sam, sam_m),
        "pixel per-category (SLIC)": ev_pix(pix_slic, slic_m),
        "pixel per-category (SAM)": ev_pix(pix_sam, sam_m),
    }
    print(f"\n=== Mean over {len(test)} urban test patches ===")
    for name, v in res.items():
        print(f"  {name:34s} NRMSE={v['NRMSE']:.4f}  Q4={v['Q4']:.4f}  "
              f"SAM={v['SAM']:.4f}  EdgeF1={v['EdgeF1']:.4f}")
    json.dump(res, open("figures/metrics_sam_pixel_urban.json", "w"), indent=2, default=float)

    idx = test[0]; sar, rgb = data[idx]
    panels = [
        (sar, "SAR", "gray"),
        (render_overlay(sar, sam_m[idx]), "SAM + category", None),
        (apply_baseline(baseline, sar), "baseline", None),
        (composite(sam_m[idx], flat_sam.predict(sam_m[idx]), sar.shape[:2]), "flat per-mask (SAM)", None),
        (pix_sam.predict(sar, sam_m[idx]), "pixel per-cat (SAM)", None),
        (rgb, "optical GT", None),
    ]
    fig, ax = plt.subplots(1, 6, figsize=(18, 3.2))
    for a, (im, t, cm) in zip(ax, panels):
        a.imshow(im, cmap=cm); a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in
               ["building", "vegetation", "road", "water", "other"]]
    ax[1].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30),
                 ncol=3, fontsize=7, frameon=False)
    fig.suptitle(f"Per-category pixel recoloring vs flat per-mask (HF row {idx})", fontsize=10)
    fig.tight_layout(); fig.savefig("figures/fig10_sam_pixel_urban.png", dpi=150, bbox_inches="tight")
    print("saved figures/fig10_sam_pixel_urban.png, figures/metrics_sam_pixel_urban.json")


if __name__ == "__main__":
    main()
