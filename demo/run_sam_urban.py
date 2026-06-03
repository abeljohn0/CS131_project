"""
Urban hypothesis test, SAM vs SLIC segmentation. Runs the full recoloring
pipeline on the 22 confirmed-urban HF patches and reports three numbers on the
same train/test split:
    - single-model pixel baseline
    - mask-conditioned with SLIC superpixels
    - mask-conditioned with SAM (ViT-H) object masks
to see whether coherent object masks let per-category recoloring win.

SAM masks are cached in sam_masks/ (slow first pass, ~16s/patch on CPU).

    .venv/bin/python -m demo.run_sam_urban
"""
from __future__ import annotations
import os, sys, json, pickle
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.segmentation import segment_with_slic, build_sam_generator, segment_with_sam
from pipeline.mask_filter import categorize_all, CATEGORIES
from pipeline.recolor import PerCategoryRegressor
from pipeline.composite import composite
from demo.run_hf_urban import (load_pair, fit_baseline, apply_baseline,
                               all_metrics, agg, CAT_COLORS, render_overlay)

CKPT = "models/sam_vit_h_4b8939.pth"
POINTS_PER_SIDE = 16
DEVICE = "cpu"


def get_sam_masks(idx, sar, gen):
    cache = f"sam_masks/hf_{idx}.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    m = segment_with_sam(sar, generator=gen)
    pickle.dump(m, open(cache, "wb"))
    return m


def build_regressor(ids, segfn):
    masks_list, rgb_list, pool = [], [], Counter()
    for i in ids:
        sar, rgb = load_pair(i)
        masks = categorize_all(sar, segfn(i, sar))
        pool.update(m["category"] for m in masks)
        masks_list.append(masks); rgb_list.append(rgb)
    return PerCategoryRegressor(1.0).fit_pooled(masks_list, rgb_list), pool


def evaluate(ids, segfn, reg):
    ms = []
    for i in ids:
        sar, rgb = load_pair(i)
        masks = categorize_all(sar, segfn(i, sar))
        pred = composite(masks, reg.predict(masks), sar.shape[:2])
        ms.append(all_metrics(pred, rgb))
    return agg(ms)


def main():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    os.makedirs("sam_masks", exist_ok=True); os.makedirs("figures", exist_ok=True)
    ids = sorted(json.load(open("data/hf_urban/urban_ids.json")))
    k = len(ids) // 2
    train, test = ids[:k], ids[k:]
    print(f"{len(ids)} urban patches -> {len(train)} train / {len(test)} test")

    gen = build_sam_generator(CKPT, model_type="vit_h", device=DEVICE,
                              points_per_side=POINTS_PER_SIDE)
    slic = lambda i, sar: segment_with_slic(sar, n_segments=140, compactness=6.0)
    sam = lambda i, sar: get_sam_masks(i, sar, gen)

    reg_slic, pool_slic = build_regressor(train, slic)
    print("SLIC train category pool:", dict(pool_slic))
    print("Running SAM on train+test patches (cached after first pass)...")
    reg_sam, pool_sam = build_regressor(train, sam)
    print("SAM  train category pool:", dict(pool_sam))
    baseline = fit_baseline(train)

    m_base = agg([all_metrics(apply_baseline(baseline, load_pair(i)[0]), load_pair(i)[1])
                  for i in test])
    m_slic = evaluate(test, slic, reg_slic)
    m_sam = evaluate(test, sam, reg_sam)

    def fmt(m): return {k: round(v, 4) for k, v in m.items()}
    print(f"\n=== Mean over {len(test)} urban test patches ===")
    print("baseline (single model) :", fmt(m_base))
    print("ours + SLIC superpixels :", fmt(m_slic))
    print("ours + SAM object masks :", fmt(m_sam))

    json.dump({"n_train": len(train), "n_test": len(test),
               "slic_pool": dict(pool_slic), "sam_pool": dict(pool_sam),
               "baseline": m_base, "ours_slic": m_slic, "ours_sam": m_sam},
              open("figures/metrics_sam_urban.json", "w"), indent=2)

    # recolor comparison figure on one representative test patch
    idx = test[0]
    sar, rgb = load_pair(idx)
    ms_slic = categorize_all(sar, slic(idx, sar))
    ms_sam = categorize_all(sar, sam(idx, sar))
    rec_base = apply_baseline(baseline, sar)
    rec_slic = composite(ms_slic, reg_slic.predict(ms_slic), sar.shape[:2])
    rec_sam = composite(ms_sam, reg_sam.predict(ms_sam), sar.shape[:2])
    panels = [(sar, "SAR", "gray"), (render_overlay(sar, ms_sam), "SAM + category", None),
              (rec_base, "baseline", None), (rec_slic, "ours (SLIC)", None),
              (rec_sam, "ours (SAM)", None), (rgb, "optical GT", None)]
    fig, ax = plt.subplots(1, 6, figsize=(18, 3.2))
    for a, (im, t, cm) in zip(ax, panels):
        a.imshow(im, cmap=cm); a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in
               ["building", "vegetation", "road", "water", "other"]]
    ax[1].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30),
                 ncol=3, fontsize=7, frameon=False)
    fig.suptitle(f"SAM vs SLIC recoloring on urban patch (HF row {idx})", fontsize=10)
    fig.tight_layout(); fig.savefig("figures/fig9_sam_urban.png", dpi=150, bbox_inches="tight")
    print("Saved figures/fig9_sam_urban.png, figures/metrics_sam_urban.json")


if __name__ == "__main__":
    main()
