"""
Run the full recoloring stack on a single HF urban patch and save every stage:
    1. SAR input
    2. SAM (ViT-H) object masks        (segmentation)
    3. category-labeled masks          (categorization)
    4. per-category pixel recoloring   (final result)
    5. optical ground truth
Writes the individual stage PNGs plus a combined panel and metrics.json to an
output folder under data/. The per-category pixel regressor is trained on the
other urban training patches; the target patch is held out.

    .venv/bin/python -m demo.run_stack 44095
"""
from __future__ import annotations
import os, sys, json, pickle
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.segmentation import build_sam_generator, segment_with_sam
from pipeline.mask_filter import categorize_all, CATEGORIES
from pipeline.recolor import PerCategoryPixelRegressor
from demo.run_hf_urban import load_pair, all_metrics, CAT_COLORS

CKPT = "models/sam_vit_h_4b8939.pth"
POINTS_PER_SIDE = 16
_GEN = [None]


def _gen():
    if _GEN[0] is None:
        print("Loading SAM ViT-H (cache miss)...")
        _GEN[0] = build_sam_generator(CKPT, model_type="vit_h", device="cpu",
                                      points_per_side=POINTS_PER_SIDE)
    return _GEN[0]


def sam_masks(idx, sar):
    cache = f"sam_masks/hf_{idx}.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    os.makedirs("sam_masks", exist_ok=True)
    m = segment_with_sam(sar, generator=_gen())
    pickle.dump(m, open(cache, "wb"))
    return m


def instance_overlay(sar, masks, seed=0, a=0.55):
    """Each SAM mask gets a distinct random colour over the SAR."""
    rng = np.random.default_rng(seed)
    base = np.stack([sar] * 3, -1).astype(np.float32)
    out = base.copy()
    for m in sorted(masks, key=lambda d: -d["area"]):
        s = m["segmentation"]; out[s] = (1 - a) * base[s] + a * rng.random(3)
    return np.clip(out, 0, 1)


def category_overlay(sar, masks, a=0.6):
    base = np.stack([sar] * 3, -1).astype(np.float32)
    out = base.copy()
    for m in sorted(masks, key=lambda d: -d["area"]):
        s = m["segmentation"]; out[s] = (1 - a) * base[s] + a * np.array(CAT_COLORS[m["category"]])
    return np.clip(out, 0, 1)


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 44095
    outdir = f"data/idx{idx}_stack"
    os.makedirs(outdir, exist_ok=True)

    ids = sorted(json.load(open("data/hf_urban/urban_ids.json")))
    k = len(ids) // 2
    train = [i for i in ids[:k] if i != idx]      # hold out the target

    # train the per-category pixel regressor on the other urban patches
    samples = []
    for i in train:
        s, r = load_pair(i)
        samples.append((s, r, categorize_all(s, sam_masks(i, s))))
    reg = PerCategoryPixelRegressor(1.0).fit(samples)

    # run the stack on the target patch
    sar, rgb = load_pair(idx)
    masks = sam_masks(idx, sar)
    cmasks = categorize_all(sar, masks)
    recolor = reg.predict(sar, cmasks)
    metrics = all_metrics(recolor, rgb)
    cats = Counter(m["category"] for m in cmasks)

    stages = [
        ("1_sar_input", np.stack([sar] * 3, -1), None),
        ("2_sam_masks", instance_overlay(sar, masks), None),
        ("3_categories", category_overlay(sar, cmasks), None),
        ("4_recolored", recolor, None),
        ("5_optical_gt", rgb, None),
    ]
    for name, img, cmap in stages:
        plt.imsave(f"{outdir}/{name}.png", np.clip(img, 0, 1), cmap=cmap)

    json.dump({"id": idx, "n_sam_masks": len(masks), "categories": dict(cats),
               "n_train_patches": len(train), "metrics_vs_gt": metrics},
              open(f"{outdir}/metrics.json", "w"), indent=2, default=float)

    titles = ["1. SAR input", f"2. SAM masks ({len(masks)})", "3. Categorized",
              f"4. Recolored\nNRMSE={metrics['NRMSE']:.3f}", "5. Optical GT"]
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.5))
    for a, (name, img, cmap), t in zip(ax, stages, titles):
        a.imshow(np.clip(img, 0, 1), cmap=cmap); a.set_title(t, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in
               ["building", "vegetation", "road", "water", "other"]]
    ax[2].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.32),
                 ncol=3, fontsize=7, frameon=False)
    fig.suptitle(f"Full stack on idx{idx}: SAM ViT-H -> categorize -> per-category pixel recolor",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(f"{outdir}/stack_panel.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"SAM masks: {len(masks)}  categories: {dict(cats)}")
    print(f"metrics vs GT: { {k: round(v, 4) for k, v in metrics.items()} }")
    print(f"Saved 5 stage PNGs + stack_panel.png + metrics.json to {outdir}/")


if __name__ == "__main__":
    main()
