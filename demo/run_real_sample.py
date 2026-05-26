"""
Run the mask-conditioned pipeline on a real paired Sentinel-1 SAR
and Sentinel-2 RGB sample, alongside the synthetic urban testbed.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.segmentation import segment_with_slic
from pipeline.mask_filter import categorize_all, CATEGORIES
from pipeline.recolor import (
    PerCategoryRegressor, FEATURE_KEYS, _mean_rgb_inside, _PRIOR_RGB,
)
from pipeline.composite import composite
from pipeline.metrics import nrmse, sam_angle, q4, edge_fidelity
from sklearn.linear_model import Ridge
from demo.synthetic import generate_urban_patch


CAT_COLORS = {
    "water":      np.array([0.10, 0.30, 0.70]),
    "road":       np.array([0.95, 0.85, 0.10]),
    "building":   np.array([0.85, 0.20, 0.20]),
    "vegetation": np.array([0.20, 0.75, 0.30]),
    "other":      np.array([0.55, 0.55, 0.55]),
}


def load_real_sample(sar_path: str, rgb_path: str
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Load the cropped Sentinel sample and convert to [0,1] floats."""
    sar = np.array(Image.open(sar_path).convert("L"), dtype=np.float32) / 255.0
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
    return sar, rgb


def fit_per_category_regressor(masks_list, rgb_list):
    """Pool training masks across multiple scenes, fit one Ridge per category."""
    by_cat = {c: ([], []) for c in CATEGORIES}
    for masks, rgb in zip(masks_list, rgb_list):
        for m in masks:
            if m.get("features") is None:
                continue
            x = np.array([m["features"][k] for k in FEATURE_KEYS],
                         dtype=np.float32)
            y = _mean_rgb_inside(rgb, m["segmentation"])
            by_cat[m["category"]][0].append(x)
            by_cat[m["category"]][1].append(y)

    reg = PerCategoryRegressor(alpha=1.0)
    for cat, (xs, ys) in by_cat.items():
        if len(xs) >= 2:
            X = np.stack(xs); Y = np.stack(ys)
            reg.models[cat] = Ridge(alpha=1.0).fit(X, Y)
            reg.fallback[cat] = Y.mean(axis=0)
        elif len(xs) == 1:
            reg.fallback[cat] = np.stack(ys).mean(axis=0)
        else:
            reg.fallback[cat] = _PRIOR_RGB[cat]
    return reg


def render_overlay(sar, masks, alpha=0.55):
    H, W = sar.shape
    overlay = np.stack([sar]*3, axis=-1).copy()
    cat_map = np.zeros((H, W, 3), dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)
    for m in masks:
        seg = m["segmentation"]
        cat_map[seg] = CAT_COLORS[m["category"]]
        covered |= seg
    overlay[covered] = (1-alpha)*overlay[covered] + alpha*cat_map[covered]
    return overlay


def main():
    os.makedirs("figures", exist_ok=True)
    n_segments = 140

    masks_list_tr, rgb_list_tr = [], []
    for s in range(1, 6):
        sar, rgb, _ = generate_urban_patch(size=256, seed=s)
        m = segment_with_slic(sar, n_segments=n_segments, compactness=6.0)
        m = categorize_all(sar, m)
        masks_list_tr.append(m); rgb_list_tr.append(rgb)
    reg = fit_per_category_regressor(masks_list_tr, rgb_list_tr)
    print("Trained per-category regressor on 5 synthetic scenes.")

    sar_real, rgb_real = load_real_sample(
        "demo/sample_s1.png", "demo/sample_s2.png"
    )
    print(f"Real Sentinel sample: SAR {sar_real.shape}, RGB {rgb_real.shape}")
    masks_real = segment_with_slic(sar_real, n_segments=n_segments, compactness=6.0)
    masks_real = categorize_all(sar_real, masks_real)
    cat_counts_real = {c: 0 for c in CATEGORIES}
    for m in masks_real:
        cat_counts_real[m["category"]] += 1
    print(f"Real-sample mask categories: {cat_counts_real}")

    pred_real_colors = reg.predict(masks_real)
    pred_real = composite(masks_real, pred_real_colors, sar_real.shape[:2])

    metrics_real = dict(
        NRMSE=nrmse(pred_real, rgb_real),
        SAM=sam_angle(pred_real, rgb_real),
        Q4=q4(pred_real, rgb_real),
        EdgeF1=edge_fidelity(pred_real, rgb_real)["f1"],
    )
    print(f"Real-data metrics: {metrics_real}")

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    axes[0].imshow(sar_real, cmap="gray")
    axes[0].set_title("(a) Sentinel-1 SAR\n(real, agricultural)")
    axes[1].imshow(render_overlay(sar_real, masks_real))
    axes[1].set_title(f"(b) Segmentation + category\n({len(masks_real)} masks)")
    axes[2].imshow(pred_real)
    axes[2].set_title("(c) Our reconstructed RGB")
    axes[3].imshow(rgb_real)
    axes[3].set_title("(d) Sentinel-2 ground truth")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    legend_handles = [Patch(facecolor=CAT_COLORS[c], label=c)
                      for c in ["building", "vegetation", "road", "water", "other"]]
    axes[1].legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=7,
                   frameon=False)
    fig.tight_layout()
    fig.savefig("figures/fig4_real_data.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/fig4_real_data.png")

    with open("figures/metrics_real.json", "w") as f:
        json.dump({"real_sample": metrics_real,
                   "real_sample_categories": cat_counts_real,
                   "note": ("Real Sentinel-1/-2 paired sample (agricultural "
                            "scene from Veeraja-Veeraesh/SAROpticalNet "
                            "preview). Not from SEN12MS-CR Bay Area urban ")}, f, indent=2)


if __name__ == "__main__":
    main()
