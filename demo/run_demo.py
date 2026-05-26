"""
End-to-end milestone demo.
"""
from __future__ import annotations
import sys, os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.segmentation import segment_with_slic
from pipeline.mask_filter import categorize_all, CATEGORIES
from pipeline.recolor import PerCategoryRegressor
from pipeline.composite import composite
from pipeline.metrics import nrmse, sam_angle, q4, edge_fidelity
from demo.synthetic import generate_urban_patch


CAT_COLORS = {
    "water":      np.array([0.10, 0.30, 0.70]),
    "road":       np.array([0.95, 0.85, 0.10]),
    "building":   np.array([0.85, 0.20, 0.20]),
    "vegetation": np.array([0.20, 0.75, 0.30]),
    "other":      np.array([0.55, 0.55, 0.55]),
}


def baseline_recolor(sar_train, rgb_train, sar_test):
    from sklearn.linear_model import Ridge

    def feats(sar):
        mu = gaussian_filter(sar, sigma=2.0)
        sq = gaussian_filter(sar * sar, sigma=2.0)
        var = np.clip(sq - mu * mu, 0, None)
        return np.stack([sar, mu, np.sqrt(var)], axis=-1)

    Xtr = feats(sar_train).reshape(-1, 3)
    Ytr = rgb_train.reshape(-1, 3)
    idx = np.random.default_rng(0).choice(Xtr.shape[0], size=20000, replace=False)
    model = Ridge(alpha=1.0).fit(Xtr[idx], Ytr[idx])
    Xte = feats(sar_test).reshape(-1, 3)
    Yhat = model.predict(Xte).reshape(*sar_test.shape, 3)
    return np.clip(Yhat, 0, 1)


def render_category_overlay(sar, masks, alpha=0.55):
    """Color each mask by its predicted category, over the SAR image."""
    H, W = sar.shape
    overlay = np.stack([sar] * 3, axis=-1).copy()
    cat_map = np.zeros((H, W, 3), dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)
    for m in masks:
        seg = m["segmentation"]
        cat_map[seg] = CAT_COLORS[m["category"]]
        covered |= seg
    overlay[covered] = (1 - alpha) * overlay[covered] + alpha * cat_map[covered]
    return overlay


def main():
    os.makedirs("figures", exist_ok=True)

    rng = np.random.default_rng(0)
    train_seeds = list(range(1, 6))
    test_seeds  = list(range(101, 121))
    n_segments = 140

    train_masks_all = []
    train_rgb_for_color: list[tuple[dict, np.ndarray]] = []
    for s in train_seeds:
        sar_tr, rgb_tr, _ = generate_urban_patch(size=256, seed=s)
        m_tr = segment_with_slic(sar_tr, n_segments=n_segments, compactness=6.0)
        m_tr = categorize_all(sar_tr, m_tr)
        for m in m_tr:
            train_masks_all.append(m)
        train_rgb_for_color.append((m_tr, rgb_tr))
    from sklearn.linear_model import Ridge
    from pipeline.recolor import FEATURE_KEYS, _mean_rgb_inside, _PRIOR_RGB

    by_cat: dict[str, tuple[list, list]] = {c: ([], []) for c in CATEGORIES}
    for masks_s, rgb_s in train_rgb_for_color:
        for m in masks_s:
            if m.get("features") is None:
                continue
            x = np.array([m["features"][k] for k in FEATURE_KEYS], dtype=np.float32)
            y = _mean_rgb_inside(rgb_s, m["segmentation"])
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
    print("Training masks pooled by category:",
          {c: len(by_cat[c][0]) for c in CATEGORIES})

    sar_tr0, rgb_tr0, _ = generate_urban_patch(size=256, seed=train_seeds[0])

    def all_metrics(pred, target):
        return {
            "NRMSE":  nrmse(pred, target),
            "SAM":    sam_angle(pred, target),
            "Q4":     q4(pred, target),
            "EdgeF1": edge_fidelity(pred, target)["f1"],
        }

    base_metrics, our_metrics = [], []
    for s in test_seeds:
        sar_te, rgb_te, _ = generate_urban_patch(size=256, seed=s)
        m_te = segment_with_slic(sar_te, n_segments=n_segments, compactness=6.0)
        m_te = categorize_all(sar_te, m_te)
        pred_colors = reg.predict(m_te)
        pred_ours = composite(m_te, pred_colors, sar_te.shape[:2])
        pred_base = baseline_recolor(sar_tr0, rgb_tr0, sar_te)
        base_metrics.append(all_metrics(pred_base, rgb_te))
        our_metrics.append(all_metrics(pred_ours, rgb_te))

    def agg(metric_list):
        keys = metric_list[0].keys()
        return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}

    m_base = agg(base_metrics)
    m_ours = agg(our_metrics)
    print(f"\n=== Averaged over {len(test_seeds)} test scenes ===")
    print("Baseline:", m_base)
    print("Ours    :", m_ours)

    with open("figures/metrics.json", "w") as f:
        json.dump({"baseline": m_base, "ours": m_ours,
                   "n_test_scenes": len(test_seeds),
                   "n_train_scenes": len(train_seeds),
                   "train_pool": {c: len(by_cat[c][0]) for c in CATEGORIES}},
                  f, indent=2)

    sar_te, rgb_te, _ = generate_urban_patch(size=256, seed=test_seeds[0])
    masks_te = segment_with_slic(sar_te, n_segments=n_segments, compactness=6.0)
    masks_te = categorize_all(sar_te, masks_te)
    cat_counts = {c: 0 for c in CATEGORIES}
    for m in masks_te:
        cat_counts[m["category"]] += 1
    print(f"Representative-scene mask categories: {cat_counts}")
    pred_colors = reg.predict(masks_te)
    pred_ours = composite(masks_te, pred_colors, sar_te.shape[:2])
    pred_base = baseline_recolor(sar_tr0, rgb_tr0, sar_te)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    axes[0].imshow(sar_te, cmap="gray"); axes[0].set_title("(a) SAR input")
    axes[1].imshow(render_category_overlay(sar_te, masks_te))
    axes[1].set_title(f"(b) Segmentation + category\n({len(masks_te)} masks)")
    axes[2].imshow(pred_ours); axes[2].set_title("(c) Mask-conditioned recolor (ours)")
    axes[3].imshow(rgb_te); axes[3].set_title("(d) Optical ground truth")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    legend_handles = [Patch(facecolor=CAT_COLORS[c], label=c)
                      for c in ["building", "vegetation", "road", "water", "other"]]
    axes[1].legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=7,
                   frameon=False)
    fig.tight_layout()
    fig.savefig("figures/fig1_pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    zy, zx = 60, 60; zs = 96
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.6))
    titles = ["SAR (input)", "Single-model baseline\n(prior cGAN-style)",
              "Mask-conditioned (ours)"]
    rows = [(sar_te, pred_base, pred_ours),
            (sar_te[zy:zy+zs, zx:zx+zs],
             pred_base[zy:zy+zs, zx:zx+zs],
             pred_ours[zy:zy+zs, zx:zx+zs])]
    for r, imgs in enumerate(rows):
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            if r == 0:
                ax.set_title(titles[c])
            ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_ylabel("full patch", fontsize=10)
    axes[1, 0].set_ylabel("zoom\n(buildings + road)", fontsize=10)
    fig.tight_layout()
    fig.savefig("figures/fig2_compare.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    from skimage.feature import canny
    from skimage.color import rgb2gray
    e_base = canny(rgb2gray(pred_base), sigma=1.0)
    e_ours = canny(rgb2gray(pred_ours), sigma=1.0)
    e_gt   = canny(rgb2gray(rgb_te),    sigma=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
    axes[0].imshow(e_base, cmap="gray_r")
    axes[0].set_title(f"Baseline edges\nF1 = {m_base['EdgeF1']:.3f}")
    axes[1].imshow(e_ours, cmap="gray_r")
    axes[1].set_title(f"Ours edges\nF1 = {m_ours['EdgeF1']:.3f}")
    axes[2].imshow(e_gt, cmap="gray_r")
    axes[2].set_title("Ground-truth edges")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("figures/fig3_edges.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Figures written to figures/")


if __name__ == "__main__":
    main()
