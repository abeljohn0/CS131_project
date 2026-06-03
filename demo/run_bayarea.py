"""
Step 1 (preprocessing) + Step 2 (run): mask-conditioned recoloring on the
real SEN12MS-CR Bay Area patches (ROIs1868_summer) downloaded under
data/sen12mscr/.

Preprocessing lives in pipeline.dataset:
  load_s1_vv  : Sentinel-1 (2-band)  -> single-channel SAR in [0,1]
  load_s2_rgb : Sentinel-2 (13-band) -> RGB ground truth in [0,1]

Unlike demo/run_real_sample.py (which trained on synthetic scenes and tested
on one real agricultural patch), this trains the per-category regressors on a
held-out split of the REAL patches and evaluates on the rest, alongside a
single-model pixel baseline on the same data.

    python -m demo.run_bayarea
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.dataset import discover_sen12mscr_pairs, load_s1_vv, load_s2_rgb
from pipeline.segmentation import segment_with_slic
from pipeline.mask_filter import categorize_all, CATEGORIES
from pipeline.recolor import PerCategoryRegressor
from pipeline.composite import composite
from pipeline.metrics import nrmse, sam_angle, q4, edge_fidelity


DATA_ROOT  = "data/sen12mscr"
N_SEGMENTS = 140
N_TRAIN    = 20            # first N pairs train; the rest are the test set

CAT_COLORS = {
    "water":      np.array([0.10, 0.30, 0.70]),
    "road":       np.array([0.95, 0.85, 0.10]),
    "building":   np.array([0.85, 0.20, 0.20]),
    "vegetation": np.array([0.20, 0.75, 0.30]),
    "other":      np.array([0.55, 0.55, 0.55]),
}


def segment_and_categorize(sar):
    masks = segment_with_slic(sar, n_segments=N_SEGMENTS, compactness=6.0)
    return categorize_all(sar, masks)


def render_overlay(sar, masks, alpha=0.55):
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


def _pixel_feats(sar):
    mu = gaussian_filter(sar, sigma=2.0)
    sq = gaussian_filter(sar * sar, sigma=2.0)
    var = np.clip(sq - mu * mu, 0, None)
    return np.stack([sar, mu, np.sqrt(var)], axis=-1)


def fit_baseline(train_pairs, max_pixels=60000):
    """Single global Ridge: per-pixel SAR features -> RGB (cGAN-style baseline)."""
    Xs, Ys = [], []
    for sp, gp in train_pairs:
        sar = load_s1_vv(sp); rgb = load_s2_rgb(gp)
        Xs.append(_pixel_feats(sar).reshape(-1, 3))
        Ys.append(rgb.reshape(-1, 3))
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    idx = np.random.default_rng(0).choice(
        X.shape[0], size=min(max_pixels, X.shape[0]), replace=False)
    # Standardize: sar and its smoothed mean are near-collinear, so the raw
    # normal equations are near-singular and alpha=1 barely regularizes them.
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[idx], Y[idx])


def apply_baseline(model, sar):
    pred = model.predict(_pixel_feats(sar).reshape(-1, 3))
    return np.clip(pred.reshape(*sar.shape, 3), 0, 1)


def all_metrics(pred, target):
    return {
        "NRMSE":  nrmse(pred, target),
        "SAM":    sam_angle(pred, target),
        "Q4":     q4(pred, target),
        "EdgeF1": edge_fidelity(pred, target)["f1"],
    }


def agg(metric_list):
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}


def main():
    # Apple's Accelerate BLAS (numpy>=2 on macOS arm64) raises spurious
    # divide/overflow/invalid flags *inside* well-conditioned matmuls (verified:
    # a plain X.T @ y on random finite data triggers it). All coefficients and
    # predictions here are finite, so silence these platform false positives.
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    os.makedirs("figures", exist_ok=True)
    pairs = discover_sen12mscr_pairs(DATA_ROOT)
    if not pairs:
        raise SystemExit(f"No SEN12MS-CR pairs found under {DATA_ROOT}/")
    train_pairs = pairs[:N_TRAIN]
    test_pairs  = pairs[N_TRAIN:]
    print(f"Discovered {len(pairs)} matched pairs -> "
          f"{len(train_pairs)} train / {len(test_pairs)} test")

    # ---- Step 2a: train per-category regressors on real patches ----
    train_masks, train_rgb = [], []
    pool = {c: 0 for c in CATEGORIES}
    for sp, gp in train_pairs:
        sar = load_s1_vv(sp); rgb = load_s2_rgb(gp)
        masks = segment_and_categorize(sar)
        for m in masks:
            pool[m["category"]] += 1
        train_masks.append(masks); train_rgb.append(rgb)
    reg = PerCategoryRegressor(alpha=1.0).fit_pooled(train_masks, train_rgb)
    print("Per-category training masks:", pool)

    baseline = fit_baseline(train_pairs)
    print("Fitted single-model pixel baseline.")

    # ---- Step 2b: evaluate on the held-out real test patches ----
    ours_m, base_m = [], []
    best = None  # most "urban" test patch for the headline figure
    for sp, gp in test_pairs:
        sar = load_s1_vv(sp); rgb = load_s2_rgb(gp)
        masks = segment_and_categorize(sar)
        pred_ours = composite(masks, reg.predict(masks), sar.shape[:2])
        pred_base = apply_baseline(baseline, sar)
        mo = all_metrics(pred_ours, rgb)
        mb = all_metrics(pred_base, rgb)
        ours_m.append(mo); base_m.append(mb)
        urban = sum(1 for m in masks if m["category"] in ("building", "road"))
        if best is None or urban > best["urban"]:
            best = dict(sp=sp, sar=sar, rgb=rgb, masks=masks,
                        pred_ours=pred_ours, pred_base=pred_base,
                        urban=urban, mo=mo)

    m_ours = agg(ours_m); m_base = agg(base_m)
    print(f"\n=== Mean over {len(test_pairs)} real test patches ===")
    print("Single-model baseline   :", {k: round(v, 4) for k, v in m_base.items()})
    print("Mask-conditioned (ours) :", {k: round(v, 4) for k, v in m_ours.items()})

    with open("figures/metrics_bayarea.json", "w") as f:
        json.dump({
            "dataset": "SEN12MS-CR ROIs1868_summer (scene 27), raw optical RGB GT",
            "n_train": len(train_pairs), "n_test": len(test_pairs),
            "train_pool": pool,
            "baseline": m_base, "ours": m_ours,
        }, f, indent=2)

    # ---- headline figure: most-urban test patch ----
    b = best
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    axes[0].imshow(b["sar"], cmap="gray")
    axes[0].set_title("(a) Sentinel-1 SAR\n(VV, normalized)")
    axes[1].imshow(render_overlay(b["sar"], b["masks"]))
    axes[1].set_title(f"(b) Segmentation + category\n({len(b['masks'])} masks)")
    axes[2].imshow(b["pred_base"])
    axes[2].set_title("(c) Single-model baseline")
    axes[3].imshow(b["pred_ours"])
    axes[3].set_title(f"(d) Mask-conditioned (ours)\nNRMSE={b['mo']['NRMSE']:.3f}")
    axes[4].imshow(b["rgb"])
    axes[4].set_title("(e) Sentinel-2 GT")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(facecolor=CAT_COLORS[c], label=c)
               for c in ["building", "vegetation", "road", "water", "other"]]
    axes[1].legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.30), ncol=3, fontsize=7, frameon=False)
    fig.suptitle(f"Real Bay Area patch {os.path.basename(b['sp'])}", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/fig5_bayarea.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- montage: SAR / ours / GT over the first few test patches ----
    n = min(6, len(test_pairs))
    fig, axes = plt.subplots(3, n, figsize=(2.1 * n, 6.4))
    for j, (sp, gp) in enumerate(test_pairs[:n]):
        sar = load_s1_vv(sp); rgb = load_s2_rgb(gp)
        masks = segment_and_categorize(sar)
        ours = composite(masks, reg.predict(masks), sar.shape[:2])
        axes[0, j].imshow(sar, cmap="gray")
        axes[1, j].imshow(ours)
        axes[2, j].imshow(rgb)
        axes[0, j].set_title(
            os.path.basename(sp).split("_")[-1].replace(".tif", ""), fontsize=7)
        for r in range(3):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("SAR", fontsize=9)
    axes[1, 0].set_ylabel("ours", fontsize=9)
    axes[2, 0].set_ylabel("GT", fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/fig6_bayarea_montage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved figures/fig5_bayarea.png, figures/fig6_bayarea_montage.png, "
          "figures/metrics_bayarea.json")


if __name__ == "__main__":
    main()
