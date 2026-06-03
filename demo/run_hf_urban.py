"""
Urban hypothesis test: run the mask-conditioned pipeline on confirmed CITY
patches scouted from the HuggingFace SEN12MS-CR mirror (data/hf_urban/).

These 22 patches were surfaced by a SAR+optical texture ranker and then
hand-confirmed as urban. They are lossy 8-bit JPEG renders (s1 = grayscale
SAR, s2 = RGB optical), so this is a QUALITATIVE test of whether per-category
recoloring beats a single model on cities (both methods are scored against the
same s2 target) -- not a faithful quantitative head-to-head vs the prior
0.239/0.350 numbers, which needs the original GeoTIFFs.

    python -m demo.run_hf_urban
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.segmentation import segment_with_slic
from pipeline.mask_filter import categorize_all, CATEGORIES
from pipeline.recolor import PerCategoryRegressor
from pipeline.composite import composite
from pipeline.metrics import nrmse, sam_angle, q4, edge_fidelity

DATA = "data/hf_urban"
N_SEG = 140
CAT_COLORS = {
    "water":      np.array([0.10, 0.30, 0.70]),
    "road":       np.array([0.95, 0.85, 0.10]),
    "building":   np.array([0.85, 0.20, 0.20]),
    "vegetation": np.array([0.20, 0.75, 0.30]),
    "other":      np.array([0.55, 0.55, 0.55]),
}


def load_pair(idx):
    s1 = np.asarray(Image.open(f"{DATA}/idx{idx}_s1.png").convert("L"), np.float32) / 255.0
    s2 = np.asarray(Image.open(f"{DATA}/idx{idx}_s2.png").convert("RGB"), np.float32) / 255.0
    return s1, s2


def seg_cat(sar):
    return categorize_all(sar, segment_with_slic(sar, n_segments=N_SEG, compactness=6.0))


def render_overlay(sar, masks, alpha=0.55):
    H, W = sar.shape
    overlay = np.stack([sar] * 3, axis=-1).copy()
    cat_map = np.zeros((H, W, 3), dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)
    for m in masks:
        seg = m["segmentation"]
        cat_map[seg] = CAT_COLORS[m["category"]]; covered |= seg
    overlay[covered] = (1 - alpha) * overlay[covered] + alpha * cat_map[covered]
    return overlay


def _pixel_feats(sar):
    mu = gaussian_filter(sar, 2.0); sq = gaussian_filter(sar * sar, 2.0)
    return np.stack([sar, mu, np.sqrt(np.clip(sq - mu * mu, 0, None))], axis=-1)


def fit_baseline(train_ids, max_pixels=60000):
    Xs, Ys = [], []
    for i in train_ids:
        sar, rgb = load_pair(i)
        Xs.append(_pixel_feats(sar).reshape(-1, 3)); Ys.append(rgb.reshape(-1, 3))
    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    idx = np.random.default_rng(0).choice(X.shape[0], size=min(max_pixels, X.shape[0]), replace=False)
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[idx], Y[idx])


def apply_baseline(model, sar):
    return np.clip(model.predict(_pixel_feats(sar).reshape(-1, 3)).reshape(*sar.shape, 3), 0, 1)


def all_metrics(pred, target):
    return {"NRMSE": nrmse(pred, target), "SAM": sam_angle(pred, target),
            "Q4": q4(pred, target), "EdgeF1": edge_fidelity(pred, target)["f1"]}


def agg(ms):
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def main():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    os.makedirs("figures", exist_ok=True)
    ids = sorted(json.load(open(f"{DATA}/urban_ids.json")))
    k = len(ids) // 2
    train_ids, test_ids = ids[:k], ids[k:]
    print(f"{len(ids)} urban patches -> {len(train_ids)} train / {len(test_ids)} test")

    train_masks, train_rgb, pool = [], [], {c: 0 for c in CATEGORIES}
    for i in train_ids:
        sar, rgb = load_pair(i); masks = seg_cat(sar)
        for m in masks:
            pool[m["category"]] += 1
        train_masks.append(masks); train_rgb.append(rgb)
    reg = PerCategoryRegressor(alpha=1.0).fit_pooled(train_masks, train_rgb)
    baseline = fit_baseline(train_ids)
    print("Per-category training masks:", pool)

    ours_m, base_m, best = [], [], None
    for i in test_ids:
        sar, rgb = load_pair(i); masks = seg_cat(sar)
        pred_ours = composite(masks, reg.predict(masks), sar.shape[:2])
        pred_base = apply_baseline(baseline, sar)
        mo, mb = all_metrics(pred_ours, rgb), all_metrics(pred_base, rgb)
        ours_m.append(mo); base_m.append(mb)
        urban = sum(1 for m in masks if m["category"] in ("building", "road"))
        if best is None or urban > best["urban"]:
            best = dict(idx=i, sar=sar, rgb=rgb, masks=masks,
                        pred_ours=pred_ours, pred_base=pred_base, urban=urban, mo=mo)

    m_ours, m_base = agg(ours_m), agg(base_m)
    print(f"\n=== Mean over {len(test_ids)} urban test patches ===")
    print("Single-model baseline   :", {k: round(v, 4) for k, v in m_base.items()})
    print("Mask-conditioned (ours) :", {k: round(v, 4) for k, v in m_ours.items()})

    json.dump({"dataset": "HuggingFace sen12mscr (lossy JPEG), 22 confirmed urban patches",
               "train_ids": train_ids, "test_ids": test_ids, "train_pool": pool,
               "baseline": m_base, "ours": m_ours},
              open("figures/metrics_hf_urban.json", "w"), indent=2)

    b = best
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    axes[0].imshow(b["sar"], cmap="gray"); axes[0].set_title("(a) SAR (HF s1)")
    axes[1].imshow(render_overlay(b["sar"], b["masks"]))
    axes[1].set_title(f"(b) Segmentation + category\n({len(b['masks'])} masks)")
    axes[2].imshow(b["pred_base"]); axes[2].set_title("(c) Single-model baseline")
    axes[3].imshow(b["pred_ours"]); axes[3].set_title(f"(d) Mask-conditioned (ours)\nNRMSE={b['mo']['NRMSE']:.3f}")
    axes[4].imshow(b["rgb"]); axes[4].set_title("(e) Optical GT (HF s2)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in
               ["building", "vegetation", "road", "water", "other"]]
    axes[1].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30),
                   ncol=3, fontsize=7, frameon=False)
    fig.suptitle(f"Confirmed urban patch (HF row {b['idx']})", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/fig7_hf_urban.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    n = len(test_ids)
    fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 6.2))
    axes = np.atleast_2d(axes)
    for j, i in enumerate(test_ids):
        sar, rgb = load_pair(i); masks = seg_cat(sar)
        ours = composite(masks, reg.predict(masks), sar.shape[:2])
        axes[0, j].imshow(sar, cmap="gray"); axes[1, j].imshow(ours); axes[2, j].imshow(rgb)
        axes[0, j].set_title(f"{i}", fontsize=7)
        for r in range(3):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("SAR", fontsize=9); axes[1, 0].set_ylabel("ours", fontsize=9)
    axes[2, 0].set_ylabel("GT", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/fig8_hf_urban_montage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/fig7_hf_urban.png, figures/fig8_hf_urban_montage.png, "
          "figures/metrics_hf_urban.json")


if __name__ == "__main__":
    main()
