"""
Full per-category PIXEL recoloring run on a 10-patch urban subset (5 train /
5 test) with DENSE SAM masks (crop layers, ~600 masks/patch) and the
'other urban area' relabel. Reports the single-model baseline vs ours (dense
SAM + per-category pixel regressor) on the held-out test patches.

Dense SAM is slow (~3 min/patch on CPU) so masks cache in sam_masks/dense_*.pkl.
The internal category key stays "vegetation" (so the rest of the pipeline is
untouched); it is only *displayed* as "other urban area" here.

    .venv/bin/python -m demo.run_sam10_urban
"""
from __future__ import annotations
import os, sys, json, time, pickle
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.segmentation import build_sam_generator, segment_with_sam
from pipeline.mask_filter import categorize_all
from pipeline.recolor import PerCategoryPixelRegressor
from demo.run_hf_urban import load_pair, fit_baseline, apply_baseline, all_metrics, agg
from demo.sam_sweep import inst_overlay

CKPT = "models/sam_vit_h_4b8939.pth"
DENSE = dict(points_per_side=32, pred_iou_thresh=0.60, stability_score_thresh=0.70,
             min_mask_region_area=0, crop_n_layers=1, crop_n_points_downscale_factor=1,
             box_nms_thresh=0.90)
TRAIN = [6616, 19842, 33075, 46303, 70552]
TEST  = [44095, 52916, 61736, 81576, 94807]
DISPLAY = {"water": ("water", [.10, .30, .70]), "road": ("road", [.95, .85, .10]),
           "building": ("building", [.85, .20, .20]),
           "vegetation": ("other urban area", [.88, .58, .25]),
           "other": ("other", [.55, .55, .55])}


def dense_masks(idx, sar, gen):
    cache = f"sam_masks/dense_hf_{idx}.pkl"
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    os.makedirs("sam_masks", exist_ok=True)
    t = time.time(); m = segment_with_sam(sar, generator=gen)
    pickle.dump(m, open(cache, "wb"))
    print(f"  SAM idx{idx}: {len(m)} masks ({time.time()-t:.0f}s)")
    return m


def cat_overlay(sar, masks, a=0.6):
    base = np.stack([sar] * 3, -1).astype(np.float32); out = base.copy()
    for m in sorted(masks, key=lambda d: -d["area"]):
        s = m["segmentation"]; out[s] = (1 - a) * base[s] + a * np.array(DISPLAY[m["category"]][1])
    return np.clip(out, 0, 1)


def main():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    os.makedirs("figures", exist_ok=True)
    gen = build_sam_generator(CKPT, model_type="vit_h", device="cpu", **DENSE)
    print(f"Train {TRAIN}  Test {TEST}")

    samples, pool = [], Counter()
    for i in TRAIN:
        s, r = load_pair(i); m = categorize_all(s, dense_masks(i, s, gen))
        pool.update(x["category"] for x in m); samples.append((s, r, m))
    reg = PerCategoryPixelRegressor(1.0).fit(samples)
    baseline = fit_baseline(TRAIN)
    print("Train category pool:", {DISPLAY[c][0]: n for c, n in pool.items()})

    ours, base, cache = [], [], {}
    for i in TEST:
        s, r = load_pair(i); m = categorize_all(s, dense_masks(i, s, gen))
        rec = reg.predict(s, m); cache[i] = (s, r, m, rec)
        ours.append(all_metrics(rec, r)); base.append(all_metrics(apply_baseline(baseline, s), r))
    m_ours, m_base = agg(ours), agg(base)

    def fmt(d): return {k: round(v, 4) for k, v in d.items()}
    print(f"\n=== Mean over {len(TEST)} held-out urban patches (dense SAM) ===")
    print("baseline (single model)      :", fmt(m_base))
    print("ours (dense SAM + per-cat px):", fmt(m_ours))
    json.dump({"train": TRAIN, "test": TEST, "sam_config": DENSE,
               "train_pool": {DISPLAY[c][0]: n for c, n in pool.items()},
               "baseline": m_base, "ours": m_ours},
              open("figures/metrics_sam10.json", "w"), indent=2)

    # headline: full stack on idx44095
    s, r, m, rec = cache[44095]
    met = all_metrics(rec, r)
    panels = [(np.stack([s] * 3, -1), "SAR input"),
              (inst_overlay(s, m), f"SAM masks ({len(m)})"),
              (cat_overlay(s, m), "categorized"),
              (np.clip(rec, 0, 1), f"recolor\nNRMSE={met['NRMSE']:.3f} EdgeF1={met['EdgeF1']:.3f}"),
              (r, "optical GT")]
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.5))
    for a, (im, t) in zip(ax, panels):
        a.imshow(np.clip(im, 0, 1)); a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])
    name2color = {disp: color for disp, color in DISPLAY.values()}
    names = ["building", "other urban area", "road", "water", "other"]
    h = [Patch(facecolor=name2color[n], label=n) for n in names]
    ax[2].legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Dense-SAM per-category pixel recolor on idx44095 (10-patch run)", fontsize=10)
    fig.tight_layout(); fig.savefig("figures/fig11_sam10_stack.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # test montage: recolor vs GT
    fig, ax = plt.subplots(2, len(TEST), figsize=(2.4 * len(TEST), 5.0)); ax = np.atleast_2d(ax)
    for j, i in enumerate(TEST):
        _, r, _, rec = cache[i]
        ax[0, j].imshow(np.clip(rec, 0, 1)); ax[1, j].imshow(r)
        ax[0, j].set_title(str(i), fontsize=8)
        for row in range(2):
            ax[row, j].set_xticks([]); ax[row, j].set_yticks([])
    ax[0, 0].set_ylabel("ours", fontsize=9); ax[1, 0].set_ylabel("GT", fontsize=9)
    fig.tight_layout(); fig.savefig("figures/fig12_sam10_montage.png", dpi=150, bbox_inches="tight")
    print("Saved figures/fig11_sam10_stack.png, fig12_sam10_montage.png, metrics_sam10.json")


if __name__ == "__main__":
    main()
