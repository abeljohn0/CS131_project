"""
Presentation assets: per-stage images for several held-out urban patches
(dense SAM + the saved 11-patch per-category pixel regressor) and a clean
baseline-vs-ours metrics bar chart. All masks are cached, so this is fast.

    .venv/bin/python -m demo.make_presentation
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.mask_filter import categorize_all
from pipeline.recolor import PerCategoryPixelRegressor  # noqa: F401 (needed to unpickle)
from demo.run_hf_urban import load_pair, fit_baseline, apply_baseline, all_metrics, agg
from demo.sam_sweep import inst_overlay

TRAIN11 = [4409, 4410, 6616, 6618, 13230, 19842, 24254, 30869, 33071, 33075, 39687]
EXAMPLES = [44095, 52916, 61736, 81576, 94807]      # held-out test patches
DISPLAY = {"water": ("water", [.10, .30, .70]), "road": ("road", [.95, .85, .10]),
           "building": ("building", [.85, .20, .20]),
           "vegetation": ("other urban area", [.88, .58, .25]),
           "other": ("other", [.55, .55, .55])}


def cat_overlay(sar, masks, a=0.6):
    base = np.stack([sar] * 3, -1).astype(np.float32); out = base.copy()
    for m in sorted(masks, key=lambda d: -d["area"]):
        s = m["segmentation"]; out[s] = (1 - a) * base[s] + a * np.array(DISPLAY[m["category"]][1])
    return np.clip(out, 0, 1)


def dense(i):
    return pickle.load(open(f"sam_masks/dense_hf_{i}.pkl", "rb"))


def main():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    reg = pickle.load(open("models/regressor_11patch.pkl", "rb"))
    baseline = fit_baseline(TRAIN11)
    name2color = {disp: color for disp, color in DISPLAY.values()}
    legend = [Patch(facecolor=name2color[n], label=n) for n in
              ["building", "other urban area", "road", "water", "other"]]

    for i in EXAMPLES:
        s, r = load_pair(i); m = categorize_all(s, dense(i)); rec = reg.predict(s, m)
        met = all_metrics(rec, r)
        od = f"data/idx{i}_stack"; os.makedirs(od, exist_ok=True)
        pfx = f"{od}/idx{i}"
        plt.imsave(f"{pfx}_1_sar_input.png", np.clip(np.stack([s] * 3, -1), 0, 1))
        plt.imsave(f"{pfx}_2_sam_masks.png", inst_overlay(s, m))
        plt.imsave(f"{pfx}_3_categories.png", cat_overlay(s, m))
        plt.imsave(f"{pfx}_4_recolored.png", np.clip(rec, 0, 1))
        plt.imsave(f"{pfx}_5_optical_gt.png", np.clip(r, 0, 1))
        panels = [(np.stack([s] * 3, -1), "1. SAR input"),
                  (inst_overlay(s, m), f"2. SAM masks ({len(m)})"),
                  (cat_overlay(s, m), "3. Categorized"),
                  (np.clip(rec, 0, 1), f"4. Recolored\nNRMSE={met['NRMSE']:.3f}"),
                  (np.clip(r, 0, 1), "5. Optical GT")]
        fig, ax = plt.subplots(1, 5, figsize=(16, 3.5))
        for a, (im, t) in zip(ax, panels):
            a.imshow(np.clip(im, 0, 1)); a.set_title(t, fontsize=10)
            a.set_xticks([]); a.set_yticks([])
        ax[2].legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.32),
                     ncol=3, fontsize=7, frameon=False)
        fig.suptitle(f"idx{i}", fontsize=11); fig.tight_layout()
        fig.savefig(f"{pfx}_stack_panel.png", dpi=160, bbox_inches="tight"); plt.close(fig)
        print(f"idx{i}: {len(m)} masks, NRMSE={met['NRMSE']:.3f} EdgeF1={met['EdgeF1']:.3f} -> {od}/")

    # ---- metrics chart: baseline vs ours on the held-out test patches ----
    mb, mo = [], []
    for i in EXAMPLES:
        s, r = load_pair(i); m = categorize_all(s, dense(i))
        mo.append(all_metrics(reg.predict(s, m), r))
        mb.append(all_metrics(apply_baseline(baseline, s), r))
    B, O = agg(mb), agg(mo)
    print("baseline:", {k: round(v, 4) for k, v in B.items()})
    print("ours    :", {k: round(v, 4) for k, v in O.items()})

    chart = [("NRMSE", "↓ lower is better"), ("Q4", "↑ higher is better"),
             ("EdgeF1", "↑ higher is better")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4))
    for ax, (mname, direction) in zip(axes, chart):
        vals = [B[mname], O[mname]]
        bars = ax.bar(["single-model\nbaseline", "mask-conditioned\n(ours)"], vals,
                      color=["#9aa0a6", "#1a73e8"], width=0.62)
        ax.set_title(f"{mname}\n{direction}", fontsize=12)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=11)
        ax.set_ylim(0, max(vals) * 1.28); ax.tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SAR→optical recoloring — 5 held-out urban patches", fontsize=13, y=1.02)
    fig.tight_layout(); fig.savefig("data/metrics_chart.png", dpi=160, bbox_inches="tight")
    print("saved data/metrics_chart.png")


if __name__ == "__main__":
    main()
