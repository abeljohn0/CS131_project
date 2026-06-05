"""
Sweep SAM automatic-mask-generator settings on ONE patch and score each, to
choose settings before running the whole set. The per-category pixel regressor
is trained once (on the other urban patches' cached masks) and held fixed, so
only the target patch's segmentation varies. A SLIC row is included as a
reference. Writes data/idx<ID>_stack/sam_sweep.png + sam_sweep_metrics.json.

    .venv/bin/python -m demo.sam_sweep 44095
"""
from __future__ import annotations
import os, sys, json, time, pickle
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from pipeline.segmentation import segment_with_slic
from pipeline.mask_filter import canonical_sar, categorize_all
from pipeline.recolor import PerCategoryPixelRegressor
from demo.run_hf_urban import load_pair, all_metrics, CAT_COLORS

CKPT = "models/sam_vit_h_4b8939.pth"

# (label, AMG kwargs) spanning prompt density and filter strictness
CONFIGS = [
    ("pps16 conservative", dict(points_per_side=16, pred_iou_thresh=0.86, stability_score_thresh=0.92, min_mask_region_area=50)),
    ("pps32 default",      dict(points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92, min_mask_region_area=50)),
    ("pps32 keep-small",   dict(points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92, min_mask_region_area=0)),
    ("pps32 aggressive",   dict(points_per_side=32, pred_iou_thresh=0.70, stability_score_thresh=0.80, min_mask_region_area=0)),
    ("pps64 aggressive",   dict(points_per_side=64, pred_iou_thresh=0.70, stability_score_thresh=0.80, min_mask_region_area=0)),
]


def to_pipeline(raw):
    return [{"segmentation": d["segmentation"].astype(bool), "area": int(d["area"]),
             "bbox": tuple(map(int, d["bbox"])), "label": i + 1}
            for i, d in enumerate(sorted(raw, key=lambda x: -x["area"]))]


def inst_overlay(sar, masks, seed=0, a=0.55):
    base = np.stack([sar] * 3, -1).astype(np.float32); out = base.copy()
    rng = np.random.default_rng(seed)
    for m in sorted(masks, key=lambda d: -d["area"]):
        s = m["segmentation"]; out[s] = (1 - a) * base[s] + a * rng.random(3)
    return np.clip(out, 0, 1)


def cat_overlay(sar, masks, a=0.6):
    base = np.stack([sar] * 3, -1).astype(np.float32); out = base.copy()
    for m in sorted(masks, key=lambda d: -d["area"]):
        s = m["segmentation"]; out[s] = (1 - a) * base[s] + a * np.array(CAT_COLORS[m["category"]])
    return np.clip(out, 0, 1)


def main():
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 44095
    outdir = f"data/idx{idx}_stack"; os.makedirs(outdir, exist_ok=True)
    ids = sorted(json.load(open("data/hf_urban/urban_ids.json")))
    k = len(ids) // 2
    train = [i for i in ids[:k] if i != idx]

    # fixed regressor: trained on the cached (pps16) SAM masks of training patches
    samples = []
    for i in train:
        s, r = load_pair(i)
        m = pickle.load(open(f"sam_masks/hf_{i}.pkl", "rb"))
        samples.append((s, r, categorize_all(s, m)))
    reg = PerCategoryPixelRegressor(1.0).fit(samples)

    sar, rgb = load_pair(idx)
    vis = np.stack([(canonical_sar(sar) * 255).astype(np.uint8)] * 3, -1)
    sam = sam_model_registry["vit_h"](checkpoint=CKPT).to("cpu")

    results = []
    for name, kw in CONFIGS:
        t = time.time()
        raw = SamAutomaticMaskGenerator(sam, **kw).generate(vis); dt = time.time() - t
        masks = categorize_all(sar, to_pipeline(raw))
        recolor = reg.predict(sar, masks)
        met = all_metrics(recolor, rgb)
        results.append((name, masks, recolor, met, Counter(m["category"] for m in masks), dt))
        print(f"{name:20s}: {len(masks):4d} masks ({dt:4.0f}s)  "
              f"NRMSE={met['NRMSE']:.3f}  Q4={met['Q4']:.3f}  EdgeF1={met['EdgeF1']:.3f}")

    # SLIC reference
    smask = categorize_all(sar, segment_with_slic(sar, n_segments=140, compactness=6.0))
    srec = reg.predict(sar, smask); smet = all_metrics(srec, rgb)
    results.append(("SLIC n=140 (ref)", smask, srec, smet,
                    Counter(m["category"] for m in smask), 0.0))
    print(f"{'SLIC n=140 (ref)':20s}: {len(smask):4d} masks         "
          f"NRMSE={smet['NRMSE']:.3f}  Q4={smet['Q4']:.3f}  EdgeF1={smet['EdgeF1']:.3f}")

    # figure: rows = settings, cols = [SAM/SLIC masks, categorized, recolored] + a GT col
    nrows = len(results)
    fig, ax = plt.subplots(nrows, 4, figsize=(13, 2.9 * nrows)); ax = np.atleast_2d(ax)
    for r, (name, masks, recolor, met, cats, dt) in enumerate(results):
        ax[r, 0].imshow(inst_overlay(sar, masks)); ax[r, 0].set_ylabel(name, fontsize=8)
        ax[r, 0].set_title(f"masks: {len(masks)}" + (f"  ({dt:.0f}s)" if dt else ""), fontsize=8)
        ax[r, 1].imshow(cat_overlay(sar, masks))
        ax[r, 2].imshow(recolor)
        ax[r, 2].set_title(f"NRMSE={met['NRMSE']:.3f}  EdgeF1={met['EdgeF1']:.3f}", fontsize=8)
        ax[r, 3].imshow(rgb)
        if r == 0:
            for c, t in zip(range(4), ["masks", "categorized", "recolored", "optical GT"]):
                ax[r, c].set_title((ax[r, c].get_title() + "\n" + t) if c in (0, 2) else t, fontsize=8)
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in
               ["building", "vegetation", "road", "water", "other"]]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"SAM settings sweep on idx{idx} (regressor fixed; NRMSE lower=better, EdgeF1 higher=better)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.99))
    fig.savefig(f"{outdir}/sam_sweep.png", dpi=150, bbox_inches="tight")

    json.dump({name: {"n_masks": len(masks), "seconds": round(dt, 1),
                      "categories": dict(cats), **{k: round(v, 4) for k, v in met.items()}}
               for (name, masks, recolor, met, cats, dt) in results},
              open(f"{outdir}/sam_sweep_metrics.json", "w"), indent=2)
    print(f"saved {outdir}/sam_sweep.png and sam_sweep_metrics.json")


if __name__ == "__main__":
    main()
