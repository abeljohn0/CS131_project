"""
Sanity test: write synthetic patches as 12-bit TIFs in the prior
project's SEN12MS-CR filename convention, read them back through
pipeline.dataset, run the full pipeline, and verify the metrics.

This is the bridge between the synthetic testbed and the real
SEN12MS-CR Bay Area data: if this round-trip works, the only thing
that changes when the user plugs in real Sentinel-1/2 patches is
the contents of the input directories.
"""
import os
import sys
import shutil
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.synthetic import generate_urban_patch
from pipeline import dataset as ds
from pipeline.segmentation import segment_with_slic
from pipeline.mask_filter import categorize_all, CATEGORIES
from pipeline.recolor import PerCategoryRegressor, FEATURE_KEYS, _mean_rgb_inside, _PRIOR_RGB
from pipeline.composite import composite
from pipeline.metrics import nrmse, sam_angle, q4, edge_fidelity
from sklearn.linear_model import Ridge


def write_tif_pair(root, split, roi_prefix, idx, patch_id, sar, gt):
    """
    Write a paired SAR/GT TIF pair following the prior project's
    naming: <ROI>_<season>_s1_<id>_p<patch>.tif and
    <ROI>_<season>_gt_<id>_p<patch>.tif.
    """
    sar_dir = os.path.join(root, f"sar_{split}")
    gt_dir  = os.path.join(root, f"gt_{split}")
    os.makedirs(sar_dir, exist_ok=True)
    os.makedirs(gt_dir,  exist_ok=True)

    # Convert [0,1] floats to 12-bit uint16 (max value 4095)
    sar_12 = np.clip(sar * 4095.0, 0, 4095).astype(np.uint16)
    gt_12  = np.clip(gt  * 4095.0, 0, 4095).astype(np.uint16)

    sar_path = os.path.join(sar_dir, f"{roi_prefix}_s1_{idx}_p{patch_id}.tif")
    gt_path  = os.path.join(gt_dir,  f"{roi_prefix}_gt_{idx}_p{patch_id}.tif")
    tifffile.imwrite(sar_path, sar_12)
    tifffile.imwrite(gt_path,  gt_12)
    return sar_path, gt_path


def main():
    root = "/tmp/SEN12MS_synth"
    if os.path.isdir(root):
        shutil.rmtree(root)

    # ---- write 3 training + 3 test scenes as TIFs ----
    print("Writing synthetic TIFs to", root)
    for s in range(1, 4):
        sar, gt, _ = generate_urban_patch(size=256, seed=s)
        write_tif_pair(root, "train", "ROIs1158_spring", 1, 100 + s, sar, gt)
    for s in range(101, 104):
        sar, gt, _ = generate_urban_patch(size=256, seed=s)
        write_tif_pair(root, "test", "ROIs1158_spring", 1, 200 + s, sar, gt)

    # ---- discover & verify loader works ----
    train_pairs = ds.discover_pairs(root, "train")
    test_pairs  = ds.discover_pairs(root, "test")
    print(f"Discovered {len(train_pairs)} train pairs, {len(test_pairs)} test pairs.")
    assert len(train_pairs) == 3 and len(test_pairs) == 3, "Pair discovery failed"

    # ---- end-to-end ----
    n_segments = 140

    # Train
    by_cat = {c: ([], []) for c in CATEGORIES}
    for sar_path, gt_path in train_pairs:
        sar_12 = ds.load_sar(sar_path)
        gt_12  = ds.load_gt(gt_path)
        sar = ds.sar_to_unit(sar_12)
        gt  = ds.rgb_to_unit(gt_12)

        masks = segment_with_slic(sar, n_segments=n_segments, compactness=6.0)
        masks = categorize_all(sar, masks)
        for m in masks:
            if m.get("features") is None:
                continue
            x = np.array([m["features"][k] for k in FEATURE_KEYS], dtype=np.float32)
            y = _mean_rgb_inside(gt, m["segmentation"])
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

    # Test
    metrics_all = []
    for sar_path, gt_path in test_pairs:
        sar_12 = ds.load_sar(sar_path); gt_12 = ds.load_gt(gt_path)
        sar = ds.sar_to_unit(sar_12); gt = ds.rgb_to_unit(gt_12)

        masks = segment_with_slic(sar, n_segments=n_segments, compactness=6.0)
        masks = categorize_all(sar, masks)
        pred_colors = reg.predict(masks)
        pred = composite(masks, pred_colors, sar.shape[:2])

        m = dict(NRMSE=nrmse(pred, gt), SAM=sam_angle(pred, gt),
                 Q4=q4(pred, gt), EdgeF1=edge_fidelity(pred, gt)["f1"])
        metrics_all.append(m)
        out_name = ds.output_filename_from_sar(sar_path, "MaskCond")
        print(f"  {out_name}: {m}")

    # Aggregate
    agg = {k: float(np.mean([m[k] for m in metrics_all])) for k in metrics_all[0]}
    print(f"\nMean over {len(metrics_all)} test patches: {agg}")
    print("\nRound-trip OK -- loader + metrics + pipeline are consistent.")


if __name__ == "__main__":
    main()
