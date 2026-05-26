"""
Stage 3: Evaluation metrics.
"""
from __future__ import annotations
import numpy as np
from skimage.feature import canny
from skimage.color import rgb2gray
from scipy.ndimage import distance_transform_edt


# below are some reused metrics

def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    num = float(np.sqrt(((target - pred) ** 2).mean()))
    den = float(np.sqrt((target ** 2).mean())) + 1e-12
    return num / den


def sam_angle(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    p = pred.reshape(-1, pred.shape[-1])
    t = target.reshape(-1, target.shape[-1])
    num = (p * t).sum(axis=1)
    den = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + eps
    cos = np.clip(num / den, -1.0, 1.0)
    return float(np.arccos(cos).mean())

# newer metricsz

def _block_quality(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = a.mean(), b.mean()
    sa2, sb2 = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    num = 4.0 * cov * ma * mb
    den = (sa2 + sb2) * (ma * ma + mb * mb)
    return num / (den + 1e-12)


def q4(pred: np.ndarray, target: np.ndarray,
       block: int = 32, stride: int = 16) -> float:
    H, W, C = target.shape
    p4 = np.concatenate([pred,   np.zeros((H, W, 1), dtype=pred.dtype)],   axis=-1)
    t4 = np.concatenate([target, np.zeros((H, W, 1), dtype=target.dtype)], axis=-1)

    vals = []
    for i in range(0, H - block + 1, stride):
        for j in range(0, W - block + 1, stride):
            qbands = []
            for c in range(t4.shape[-1]):
                a = p4[i:i+block, j:j+block, c].ravel()
                b = t4[i:i+block, j:j+block, c].ravel()
                if b.std() < 1e-8 and a.std() < 1e-8:
                    qbands.append(1.0)
                else:
                    qbands.append(_block_quality(a, b))
            vals.append(np.mean(qbands))
    return float(np.mean(vals)) if vals else 0.0

def edge_fidelity(pred: np.ndarray, target: np.ndarray,
                  sigma: float = 1.0, tolerance: int = 2) -> dict:
    pg = rgb2gray(pred) if pred.ndim == 3 else pred
    tg = rgb2gray(target) if target.ndim == 3 else target

    pe = canny(pg, sigma=sigma)
    te = canny(tg, sigma=sigma)

    if pe.sum() == 0 or te.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "pred_edges": int(pe.sum()), "target_edges": int(te.sum())}

    dt_target = distance_transform_edt(~te)
    dt_pred = distance_transform_edt(~pe)

    tp_pred = (dt_target[pe] <= tolerance).sum()
    precision = tp_pred / pe.sum()

    tp_target = (dt_pred[te] <= tolerance).sum()
    recall = tp_target / te.sum()

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_edges": int(pe.sum()),
        "target_edges": int(te.sum()),
    }
