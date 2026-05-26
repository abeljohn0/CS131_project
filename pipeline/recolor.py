"""
Stage 2b: Per-category SAR-to-color recoloring.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from .mask_filter import CATEGORIES


FEATURE_KEYS = ["sar_mean", "sar_std", "gradient_mean",
                "elongation", "solidity"]


def _features_to_vector(feats: dict) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_KEYS], dtype=np.float32)


def _mean_rgb_inside(rgb: np.ndarray, seg: np.ndarray) -> np.ndarray:
    return rgb[seg].mean(axis=0)


class PerCategoryRegressor:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models: dict[str, Ridge] = {}
        self.fallback: dict[str, np.ndarray] = {}

    def fit(self, masks: list[dict], optical: np.ndarray) -> "PerCategoryRegressor":
        by_cat: dict[str, tuple[list, list]] = {c: ([], []) for c in CATEGORIES}
        for m in masks:
            if m.get("features") is None:
                continue
            x = _features_to_vector(m["features"])
            y = _mean_rgb_inside(optical, m["segmentation"])
            by_cat[m["category"]][0].append(x)
            by_cat[m["category"]][1].append(y)

        for cat, (xs, ys) in by_cat.items():
            if len(xs) >= 2:
                X = np.stack(xs)
                Y = np.stack(ys)
                model = Ridge(alpha=self.alpha)
                model.fit(X, Y)
                self.models[cat] = model
                self.fallback[cat] = Y.mean(axis=0)
            elif len(xs) == 1:
                self.fallback[cat] = np.stack(ys).mean(axis=0)
            else:
                self.fallback[cat] = _PRIOR_RGB[cat]
        return self

    def predict(self, masks: list[dict]) -> dict[int, np.ndarray]:
        out = {}
        for i, m in enumerate(masks):
            cat = m["category"]
            if m.get("features") is None:
                out[i] = _PRIOR_RGB[cat]
                continue
            if cat in self.models:
                x = _features_to_vector(m["features"]).reshape(1, -1)
                pred = self.models[cat].predict(x)[0]
            else:
                pred = self.fallback.get(cat, _PRIOR_RGB[cat])
            out[i] = np.clip(pred, 0, 1)
        return out


_PRIOR_RGB: dict[str, np.ndarray] = {
    "water":      np.array([0.10, 0.18, 0.30]),
    "road":       np.array([0.28, 0.28, 0.28]),
    "building":   np.array([0.55, 0.52, 0.50]),
    "vegetation": np.array([0.22, 0.38, 0.18]),
    "other":      np.array([0.45, 0.42, 0.38]),
}
