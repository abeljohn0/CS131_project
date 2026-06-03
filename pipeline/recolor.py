"""
Stage 2b: Per-category SAR-to-color recoloring.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from .mask_filter import CATEGORIES, CATEGORY_TO_IDX


def _new_model(alpha: float):
    """Standardize the disparate-scale mask features before ridge regression."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


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
                self.models[cat] = _new_model(self.alpha).fit(X, Y)
                self.fallback[cat] = Y.mean(axis=0)
            elif len(xs) == 1:
                self.fallback[cat] = np.stack(ys).mean(axis=0)
            else:
                self.fallback[cat] = _PRIOR_RGB[cat]
        return self

    def fit_pooled(self, masks_list: list[list[dict]],
                   rgb_list: list[np.ndarray]) -> "PerCategoryRegressor":
        """Fit one Ridge per category, pooling masks across multiple scenes."""
        by_cat: dict[str, tuple[list, list]] = {c: ([], []) for c in CATEGORIES}
        for masks, optical in zip(masks_list, rgb_list):
            for m in masks:
                if m.get("features") is None:
                    continue
                by_cat[m["category"]][0].append(_features_to_vector(m["features"]))
                by_cat[m["category"]][1].append(
                    _mean_rgb_inside(optical, m["segmentation"]))
        for cat, (xs, ys) in by_cat.items():
            if len(xs) >= 2:
                self.models[cat] = _new_model(self.alpha).fit(
                    np.stack(xs), np.stack(ys))
                self.fallback[cat] = np.stack(ys).mean(axis=0)
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


# ---------------------------------------------------------------------------
# Per-category PER-PIXEL recoloring.
#
# The flat PerCategoryRegressor paints one mean colour per mask, which throws
# away intra-object variation -- harmless for tiny SLIC superpixels but costly
# for SAM's large coherent objects. This variant instead colours every pixel
# from its *local* SAR features using a regressor specialised to the category of
# the mask it falls in, with a single global model for pixels no mask covers.
# It is the single-model pixel baseline, split into one model per category.
# ---------------------------------------------------------------------------

def pixel_features(sar: np.ndarray) -> np.ndarray:
    """Per-pixel SAR features (same triple as the single-model baseline)."""
    mu = gaussian_filter(sar, sigma=2.0)
    sq = gaussian_filter(sar * sar, sigma=2.0)
    var = np.clip(sq - mu * mu, 0, None)
    return np.stack([sar, mu, np.sqrt(var)], axis=-1).astype(np.float32)


def category_pixel_map(shape, masks: list[dict]) -> np.ndarray:
    """(H,W) int map of each pixel's category index; -1 where no mask covers it.
    Larger masks are painted first so smaller (more specific) ones override."""
    cm = np.full(shape[:2], -1, dtype=np.int64)
    for m in sorted(masks, key=lambda d: -d["area"]):
        cm[m["segmentation"]] = CATEGORY_TO_IDX[m["category"]]
    return cm


class PerCategoryPixelRegressor:
    def __init__(self, alpha: float = 1.0, max_pixels_per_cat: int = 40000,
                 min_pixels: int = 200, seed: int = 0):
        self.alpha = alpha
        self.max_pixels_per_cat = max_pixels_per_cat
        self.min_pixels = min_pixels
        self.rng = np.random.default_rng(seed)
        self.models: dict[str, object] = {}
        self.global_model = None

    def _sub(self, n):
        if n <= self.max_pixels_per_cat:
            return slice(None)
        return self.rng.choice(n, size=self.max_pixels_per_cat, replace=False)

    def fit(self, samples: list[tuple]) -> "PerCategoryPixelRegressor":
        """samples: list of (sar, rgb, categorized_masks)."""
        by_cat = {c: ([], []) for c in CATEGORIES}
        gX, gY = [], []
        for sar, rgb, masks in samples:
            F = pixel_features(sar).reshape(-1, 3)
            Y = rgb.reshape(-1, 3)
            cm = category_pixel_map(sar.shape, masks).reshape(-1)
            gX.append(F); gY.append(Y)
            for ci, c in enumerate(CATEGORIES):
                idx = np.where(cm == ci)[0]
                if idx.size:
                    by_cat[c][0].append(F[idx]); by_cat[c][1].append(Y[idx])
        GX = np.concatenate(gX); GY = np.concatenate(gY)
        gi = self._sub(GX.shape[0])
        self.global_model = make_pipeline(
            StandardScaler(), Ridge(alpha=self.alpha)).fit(GX[gi], GY[gi])
        for c in CATEGORIES:
            if not by_cat[c][0]:
                continue
            X = np.concatenate(by_cat[c][0]); Y = np.concatenate(by_cat[c][1])
            if X.shape[0] < self.min_pixels:
                continue                          # too few -> fall back to global
            si = self._sub(X.shape[0])
            self.models[c] = make_pipeline(
                StandardScaler(), Ridge(alpha=self.alpha)).fit(X[si], Y[si])
        return self

    def predict(self, sar: np.ndarray, masks: list[dict]) -> np.ndarray:
        H, W = sar.shape[:2]
        F = pixel_features(sar).reshape(-1, 3)
        cm = category_pixel_map(sar.shape, masks).reshape(-1)
        out = self.global_model.predict(F)        # default for uncovered pixels
        for ci, c in enumerate(CATEGORIES):
            if c not in self.models:
                continue
            idx = np.where(cm == ci)[0]
            if idx.size:
                out[idx] = self.models[c].predict(F[idx])
        return np.clip(out.reshape(H, W, 3), 0, 1).astype(np.float32)
