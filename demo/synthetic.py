"""
Synthetic urban SAR/optical patch generator.
"""
from __future__ import annotations
import numpy as np


def _add_rect(canvas_sar, canvas_rgb, label, x, y, w, h,
              sar_val, rgb, sar_texture=0.02):
    H, W = canvas_sar.shape
    x2, y2 = min(x + w, W), min(y + h, H)
    x, y = max(x, 0), max(y, 0)
    if x >= x2 or y >= y2:
        return
    region_sar = sar_val + sar_texture * np.random.randn(y2 - y, x2 - x)
    canvas_sar[y:y2, x:x2] = region_sar
    canvas_rgb[y:y2, x:x2, :] = rgb
    label[y:y2, x:x2] = _CLASS_TO_ID[_LAST_CLASS[0]]


_CLASS_TO_ID = {"bg": 0, "water": 1, "road": 2, "building": 3,
                "vegetation": 4}
_LAST_CLASS = ["bg"]


def _set_class(name):
    _LAST_CLASS[0] = name


def generate_urban_patch(size: int = 256, seed: int = 0
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        sar     : (H, W) float32 in [0, 1]
        optical : (H, W, 3) float32 in [0, 1]
        label   : (H, W) int (ground-truth class, for evaluation only)
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    H = W = size

    sar = 0.35 + 0.02 * rng.standard_normal((H, W))
    rgb = np.tile(np.array([0.45, 0.40, 0.32], dtype=np.float32),
                  (H, W, 1))
    label = np.zeros((H, W), dtype=np.int32)

    _set_class("water")
    if rng.random() < 0.7:
        cy = rng.integers(int(0.6 * H), H)
        amplitude = rng.integers(5, 20)
        period = rng.integers(40, 90)
        for x in range(W):
            top = int(cy + amplitude * np.sin(2 * np.pi * x / period))
            top = max(0, min(H, top))
            sar[top:, x] = 0.05 + 0.01 * rng.standard_normal(H - top)
            rgb[top:, x, :] = [0.08, 0.18, 0.32]
            label[top:, x] = _CLASS_TO_ID["water"]

    _set_class("vegetation")
    n_veg = rng.integers(3, 7)
    for _ in range(n_veg):
        cx = rng.integers(0, W); cy = rng.integers(0, int(0.7 * H))
        rx, ry = rng.integers(15, 45), rng.integers(15, 45)
        yy, xx = np.ogrid[:H, :W]
        ellipse = ((xx - cx) / rx)**2 + ((yy - cy) / ry)**2 <= 1
        texture = 0.42 + 0.10 * rng.standard_normal(ellipse.sum())
        sar[ellipse] = texture
        veg_color = np.array([0.18, 0.40, 0.18]) + 0.04 * rng.standard_normal(3)
        rgb[ellipse] = veg_color
        label[ellipse] = _CLASS_TO_ID["vegetation"]

    _set_class("road")
    n_h = rng.integers(2, 4); n_v = rng.integers(2, 4)
    rys = sorted(rng.integers(20, H - 20, size=n_h))
    rxs = sorted(rng.integers(20, W - 20, size=n_v))
    road_w = 5
    for ry in rys:
        sar[ry:ry+road_w, :] = 0.18 + 0.01 * rng.standard_normal((road_w, W))
        rgb[ry:ry+road_w, :, :] = [0.28, 0.28, 0.28]
        label[ry:ry+road_w, :] = _CLASS_TO_ID["road"]
    for rx in rxs:
        sar[:, rx:rx+road_w] = 0.18 + 0.01 * rng.standard_normal((H, road_w))
        rgb[:, rx:rx+road_w, :] = [0.28, 0.28, 0.28]
        label[:, rx:rx+road_w] = _CLASS_TO_ID["road"]

    _set_class("building")
    rys_ext = [0] + list(rys) + [H]
    rxs_ext = [0] + list(rxs) + [W]
    for i in range(len(rys_ext) - 1):
        for j in range(len(rxs_ext) - 1):
            y0, y1 = rys_ext[i] + road_w, rys_ext[i+1]
            x0, x1 = rxs_ext[j] + road_w, rxs_ext[j+1]
            block_h, block_w = y1 - y0, x1 - x0
            if block_h < 8 or block_w < 8:
                continue
            n_b = rng.integers(1, 4)
            for _ in range(n_b):
                bw = rng.integers(min(8, block_w), max(9, min(28, block_w)))
                bh = rng.integers(min(8, block_h), max(9, min(28, block_h)))
                bx = rng.integers(x0, max(x0 + 1, x1 - bw))
                by = rng.integers(y0, max(y0 + 1, y1 - bh))
                sar_b = 0.75 + 0.04 * rng.standard_normal((bh, bw))
                roof_shade = 0.45 + 0.15 * rng.random()
                rgb_b = np.array([roof_shade, roof_shade * 0.95,
                                  roof_shade * 0.90])
                sar[by:by+bh, bx:bx+bw] = sar_b
                rgb[by:by+bh, bx:bx+bw, :] = rgb_b
                label[by:by+bh, bx:bx+bw] = _CLASS_TO_ID["building"]

    speckle = np.sqrt(-2 * np.log(np.clip(rng.random((H, W)), 1e-6, 1.0)))
    speckle = speckle / speckle.mean()
    sar = sar * (0.85 + 0.15 * speckle)
    sar = np.clip(sar, 0, 1).astype(np.float32)
    rgb = np.clip(rgb, 0, 1).astype(np.float32)
    return sar, rgb, label
