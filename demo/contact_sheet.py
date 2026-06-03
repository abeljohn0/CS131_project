"""
Contact sheets of the downloaded SEN12MS-CR patches, for visual labeling
(e.g. deciding which patches are urban). Writes figures/contact_sheet_rgb.png
and figures/contact_sheet_sar.png, each thumbnail titled with its patch id.

    python -m demo.contact_sheet
"""
from __future__ import annotations
import os, sys, math, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.dataset import discover_sen12mscr_pairs, load_s1_vv, load_s2_rgb

DATA_ROOT = "data/sen12mscr"
COLS = 8


def _pnum(path: str) -> int:
    m = re.search(r"_p(\d+)\.tif$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _sheet(pairs, loader, kind, cmap=None):
    n = len(pairs)
    rows = math.ceil(n / COLS)
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 1.7, rows * 1.8))
    axes = np.atleast_2d(axes)
    for k, (sp, gp) in enumerate(pairs):
        r, c = divmod(k, COLS)
        ax = axes[r, c]
        ax.imshow(loader(sp if kind == "sar" else gp), cmap=cmap)
        ax.set_title(f"p{_pnum(sp)}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, rows * COLS):
        r, c = divmod(k, COLS)
        axes[r, c].axis("off")
    fig.suptitle(f"SEN12MS-CR ROIs1868_summer scene 27 — {kind.upper()} "
                 f"({n} patches)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = f"figures/contact_sheet_{kind}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    os.makedirs("figures", exist_ok=True)
    pairs = discover_sen12mscr_pairs(DATA_ROOT)
    if not pairs:
        raise SystemExit(f"No pairs under {DATA_ROOT}/")
    pairs = sorted(pairs, key=lambda p: _pnum(p[0]))
    rgb = _sheet(pairs, load_s2_rgb, "rgb")
    sar = _sheet(pairs, load_s1_vv, "sar", cmap="gray")
    ids = ", ".join(f"p{_pnum(sp)}" for sp, _ in pairs)
    print(f"{len(pairs)} patches.\nWrote {rgb} and {sar}\nPatch ids: {ids}")


if __name__ == "__main__":
    main()
