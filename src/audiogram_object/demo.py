# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np

from audiogram_object.symbols import (
    SymbolPath,
    DEFAULT_RENDER_CONFIG,
    DEFAULT_GEOM,
    NR_START,
    get_symbol,
)

from audiogram_object.render_mpl import draw_air_symbol, draw_bone_symbol

from matplotlib.path import Path
from matplotlib.patches import PathPatch

def draw_symbol_local(ax, symbol: SymbolPath, x: float, y: float, *,
                      scale: float = 1.0, color="black", lw: float = 2.0, zorder: int = 3):
    """
    Debug-only local renderer in a linear coordinate system.
    Treats symbol verts as local units and applies a uniform scale.
    """
    verts = (symbol.verts - np.array(symbol.anchor, dtype=float)) * scale
    xs = x + verts[:, 0]
    ys = y + verts[:, 1]
    path = Path(np.column_stack([xs, ys]), np.asarray(symbol.codes, dtype=int))
    patch = PathPatch(path, fill=False, edgecolor=color, linewidth=lw, zorder=zorder)
    ax.add_patch(patch)
    return patch

#---NR Symbol Debuggery---#

def demo_nr_gallery(cfg=DEFAULT_RENDER_CONFIG):
    items = [
        ("O",   dict(kind="air",  ear="right", masked=False)),
        ("X",   dict(kind="air",  ear="left",  masked=False)),
        ("TRI", dict(kind="air",  ear="right", masked=True)),
        ("SQ",  dict(kind="air",  ear="left",  masked=True)),
        ("CHEV_R",  dict(kind="bone", ear="right", masked=False)),
        ("CHEV_L",  dict(kind="bone", ear="left",  masked=False)),
        ("BRACK_R", dict(kind="bone", ear="right", masked=True)),
        ("BRACK_L", dict(kind="bone", ear="left",  masked=True)),
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, len(items) + 1)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    for i, (label, params) in enumerate(items, start=1):
        x0 = i

        # RIGHT ear composite (arrow down-left)
        sym_r = get_symbol(**params, nr=True)
        if params["kind"] == "air":
            draw_air_symbol(ax, sym_r, x0, 2.2, cfg=cfg, color="red", zorder=3)
        else:
            draw_bone_symbol(ax, sym_r, x0, 2.2, cfg=cfg, color="red", zorder=3)
        ax.text(x0, 2.75, f"{label} + NR (R)", ha="center", va="bottom", fontsize=8)

        # LEFT ear composite (arrow down-right)
        params_l = dict(params)
        params_l["ear"] = "left"
        sym_l = get_symbol(**params_l, nr=True)
        if params_l["kind"] == "air":
            draw_air_symbol(ax, sym_l, x0, 0.9, cfg=cfg, color="blue", zorder=3)
        else:
            draw_bone_symbol(ax, sym_l, x0, 0.9, cfg=cfg, color="blue", zorder=3)
        ax.text(x0, 1.35, f"{label} + NR (L)", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()

    #---NR Symbol Gallery---#
demo_nr_gallery()

def demo_audiogram_plot(cfg=DEFAULT_RENDER_CONFIG):
    fig, ax = plt.subplots(figsize=(7, 5))

    # --- Axes formatting FIRST (critical for display-space markers) ---
    ax.set_xscale("log")
    ax.set_xticks([250, 500, 1000, 2000, 4000, 8000])
    ax.set_xticklabels(["250", "500", "1k", "2k", "4k", "8k"])
    ax.tick_params(axis="x", which="both", bottom=False, top=True,
                   labelbottom=False, labeltop=True)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Frequency (Hz)")

    ax.set_ylim(120, -10)
    ax.set_yticks(np.arange(-10, 121, 10))
    ax.set_ylabel("Hearing Level (dB HL)")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    for f in [250, 500, 1000, 2000, 4000, 8000]:
        ax.axvline(f, linewidth=1, alpha=0.25)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    # --- Data AFTER axes are set ---
    freqs = np.array([250, 500, 1000, 2000, 4000, 8000])
    thr_air_r = np.array([15, 20, 25, 35, 55, 60])
    thr_air_l = np.array([10, 15, 30, 40, 50, 65])

    nr_r_idx = {4}  # 4k
    nr_l_idx = {5}  # 8k

    # --- Air symbols ---
    for i, (f, y) in enumerate(zip(freqs, thr_air_r)):
        sym = get_symbol(kind="air", ear="right", masked=False, nr=(i in nr_r_idx))
        draw_air_symbol(ax, sym, float(f), float(y), cfg=cfg, color="red", zorder=4)

    for i, (f, y) in enumerate(zip(freqs, thr_air_l)):
        sym = get_symbol(kind="air", ear="left", masked=False, nr=(i in nr_l_idx))
        draw_air_symbol(ax, sym, float(f), float(y), cfg=cfg, color="blue", zorder=4)

    # --- Bone examples ---
    bone_points = [
        (1000, 25, "right", False, "red",  False),
        (2000, 35, "left",  False, "blue", False),
        (4000, 55, "right", True,  "red",  True),
        (500,  15, "left",  True,  "blue", False),
    ]
    for f, y, ear, masked, color, nr in bone_points:
        sym = get_symbol(kind="bone", ear=ear, masked=masked, nr=nr)
        draw_bone_symbol(ax, sym, float(f), float(y), cfg=cfg, color=color, zorder=3)

    ax.set_title("Audiogram demo — frozen geometry symbols + NR composites", pad=16)

    plt.tight_layout()
    plt.show()

demo_audiogram_plot()