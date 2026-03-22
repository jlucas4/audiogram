"""High-level Matplotlib plotting helpers for audiograms.

This module is intentionally *Matplotlib-specific* and sits above the low-level
symbol renderer (`render_mpl.py`).

Goals:
- One canonical "audiogram canvas" (axes setup) with ANSI-like defaults.
- Named presets ("ansi", "ansi_uhf", "ansi_tiny", ...).
- Plot helpers that consume your frozen symbols + render config.

Geometry lives in `symbols.py`. Low-level drawing lives in `render_mpl.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from . import symbols as sym
from . import render_mpl as rm


@dataclass(frozen=True)
class AudiogramAxesConfig:
    """Axes/canvas settings for an audiogram plot (Matplotlib)."""

    # Data domain
    xlim: tuple[float, float] = (200.0, 10000.0)
    ylim: tuple[float, float] = (120.0, -10.0)  # inverted y

    # Frequency ticks
    xticks: tuple[int, ...] = (250, 500, 1000, 2000, 4000, 8000)
    xtick_labels: tuple[str, ...] = ("250", "500", "1000", "2000", "4000", "8000")

    # dB ticks
    yticks: tuple[int, ...] = tuple(range(-10, 121, 10))

    # Label placement (clinical style)
    x_label: str = "Frequency (Hz)"
    y_label: str = "Hearing Level (dB HL)"
    top_x_ticks: bool = True
    right_y_ticks: bool = False
    show_x_minor_ticks: bool = False

    # Grid
    show_vlines: bool = True
    vline_alpha: float = 0.25
    vline_lw: float = 1.0

    show_y_grid: bool = True
    y_grid_ls: str = "--"
    y_grid_alpha: float = 0.35

    # Shaded background bands (y0, y1, facecolor, alpha)
    shade_bands: tuple[tuple[float, float, str, float], ...] = ()

    # Tick label rotation for ultra-high frequencies (e.g., 10k+)
    angled_xticks_from: int | None = None  # rotate tick labels for ticks >= this value
    angled_xtick_rotation: float = 45.0
    angled_xtick_ha: str = "left"

    # Optional two-tier horizontal grid styling
    y_grid_major_every: int | None = 20
    y_grid_major_lw: float = 1.2
    y_grid_minor_lw: float = 0.8
    y_grid_major_alpha: float = 0.55
    y_grid_minor_alpha: float = 0.35


@dataclass(frozen=True)
class AudiogramPlotStyle:
    """High-level plot style preset (axes + palette + default figure size)."""

    axes: AudiogramAxesConfig
    figsize: tuple[float, float] = (7.5, 5.5)
    color_right: str = "red"
    color_left: str = "blue"


# --- Presets ---

ANSI_BASE = AudiogramAxesConfig()

ANSI_UHF = AudiogramAxesConfig(
    xlim=(200.0, 20000.0),
    xticks=(250, 500, 1000, 2000, 4000, 8000, 10000, 12500, 14000, 16000),
    xtick_labels=("250", "500", "1000", "2000", "4000", "8000", "10000", "12500", "14000", "16000"),
)

ANSI_PED = AudiogramAxesConfig(
    xlim=(110.0, 10000.0),
    xticks=(125, 250, 500, 1000, 2000, 4000, 8000),
    xtick_labels=("125", "250", "500", "1000", "2000", "4000", "8000"),
)

# Michigan Ear Institute-like canvas: stronger grid, shaded upper band, angled UHF labels
MEI_BASE = AudiogramAxesConfig(
    xlim=(110.0, 20000.0),
    xticks=(125, 250, 500, 1000, 2000, 4000, 8000, 10000, 12500, 14000, 16000),
    xtick_labels=("125", "250", "500", "1000", "2000", "4000", "8000", "10000", "12500", "14000", "16000"),
    right_y_ticks=True,
    # light shaded region near the top (normal hearing band)
    shade_bands=(( -10.0, 25.0, "#dbe9f6", 0.55),),
    angled_xticks_from=10000,
    y_grid_major_every=20,
    y_grid_major_lw=1.4,
    y_grid_minor_lw=0.9,
    y_grid_major_alpha=0.60,
    y_grid_minor_alpha=0.35,
    show_vlines=True,
    vline_alpha=0.35,
    vline_lw=1.1,
)

AXES_PRESETS: dict[str, AudiogramAxesConfig] = {
    "ansi": ANSI_BASE,
    "ansi_uhf": ANSI_UHF,
    "ansi_ped": ANSI_PED,
    "mei": MEI_BASE,
}

# Add aliases
AXES_PRESETS["uhf"] = ANSI_UHF
AXES_PRESETS["standard"] = ANSI_BASE
AXES_PRESETS["michigan"] = MEI_BASE


def get_axes_preset(preset: str | AudiogramAxesConfig | None) -> AudiogramAxesConfig:
    if preset is None:
        return ANSI_BASE
    if isinstance(preset, AudiogramAxesConfig):
        return preset
    key = preset.lower().strip()
    if key not in AXES_PRESETS:
        raise KeyError(f"Unknown axes preset '{preset}'. Options: {sorted(AXES_PRESETS)}")
    return AXES_PRESETS[key]


# --- Plot style presets (axes + palette + default figsize) ---

PLOT_STYLE_PRESETS: dict[str, AudiogramPlotStyle] = {
    "ansi": AudiogramPlotStyle(axes=ANSI_BASE, figsize=(7.5, 5.5), color_right="red", color_left="blue"),
    "ansi_uhf": AudiogramPlotStyle(axes=ANSI_UHF, figsize=(7.5, 5.5), color_right="red", color_left="blue"),

    # Black & white: use grayscale palette; glyphs still encode laterality.
    "ansi_bw": AudiogramPlotStyle(axes=ANSI_BASE, figsize=(7.5, 5.5), color_right="black", color_left="0.35"),

    # Figure-size variants
    "ansi_small": AudiogramPlotStyle(axes=ANSI_BASE, figsize=(4.2, 3.2), color_right="red", color_left="blue"),
    "ansi_large": AudiogramPlotStyle(axes=ANSI_BASE, figsize=(12.0, 8.0), color_right="red", color_left="blue"),

    # MEI styles
    "mei": AudiogramPlotStyle(axes=MEI_BASE, figsize=(7.5, 5.5), color_right="red", color_left="blue"),
    "mei_bw": AudiogramPlotStyle(axes=MEI_BASE, figsize=(7.5, 5.5), color_right="black", color_left="0.35"),
}

# Convenience aliases
PLOT_STYLE_PRESETS["standard"] = PLOT_STYLE_PRESETS["ansi"]
PLOT_STYLE_PRESETS["uhf"] = PLOT_STYLE_PRESETS["ansi_uhf"]
PLOT_STYLE_PRESETS["bw"] = PLOT_STYLE_PRESETS["ansi_bw"]
PLOT_STYLE_PRESETS["small"] = PLOT_STYLE_PRESETS["ansi_small"]
PLOT_STYLE_PRESETS["large"] = PLOT_STYLE_PRESETS["ansi_large"]
PLOT_STYLE_PRESETS["michigan"] = PLOT_STYLE_PRESETS["mei"]
PLOT_STYLE_PRESETS["mei-bw"] = PLOT_STYLE_PRESETS["mei_bw"]


def get_plot_style(style: str | AudiogramPlotStyle | None) -> AudiogramPlotStyle | None:
    if style is None:
        return None
    if isinstance(style, AudiogramPlotStyle):
        return style
    key = style.lower().strip()
    if key not in PLOT_STYLE_PRESETS:
        raise KeyError(f"Unknown plot style '{style}'. Options: {sorted(PLOT_STYLE_PRESETS)}")
    return PLOT_STYLE_PRESETS[key]


def setup_audiogram_axes(
    ax: plt.Axes,
    *,
    preset: str | AudiogramAxesConfig | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Configure an axes to look like a standard audiogram canvas."""
    p = get_axes_preset(preset)

    ax.set_xscale("log")
    ax.set_xlim(*p.xlim)  # IMPORTANT: always lock x-limits on log plots

    ax.set_ylim(*p.ylim)

    # Optional shaded background bands (draw first so grid/lines sit on top)
    for y0, y1, fc, a in p.shade_bands:
        ax.axhspan(y0, y1, facecolor=fc, alpha=a, zorder=0)

    ax.set_xticks(list(p.xticks))
    ax.set_xticklabels(list(p.xtick_labels))

    # Optionally rotate UHF tick labels (e.g., 10k/12.5k/14k/16k)
    if p.angled_xticks_from is not None:
        for tick, txt in zip(p.xticks, ax.get_xticklabels()):
            if int(tick) >= int(p.angled_xticks_from):
                txt.set_rotation(p.angled_xtick_rotation)
                txt.set_ha(p.angled_xtick_ha)
                txt.set_va("bottom")

    # Clinical layout: show freq ticks at top
    ax.tick_params(
        axis="x",
        which="both",
        bottom=not p.top_x_ticks,
        top=p.top_x_ticks,
        labelbottom=not p.top_x_ticks,
        labeltop=p.top_x_ticks,
    )
    # Log-scaled axes default to showing many minor ticks; audiograms usually hide them.
    if not p.show_x_minor_ticks:
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis="x", which="minor", top=False, bottom=False)

    ax.xaxis.set_label_position("top" if p.top_x_ticks else "bottom")
    ax.set_xlabel(p.x_label, labelpad=10)

    ax.set_yticks(list(p.yticks))
    ax.set_ylabel(p.y_label)

    if p.right_y_ticks:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    # Grid / guide lines
    if p.show_vlines:
        for f in p.xticks:
            ax.axvline(float(f), linewidth=p.vline_lw, alpha=p.vline_alpha)

    if p.show_y_grid:
        # Draw custom horizontal grid lines so we can emphasize majors (e.g., every 20 dB)
        for y in p.yticks:
            is_major = False
            if p.y_grid_major_every is not None and p.y_grid_major_every > 0:
                is_major = (y % p.y_grid_major_every) == 0
            lw = p.y_grid_major_lw if is_major else p.y_grid_minor_lw
            a = p.y_grid_major_alpha if is_major else p.y_grid_minor_alpha
            ax.axhline(float(y), linestyle=p.y_grid_ls, linewidth=lw, alpha=a, zorder=0.5)

    # Never force aspect='equal' on audiograms.
    ax.set_aspect("auto")

    if title:
        ax.set_title(title, pad=14)

    return ax


def new_audiogram_canvas(
    *,
    style: str | AudiogramPlotStyle | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    preset: str | AudiogramAxesConfig | None = None,
    title: str | None = None,
    constrained_layout: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a Figure+Axes and apply the audiogram canvas preset."""
    st = get_plot_style(style)
    if st is not None:
        if figsize is None:
            figsize = st.figsize
        if preset is None:
            preset = st.axes
    if figsize is None:
        figsize = (7.5, 5.5)

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
        constrained_layout=constrained_layout,
    )
    setup_audiogram_axes(ax, preset=preset, title=title)
    return fig, ax


def plot_binaural_two_panel(
    *,
    left: Mapping[int, float] | Sequence[tuple[int, float]],
    right: Mapping[int, float] | Sequence[tuple[int, float]],
    cfg: sym.AudiogramRenderConfig = sym.DEFAULT_RENDER_CONFIG,
    style: str | AudiogramPlotStyle | None = None,
    preset: str | AudiogramAxesConfig | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    constrained_layout: bool = True,
    left_masked: Iterable[int] = (),
    right_masked: Iterable[int] = (),
    left_nr: Iterable[int] = (),
    right_nr: Iterable[int] = (),
    sharey: bool = True,
    show_ylabel_right: bool = True,
    show_yticks_both: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Two-panel audiogram (clinic style): Right ear panel on the LEFT, Left ear panel on the RIGHT.

    Returns
    - (fig, (ax_right_panel, ax_left_panel))

    Notes
    - Both panels use the same audiogram canvas preset.
    - By default, panels share the y-axis range.
    - `show_ylabel_right=False` hides the y-axis label on the right panel to reduce clutter.
    - When `sharey=True`, Matplotlib hides y tick labels on the right panel unless forced on.
      This function can force them on when `show_yticks_both=True`.
    - preset: optional axes-only override (wins over style for axis configuration)
    """

    st = get_plot_style(style)
    axes_preset = preset
    if axes_preset is None and st is not None:
        axes_preset = st.axes

    if figsize is None:
        figsize = st.figsize if st is not None else (11.0, 5.5)
    color_r = (st.color_right if st is not None else "red")
    color_l = (st.color_left if st is not None else "blue")

    fig, (ax_r, ax_l) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=constrained_layout,
        sharey=sharey,
    )

    # Configure axes (do NOT set per-panel titles; we draw boxed panel labels instead)
    setup_audiogram_axes(ax_r, preset=axes_preset, title=None)
    setup_audiogram_axes(ax_l, preset=axes_preset, title=None)

    # Defensive: ensure no stale titles remain (e.g., if caller reuses axes)
    ax_r.set_title("")
    ax_l.set_title("")

    def _panel_box_label(ax: plt.Axes, text: str) -> None:
        ax.text(
            0.5,
            1.18,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="square,pad=0.35",
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
            ),
            zorder=20,
            clip_on=False,
        )

    _panel_box_label(ax_r, "RIGHT EAR")
    _panel_box_label(ax_l, "LEFT EAR")

    # Matplotlib hides y tick labels on the second axis when sharey=True.
    # Clinic-style two-panel audiograms usually show y ticks on BOTH panels.
    if show_yticks_both:
        ax_r.tick_params(axis="y", which="both", left=True, labelleft=True)
        ax_l.tick_params(axis="y", which="both", left=True, labelleft=True)

    if title:
        fig.suptitle(title)

    # Optional cleanup for right panel y-label
    if not show_ylabel_right:
        ax_l.set_ylabel("")

    # Plot right ear on LEFT panel
    plot_ear(
        ax_r,
        right,
        ear="right",
        cfg=cfg,
        masked_freqs=right_masked,
        nr_freqs=right_nr,
        color=color_r,
    )

    # Plot left ear on RIGHT panel
    plot_ear(
        ax_l,
        left,
        ear="left",
        cfg=cfg,
        masked_freqs=left_masked,
        nr_freqs=left_nr,
        color=color_l,
    )

    return fig, (ax_r, ax_l)


def _as_sorted_pairs(thresholds: Mapping[int, float] | Sequence[tuple[int, float]]):
    if isinstance(thresholds, Mapping):
        items = list(thresholds.items())
    else:
        items = list(thresholds)
    items = [(int(f), float(v)) for f, v in items]
    items.sort(key=lambda t: t[0])
    return items


def plot_ear(
    ax: plt.Axes,
    thresholds: Mapping[int, float] | Sequence[tuple[int, float]],
    *,
    ear: str,
    cfg: sym.AudiogramRenderConfig = sym.DEFAULT_RENDER_CONFIG,
    masked_freqs: Iterable[int] = (),
    nr_freqs: Iterable[int] = (),
    color: str | None = None,
    zorder_air: int = 4,
    zorder_bone: int = 3,
    show_air: bool = True,
    show_bone: bool = False,
    bone_thresholds: Mapping[int, float] | Sequence[tuple[int, float]] | None = None,
):
    """Plot one ear worth of thresholds onto an already-configured audiogram axes.

    Notes
    - `thresholds` is typically air-conduction.
    - For now, `masked_freqs` toggles the air glyph (square/triangle).
    - `nr_freqs` toggles NR composite overlays.
    - Optional `bone_thresholds` allows plotting bone points separately.

    This stays deliberately simple; you can build richer APIs later.
    """
    ear = ear.lower()
    if ear not in ("left", "right"):
        raise ValueError("ear must be 'left' or 'right'")

    if color is None:
        color = "blue" if ear == "left" else "red"

    masked_set = set(int(f) for f in masked_freqs)
    nr_set = set(int(f) for f in nr_freqs)

    air_items = _as_sorted_pairs(thresholds)

    if show_air:
        for f, y in air_items:
            s = sym.get_symbol(kind="air", ear=ear, masked=(f in masked_set), nr=(f in nr_set), cfg=cfg)
            rm.add_air_symbol(ax, s, float(f), float(y), cfg=cfg, color=color, zorder=zorder_air)

    if show_bone and bone_thresholds is not None:
        bone_items = _as_sorted_pairs(bone_thresholds)
        for f, y in bone_items:
            s = sym.get_symbol(kind="bone", ear=ear, masked=(f in masked_set), nr=(f in nr_set), cfg=cfg)
            rm.add_bone_symbol(ax, s, float(f), float(y), cfg=cfg, color=color, zorder=zorder_bone)


def plot_binaural(
    *,
    left: Mapping[int, float] | Sequence[tuple[int, float]],
    right: Mapping[int, float] | Sequence[tuple[int, float]],
    cfg: sym.AudiogramRenderConfig = sym.DEFAULT_RENDER_CONFIG,
    style: str | AudiogramPlotStyle | None = None,
    preset: str | AudiogramAxesConfig | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    constrained_layout: bool = True,
    left_masked: Iterable[int] = (),
    right_masked: Iterable[int] = (),
    left_nr: Iterable[int] = (),
    right_nr: Iterable[int] = (),
) -> tuple[plt.Figure, plt.Axes]:
    """Convenience: create a canvas and plot left+right air thresholds.

    Notes
    - preset: optional axes-only override (wins over style for axis configuration)
    """
    st = get_plot_style(style)
    axes_preset = preset
    if axes_preset is None and st is not None:
        axes_preset = st.axes

    if figsize is None:
        figsize = st.figsize if st is not None else (7.5, 5.5)
    color_r = (st.color_right if st is not None else "red")
    color_l = (st.color_left if st is not None else "blue")

    fig, ax = new_audiogram_canvas(
        style=style,
        figsize=figsize,
        dpi=dpi,
        preset=axes_preset,
        title=title,
        constrained_layout=constrained_layout,
    )

    plot_ear(ax, right, ear="right", cfg=cfg, masked_freqs=right_masked, nr_freqs=right_nr, color=color_r)
    plot_ear(ax, left, ear="left", cfg=cfg, masked_freqs=left_masked, nr_freqs=left_nr, color=color_l)

    return fig, ax
