"""High-level Matplotlib plotting helpers for audiograms.

This module is intentionally *Matplotlib-specific* and sits above the low-level
symbol renderer (`render_mpl.py`).

Goals:
- One canonical "audiogram canvas" (axes setup) with ANSI-like defaults.
- Named presets ("ansi", "ehf", "mei", ...).
- Plot helpers that consume your frozen symbols + render config.

Geometry lives in `symbols.py`. Low-level drawing lives in `render_mpl.py`.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt

from . import symbols as sym
from . import render_mpl as rm

# Canvas geometry lives in the backend-neutral `canvas` module so the plot-spec
# exporter can build it without importing Matplotlib. Re-exported here so
# existing `audiogram.plot_mpl.<name>` imports keep working.
from .canvas import (  # noqa: F401
    AudiogramAxesConfig,
    AudiogramPlotStyle,
    ANSI_BASE,
    EHF,
    MEI_BASE,
    AXES_PRESETS,
    PLOT_STYLE_PRESETS,
    DEFAULT_PLOT_STYLE,
    get_axes_preset,
    get_plot_style,
    air_line_segments,
)


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

    # Rotate UHF tick labels — must happen AFTER tick_params moves labels to
    # the top axis, since that creates new Text objects.
    if p.angled_xticks_from is not None:
        uhf_set = {int(t) for t in p.xticks if int(t) >= int(p.angled_xticks_from)}
        for tick_val, txt in zip(p.xticks, ax.xaxis.get_majorticklabels()):
            if int(tick_val) in uhf_set:
                txt.set_rotation(p.angled_xtick_rotation)
                txt.set_ha(p.angled_xtick_ha)
                txt.set_va("bottom")

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
    if figsize is None:
        figsize = st.figsize
    if preset is None:
        preset = st.axes

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
    show_bone: bool = False,
    left_bone: Mapping[int, float] | None = None,
    right_bone: Mapping[int, float] | None = None,
    left_bone_masked: Iterable[int] = (),
    right_bone_masked: Iterable[int] = (),
    left_bone_nr: Iterable[int] = (),
    right_bone_nr: Iterable[int] = (),
    show_air: bool = True,
    left_soundfield: Mapping[int, float] | None = None,
    right_soundfield: Mapping[int, float] | None = None,
    soundfield_color: str | None = None,
    left_ci: Mapping[int, float] | None = None,
    right_ci: Mapping[int, float] | None = None,
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
    if axes_preset is None:
        axes_preset = st.axes

    if figsize is None:
        w, h = st.figsize
        figsize = (w * 2 - 2, h)
    color_r = st.color_right
    color_l = st.color_left
    color_sf = soundfield_color or st.color_soundfield
    tm_fontsize = st.text_marker_fontsize
    tm_fontweight = st.text_marker_fontweight

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
            1.28,
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
        fig.suptitle(title, y=1.05, fontsize=13, fontweight="bold")

    # Optional cleanup for right panel y-label
    if not show_ylabel_right:
        ax_l.set_ylabel("")

    # Plot right ear on LEFT panel
    plot_ear(
        ax_r, right, ear="right", cfg=cfg,
        show_air=show_air,
        masked_freqs=right_masked, nr_freqs=right_nr, color=color_r,
        show_bone=show_bone, bone_thresholds=right_bone,
        bone_masked_freqs=right_bone_masked, bone_nr_freqs=right_bone_nr,
        soundfield_thresholds=right_soundfield, soundfield_color=color_sf,
        ci_thresholds=right_ci,
        text_marker_fontsize=tm_fontsize, text_marker_fontweight=tm_fontweight,
    )

    # Plot left ear on RIGHT panel
    plot_ear(
        ax_l, left, ear="left", cfg=cfg,
        show_air=show_air,
        masked_freqs=left_masked, nr_freqs=left_nr, color=color_l,
        show_bone=show_bone, bone_thresholds=left_bone,
        bone_masked_freqs=left_bone_masked, bone_nr_freqs=left_bone_nr,
        soundfield_thresholds=left_soundfield, soundfield_color=color_sf,
        ci_thresholds=left_ci,
        text_marker_fontsize=tm_fontsize, text_marker_fontweight=tm_fontweight,
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


def _plot_text_series(
    ax: plt.Axes,
    thresholds: Mapping[int, float] | Sequence[tuple[int, float]],
    *,
    label: str,
    color: str,
    linewidth: float = 1.5,
    fontsize: float = 9.0,
    fontweight: str = "bold",
    zorder: int = 4,
):
    """Plot a threshold series using text labels ('S', 'CI') instead of path symbols.

    Draws connecting lines between all points, then places text labels at each
    threshold, centered on the data point with a white background for legibility.
    """
    items = _as_sorted_pairs(thresholds)
    if not items:
        return

    if len(items) > 1:
        xs = [float(f) for f, _ in items]
        ys = [float(y) for _, y in items]
        ax.plot(
            xs, ys,
            color=color,
            linewidth=linewidth,
            zorder=zorder - 1,
            solid_capstyle="round",
        )

    for f, y in items:
        ax.text(
            float(f), float(y), label,
            color=color,
            fontsize=fontsize,
            fontweight=fontweight,
            ha="center",
            va="center",
            zorder=zorder,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )


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
    bone_masked_freqs: Iterable[int] = (),
    bone_nr_freqs: Iterable[int] = (),
    soundfield_thresholds: Mapping[int, float] | None = None,
    soundfield_color: str | None = None,
    ci_thresholds: Mapping[int, float] | None = None,
    ci_color: str | None = None,
    text_marker_fontsize: float = 9.0,
    text_marker_fontweight: str = "bold",
):
    """Plot one ear worth of thresholds onto an already-configured audiogram axes.

    Parameters
    ----------
    thresholds
        Air conduction thresholds (freq_hz -> dB HL).
    masked_freqs / nr_freqs
        Air conduction frequencies that are masked or NR.
    bone_thresholds
        Bone conduction thresholds (freq_hz -> dB HL). Only plotted when show_bone=True.
    bone_masked_freqs / bone_nr_freqs
        Bone conduction frequencies that are masked or NR.
    soundfield_thresholds
        Soundfield thresholds (freq_hz -> dB HL). Plotted as 'S' text markers.
    soundfield_color
        Color for soundfield series. Defaults to green.
    ci_thresholds
        Cochlear implant aided thresholds (freq_hz -> dB HL). Plotted as 'CI' text markers.
    ci_color
        Color for CI series. Defaults to the ear color.
    text_marker_fontsize
        Font size for text-based markers (S, CI).
    text_marker_fontweight
        Font weight for text-based markers (S, CI).
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
        # An NR breaks the air line: neighbours on opposite sides of an NR are
        # NOT connected across it (the line splits into segments, and the NR
        # symbol floats unconnected). See canvas.air_line_segments.
        for seg in air_line_segments(air_items, nr_set):
            if len(seg) > 1:
                ax.plot(
                    [float(f) for f, _ in seg],
                    [float(y) for _, y in seg],
                    color=color,
                    linewidth=cfg.style.linewidth,
                    zorder=zorder_air - 1,
                    solid_capstyle="round",
                )

        for f, y in air_items:
            s = sym.get_symbol(kind="air", ear=ear, masked=(f in masked_set), nr=(f in nr_set), cfg=cfg)
            rm.add_air_symbol(ax, s, float(f), float(y), cfg=cfg, color=color, zorder=zorder_air, fill=True, facecolor="white")

    if show_bone and bone_thresholds is not None:
        bone_masked_set = set(int(f) for f in bone_masked_freqs)
        bone_nr_set = set(int(f) for f in bone_nr_freqs)
        bone_items = _as_sorted_pairs(bone_thresholds)
        for f, y in bone_items:
            s = sym.get_symbol(kind="bone", ear=ear, masked=(f in bone_masked_set), nr=(f in bone_nr_set), cfg=cfg)
            rm.add_bone_symbol(ax, s, float(f), float(y), cfg=cfg, color=color, zorder=zorder_bone)

    if soundfield_thresholds:
        _plot_text_series(
            ax, soundfield_thresholds,
            label="S",
            color=soundfield_color or "#2ca02c",
            linewidth=cfg.style.linewidth,
            fontsize=text_marker_fontsize,
            fontweight=text_marker_fontweight,
            zorder=zorder_air + 1,
        )

    if ci_thresholds:
        _plot_text_series(
            ax, ci_thresholds,
            label="CI",
            color=ci_color or color,
            linewidth=cfg.style.linewidth,
            fontsize=text_marker_fontsize,
            fontweight=text_marker_fontweight,
            zorder=zorder_air + 1,
        )


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
    show_bone: bool = False,
    left_bone: Mapping[int, float] | None = None,
    right_bone: Mapping[int, float] | None = None,
    left_bone_masked: Iterable[int] = (),
    right_bone_masked: Iterable[int] = (),
    left_bone_nr: Iterable[int] = (),
    right_bone_nr: Iterable[int] = (),
    show_air: bool = True,
    left_soundfield: Mapping[int, float] | None = None,
    right_soundfield: Mapping[int, float] | None = None,
    soundfield_color: str | None = None,
    left_ci: Mapping[int, float] | None = None,
    right_ci: Mapping[int, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a canvas and plot left+right thresholds on a single axes."""
    st = get_plot_style(style)
    axes_preset = preset
    if axes_preset is None:
        axes_preset = st.axes

    if figsize is None:
        figsize = st.figsize
    color_r = st.color_right
    color_l = st.color_left
    color_sf = soundfield_color or st.color_soundfield
    tm_fontsize = st.text_marker_fontsize
    tm_fontweight = st.text_marker_fontweight

    fig, ax = new_audiogram_canvas(
        style=style,
        figsize=figsize,
        dpi=dpi,
        preset=axes_preset,
        title=title,
        constrained_layout=constrained_layout,
    )

    plot_ear(
        ax, right, ear="right", cfg=cfg,
        show_air=show_air,
        masked_freqs=right_masked, nr_freqs=right_nr, color=color_r,
        show_bone=show_bone, bone_thresholds=right_bone,
        bone_masked_freqs=right_bone_masked, bone_nr_freqs=right_bone_nr,
        soundfield_thresholds=right_soundfield, soundfield_color=color_sf,
        ci_thresholds=right_ci,
        text_marker_fontsize=tm_fontsize, text_marker_fontweight=tm_fontweight,
    )
    plot_ear(
        ax, left, ear="left", cfg=cfg,
        show_air=show_air,
        masked_freqs=left_masked, nr_freqs=left_nr, color=color_l,
        show_bone=show_bone, bone_thresholds=left_bone,
        bone_masked_freqs=left_bone_masked, bone_nr_freqs=left_bone_nr,
        soundfield_thresholds=left_soundfield, soundfield_color=color_sf,
        ci_thresholds=left_ci,
        text_marker_fontsize=tm_fontsize, text_marker_fontweight=tm_fontweight,
    )

    return fig, ax
