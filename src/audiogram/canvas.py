"""Backend-neutral audiogram canvas geometry: axes presets, style presets, and
line-connection policy.

This module holds the *data* that defines an audiogram canvas — axis limits,
ticks, grid weights, palettes, figure sizes — with no dependency on any
rendering backend. Both the Matplotlib renderer (`plot_mpl`) and the JSON
plot-spec exporter (`plot_spec`) import from here, so neither owns the geometry
and the two can never drift. In particular, `plot_spec` (and the web/API stack
that serves it) no longer transitively imports Matplotlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AudiogramAxesConfig:
    """Axes/canvas settings for an audiogram plot."""

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
    color_soundfield: str = "#2ca02c"
    text_marker_fontsize: float = 15.0
    text_marker_fontweight: str = "bold"


# --- Presets ---

ANSI_BASE = AudiogramAxesConfig(
    ylim=(125.0, -10.0),
)

# Extended high-frequency supplemental panel (8k–20k)
EHF = AudiogramAxesConfig(
    xlim=(7000.0, 22000.0),
    ylim=(125.0, -10.0),
    xticks=(8000, 10000, 11200, 12500, 14000, 16000, 18000, 20000),
    xtick_labels=("8000", "10000", "11200", "12500", "14000", "16000", "18000", "20000"),
)

# Michigan Ear Institute-like canvas: stronger grid, shaded upper band, angled UHF labels
MEI_BASE = AudiogramAxesConfig(
    xlim=(110.0, 20000.0),
    ylim=(125.0, -20.0),
    xticks=(125, 250, 500, 1000, 2000, 4000, 8000, 10000, 12500, 14000, 16000),
    xtick_labels=("125", "250", "500", "1000", "2000", "4000", "8000", "10000", "12500", "14000", "16000"),
    yticks=tuple(range(-20, 121, 10)),
    right_y_ticks=False,
    shade_bands=((-10.0, 25.0, "#dbe9f6", 0.55),),
    angled_xticks_from=10000,
    angled_xtick_rotation=55.0,
    angled_xtick_ha="left",
    y_grid_ls="-",
    y_grid_major_every=20,
    y_grid_major_lw=1.4,
    y_grid_minor_lw=0.7,
    y_grid_major_alpha=0.50,
    y_grid_minor_alpha=0.25,
    show_vlines=True,
    vline_alpha=0.30,
    vline_lw=0.9,
)

AXES_PRESETS: dict[str, AudiogramAxesConfig] = {
    "ansi": ANSI_BASE,
    "ehf": EHF,
    "mei": MEI_BASE,
}

# Add aliases
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

    # Black & white: use grayscale palette; glyphs still encode laterality.
    "ansi_bw": AudiogramPlotStyle(axes=ANSI_BASE, figsize=(7.5, 5.5), color_right="black", color_left="0.35"),

    # Figure-size variants
    "ansi_small": AudiogramPlotStyle(axes=ANSI_BASE, figsize=(4.2, 3.2), color_right="red", color_left="blue"),
    "ansi_large": AudiogramPlotStyle(axes=ANSI_BASE, figsize=(12.0, 8.0), color_right="red", color_left="blue"),

    # Extended high-frequency supplemental
    "ehf": AudiogramPlotStyle(axes=EHF, figsize=(7.5, 5.5), color_right="red", color_left="blue"),

    # MEI styles
    "mei": AudiogramPlotStyle(axes=MEI_BASE, figsize=(9.0, 5.5), color_right="red", color_left="blue"),
    "mei_bw": AudiogramPlotStyle(axes=MEI_BASE, figsize=(9.0, 5.5), color_right="black", color_left="0.35"),
}

# Convenience aliases
PLOT_STYLE_PRESETS["standard"] = PLOT_STYLE_PRESETS["ansi"]
PLOT_STYLE_PRESETS["bw"] = PLOT_STYLE_PRESETS["ansi_bw"]
PLOT_STYLE_PRESETS["small"] = PLOT_STYLE_PRESETS["ansi_small"]
PLOT_STYLE_PRESETS["large"] = PLOT_STYLE_PRESETS["ansi_large"]
PLOT_STYLE_PRESETS["michigan"] = PLOT_STYLE_PRESETS["mei"]
PLOT_STYLE_PRESETS["mei-bw"] = PLOT_STYLE_PRESETS["mei_bw"]


DEFAULT_PLOT_STYLE = PLOT_STYLE_PRESETS["ansi"]


def get_plot_style(style: str | AudiogramPlotStyle | None) -> AudiogramPlotStyle:
    if style is None:
        return DEFAULT_PLOT_STYLE
    if isinstance(style, AudiogramPlotStyle):
        return style
    key = style.lower().strip()
    if key not in PLOT_STYLE_PRESETS:
        raise KeyError(f"Unknown plot style '{style}'. Options: {sorted(PLOT_STYLE_PRESETS)}")
    return PLOT_STYLE_PRESETS[key]


# --- Line-connection policy ---


def air_line_segments(
    items: Sequence[tuple[int, float]],
    nr_freqs: set[int],
) -> list[list[tuple[int, float]]]:
    """Split air-conduction points into connected segments, breaking at NR.

    A no-response (NR) threshold is a hard break in the connecting line, NOT a
    point to route the line around: given T1-T2-NR-T3-T4 (in frequency order),
    the result is [[T1, T2], [T3, T4]] — T2 does not connect to T3, and the NR
    point belongs to no segment (it is drawn as a free-floating symbol).

    `items` must be (freq_hz, db) pairs; they are sorted by frequency here.
    Returns the list of segments (each a list of points, including
    single-point segments — callers draw a line only when len >= 2).
    """
    ordered = sorted(((int(f), float(y)) for f, y in items), key=lambda t: t[0])
    segments: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for f, y in ordered:
        if f in nr_freqs:
            if current:
                segments.append(current)
                current = []
            continue  # NR floats alone — not added to any segment
        current.append((f, y))
    if current:
        segments.append(current)
    return segments
