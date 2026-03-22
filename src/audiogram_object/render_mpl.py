from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.transforms import IdentityTransform
from matplotlib.artist import Artist

from .symbols import SymbolPath, SymbolComposite, DEFAULT_RENDER_CONFIG


def _mpl_path(symbol: SymbolPath | SymbolComposite) -> Path:
    """Convert SymbolPath (local coords) -> matplotlib Path.

    Note: If a SymbolComposite is passed accidentally, we fall back to its base.
    Composites should generally be rendered as base + overlays (see draw_symbol/add_symbol).
    """
    # Duck-typed composite handling: avoid isinstance() because notebooks/autoreload
    # can create multiple SymbolComposite class identities.
    base = getattr(symbol, "base", None)
    if base is not None and not hasattr(symbol, "verts"):
        symbol = base

    verts_local = symbol.verts - np.array(symbol.anchor, dtype=float)
    codes = np.asarray(symbol.codes, dtype=int)
    if len(codes):
        codes = codes.copy()
        codes[0] = Path.MOVETO
    return Path(verts_local, codes)


def _path_in_display(ax, symbol: SymbolPath | SymbolComposite, x: float, y: float, *, size_pt: float) -> Path:
    """Return a matplotlib Path whose vertices are in DISPLAY coordinates (pixels)."""
    base = getattr(symbol, "base", None)
    if base is not None and not hasattr(symbol, "verts"):
        symbol = base

    fig = ax.figure
    s_px = (float(size_pt) * fig.dpi) / 72.0  # points -> pixels
    scale = s_px / 2.0

    p = _mpl_path(symbol)
    verts = p.vertices.copy() * scale

    x_disp, y_disp = ax.transData.transform((x, y))
    verts[:, 0] += x_disp
    verts[:, 1] += y_disp

    return Path(verts, p.codes)


def _draw_symbol_display(
    ax,
    symbol: SymbolPath,
    x: float,
    y: float,
    *,
    color: str,
    size_pt: float,
    linewidth: float,
    zorder: int,
    fill: bool,
    facecolor: str | None,
):
    """Draw `symbol` centered at (x,y) with geometry scaled in *points*.

    We render in display coordinates (pixels) using IdentityTransform so that:
    - glyph shape stays stable under log-x / aspect changes
    - overlay glyphs (e.g., NR arrow) do NOT shrink the base glyph

    `size_pt` is interpreted as ~total glyph height for local coords spanning [-1,+1].
    """
    fig = ax.figure

    # points -> pixels
    s_px = (float(size_pt) * fig.dpi) / 72.0
    scale = s_px / 2.0  # local coords ~[-1,+1]

    p = _mpl_path(symbol)
    verts = p.vertices.copy() * scale

    x_disp, y_disp = ax.transData.transform((x, y))
    verts[:, 0] += x_disp
    verts[:, 1] += y_disp

    disp_path = Path(verts, p.codes)

    fc = (facecolor if facecolor is not None else color) if fill else "none"
    ec = color

    patch = PathPatch(
        disp_path,
        facecolor=fc,
        edgecolor=ec,
        fill=fill,
        linewidth=linewidth,
        zorder=zorder,
        transform=IdentityTransform(),
        clip_on=True,
    )
    ax.add_artist(patch)
    return patch


class SymbolArtist(Artist):
    """Artist that draws a SymbolPath or SymbolComposite in display space.

    Unlike one-off PathPatches, this recomputes display-space paths on every draw,
    so symbols remain correctly positioned after `tight_layout`, legend changes,
    window resizes, or any other layout/transform updates.
    """

    def __init__(
        self,
        ax,
        symbol: SymbolPath | SymbolComposite,
        x: float,
        y: float,
        *,
        color: str = "black",
        size_pt: float = 12.0,
        linewidth: float = 2.0,
        overlay_linewidth: float | None = None,
        zorder: int = 4,
        fill: bool = False,
        facecolor: str | None = None,
    ):
        super().__init__()
        self.ax = ax
        self.symbol = symbol
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.size_pt = float(size_pt)
        self.linewidth = float(linewidth)
        if overlay_linewidth is None:
            overlay_linewidth = self.linewidth * 0.65
        self.overlay_linewidth = float(overlay_linewidth)
        self._fill = bool(fill)
        self._facecolor = facecolor
        self.set_zorder(zorder)

        # We keep internal PathPatches and draw them ourselves.
        self._base_patch = PathPatch(
            Path(np.zeros((1, 2)), [Path.MOVETO]),
            facecolor="none",
            edgecolor=self.color,
            fill=False,
            linewidth=self.linewidth,
            transform=IdentityTransform(),
            clip_on=True,
        )
        self._overlay_patches: list[PathPatch] = []

        # Clip to the axes patch so glyphs don't spill outside the plotting area.
        self._base_patch.set_clip_path(ax.patch)

        overlays = getattr(symbol, "overlays", None)
        if overlays is not None:
            for _ in overlays:
                p = PathPatch(
                    Path(np.zeros((1, 2)), [Path.MOVETO]),
                    facecolor="none",
                    edgecolor=self.color,
                    fill=False,
                    linewidth=self.overlay_linewidth,
                    transform=IdentityTransform(),
                    clip_on=True,
                )
                p.set_clip_path(ax.patch)
                self._overlay_patches.append(p)

    # --- convenience setters for notebook tuning ---
    def set_data(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
        self.stale = True

    def set_symbol(self, symbol: SymbolPath | SymbolComposite):
        self.symbol = symbol
        # Rebuild overlay patch list if overlay count changes.
        self._overlay_patches = []
        overlays = getattr(symbol, "overlays", None)
        if overlays is not None:
            for _ in overlays:
                p = PathPatch(
                    Path(np.zeros((1, 2)), [Path.MOVETO]),
                    facecolor="none",
                    edgecolor=self.color,
                    fill=False,
                    linewidth=self.overlay_linewidth,
                    transform=IdentityTransform(),
                    clip_on=True,
                )
                p.set_clip_path(self.ax.patch)
                self._overlay_patches.append(p)
        self.stale = True

    def set_style(self, *, color: str | None = None, size_pt: float | None = None, linewidth: float | None = None, overlay_linewidth: float | None = None, fill: bool | None = None, facecolor: str | None = None):
        if color is not None:
            self.color = color
        if size_pt is not None:
            self.size_pt = float(size_pt)
        if linewidth is not None:
            self.linewidth = float(linewidth)
            if overlay_linewidth is None:
                self.overlay_linewidth = self.linewidth * 0.65
        if overlay_linewidth is not None:
            self.overlay_linewidth = float(overlay_linewidth)
        if fill is not None:
            self._fill = bool(fill)
        if facecolor is not None:
            self._facecolor = facecolor
        self.stale = True

    def draw(self, renderer):
        ax = self.ax

        sym_obj = self.symbol
        base = getattr(sym_obj, "base", None)
        if base is None:
            base = sym_obj
            overlays = ()
        else:
            overlays = tuple(getattr(sym_obj, "overlays", ()))

        base_path = _path_in_display(ax, base, self.x, self.y, size_pt=self.size_pt)
        self._base_patch.set_path(base_path)
        self._base_patch.set_edgecolor(self.color)
        self._base_patch.set_linewidth(self.linewidth)

        fc = (self._facecolor if self._facecolor is not None else self.color) if self._fill else "none"
        self._base_patch.set_facecolor(fc)
        self._base_patch.set_fill(self._fill)

        self._base_patch.draw(renderer)

        # Overlays (stroke only)
        for i, ov in enumerate(overlays):
            if i >= len(self._overlay_patches):
                break
            ov_path = _path_in_display(ax, ov, self.x, self.y, size_pt=self.size_pt)
            p = self._overlay_patches[i]
            p.set_path(ov_path)
            p.set_edgecolor(self.color)
            p.set_linewidth(self.overlay_linewidth)
            p.set_facecolor("none")
            p.set_fill(False)
            p.draw(renderer)

        self.stale = False


def draw_symbol(
    ax,
    symbol: SymbolPath | SymbolComposite,
    x: float,
    y: float,
    *,
    cfg=DEFAULT_RENDER_CONFIG,
    color: str = "black",
    size_pt: float | None = None,
    linewidth: float | None = None,
    overlay_linewidth: float | None = None,
    zorder: int = 4,
    fill: bool = False,
    facecolor: str | None = None,
):
    """Draw a SymbolPath or SymbolComposite at data coords (x, y).

    Important: We draw in display space (pixels) so glyph sizing is in points and
    overlays (NR arrows) do not change the base glyph size.
    """
    # Fallbacks: pull from cfg.style when not explicitly provided.
    # Note: generic draw_symbol uses air_size_pt as its default size.
    if size_pt is None:
        size_pt = cfg.style.air_size_pt
    if linewidth is None:
        linewidth = cfg.style.linewidth
    if overlay_linewidth is None:
        overlay_linewidth = getattr(cfg.style, "nr_linewidth", float(linewidth) * 0.65)

    if getattr(symbol, "base", None) is not None:
        # Base first (can be filled)
        base_artist = _draw_symbol_display(
            ax,
            symbol.base,
            x,
            y,
            color=color,
            size_pt=size_pt,
            linewidth=linewidth,
            zorder=zorder,
            fill=fill,
            facecolor=facecolor,
        )
        # Overlays (always stroke-only)
        overlay_artists = []
        for ov in symbol.overlays:
            overlay_artists.append(
                _draw_symbol_display(
                    ax,
                    ov,
                    x,
                    y,
                    color=color,
                    size_pt=size_pt,
                    linewidth=float(overlay_linewidth),
                    zorder=zorder,
                    fill=False,
                    facecolor=None,
                )
            )
        return (base_artist, *overlay_artists)

    return _draw_symbol_display(
        ax,
        symbol,
        x,
        y,
        color=color,
        size_pt=size_pt,
        linewidth=linewidth,
        zorder=zorder,
        fill=fill,
        facecolor=facecolor,
    )


def add_symbol(
    ax,
    symbol: SymbolPath | SymbolComposite,
    x: float,
    y: float,
    *,
    cfg=DEFAULT_RENDER_CONFIG,
    color: str = "black",
    size_pt: float | None = None,
    linewidth: float | None = None,
    overlay_linewidth: float | None = None,
    zorder: int = 4,
    fill: bool = False,
    facecolor: str | None = None,
):
    """Add a layout-safe symbol artist to the axes and return it."""
    # Fallbacks: pull from cfg.style when not explicitly provided.
    # Note: generic add_symbol uses air_size_pt as its default size.
    if size_pt is None:
        size_pt = cfg.style.air_size_pt
    if linewidth is None:
        linewidth = cfg.style.linewidth
    if overlay_linewidth is None:
        overlay_linewidth = getattr(cfg.style, "nr_linewidth", float(linewidth) * 0.65)

    art = SymbolArtist(
        ax,
        symbol,
        x,
        y,
        color=color,
        size_pt=float(size_pt),
        linewidth=float(linewidth),
        overlay_linewidth=float(overlay_linewidth),
        zorder=zorder,
        fill=fill,
        facecolor=facecolor,
    )
    ax.add_artist(art)
    return art


# --- Convenience wrappers ---
def draw_air_symbol(
    ax,
    symbol,
    x,
    y,
    *,
    cfg=DEFAULT_RENDER_CONFIG,
    color="black",
    zorder=4,
    fill: bool = False,
    facecolor: str | None = None,
):
    return draw_symbol(
        ax,
        symbol,
        x,
        y,
        cfg=cfg,
        color=color,
        size_pt=cfg.style.air_size_pt,
        zorder=zorder,
        fill=fill,
        facecolor=facecolor,
    )


def draw_bone_symbol(
    ax,
    symbol,
    x,
    y,
    *,
    cfg=DEFAULT_RENDER_CONFIG,
    color="black",
    zorder=4,
    fill: bool = False,
    facecolor: str | None = None,
):
    return draw_symbol(
        ax,
        symbol,
        x,
        y,
        cfg=cfg,
        color=color,
        size_pt=cfg.style.bone_size_pt,
        zorder=zorder,
        fill=fill,
        facecolor=facecolor,
    )


def add_air_symbol(ax, symbol, x, y, *, cfg=DEFAULT_RENDER_CONFIG, color="black", zorder=4, fill: bool = False, facecolor: str | None = None):
    return add_symbol(
        ax,
        symbol,
        x,
        y,
        cfg=cfg,
        color=color,
        size_pt=cfg.style.air_size_pt,
        zorder=zorder,
        fill=fill,
        facecolor=facecolor,
    )


def add_bone_symbol(ax, symbol, x, y, *, cfg=DEFAULT_RENDER_CONFIG, color="black", zorder=4, fill: bool = False, facecolor: str | None = None):
    return add_symbol(
        ax,
        symbol,
        x,
        y,
        cfg=cfg,
        color=color,
        size_pt=cfg.style.bone_size_pt,
        zorder=zorder,
        fill=fill,
        facecolor=facecolor,
    )


# Backwards compat for older notebooks
draw_symbol_cfg = draw_symbol