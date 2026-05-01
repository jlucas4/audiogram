# --- imports ---
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from functools import lru_cache
from enum import IntEnum

import numpy as np


# Public entry points:
# - get_symbol(...)
# - clear_symbol_caches()
#
# Rendering lives in render_mpl.py (matplotlib) and other backends.


class Cmd(IntEnum):
    MOVETO = 1
    LINETO = 2
    CLOSE = 79


@dataclass(frozen=True)
class NRStartConfig:
    """Per-glyph attachment point (in local glyph coords) for NR arrow tail."""

    O: tuple[float, float] = (-0.80, -0.60)
    X: tuple[float, float] = (0.6, -0.6)
    TRI: tuple[float, float] = (-1.0, -1.0)
    SQ: tuple[float, float] = (1.0, -1.0)
    CHEV_R: tuple[float, float] = (-0.30, -1.1)
    CHEV_L: tuple[float, float] = (0.30, -1.1)
    BRACK_R: tuple[float, float] = (-0.30, -1.2)
    BRACK_L: tuple[float, float] = (0.30, -1.2)

    def as_dict(self) -> Mapping[str, tuple[float, float]]:
        return {
            "O": self.O,
            "X": self.X,
            "TRI": self.TRI,
            "SQ": self.SQ,
            "CHEV_R": self.CHEV_R,
            "CHEV_L": self.CHEV_L,
            "BRACK_R": self.BRACK_R,
            "BRACK_L": self.BRACK_L,
        }


@dataclass(frozen=True)
class NRScaleConfig:
    """Optional per-glyph multiplier for NR arrow geometry.

    Useful when bone glyph extents (e.g. chevron_tip=1.75) are larger than air
    glyph extents (~1.0), so a single global stem/head length can look too small.
    """

    O: float = 1.00
    X: float = 1.00
    TRI: float = 1.00
    SQ: float = 1.00
    CHEV_R: float = 1.00
    CHEV_L: float = 1.00
    BRACK_R: float = 1.00
    BRACK_L: float = 1.00

    def as_dict(self) -> Mapping[str, float]:
        return {
            "O": self.O,
            "X": self.X,
            "TRI": self.TRI,
            "SQ": self.SQ,
            "CHEV_R": self.CHEV_R,
            "CHEV_L": self.CHEV_L,
            "BRACK_R": self.BRACK_R,
            "BRACK_L": self.BRACK_L,
        }


# --- data structures ---
@dataclass
class SymbolPath:
    """Geometry for a symbol in LOCAL coordinates.

    - verts: (N, 2) array in "symbol space"
    - codes: (N,) path codes (MOVETO, LINETO, CLOSE)
    - anchor: where local (0,0) should sit conceptually
    """

    verts: np.ndarray
    codes: np.ndarray
    anchor: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class SymbolComposite:
    """A symbol composed of a base glyph plus optional overlay glyphs.

    We use this for NR (no response) so the base glyph keeps its size even when
    an arrow is present. Rendering backends should draw `base` first, then each
    overlay in `overlays` using the same marker sizing.
    """

    base: SymbolPath
    overlays: tuple[SymbolPath, ...] = ()


@dataclass(frozen=True)
class SymbolGeometryConfig:
    # bone chevrons
    chevron_tip: float = 1.75
    chevron_gap: float = 0.30
    chevron_half_height: float = 1.2

    # bone brackets
    bracket_half_height: float = 1.2
    bracket_depth: float = 1.00

    # air circle fidelity (geometry)
    circle_n: int = 36

    # NR arrow geometry (local coords)
    nr_stem_len: float = 1.0
    nr_head_len: float = 0.5
    nr_head_angle_deg: float = 32.0

    # NR tail attachment points (per base glyph)
    nr_start: NRStartConfig = NRStartConfig()

    # Optional per-glyph multiplier for NR arrow geometry
    nr_scale: NRScaleConfig = NRScaleConfig()


@dataclass(frozen=True)
class RenderStyleConfig:
    """Per-plot styling for rendering.

    Sizes are in *points* (used by renderers) so glyphs are geometrically stable
    under figure resizing and independent of data scaling.
    """

    # Air markers (O/X/^/s)
    air_size_pt: float = 14.0

    # Bone markers (chevrons/brackets)
    bone_size_pt: float = 18.0

    # Stroke styling
    linewidth: float = 1.5


@dataclass(frozen=True)
class AudiogramRenderConfig:
    geom: SymbolGeometryConfig = SymbolGeometryConfig()
    style: RenderStyleConfig = RenderStyleConfig()


SYMBOL_SET_VERSION = "1.0"

# Geometry is frozen/versioned (SymbolGeometryConfig). Style is per-plot (RenderStyleConfig).
DEFAULT_GEOM = SymbolGeometryConfig()
DEFAULT_STYLE = RenderStyleConfig()
DEFAULT_RENDER_CONFIG = AudiogramRenderConfig(geom=DEFAULT_GEOM, style=DEFAULT_STYLE)

TINY_STYLE = RenderStyleConfig(
    air_size_pt=7.0,
    bone_size_pt=9.0,
    linewidth=1.0,
)

TINY_RENDER_CONFIG = AudiogramRenderConfig(
    geom=DEFAULT_GEOM,
    style=TINY_STYLE,
)

# --- symbol builders (air) ---
def _symbol_x(size: float = 1.0) -> SymbolPath:
    a = float(size)
    verts = np.array(
        [
            [-a, -a],
            [a, a],
            [-a, a],
            [a, -a],
        ],
        dtype=float,
    )
    codes = np.array([Cmd.MOVETO, Cmd.LINETO, Cmd.MOVETO, Cmd.LINETO], dtype=int)
    return SymbolPath(verts, codes)


def _symbol_square(size: float = 1.0) -> SymbolPath:
    a = float(size)
    verts = np.array(
        [
            [-a, -a],
            [a, -a],
            [a, a],
            [-a, a],
            [-a, -a],
        ],
        dtype=float,
    )
    codes = np.array([Cmd.MOVETO, Cmd.LINETO, Cmd.LINETO, Cmd.LINETO, Cmd.CLOSE], dtype=int)
    return SymbolPath(verts, codes)


def _symbol_triangle_up(size: float = 1.0) -> SymbolPath:
    a = float(size)
    verts = np.array(
        [
            [0.0, a],
            [-a, -a],
            [a, -a],
            [0.0, a],
        ],
        dtype=float,
    )
    codes = np.array([Cmd.MOVETO, Cmd.LINETO, Cmd.LINETO, Cmd.CLOSE], dtype=int)
    return SymbolPath(verts, codes)


def _symbol_circle(radius: float = 1.0, n: int = 36) -> SymbolPath:
    r = float(radius)
    t = np.linspace(0, 2 * np.pi, int(n), endpoint=False)
    xy = np.column_stack([r * np.cos(t), r * np.sin(t)])
    verts = np.vstack([xy, xy[0]])  # close
    codes = np.array([Cmd.MOVETO] + [Cmd.LINETO] * (len(verts) - 2) + [Cmd.CLOSE], dtype=int)
    return SymbolPath(verts, codes)


# --- symbol builders (bone) ---
def _symbol_bone_unmasked_right(geom: SymbolGeometryConfig) -> SymbolPath:
    """Right ear unmasked bone: '<' on the LEFT side of anchor."""
    tip = geom.chevron_tip
    gap = geom.chevron_gap
    hh = geom.chevron_half_height

    x_tip = -tip
    x_arm = -gap

    verts = np.array([[x_arm, -hh], [x_tip, 0.0], [x_arm, +hh]], dtype=float)
    codes = np.array([Cmd.MOVETO, Cmd.LINETO, Cmd.LINETO], dtype=int)
    return SymbolPath(verts=verts, codes=codes, anchor=(0.0, 0.0))


def _symbol_bone_unmasked_left(geom: SymbolGeometryConfig) -> SymbolPath:
    """Left ear unmasked bone: '>' on the RIGHT side of anchor."""
    tip = geom.chevron_tip
    gap = geom.chevron_gap
    hh = geom.chevron_half_height

    x_tip = +tip
    x_arm = +gap

    verts = np.array([[x_arm, -hh], [x_tip, 0.0], [x_arm, +hh]], dtype=float)
    codes = np.array([Cmd.MOVETO, Cmd.LINETO, Cmd.LINETO], dtype=int)
    return SymbolPath(verts=verts, codes=codes, anchor=(0.0, 0.0))


def _symbol_bone_masked_right(geom: SymbolGeometryConfig) -> SymbolPath:
    """Right ear masked bone: '[' open to the RIGHT; bracket extends negative x."""
    hh = geom.bracket_half_height
    d = geom.bracket_depth
    g = geom.chevron_gap  # universal clearance gap

    x_open = -g
    x_bar = -(g + d)

    verts = np.array(
        [
            [x_open, +hh],
            [x_bar, +hh],
            [x_bar, +hh],
            [x_bar, -hh],
            [x_bar, -hh],
            [x_open, -hh],
        ],
        dtype=float,
    )

    codes = np.array(
        [
            Cmd.MOVETO,
            Cmd.LINETO,
            Cmd.MOVETO,
            Cmd.LINETO,
            Cmd.MOVETO,
            Cmd.LINETO,
        ],
        dtype=int,
    )
    return SymbolPath(verts=verts, codes=codes, anchor=(0.0, 0.0))


def _symbol_bone_masked_left(geom: SymbolGeometryConfig) -> SymbolPath:
    """Left ear masked bone: ']' open to the LEFT; bracket extends positive x."""
    base = _symbol_bone_masked_right(geom)
    verts = base.verts.copy()
    verts[:, 0] *= -1.0
    return SymbolPath(verts=verts, codes=base.codes, anchor=base.anchor)


# --- symbol set builder (per-geometry, cached) ---
@lru_cache(maxsize=32)
def _symbol_set(geom: SymbolGeometryConfig) -> Mapping[str, SymbolPath]:
    """Build (and cache) the base glyphs for a given geometry config."""
    return {
        # air
        "O": _symbol_circle(radius=1.0, n=geom.circle_n),
        "X": _symbol_x(size=1.0),
        "TRI": _symbol_triangle_up(size=1.0),
        "SQ": _symbol_square(size=1.0),
        # bone
        "CHEV_R": _symbol_bone_unmasked_right(geom),
        "CHEV_L": _symbol_bone_unmasked_left(geom),
        "BRACK_R": _symbol_bone_masked_right(geom),
        "BRACK_L": _symbol_bone_masked_left(geom),
    }


def _symbol_nr_arrow(
    geom: SymbolGeometryConfig,
    *,
    ear: str,
    start=(0.0, 0.0),
    scale: float = 1.0,
) -> SymbolPath:
    ear = ear.lower()
    if ear not in ("right", "left"):
        raise ValueError("ear must be 'right' or 'left'")

    # Right ear NR: down-left; Left ear NR: down-right
    d = (
        (np.array([-1, -1], dtype=float) / np.sqrt(2))
        if ear == "right"
        else (np.array([1, -1], dtype=float) / np.sqrt(2))
    )

    tail = np.array(start, dtype=float)
    stem_len = geom.nr_stem_len * float(scale)
    head_len = geom.nr_head_len * float(scale)

    tip = tail + stem_len * d

    theta = np.deg2rad(geom.nr_head_angle_deg)
    Rpos = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Rneg = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])

    back = -d * head_len
    wing1 = tip + Rpos @ back
    wing2 = tip + Rneg @ back

    verts = np.vstack([tail, tip, tip, wing1, tip, wing2])
    codes = np.array(
        [
            Cmd.MOVETO,
            Cmd.LINETO,
            Cmd.MOVETO,
            Cmd.LINETO,
            Cmd.MOVETO,
            Cmd.LINETO,
        ],
        dtype=int,
    )

    return SymbolPath(verts=verts, codes=codes, anchor=(0.0, 0.0))


def _get_air_base(*, geom: SymbolGeometryConfig, ear: str, masked: bool):
    s = _symbol_set(geom)
    ear = ear.lower()
    if masked:
        return ("TRI", s["TRI"]) if ear == "right" else ("SQ", s["SQ"])
    return ("O", s["O"]) if ear == "right" else ("X", s["X"])


def _get_bone_base(*, geom: SymbolGeometryConfig, ear: str, masked: bool):
    s = _symbol_set(geom)
    ear = ear.lower()
    if masked:
        return ("BRACK_R", s["BRACK_R"]) if ear == "right" else ("BRACK_L", s["BRACK_L"])
    return ("CHEV_R", s["CHEV_R"]) if ear == "right" else ("CHEV_L", s["CHEV_L"])


# Semantics:
# air unmasked: right=O, left=X
# air masked: right=TRI, left=SQ
# bone unmasked: right=<, left=>
# bone masked: right=[, left=]
# NR: arrow down-left for right, down-right for left


@lru_cache(maxsize=512)
def _get_symbol_cached(
    geom: SymbolGeometryConfig,
    kind: str,
    ear: str,
    masked: bool,
    nr: bool,
) -> SymbolComposite:
    kind_l = kind.lower()
    if kind_l == "air":
        key, base = _get_air_base(geom=geom, ear=ear, masked=masked)
    elif kind_l == "bone":
        key, base = _get_bone_base(geom=geom, ear=ear, masked=masked)
    else:
        raise ValueError("kind must be 'air' or 'bone'")

    if not nr:
        return SymbolComposite(base=base, overlays=())

    start = geom.nr_start.as_dict()[key]
    scale = geom.nr_scale.as_dict().get(key, 1.0)
    arrow = _symbol_nr_arrow(geom, ear=ear, start=start, scale=scale)
    return SymbolComposite(base=base, overlays=(arrow,))


def get_symbol(
    *,
    kind: str,
    ear: str,
    masked: bool,
    nr: bool,
    cfg: AudiogramRenderConfig = DEFAULT_RENDER_CONFIG,
) -> SymbolComposite:
    return _get_symbol_cached(cfg.geom, kind, ear, masked, nr)



def clear_symbol_caches() -> None:
    """Clear internal caches (useful during notebook iteration)."""
    _symbol_set.cache_clear()
    _get_symbol_cached.cache_clear()
