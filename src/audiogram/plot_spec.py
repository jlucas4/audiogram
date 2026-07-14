"""Export the audiogram plot geometry as a backend-neutral JSON spec.

Motivation: any renderer other than this library's own Matplotlib backend
(e.g. a web/SVG front-end) must not re-implement audiogram geometry — axis
mappings, symbol conventions, and glyph shapes drift when duplicated. This
module serializes the *same* frozen geometry that `render_mpl` draws:
glyph paths come verbatim from `symbols.get_symbol`, axes from the
`plot_mpl` presets. A consumer renders the spec; it never computes clinical
geometry itself.

Coordinate conventions
----------------------
- Glyph paths are in LOCAL units spanning roughly [-1, +1] with (0, 0) at the
  threshold point, y increasing UP (mathematical convention) in `verts`.
- `svg_d` is the same path pre-flipped to SVG's y-down convention, so a web
  client can use it directly:  <path d={svg_d} transform="translate(x y) scale(s)">
- On-screen size: a glyph drawn at `size_pt` points is scaled by
  s = size_px / 2, where size_px = size_pt * dpi / 72 (Matplotlib) or
  size_pt * 96 / 72 (CSS pixels). This matches `render_mpl._path_in_display`.
- The frequency axis is log10-scaled; the dB axis is linear and INVERTED
  (`ylim` is given as (bottom, top) = (max_db, min_db), matching Matplotlib).

Example
-------
Imported as a submodule (like ``plot_mpl``, so ``import audiogram`` stays free
of Matplotlib)::

    from audiogram.plot_spec import build_plot_spec
    spec = build_plot_spec(style="ansi")
    spec["symbols"]["air_right_unmasked"]["base"]["svg_d"]  # 'M 1.000000 ...'
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from . import symbols as sym
from .plot_mpl import (
    AXES_PRESETS,
    PLOT_STYLE_PRESETS,
    get_plot_style,
)

PLOT_SPEC_VERSION = "1.0"

# Glyph-key semantics (mirrors the table at the bottom of symbols.py).
SYMBOL_SEMANTICS: dict[str, dict[str, dict[str, str]]] = {
    "air": {
        "right": {"unmasked": "O", "masked": "TRI"},
        "left": {"unmasked": "X", "masked": "SQ"},
    },
    "bone": {
        "right": {"unmasked": "CHEV_R", "masked": "BRACK_R"},
        "left": {"unmasked": "CHEV_L", "masked": "BRACK_L"},
    },
}


def _path_to_dict(path: sym.SymbolPath, *, flip_y_for_svg: bool) -> dict[str, Any]:
    """Serialize one SymbolPath: raw verts/codes plus an SVG `d` string."""
    verts = np.asarray(path.verts, dtype=float)
    codes = np.asarray(path.codes, dtype=int)

    parts: list[str] = []
    for (x, y), code in zip(verts, codes):
        ys = -y if flip_y_for_svg else y
        if code == sym.Cmd.MOVETO:
            parts.append(f"M {x:.6f} {ys:.6f}")
        elif code == sym.Cmd.LINETO:
            parts.append(f"L {x:.6f} {ys:.6f}")
        elif code == sym.Cmd.CLOSE:
            parts.append("Z")  # SVG Z ignores coordinates, as does CLOSEPOLY
        else:  # pragma: no cover - Cmd only defines the three above
            raise ValueError(f"Unknown path code {code!r}")

    return {
        "verts": [[float(x), float(y)] for x, y in verts],
        "codes": [int(c) for c in codes],
        "svg_d": " ".join(parts),
        "anchor": [float(path.anchor[0]), float(path.anchor[1])],
    }


def _symbol_entry(
    *, kind: str, ear: str, masked: bool, nr: bool, cfg: sym.AudiogramRenderConfig
) -> dict[str, Any]:
    composite = sym.get_symbol(kind=kind, ear=ear, masked=masked, nr=nr, cfg=cfg)
    glyph_key = SYMBOL_SEMANTICS[kind][ear]["masked" if masked else "unmasked"]
    return {
        "kind": kind,
        "ear": ear,
        "masked": masked,
        "nr": nr,
        "glyph": glyph_key,
        "size_pt": cfg.style.bone_size_pt if kind == "bone" else cfg.style.air_size_pt,
        "base": _path_to_dict(composite.base, flip_y_for_svg=True),
        "overlays": [
            _path_to_dict(p, flip_y_for_svg=True) for p in composite.overlays
        ],
    }


def build_plot_spec(
    style: str | None = None,
    *,
    cfg: sym.AudiogramRenderConfig = sym.DEFAULT_RENDER_CONFIG,
) -> dict[str, Any]:
    """Build a JSON-serializable plot spec for external renderers.

    Parameters
    ----------
    style
        A plot-style preset name (see ``PLOT_STYLE_PRESETS``; e.g. "ansi",
        "mei"). None uses the default style.
    cfg
        Render config supplying symbol geometry and point sizes.

    Returns
    -------
    dict with keys: ``spec_version``, ``symbol_set_version``,
    ``library_version``, ``style_name``, ``axes``, ``colors``, ``sizing``,
    ``symbols``, ``text_markers``, ``lines``, ``semantics``.
    """
    from . import __version__

    st = get_plot_style(style)
    axes = asdict(st.axes)
    # Nested dataclass fields arrive as plain dicts via asdict; tuples become
    # lists on json round-trip anyway, so normalize now for a stable shape.
    axes = {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in axes.items()
    }
    axes["shade_bands"] = [list(b) for b in (st.axes.shade_bands or ())]
    axes["xscale"] = "log"
    axes["y_inverted"] = True

    symbols: dict[str, dict[str, Any]] = {}
    for kind in ("air", "bone"):
        for ear in ("right", "left"):
            for masked in (False, True):
                for nr in (False, True):
                    name = (
                        f"{kind}_{ear}_{'masked' if masked else 'unmasked'}"
                        f"{'_nr' if nr else ''}"
                    )
                    symbols[name] = _symbol_entry(
                        kind=kind, ear=ear, masked=masked, nr=nr, cfg=cfg
                    )

    return {
        "spec_version": PLOT_SPEC_VERSION,
        "symbol_set_version": sym.SYMBOL_SET_VERSION,
        "library_version": __version__,
        "style_name": style or "default",
        "available_styles": sorted(PLOT_STYLE_PRESETS),
        "available_axes_presets": sorted(AXES_PRESETS),
        "axes": axes,
        "colors": {
            "right": st.color_right,
            "left": st.color_left,
            "soundfield": st.color_soundfield,
        },
        "sizing": {
            "air_size_pt": cfg.style.air_size_pt,
            "bone_size_pt": cfg.style.bone_size_pt,
            "linewidth_pt": cfg.style.linewidth,
            # Matches render_mpl._path_in_display: local verts * (size_px / 2),
            # size_px = size_pt * dpi / 72 (use dpi=96 for CSS pixels).
            "scale_rule": "screen = local * (size_pt * dpi / 72) / 2",
            "svg_y_down": True,
        },
        "symbols": symbols,
        "text_markers": {
            "soundfield": {"label": "S", "fontsize": st.text_marker_fontsize,
                           "fontweight": st.text_marker_fontweight},
            "ci": {"label": "CI", "fontsize": st.text_marker_fontsize,
                   "fontweight": st.text_marker_fontweight},
        },
        "lines": {
            # Mirrors plot_mpl.plot_ear: air points connect in frequency order
            # excluding NR; bone symbols are never connected; text-marker
            # series (S/CI) connect all points.
            "connect_air": True,
            "connect_air_excludes_nr": True,
            "connect_bone": False,
            "connect_text_markers": True,
        },
        "semantics": {
            "glyphs": SYMBOL_SEMANTICS,
            "nr_arrow_direction": {"right": "down-left", "left": "down-right"},
        },
    }
