"""Tests for the backend-neutral plot spec export."""
import json
import re

import numpy as np
import pytest

from audiogram import symbols as sym
from audiogram.plot_mpl import PLOT_STYLE_PRESETS, get_plot_style
from audiogram.plot_spec import PLOT_SPEC_VERSION, SYMBOL_SEMANTICS, build_plot_spec


ALL_COMBOS = [
    (kind, ear, masked, nr)
    for kind in ("air", "bone")
    for ear in ("right", "left")
    for masked in (False, True)
    for nr in (False, True)
]


def _combo_name(kind, ear, masked, nr):
    return f"{kind}_{ear}_{'masked' if masked else 'unmasked'}{'_nr' if nr else ''}"


def _parse_svg_d(d):
    """Parse an M/L/Z-only SVG path back into (verts, codes)."""
    verts, codes = [], []
    tokens = re.findall(r"([MLZ])((?:\s+-?\d+\.?\d*){0,2})", d)
    for cmd, nums in tokens:
        if cmd == "Z":
            codes.append(int(sym.Cmd.CLOSE))
            continue
        x, y = (float(v) for v in nums.split())
        verts.append([x, y])
        codes.append(int(sym.Cmd.MOVETO if cmd == "M" else sym.Cmd.LINETO))
    return np.array(verts, dtype=float), codes


class TestSpecShape:
    def test_json_serializable(self):
        spec = build_plot_spec()
        round_tripped = json.loads(json.dumps(spec))
        assert round_tripped["spec_version"] == PLOT_SPEC_VERSION

    def test_every_symbol_combo_present(self):
        spec = build_plot_spec()
        for combo in ALL_COMBOS:
            assert _combo_name(*combo) in spec["symbols"]
        assert len(spec["symbols"]) == len(ALL_COMBOS)

    def test_axes_from_preset(self):
        spec = build_plot_spec(style="mei")
        assert spec["axes"]["xscale"] == "log"
        assert spec["axes"]["y_inverted"] is True
        # MEI preset: inverted ylim (bottom, top) and its distinctive ticks
        assert spec["axes"]["ylim"] == [125.0, -20.0]
        assert 12500 in spec["axes"]["xticks"]
        assert spec["axes"]["shade_bands"] == [[-10.0, 25.0, "#dbe9f6", 0.55]]

    def test_colors_follow_style(self):
        assert build_plot_spec(style="ansi")["colors"]["right"] == "red"
        assert build_plot_spec(style="ansi_bw")["colors"]["right"] == "black"

    def test_all_named_styles_build(self):
        for name in PLOT_STYLE_PRESETS:
            spec = build_plot_spec(style=name)
            st = get_plot_style(name)
            assert spec["colors"]["left"] == st.color_left

    def test_unknown_style_raises(self):
        with pytest.raises(KeyError):
            build_plot_spec(style="nope")


class TestGoldenGeometry:
    """The spec must carry the EXACT vertices the mpl renderer draws."""

    @pytest.mark.parametrize("kind,ear,masked,nr", ALL_COMBOS)
    def test_verts_match_get_symbol(self, kind, ear, masked, nr):
        spec = build_plot_spec()
        entry = spec["symbols"][_combo_name(kind, ear, masked, nr)]
        composite = sym.get_symbol(kind=kind, ear=ear, masked=masked, nr=nr)

        np.testing.assert_allclose(
            np.array(entry["base"]["verts"]), composite.base.verts, atol=0
        )
        assert entry["base"]["codes"] == [int(c) for c in composite.base.codes]
        assert len(entry["overlays"]) == len(composite.overlays)
        for got, want in zip(entry["overlays"], composite.overlays):
            np.testing.assert_allclose(np.array(got["verts"]), want.verts, atol=0)

    @pytest.mark.parametrize("kind,ear,masked,nr", ALL_COMBOS)
    def test_svg_d_is_y_flipped_verts(self, kind, ear, masked, nr):
        spec = build_plot_spec()
        entry = spec["symbols"][_combo_name(kind, ear, masked, nr)]
        composite = sym.get_symbol(kind=kind, ear=ear, masked=masked, nr=nr)

        d_verts, d_codes = _parse_svg_d(entry["base"]["svg_d"])
        want = composite.base.verts.copy()
        want[:, 1] *= -1.0  # svg y-down
        # CLOSE vertices are dropped from the d string (Z carries no coords)
        keep = [i for i, c in enumerate(composite.base.codes) if c != sym.Cmd.CLOSE]
        np.testing.assert_allclose(d_verts, want[keep], atol=1e-6)
        assert d_codes == [int(c) for c in composite.base.codes]

    def test_nr_adds_overlay(self):
        spec = build_plot_spec()
        assert spec["symbols"]["air_right_unmasked"]["overlays"] == []
        assert len(spec["symbols"]["air_right_unmasked_nr"]["overlays"]) == 1

    def test_nr_arrow_direction(self):
        """Right-ear NR arrow tip must sit down-LEFT of its tail (in y-up
        coords: negative x and negative y displacement); left-ear down-RIGHT."""
        spec = build_plot_spec()
        for ear, x_sign in (("right", -1), ("left", 1)):
            arrow = spec["symbols"][f"air_{ear}_unmasked_nr"]["overlays"][0]
            tail, tip = np.array(arrow["verts"][0]), np.array(arrow["verts"][1])
            dx, dy = tip - tail
            assert np.sign(dx) == x_sign
            assert dy < 0

    def test_glyph_semantics(self):
        spec = build_plot_spec()
        assert spec["semantics"]["glyphs"] == SYMBOL_SEMANTICS
        assert spec["symbols"]["air_right_unmasked"]["glyph"] == "O"
        assert spec["symbols"]["air_left_unmasked"]["glyph"] == "X"
        assert spec["symbols"]["bone_right_masked"]["glyph"] == "BRACK_R"

    def test_sizes_by_kind(self):
        spec = build_plot_spec()
        assert (spec["symbols"]["air_right_unmasked"]["size_pt"]
                == spec["sizing"]["air_size_pt"])
        assert (spec["symbols"]["bone_right_unmasked"]["size_pt"]
                == spec["sizing"]["bone_size_pt"])
