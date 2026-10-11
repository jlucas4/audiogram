"""Tri-state ``tested`` on EarAudiogram and the DNT panel marker (spec + matplotlib)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from audiogram import EarAudiogram
from audiogram.canvas import get_plot_style
from audiogram.plot_mpl import plot_binaural, plot_binaural_two_panel
from audiogram.plot_spec import PLOT_SPEC_VERSION, build_plot_spec


class TestTestedFlag:
    def test_default_is_not_stated(self):
        ear = EarAudiogram(air={1000: 20})
        assert ear.tested is None
        assert "tested" not in ear.to_dict()                 # absent, never implicitly False

    def test_round_trip_both_values(self):
        for v in (True, False):
            ear = EarAudiogram(air={} if v is False else {1000: 20}, tested=v)
            d = json.loads(json.dumps(ear.to_dict()))
            assert d["tested"] is v
            assert EarAudiogram.from_dict(d).tested is v

    def test_repr_mentions_it_only_when_stated(self):
        assert "tested" not in repr(EarAudiogram(air={1000: 20}))
        assert "tested=False" in repr(EarAudiogram(tested=False))


class TestDntMarker:
    def test_spec_carries_the_marker(self):
        spec = build_plot_spec("mei")
        assert PLOT_SPEC_VERSION == "1.2" and spec["spec_version"] == "1.2"
        m = spec["dnt_marker"]
        assert m["label"] == "DNT" and m["rotation_deg"] == 45.0 and m["rotation_sense"] == "counter-clockwise"
        assert 0 < m["alpha"] < 1 and 0 < m["fill_fraction"] <= 1 and m["when"] == "ear.tested is False"
        assert "dnt" in spec["semantics"]

    def test_single_axes_draws_dnt_for_the_untested_ear(self):
        fig, ax = plot_binaural(left={}, right={1000: 20}, left_dnt=True, style="mei")
        texts = [t for t in ax.texts if t.get_text() == "DNT"]
        assert len(texts) == 1
        t = texts[0]
        assert t.get_rotation() == 45.0 and t.get_alpha() == get_plot_style("mei").dnt_alpha and t.get_color() == "blue"
        plt.close(fig)

    def test_two_panel_draws_in_the_right_panel_only(self):
        fig, (ax_r, ax_l) = plot_binaural_two_panel(left={1000: 20}, right={}, right_dnt=True, style="mei")
        assert [t.get_text() for t in ax_r.texts if t.get_text() == "DNT"] == ["DNT"]
        assert not [t for t in ax_l.texts if t.get_text() == "DNT"]
        plt.close(fig)

    def test_no_marker_when_tested_is_not_stated(self):
        fig, ax = plot_binaural(left={}, right={1000: 20}, style="mei")
        assert not [t for t in ax.texts if t.get_text() == "DNT"]
        plt.close(fig)
