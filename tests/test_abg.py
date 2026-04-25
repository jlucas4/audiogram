"""Tests for air-bone gap, ABG PTA, loss type classification, and summary integration."""
import warnings

import pytest

from audiogram_object import (
    EarAudiogram,
    BinauralAudiogram,
    ThresholdPoint,
    abg_from_thresholds,
    abg_pta,
    loss_type,
    LOSS_TYPES,
    VALID_STANDARDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _conductive_ear():
    return EarAudiogram(
        air={500: 45, 1000: 50, 2000: 55, 4000: 60},
        bone={500: 10, 1000: 15, 2000: 15, 4000: 20},
    )


def _snhl_ear():
    return EarAudiogram(
        air={500: 45, 1000: 50, 2000: 55, 4000: 60},
        bone={500: 40, 1000: 45, 2000: 50, 4000: 55},
    )


def _mixed_ear():
    return EarAudiogram(
        air={500: 55, 1000: 60, 2000: 65, 4000: 70},
        bone={500: 35, 1000: 40, 2000: 45, 4000: 50},
    )


def _normal_ear():
    return EarAudiogram(
        air={500: 10, 1000: 15, 2000: 10, 4000: 15},
        bone={500: 5, 1000: 10, 2000: 5, 4000: 10},
    )


# ---------------------------------------------------------------------------
# abg_from_thresholds (pure function)
# ---------------------------------------------------------------------------

class TestAbgFromThresholds:
    def test_basic(self):
        air = {500: 40.0, 1000: 50.0, 2000: 60.0}
        bone = {500: 10.0, 1000: 20.0, 2000: 30.0}
        result = abg_from_thresholds(air, bone)
        assert result == {500: 30.0, 1000: 30.0, 2000: 30.0}

    def test_only_common_frequencies(self):
        air = {500: 40.0, 1000: 50.0, 2000: 60.0, 4000: 70.0}
        bone = {500: 10.0, 2000: 30.0}
        result = abg_from_thresholds(air, bone)
        assert result == {500: 30.0, 2000: 30.0}
        assert 1000 not in result
        assert 4000 not in result

    def test_no_common_frequencies(self):
        air = {500: 40.0, 1000: 50.0}
        bone = {2000: 30.0, 4000: 40.0}
        assert abg_from_thresholds(air, bone) == {}

    def test_empty_air(self):
        assert abg_from_thresholds({}, {500: 10.0}) == {}

    def test_empty_bone(self):
        assert abg_from_thresholds({500: 40.0}, {}) == {}

    def test_both_empty(self):
        assert abg_from_thresholds({}, {}) == {}

    def test_zero_gap(self):
        air = {500: 30.0, 1000: 40.0}
        bone = {500: 30.0, 1000: 40.0}
        result = abg_from_thresholds(air, bone)
        assert result == {500: 0.0, 1000: 0.0}

    def test_negative_gap(self):
        air = {500: 10.0}
        bone = {500: 15.0}
        result = abg_from_thresholds(air, bone)
        assert result == {500: -5.0}

    def test_sorted_output(self):
        air = {4000: 70.0, 500: 40.0, 2000: 60.0, 1000: 50.0}
        bone = {1000: 20.0, 4000: 40.0, 500: 10.0, 2000: 30.0}
        result = abg_from_thresholds(air, bone)
        assert list(result.keys()) == [500, 1000, 2000, 4000]


class TestAbgMaskWarning:
    def test_no_warning_by_default(self):
        air = {500: 40.0}
        bone = {500: 10.0}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            abg_from_thresholds(air, bone)
            assert len(w) == 0

    def test_no_warning_when_all_masked(self):
        air = {500: 40.0, 1000: 50.0}
        bone = {500: 10.0, 1000: 20.0}
        bone_masked = {500: True, 1000: True}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            abg_from_thresholds(air, bone, mask_warning=True, bone_masked=bone_masked)
            assert len(w) == 0

    def test_warning_when_unmasked(self):
        air = {500: 40.0, 1000: 50.0}
        bone = {500: 10.0, 1000: 20.0}
        bone_masked = {500: False, 1000: True}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            abg_from_thresholds(air, bone, mask_warning=True, bone_masked=bone_masked)
            assert len(w) == 1
            assert "unmasked" in str(w[0].message).lower()
            assert "500" in str(w[0].message)

    def test_warning_lists_all_unmasked(self):
        air = {500: 40.0, 1000: 50.0, 2000: 60.0}
        bone = {500: 10.0, 1000: 20.0, 2000: 30.0}
        bone_masked = {500: False, 1000: False, 2000: True}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            abg_from_thresholds(air, bone, mask_warning=True, bone_masked=bone_masked)
            assert len(w) == 1
            assert "500" in str(w[0].message)
            assert "1000" in str(w[0].message)

    def test_mask_warning_true_but_no_bone_masked_dict(self):
        air = {500: 40.0}
        bone = {500: 10.0}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            abg_from_thresholds(air, bone, mask_warning=True, bone_masked=None)
            assert len(w) == 0


# ---------------------------------------------------------------------------
# abg_pta (pure function)
# ---------------------------------------------------------------------------

class TestAbgPta:
    def test_who_default_freqs(self):
        air = {500: 40.0, 1000: 50.0, 2000: 60.0, 4000: 70.0}
        bone = {500: 10.0, 1000: 20.0, 2000: 30.0, 4000: 40.0}
        result = abg_pta(air, bone)
        assert result == 30.0

    def test_who_custom_freqs(self):
        air = {500: 40.0, 1000: 50.0, 2000: 60.0}
        bone = {500: 10.0, 1000: 20.0, 2000: 30.0}
        result = abg_pta(air, bone, freqs=(500, 1000, 2000))
        assert result == 30.0

    def test_who_partial_overlap(self):
        air = {500: 40.0, 1000: 50.0, 2000: 60.0}
        bone = {500: 10.0, 1000: 20.0}
        result = abg_pta(air, bone, freqs=(500, 1000, 2000))
        assert result == 30.0

    def test_who_require_all_missing(self):
        air = {500: 40.0, 1000: 50.0}
        bone = {500: 10.0, 1000: 20.0}
        result = abg_pta(air, bone, require_all=True)
        assert result is None

    def test_who_require_all_satisfied(self):
        air = {500: 40.0, 1000: 50.0, 2000: 60.0, 4000: 70.0}
        bone = {500: 10.0, 1000: 20.0, 2000: 30.0, 4000: 40.0}
        result = abg_pta(air, bone, require_all=True)
        assert result == 30.0

    def test_aao_hns_with_3000(self):
        air = {500: 40.0, 1000: 50.0, 2000: 60.0, 3000: 65.0}
        bone = {500: 10.0, 1000: 20.0, 2000: 30.0, 3000: 35.0}
        result = abg_pta(air, bone, standard="aao_hns")
        assert result == 30.0

    def test_aao_hns_fallback(self):
        air = {500: 40.0, 1000: 50.0, 2000: 60.0, 4000: 70.0}
        bone = {500: 10.0, 1000: 20.0, 2000: 30.0, 4000: 40.0}
        # No 3000 — uses avg(gap_2000, gap_4000) = avg(30, 30) = 30
        result = abg_pta(air, bone, standard="aao_hns")
        assert result == 30.0

    def test_aao_hns_fallback_asymmetric(self):
        air = {500: 40.0, 1000: 50.0, 2000: 50.0, 4000: 70.0}
        bone = {500: 10.0, 1000: 20.0, 2000: 30.0, 4000: 40.0}
        # Gaps: 500=30, 1000=30, 2000=20, 4000=30
        # 3000 fallback: avg(20, 30) = 25
        # ABG PTA = (30 + 30 + 20 + 25) / 4 = 26.25
        result = abg_pta(air, bone, standard="aao_hns")
        assert result == 26.25

    def test_no_common_freqs(self):
        air = {500: 40.0}
        bone = {1000: 20.0}
        assert abg_pta(air, bone) is None

    def test_empty(self):
        assert abg_pta({}, {}) is None

    def test_invalid_standard(self):
        with pytest.raises(ValueError, match="Unknown standard"):
            abg_pta({500: 40.0}, {500: 10.0}, standard="bogus")


# ---------------------------------------------------------------------------
# loss_type (pure function)
# ---------------------------------------------------------------------------

class TestLossType:
    def test_normal(self):
        assert loss_type(20.0, 15.0, 5.0) == "normal"

    def test_normal_at_boundary(self):
        assert loss_type(25.0, 20.0, 5.0) == "normal"

    def test_sensorineural(self):
        assert loss_type(50.0, 45.0, 5.0) == "sensorineural"

    def test_sensorineural_abg_just_under(self):
        assert loss_type(50.0, 41.0, 9.9) == "sensorineural"

    def test_conductive(self):
        assert loss_type(50.0, 15.0, 35.0) == "conductive"

    def test_conductive_bone_at_boundary(self):
        assert loss_type(50.0, 25.0, 25.0) == "conductive"

    def test_mixed(self):
        assert loss_type(60.0, 40.0, 20.0) == "mixed"

    def test_mixed_bone_just_above(self):
        assert loss_type(60.0, 25.1, 34.9) == "mixed"

    def test_abg_at_boundary_10(self):
        # ABG exactly 10 → conductive (>= 10 cutoff)
        assert loss_type(50.0, 20.0, 10.0) == "conductive"

    def test_none_air(self):
        assert loss_type(None, 20.0, 5.0) is None

    def test_none_bone(self):
        assert loss_type(50.0, None, 5.0) is None

    def test_none_abg(self):
        assert loss_type(50.0, 20.0, None) is None

    def test_all_none(self):
        assert loss_type(None, None, None) is None

    def test_loss_types_constant(self):
        assert set(LOSS_TYPES) == {"normal", "sensorineural", "conductive", "mixed"}


# ---------------------------------------------------------------------------
# EarAudiogram convenience methods
# ---------------------------------------------------------------------------

class TestEarAudiogramAbg:
    def test_abg_per_frequency(self):
        ear = _conductive_ear()
        abg = ear.abg()
        assert abg == {500: 35.0, 1000: 35.0, 2000: 40.0, 4000: 40.0}

    def test_abg_no_bone(self):
        ear = EarAudiogram(air={500: 40.0, 1000: 50.0})
        assert ear.abg() == {}

    def test_abg_no_air(self):
        ear = EarAudiogram(bone={500: 10.0})
        assert ear.abg() == {}

    def test_abg_pta_who(self):
        ear = _conductive_ear()
        result = ear.abg_pta()
        assert result == 37.5

    def test_abg_pta_aao_hns(self):
        ear = _conductive_ear()
        result = ear.abg_pta(standard="aao_hns")
        # Gaps: 500=35, 1000=35, 2000=40, 4000=40
        # No 3000 → fallback avg(40, 40) = 40
        # ABG PTA = (35 + 35 + 40 + 40) / 4 = 37.5
        assert result == 37.5

    def test_abg_pta_no_bone(self):
        ear = EarAudiogram(air={500: 40.0, 1000: 50.0})
        assert ear.abg_pta() is None

    def test_loss_type_conductive(self):
        assert _conductive_ear().loss_type() == "conductive"

    def test_loss_type_sensorineural(self):
        assert _snhl_ear().loss_type() == "sensorineural"

    def test_loss_type_mixed(self):
        assert _mixed_ear().loss_type() == "mixed"

    def test_loss_type_normal(self):
        assert _normal_ear().loss_type() == "normal"

    def test_loss_type_no_bone(self):
        ear = EarAudiogram(air={500: 40.0, 1000: 50.0, 2000: 60.0})
        assert ear.loss_type() is None

    def test_loss_type_aao_hns(self):
        assert _conductive_ear().loss_type(standard="aao_hns") == "conductive"
        assert _snhl_ear().loss_type(standard="aao_hns") == "sensorineural"
        assert _mixed_ear().loss_type(standard="aao_hns") == "mixed"
        assert _normal_ear().loss_type(standard="aao_hns") == "normal"

    def test_mask_warning_via_object(self):
        ear = EarAudiogram(
            air={500: 40.0, 1000: 50.0},
            bone={
                500: ThresholdPoint(10.0),
                1000: ThresholdPoint(20.0, masked=True),
            },
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ear.abg(mask_warning=True)
            assert len(w) == 1
            assert "500" in str(w[0].message)
            assert "1000" not in str(w[0].message)

    def test_no_mask_warning_when_all_masked(self):
        ear = EarAudiogram(
            air={500: 40.0, 1000: 50.0},
            bone={
                500: ThresholdPoint(10.0, masked=True),
                1000: ThresholdPoint(20.0, masked=True),
            },
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ear.abg(mask_warning=True)
            assert len(w) == 0


# ---------------------------------------------------------------------------
# Summary integration
# ---------------------------------------------------------------------------

class TestAbgSummary:
    def test_summary_includes_abg(self):
        ba = BinauralAudiogram(left=_snhl_ear(), right=_conductive_ear())
        s = ba.summary(include=["abg"])
        assert "left_abg_pta" in s
        assert "right_abg_pta" in s
        assert "left_loss_type" in s
        assert "right_loss_type" in s

    def test_summary_loss_types(self):
        ba = BinauralAudiogram(left=_snhl_ear(), right=_conductive_ear())
        s = ba.summary(include=["abg"])
        assert s["left_loss_type"] == "sensorineural"
        assert s["right_loss_type"] == "conductive"

    def test_summary_standard_kwarg(self):
        ba = BinauralAudiogram(left=_snhl_ear(), right=_conductive_ear())
        s_who = ba.summary(include=["abg"])
        s_aao = ba.summary(include=["abg"], standard="aao_hns")
        # Both should classify correctly regardless of standard
        assert s_who["left_loss_type"] == "sensorineural"
        assert s_aao["left_loss_type"] == "sensorineural"
        assert s_who["right_loss_type"] == "conductive"
        assert s_aao["right_loss_type"] == "conductive"

    def test_summary_severity_uses_standard(self):
        ba = BinauralAudiogram(left=_snhl_ear(), right=_conductive_ear())
        s_who = ba.summary(include=["severity"])
        s_aao = ba.summary(include=["severity"], standard="aao_hns")
        # Both produce severity values (may differ depending on frequencies used)
        assert s_who["left_severity"] is not None
        assert s_aao["left_severity"] is not None

    def test_summary_no_bone(self):
        ba = BinauralAudiogram(
            left=EarAudiogram(air={500: 40.0, 1000: 50.0, 2000: 60.0}),
            right=EarAudiogram(air={500: 30.0, 1000: 40.0, 2000: 50.0}),
        )
        s = ba.summary(include=["abg"])
        assert s["left_abg_pta"] is None
        assert s["right_abg_pta"] is None
        assert s["left_loss_type"] is None
        assert s["right_loss_type"] is None

    def test_summary_exclude_abg(self):
        ba = BinauralAudiogram(left=_snhl_ear(), right=_conductive_ear())
        s = ba.summary(exclude=["abg"])
        assert "left_abg_pta" not in s
        assert "left_loss_type" not in s

    def test_summary_mixed_ears(self):
        ba = BinauralAudiogram(left=_mixed_ear(), right=_normal_ear())
        s = ba.summary(include=["abg"])
        assert s["left_loss_type"] == "mixed"
        assert s["right_loss_type"] == "normal"

    def test_full_summary_does_not_error(self):
        ba = BinauralAudiogram(left=_conductive_ear(), right=_snhl_ear())
        s = ba.summary()
        assert isinstance(s, dict)
        assert len(s) > 0
