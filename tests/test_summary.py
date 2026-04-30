"""Tests for summary metric registry and compute_summary."""
import pytest

from audiogram_object import (
    EarAudiogram, BinauralAudiogram, WordRecognitionScore,
    register_summary_metric, unregister_summary_metric, SUMMARY_METRICS,
)


@pytest.fixture
def simple_ba():
    left = EarAudiogram(air={500: 20.0, 1000: 25.0, 2000: 30.0, 4000: 35.0})
    right = EarAudiogram(air={500: 15.0, 1000: 20.0, 2000: 25.0, 4000: 30.0})
    return BinauralAudiogram(left, right)


class TestSummaryBasic:
    def test_summary_returns_dict(self, simple_ba):
        s = simple_ba.summary()
        assert isinstance(s, dict)

    def test_summary_has_pta(self, simple_ba):
        s = simple_ba.summary()
        assert "pta_left" in s
        assert "pta_right" in s
        assert "better_ear_pta" in s
        assert "worse_ear_pta" in s

    def test_summary_pta_values(self, simple_ba):
        s = simple_ba.summary()
        assert s["pta_left"] == pytest.approx(25.0)
        assert s["pta_right"] == pytest.approx(20.0)
        assert s["better_ear_pta"] == pytest.approx(20.0)
        assert s["worse_ear_pta"] == pytest.approx(25.0)

    def test_summary_has_severity(self, simple_ba):
        s = simple_ba.summary()
        assert "left_severity" in s
        assert "right_severity" in s

    def test_summary_has_frequency_count(self, simple_ba):
        s = simple_ba.summary()
        assert s["left_air_freq_count"] == 4
        assert s["right_air_freq_count"] == 4
        assert s["left_bone_freq_count"] == 0
        assert s["right_bone_freq_count"] == 0

    def test_summary_has_asymmetry(self, simple_ba):
        s = simple_ba.summary()
        assert "asymmetric_any_15db" in s

    def test_summary_has_all_asymmetry_criteria(self, simple_ba):
        s = simple_ba.summary()
        from audiogram_object.asymmetry import ASYMMETRY_CRITERIA
        for name in ASYMMETRY_CRITERIA:
            assert f"asymmetric_{name}" in s

    def test_summary_wrs_15pct_with_data(self):
        left = EarAudiogram(
            air={500: 20.0, 1000: 25.0, 2000: 30.0},
            wrs=[WordRecognitionScore(92.0, 70.0)],
        )
        right = EarAudiogram(
            air={500: 20.0, 1000: 25.0, 2000: 30.0},
            wrs=[WordRecognitionScore(72.0, 70.0)],
        )
        ba = BinauralAudiogram(left, right)
        s = ba.summary()
        assert s["asymmetric_wrs_15pct"] is True

    def test_summary_wrs_15pct_none_without_data(self, simple_ba):
        s = simple_ba.summary()
        assert s["asymmetric_wrs_15pct"] is None


class TestSummaryFiltering:
    def test_include(self, simple_ba):
        s = simple_ba.summary(include=["pta"])
        assert "pta_left" in s
        assert "left_severity" not in s
        assert "asymmetric_any_15db" not in s

    def test_exclude(self, simple_ba):
        s = simple_ba.summary(exclude=["frequency_count", "asymmetry"])
        assert "pta_left" in s
        assert "left_air_freq_count" not in s
        assert "asymmetric_any_15db" not in s

    def test_include_multiple(self, simple_ba):
        s = simple_ba.summary(include=["pta", "severity"])
        assert "pta_left" in s
        assert "left_severity" in s
        assert "left_air_freq_count" not in s


class TestSummaryRegistry:
    def test_register_and_use(self, simple_ba):
        def custom_metric(ba):
            return {"custom_value": 42}

        register_summary_metric("custom_test", custom_metric)
        try:
            s = simple_ba.summary()
            assert s["custom_value"] == 42
        finally:
            unregister_summary_metric("custom_test")

    def test_unregister(self, simple_ba):
        def dummy(ba):
            return {"dummy_val": 1}

        register_summary_metric("dummy_test", dummy)
        unregister_summary_metric("dummy_test")
        s = simple_ba.summary()
        assert "dummy_val" not in s

    def test_unregister_nonexistent_no_error(self):
        unregister_summary_metric("nonexistent_metric_xyz")

    def test_registered_metrics_present(self):
        assert "pta" in SUMMARY_METRICS
        assert "severity" in SUMMARY_METRICS
        assert "speech" in SUMMARY_METRICS
        assert "asymmetry" in SUMMARY_METRICS
        assert "frequency_count" in SUMMARY_METRICS


class TestSpeechSummary:
    def test_speech_fields_present(self, speech_ba):
        s = speech_ba.summary(include=["speech"])
        assert "left_srt" in s
        assert "right_srt" in s
        assert "left_sat" in s
        assert "right_sat" in s
        assert "left_wrs" in s
        assert "right_wrs" in s
        assert "left_wrs_level" in s
        assert "right_wrs_level" in s
        assert "left_srt_pta_agreement" in s

    def test_speech_values(self, speech_ba):
        s = speech_ba.summary(include=["speech"])
        assert s["left_srt"] == 28.0
        assert s["right_srt"] == 52.0
        assert s["left_sat"] is None
        assert s["right_sat"] == 45.0
        assert s["left_wrs"] == 92.0
        assert s["right_wrs"] == 68.0

    def test_speech_none_when_empty(self, simple_ba):
        s = simple_ba.summary(include=["speech"])
        assert s["left_srt"] is None
        assert s["left_wrs"] is None
        assert s["left_wrs_level"] is None


class TestLongFormSpeechRoundTrip:
    def test_speech_rows_output(self, speech_ba):
        rows = speech_ba.to_speech_rows()
        measures = {r["measure"] for r in rows}
        assert measures == {"srt", "sat", "wrs"}

    def test_speech_rows_count(self, speech_ba):
        rows = speech_ba.to_speech_rows()
        assert len(rows) == 6

    def test_speech_rows_metadata(self, speech_ba):
        rows = speech_ba.to_speech_rows()
        assert all(r["audiogram_id"] == "speech-001" for r in rows)

    def test_speech_rows_no_meta(self, speech_ba):
        rows = speech_ba.to_speech_rows(include_meta=False)
        assert "subject_id" not in rows[0]
        assert "audiogram_id" in rows[0]

    def test_combined_long_rows(self, speech_ba):
        rows = speech_ba.to_long_rows(include_speech=True)
        types = {r["measure_type"] for r in rows}
        assert types == {"threshold", "speech"}

    def test_combined_long_rows_round_trip(self, speech_ba):
        rows = speech_ba.to_long_rows(include_speech=True)
        restored = BinauralAudiogram.from_long_rows(rows)
        assert restored.left.srt == 28.0
        assert len(restored.left.wrs) == 2
        assert restored.right.sat == 45.0
        assert restored.right.wrs[0].masked is True
        assert restored.left.air[500].threshold_db == 25.0

    def test_threshold_only_long_rows_no_measure_type(self, speech_ba):
        rows = speech_ba.to_long_rows(include_speech=False)
        assert "measure_type" not in rows[0]

    def test_speech_only_rows_from_long_rows(self):
        rows = [
            {"audiogram_id": "s-1", "ear": "left", "measure": "srt", "value": 25.0,
             "level_db": None, "masked": False, "word_list": None, "freq_hz": None,
             "threshold_db": None, "pathway": None, "nr": None,
             "measure_type": "speech"},
            {"audiogram_id": "s-1", "ear": "left", "measure": "wrs", "value": 92.0,
             "level_db": 70.0, "masked": False, "word_list": "CNC", "freq_hz": None,
             "threshold_db": None, "pathway": None, "nr": None,
             "measure_type": "speech"},
        ]
        restored = BinauralAudiogram.from_long_rows(rows)
        assert restored.left.srt == 25.0
        assert len(restored.left.wrs) == 1
        assert restored.left.wrs[0].score_pct == 92.0
