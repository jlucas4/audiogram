"""Tests for wide-format I/O and schema wide column helpers."""
import pytest

from audiogram_object import (
    ThresholdPoint, WordRecognitionScore, EarAudiogram, BinauralAudiogram,
    wide_column_name, parse_wide_column, canonical_wide_columns, apply_column_map,
    enrich_wide_rows,
)


class TestWideColumnName:
    def test_basic(self):
        assert wide_column_name("right", "air", 500) == "r_air_500"
        assert wide_column_name("left", "bone", 1000) == "l_bone_1000"

    def test_with_suffix(self):
        assert wide_column_name("right", "air", 4000, "masked") == "r_air_4000_masked"
        assert wide_column_name("left", "air", 250, "nr") == "l_air_250_nr"

    def test_abbreviations_accepted(self):
        assert wide_column_name("r", "air", 500) == "r_air_500"
        assert wide_column_name("l", "bone", 1000) == "l_bone_1000"


class TestParseWideColumn:
    def test_basic(self):
        result = parse_wide_column("r_air_500")
        assert result == {"ear": "right", "pathway": "air", "freq_hz": 500, "field": "threshold"}

    def test_bone(self):
        result = parse_wide_column("l_bone_1000")
        assert result["ear"] == "left"
        assert result["pathway"] == "bone"

    def test_masked_suffix(self):
        result = parse_wide_column("r_air_4000_masked")
        assert result["field"] == "masked"

    def test_nr_suffix(self):
        result = parse_wide_column("l_air_250_nr")
        assert result["field"] == "nr"

    def test_non_matching_returns_none(self):
        assert parse_wide_column("patient_id") is None
        assert parse_wide_column("x_air_500") is None
        assert parse_wide_column("r_water_500") is None
        assert parse_wide_column("r_air") is None

    def test_invalid_suffix_returns_none(self):
        assert parse_wide_column("r_air_500_bogus") is None

    def test_non_numeric_freq_returns_none(self):
        assert parse_wide_column("r_air_abc") is None

    def test_too_many_parts_returns_none(self):
        assert parse_wide_column("r_air_500_masked_extra") is None


class TestCanonicalWideColumns:
    def test_default_air_only(self):
        cols = canonical_wide_columns([500, 1000])
        assert cols == ["r_air_500", "r_air_1000", "l_air_500", "l_air_1000"]

    def test_both_pathways(self):
        cols = canonical_wide_columns([500], pathways=["air", "bone"])
        assert "r_air_500" in cols
        assert "r_bone_500" in cols
        assert "l_air_500" in cols
        assert "l_bone_500" in cols

    def test_include_masked(self):
        cols = canonical_wide_columns([500], include_masked=True)
        assert "r_air_500" in cols
        assert "r_air_500_masked" in cols
        assert "l_air_500_masked" in cols

    def test_include_nr(self):
        cols = canonical_wide_columns([500], include_nr=True)
        assert "r_air_500_nr" in cols

    def test_sorted_frequencies(self):
        cols = canonical_wide_columns([2000, 500, 1000])
        freqs_r = [c for c in cols if c.startswith("r_")]
        assert freqs_r == ["r_air_500", "r_air_1000", "r_air_2000"]

    def test_defaults_to_standard_frequencies(self):
        cols = canonical_wide_columns()
        assert "r_air_500" in cols
        assert "l_air_8000" in cols


class TestApplyColumnMap:
    def test_basic_mapping(self):
        row = {"R_AC_500": 25.0, "PatientID": "pt-1"}
        mapped = apply_column_map(row, {"R_AC_500": "r_air_500", "PatientID": "subject_id"})
        assert mapped == {"r_air_500": 25.0, "subject_id": "pt-1"}

    def test_unmapped_keys_pass_through(self):
        row = {"R_AC_500": 25.0, "extra_col": "hello"}
        mapped = apply_column_map(row, {"R_AC_500": "r_air_500"})
        assert mapped["extra_col"] == "hello"

    def test_empty_map(self):
        row = {"a": 1}
        assert apply_column_map(row, {}) == row


class TestWideRoundTrip:
    def test_air_only_round_trip(self):
        left = EarAudiogram(air={500: 20.0, 1000: 25.0, 2000: 30.0})
        right = EarAudiogram(air={500: 15.0, 1000: 20.0, 2000: 25.0})
        ba = BinauralAudiogram(left, right, audiogram_id="w-001")
        wide = ba.to_wide_row()
        restored = BinauralAudiogram.from_wide_row(wide)
        assert restored.left.air[500].threshold_db == 20.0
        assert restored.right.air[2000].threshold_db == 25.0
        assert restored.audiogram_id == "w-001"

    def test_bone_and_masked_round_trip(self):
        left = EarAudiogram(
            air={500: ThresholdPoint(25.0)},
            bone={500: ThresholdPoint(15.0, masked=True)},
        )
        right = EarAudiogram(air={500: ThresholdPoint(20.0)})
        ba = BinauralAudiogram(left, right)
        wide = ba.to_wide_row()
        restored = BinauralAudiogram.from_wide_row(wide)
        assert restored.left.bone[500].masked is True
        assert restored.left.bone[500].threshold_db == 15.0

    def test_nr_round_trip(self):
        right = EarAudiogram(air={4000: ThresholdPoint(120.0, nr=True)})
        ba = BinauralAudiogram(EarAudiogram(), right)
        wide = ba.to_wide_row()
        restored = BinauralAudiogram.from_wide_row(wide)
        assert restored.right.air[4000].nr is True

    def test_metadata_in_wide_row(self):
        ba = BinauralAudiogram(
            EarAudiogram(air={500: 20.0}),
            EarAudiogram(air={500: 25.0}),
            audiogram_id="w-meta",
            subject_id="pt-42",
            performed_at="2024-06-15",
            source="clinic",
        )
        wide = ba.to_wide_row()
        assert wide["audiogram_id"] == "w-meta"
        assert wide["subject_id"] == "pt-42"

    def test_metadata_overrides(self):
        ba = BinauralAudiogram(
            EarAudiogram(air={500: 20.0}),
            EarAudiogram(air={500: 25.0}),
        )
        wide = ba.to_wide_row()
        restored = BinauralAudiogram.from_wide_row(
            wide, audiogram_id="override", subject_id="pt-override",
        )
        assert restored.audiogram_id == "override"
        assert restored.subject_id == "pt-override"

    def test_exclude_meta(self):
        ba = BinauralAudiogram(
            EarAudiogram(air={500: 20.0}),
            EarAudiogram(air={500: 25.0}),
            subject_id="pt-1",
        )
        wide = ba.to_wide_row(include_meta=False)
        assert "subject_id" not in wide

    def test_missing_values_skipped(self):
        row = {"r_air_500": 25.0, "r_air_1000": None, "r_air_2000": ""}
        ba = BinauralAudiogram.from_wide_row(row)
        assert 500 in ba.right.air
        assert 1000 not in ba.right.air
        assert 2000 not in ba.right.air

    def test_column_map_ingest(self):
        row = {"R_AC_500": 25.0, "R_AC_1K": 30.0, "L_AC_500": 20.0}
        col_map = {
            "R_AC_500": "r_air_500",
            "R_AC_1K": "r_air_1000",
            "L_AC_500": "l_air_500",
        }
        ba = BinauralAudiogram.from_wide_row(row, column_map=col_map)
        assert ba.right.air[500].threshold_db == 25.0
        assert ba.right.air[1000].threshold_db == 30.0
        assert ba.left.air[500].threshold_db == 20.0

    def test_irrelevant_columns_ignored(self):
        row = {
            "r_air_500": 25.0,
            "diagnosis": "SNHL",
            "visit_notes": "follow up in 6 months",
        }
        ba = BinauralAudiogram.from_wide_row(row)
        assert ba.right.air[500].threshold_db == 25.0
        assert len(ba.right.air) == 1


class TestWideSpeech:
    def test_single_wrs_round_trip(self):
        ear = EarAudiogram(
            air={500: 20.0},
            srt=25.0,
            wrs=[WordRecognitionScore(92.0, 70.0, word_list="CNC")],
        )
        ba = BinauralAudiogram(ear, EarAudiogram())
        wide = ba.to_wide_row()
        assert wide["l_srt"] == 25.0
        assert wide["l_wrs"] == 92.0
        assert wide["l_wrs_level"] == 70.0
        assert wide["l_wrs_list"] == "CNC"
        restored = BinauralAudiogram.from_wide_row(wide)
        assert restored.left.srt == 25.0
        assert len(restored.left.wrs) == 1
        assert restored.left.wrs[0].score_pct == 92.0
        assert restored.left.wrs[0].word_list == "CNC"

    def test_multiple_wrs_round_trip(self):
        ear = EarAudiogram(
            air={500: 20.0},
            wrs=[
                WordRecognitionScore(92.0, 70.0),
                WordRecognitionScore(80.0, 50.0),
            ],
        )
        ba = BinauralAudiogram(ear, EarAudiogram())
        wide = ba.to_wide_row()
        assert "l_wrs_1" in wide
        assert "l_wrs_2" in wide
        assert "l_wrs" not in wide
        restored = BinauralAudiogram.from_wide_row(wide)
        assert len(restored.left.wrs) == 2

    def test_sat_round_trip(self):
        ear = EarAudiogram(air={500: 20.0}, sat=15.0)
        ba = BinauralAudiogram(EarAudiogram(), ear)
        wide = ba.to_wide_row()
        assert wide["r_sat"] == 15.0
        restored = BinauralAudiogram.from_wide_row(wide)
        assert restored.right.sat == 15.0

    def test_masked_wrs(self):
        ear = EarAudiogram(
            air={500: 40.0},
            wrs=[WordRecognitionScore(68.0, 90.0, masked=True)],
        )
        ba = BinauralAudiogram(EarAudiogram(), ear)
        wide = ba.to_wide_row()
        assert wide.get("r_wrs_masked") is True
        restored = BinauralAudiogram.from_wide_row(wide)
        assert restored.right.wrs[0].masked is True

    def test_full_clinical_wide_round_trip(self, speech_ba):
        wide = speech_ba.to_wide_row()
        restored = BinauralAudiogram.from_wide_row(wide)
        assert restored.left.srt == 28.0
        assert len(restored.left.wrs) == 2
        assert restored.right.sat == 45.0
        assert restored.right.wrs[0].masked is True


class TestEnrichWideRows:
    def _make_rows(self):
        return [
            {"r_air_500": 25.0, "r_air_1000": 30.0, "r_air_2000": 35.0,
             "l_air_500": 20.0, "l_air_1000": 25.0, "l_air_2000": 30.0,
             "subject_id": "pt-1"},
            {"r_air_500": 40.0, "r_air_1000": 45.0, "r_air_2000": 50.0,
             "l_air_500": 35.0, "l_air_1000": 40.0, "l_air_2000": 45.0,
             "subject_id": "pt-2"},
        ]

    def test_returns_list_of_dicts(self):
        result = enrich_wide_rows(self._make_rows())
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_original_data_preserved(self):
        result = enrich_wide_rows(self._make_rows())
        assert result[0]["subject_id"] == "pt-1"
        assert result[0]["r_air_500"] == 25.0

    def test_metrics_merged(self):
        result = enrich_wide_rows(self._make_rows())
        assert "pta_left" in result[0]
        assert "pta_right" in result[0]

    def test_pta_values_correct(self):
        result = enrich_wide_rows(self._make_rows())
        assert result[0]["pta_right"] == pytest.approx(30.0)
        assert result[0]["pta_left"] == pytest.approx(25.0)

    def test_include_filter(self):
        result = enrich_wide_rows(self._make_rows(), include=["pta"])
        assert "pta_left" in result[0]
        assert "left_severity" not in result[0]

    def test_exclude_filter(self):
        result = enrich_wide_rows(self._make_rows(), exclude=["asymmetry"])
        assert "pta_left" in result[0]
        for key in result[0]:
            assert "asymmetry" not in key.lower() or key in ("subject_id",)

    def test_standard_kwarg(self):
        result = enrich_wide_rows(self._make_rows(), standard="aao_hns")
        assert "pta_left" in result[0]

    def test_column_map(self):
        rows = [{"R_AC_500": 25.0, "R_AC_1K": 30.0, "R_AC_2K": 35.0}]
        col_map = {"R_AC_500": "r_air_500", "R_AC_1K": "r_air_1000", "R_AC_2K": "r_air_2000"}
        result = enrich_wide_rows(rows, column_map=col_map)
        assert "pta_right" in result[0]
        assert result[0]["pta_right"] == pytest.approx(30.0)

    def test_empty_input(self):
        assert enrich_wide_rows([]) == []

    def test_via_classmethod(self):
        result = BinauralAudiogram.enrich_wide_rows(self._make_rows())
        assert len(result) == 2
        assert "pta_left" in result[0]
