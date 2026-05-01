"""Tests for soundfield and cochlear-implant aided thresholds."""
import pytest

from audiogram import (
    ThresholdPoint, EarAudiogram, BinauralAudiogram,
    wide_column_name, parse_wide_column, canonical_wide_columns,
)


class TestEarAudiogramSoundfield:
    def test_defaults_empty(self):
        ear = EarAudiogram(air={500: 20.0})
        assert ear.soundfield == {}
        assert ear.ci == {}

    def test_soundfield_stored(self):
        ear = EarAudiogram(
            air={500: 60.0},
            soundfield={500: 30.0, 1000: 35.0, 2000: 40.0},
        )
        assert len(ear.soundfield) == 3
        assert ear.soundfield[500].threshold_db == 30.0

    def test_ci_stored(self):
        ear = EarAudiogram(
            air={500: 80.0},
            ci={500: ThresholdPoint(25.0), 1000: ThresholdPoint(30.0)},
        )
        assert len(ear.ci) == 2
        assert ear.ci[500].threshold_db == 25.0

    def test_float_coercion(self):
        ear = EarAudiogram(soundfield={500: 30.0})
        assert isinstance(ear.soundfield[500], ThresholdPoint)

    def test_available_frequencies_soundfield(self):
        ear = EarAudiogram(soundfield={500: 30.0, 2000: 40.0, 1000: 35.0})
        assert ear.available_frequencies("soundfield") == [500, 1000, 2000]

    def test_available_frequencies_ci(self):
        ear = EarAudiogram(ci={1000: 25.0, 4000: 35.0})
        assert ear.available_frequencies("ci") == [1000, 4000]

    def test_pta_soundfield(self):
        ear = EarAudiogram(soundfield={500: 20.0, 1000: 30.0, 2000: 40.0})
        assert ear.pta(pathway="soundfield") == pytest.approx(30.0)

    def test_pta_ci(self):
        ear = EarAudiogram(ci={500: 20.0, 1000: 30.0, 2000: 40.0})
        assert ear.pta(pathway="ci") == pytest.approx(30.0)

    def test_pathway_data_invalid_raises(self):
        ear = EarAudiogram(air={500: 20.0})
        with pytest.raises(ValueError, match="pathway"):
            ear.available_frequencies("invalid")


class TestEarAudiogramSfCiEquality:
    def test_eq_with_soundfield(self):
        a = EarAudiogram(air={500: 20.0}, soundfield={500: 30.0})
        b = EarAudiogram(air={500: 20.0}, soundfield={500: 30.0})
        assert a == b

    def test_neq_different_soundfield(self):
        a = EarAudiogram(air={500: 20.0}, soundfield={500: 30.0})
        b = EarAudiogram(air={500: 20.0}, soundfield={500: 35.0})
        assert a != b

    def test_neq_soundfield_vs_empty(self):
        a = EarAudiogram(air={500: 20.0}, soundfield={500: 30.0})
        b = EarAudiogram(air={500: 20.0})
        assert a != b

    def test_eq_with_ci(self):
        a = EarAudiogram(air={500: 80.0}, ci={500: 25.0})
        b = EarAudiogram(air={500: 80.0}, ci={500: 25.0})
        assert a == b

    def test_neq_different_ci(self):
        a = EarAudiogram(air={500: 80.0}, ci={500: 25.0})
        b = EarAudiogram(air={500: 80.0}, ci={500: 30.0})
        assert a != b


class TestEarAudiogramSfCiRepr:
    def test_repr_includes_soundfield(self):
        ear = EarAudiogram(air={500: 20.0}, soundfield={500: 30.0})
        assert "soundfield=" in repr(ear)

    def test_repr_includes_ci(self):
        ear = EarAudiogram(air={500: 20.0}, ci={500: 25.0})
        assert "ci=" in repr(ear)

    def test_repr_excludes_when_empty(self):
        ear = EarAudiogram(air={500: 20.0})
        assert "soundfield" not in repr(ear)
        assert "ci=" not in repr(ear)


class TestEarAudiogramSfCiSerialization:
    def test_dict_round_trip_soundfield(self):
        ear = EarAudiogram(
            air={500: 20.0},
            soundfield={500: ThresholdPoint(30.0), 1000: ThresholdPoint(35.0)},
        )
        restored = EarAudiogram.from_dict(ear.to_dict())
        assert restored.soundfield[500].threshold_db == 30.0
        assert restored.soundfield[1000].threshold_db == 35.0
        assert ear == restored

    def test_dict_round_trip_ci(self):
        ear = EarAudiogram(
            air={500: 80.0},
            ci={500: ThresholdPoint(25.0, masked=True)},
        )
        restored = EarAudiogram.from_dict(ear.to_dict())
        assert restored.ci[500].threshold_db == 25.0
        assert restored.ci[500].masked is True
        assert ear == restored

    def test_dict_omits_when_empty(self):
        ear = EarAudiogram(air={500: 20.0})
        d = ear.to_dict()
        assert "soundfield" not in d
        assert "ci" not in d

    def test_json_round_trip(self):
        ear = EarAudiogram(
            air={500: 20.0}, soundfield={500: 30.0}, ci={1000: 25.0},
        )
        restored = EarAudiogram.from_json(ear.to_json())
        assert ear == restored


class TestBinauralSfCiDictRoundTrip:
    def test_dict_round_trip(self):
        left = EarAudiogram(
            air={500: 60.0},
            soundfield={500: 30.0, 1000: 35.0},
        )
        right = EarAudiogram(
            air={500: 80.0},
            ci={500: 25.0, 1000: 30.0, 2000: 35.0},
        )
        ba = BinauralAudiogram(left, right, audiogram_id="sf-ci-001")
        restored = BinauralAudiogram.from_dict(ba.to_dict())
        assert ba == restored
        assert restored.left.soundfield[500].threshold_db == 30.0
        assert restored.right.ci[1000].threshold_db == 30.0

    def test_json_round_trip(self):
        left = EarAudiogram(air={500: 60.0}, soundfield={500: 30.0})
        right = EarAudiogram(air={500: 80.0}, ci={500: 25.0})
        ba = BinauralAudiogram(left, right)
        restored = BinauralAudiogram.from_json(ba.to_json())
        assert ba == restored


class TestBinauralSfCiLongRows:
    def test_long_rows_include_sf_ci(self):
        left = EarAudiogram(air={500: 60.0}, soundfield={500: 30.0})
        right = EarAudiogram(air={500: 80.0}, ci={500: 25.0})
        ba = BinauralAudiogram(left, right, audiogram_id="lr-001")
        rows = ba.to_long_rows()
        pathways = {r["pathway"] for r in rows}
        assert pathways == {"air", "soundfield", "ci"}
        assert len(rows) == 4

    def test_long_rows_round_trip(self):
        left = EarAudiogram(
            air={500: 60.0, 1000: 65.0},
            soundfield={500: 30.0, 1000: 35.0},
        )
        right = EarAudiogram(
            air={500: 80.0},
            ci={500: 25.0, 1000: 30.0},
        )
        ba = BinauralAudiogram(left, right, audiogram_id="lr-002")
        rows = ba.to_long_rows()
        restored = BinauralAudiogram.from_long_rows(rows)
        assert restored.left.soundfield[500].threshold_db == 30.0
        assert restored.right.ci[1000].threshold_db == 30.0
        assert len(restored.left.air) == 2


class TestBinauralSfCiWideFormat:
    def test_wide_column_name_sf(self):
        assert wide_column_name("right", "soundfield", 500) == "r_sf_500"
        assert wide_column_name("left", "soundfield", 1000) == "l_sf_1000"

    def test_wide_column_name_ci(self):
        assert wide_column_name("right", "ci", 500) == "r_ci_500"
        assert wide_column_name("left", "ci", 2000) == "l_ci_2000"

    def test_parse_wide_column_sf(self):
        result = parse_wide_column("r_sf_500")
        assert result == {"ear": "right", "pathway": "soundfield", "freq_hz": 500, "field": "threshold"}

    def test_parse_wide_column_ci(self):
        result = parse_wide_column("l_ci_1000")
        assert result == {"ear": "left", "pathway": "ci", "freq_hz": 1000, "field": "threshold"}

    def test_canonical_wide_columns_sf(self):
        cols = canonical_wide_columns([500], pathways=["soundfield"])
        assert "r_sf_500" in cols
        assert "l_sf_500" in cols

    def test_wide_round_trip_sf(self):
        left = EarAudiogram(air={500: 60.0}, soundfield={500: 30.0, 1000: 35.0})
        right = EarAudiogram(air={500: 80.0})
        ba = BinauralAudiogram(left, right, audiogram_id="w-sf")
        wide = ba.to_wide_row()
        assert wide["l_sf_500"] == 30.0
        assert wide["l_sf_1000"] == 35.0
        restored = BinauralAudiogram.from_wide_row(wide)
        assert restored.left.soundfield[500].threshold_db == 30.0
        assert restored.left.soundfield[1000].threshold_db == 35.0

    def test_wide_round_trip_ci(self):
        right = EarAudiogram(air={500: 80.0}, ci={500: 25.0, 1000: 30.0})
        ba = BinauralAudiogram(EarAudiogram(), right, audiogram_id="w-ci")
        wide = ba.to_wide_row()
        assert wide["r_ci_500"] == 25.0
        restored = BinauralAudiogram.from_wide_row(wide)
        assert restored.right.ci[500].threshold_db == 25.0
        assert restored.right.ci[1000].threshold_db == 30.0


class TestSfCiNotInMetrics:
    """Soundfield/CI thresholds should not contaminate air-based PTA or severity."""

    def test_pta_ignores_soundfield(self):
        ear = EarAudiogram(
            air={500: 60.0, 1000: 65.0, 2000: 70.0},
            soundfield={500: 20.0, 1000: 25.0, 2000: 30.0},
        )
        assert ear.pta() == pytest.approx(65.0)

    def test_severity_ignores_ci(self):
        ear = EarAudiogram(
            air={500: 60.0, 1000: 65.0, 2000: 70.0, 4000: 75.0},
            ci={500: 20.0, 1000: 25.0, 2000: 30.0, 4000: 35.0},
        )
        assert ear.severity() == "moderately_severe"

    def test_binaural_pta_ignores_sf_ci(self):
        left = EarAudiogram(air={500: 60.0, 1000: 65.0, 2000: 70.0}, soundfield={500: 20.0})
        right = EarAudiogram(air={500: 40.0, 1000: 45.0, 2000: 50.0}, ci={500: 15.0})
        ba = BinauralAudiogram(left, right)
        ptas = ba.pta()
        assert ptas["left"] == pytest.approx(65.0)
        assert ptas["right"] == pytest.approx(45.0)
