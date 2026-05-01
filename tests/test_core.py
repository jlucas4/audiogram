"""Tests for ThresholdPoint, EarAudiogram, and BinauralAudiogram."""
import json
import pytest

from audiogram import ThresholdPoint, EarAudiogram, BinauralAudiogram


# ---------------------------------------------------------------------------
# ThresholdPoint
# ---------------------------------------------------------------------------

class TestThresholdPoint:
    def test_defaults(self):
        pt = ThresholdPoint(25.0)
        assert pt.threshold_db == 25.0
        assert pt.masked is False
        assert pt.nr is False

    def test_explicit_fields(self):
        pt = ThresholdPoint(120.0, masked=True, nr=True)
        assert pt.masked is True
        assert pt.nr is True

    def test_to_dict(self):
        pt = ThresholdPoint(30.0, masked=True)
        d = pt.to_dict()
        assert d == {"threshold_db": 30.0, "masked": True, "nr": False}

    def test_from_dict_round_trip(self):
        pt = ThresholdPoint(45.0, nr=True)
        assert ThresholdPoint.from_dict(pt.to_dict()) == pt

    def test_from_dict_defaults_masked_nr(self):
        pt = ThresholdPoint.from_dict({"threshold_db": 20.0})
        assert pt.masked is False
        assert pt.nr is False


# ---------------------------------------------------------------------------
# EarAudiogram
# ---------------------------------------------------------------------------

class TestEarAudiogram:
    def test_float_coercion(self):
        ear = EarAudiogram(air={500: 25.0, 1000: 30.0})
        assert isinstance(ear.air[500], ThresholdPoint)
        assert ear.air[500].threshold_db == 25.0
        assert ear.air[500].masked is False

    def test_threshold_point_passthrough(self):
        pt = ThresholdPoint(40.0, masked=True)
        ear = EarAudiogram(air={1000: pt})
        assert ear.air[1000] is pt

    def test_empty_bone_by_default(self):
        ear = EarAudiogram(air={500: 20.0})
        assert ear.bone == {}

    def test_available_frequencies_sorted(self):
        ear = EarAudiogram(air={4000: 50.0, 500: 20.0, 1000: 25.0})
        assert ear.available_frequencies() == [500, 1000, 4000]

    def test_available_frequencies_bone(self):
        ear = EarAudiogram(
            air={500: 20.0},
            bone={1000: ThresholdPoint(15.0), 2000: ThresholdPoint(20.0)},
        )
        assert ear.available_frequencies("bone") == [1000, 2000]

    def test_available_frequencies_invalid_pathway(self):
        ear = EarAudiogram(air={500: 20.0})
        with pytest.raises(ValueError):
            ear.available_frequencies("invalid")

    def test_pta_standard(self):
        ear = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0})
        assert ear.pta() == pytest.approx(30.0)

    def test_pta_partial(self):
        """Frequencies present averaged; missing ones skipped by default."""
        ear = EarAudiogram(air={500: 20.0, 1000: 40.0})
        assert ear.pta(freqs=(500, 1000, 2000)) == pytest.approx(30.0)

    def test_pta_require_all_missing(self):
        ear = EarAudiogram(air={500: 20.0, 1000: 40.0})
        assert ear.pta(freqs=(500, 1000, 2000), require_all=True) is None

    def test_pta_require_all_satisfied(self):
        ear = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0})
        assert ear.pta(freqs=(500, 1000, 2000), require_all=True) == pytest.approx(30.0)

    def test_pta_empty(self):
        ear = EarAudiogram()
        assert ear.pta() is None

    def test_pta_bone_pathway(self):
        ear = EarAudiogram(
            air={500: 50.0},
            bone={500: ThresholdPoint(20.0), 1000: ThresholdPoint(30.0), 2000: ThresholdPoint(40.0)},
        )
        assert ear.pta(pathway="bone") == pytest.approx(30.0)

    def test_pta_4tone(self):
        ear = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0, 4000: 50.0})
        assert ear.pta(standard="4tone") == pytest.approx(35.0)

    def test_pta_aao_hns(self):
        ear = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0, 3000: 50.0})
        assert ear.pta(standard="aao_hns") == pytest.approx(35.0)

    def test_pta_freqs_overrides_standard(self):
        ear = EarAudiogram(air={500: 10.0, 1000: 30.0, 2000: 50.0})
        assert ear.pta(standard="4tone", freqs=(500,)) == pytest.approx(10.0)

    def test_to_dict_structure(self):
        ear = EarAudiogram(
            air={500: ThresholdPoint(20.0)},
            bone={500: ThresholdPoint(15.0, masked=True)},
        )
        d = ear.to_dict()
        assert 500 in d["air"]
        assert d["air"][500] == {"threshold_db": 20.0, "masked": False, "nr": False}
        assert d["bone"][500]["masked"] is True

    def test_dict_round_trip(self):
        ear = EarAudiogram(
            air={500: ThresholdPoint(25.0), 1000: ThresholdPoint(120.0, nr=True)},
            bone={1000: ThresholdPoint(20.0, masked=True)},
        )
        restored = EarAudiogram.from_dict(ear.to_dict())
        assert restored.air[500].threshold_db == 25.0
        assert restored.air[1000].nr is True
        assert restored.bone[1000].masked is True

    def test_json_round_trip(self):
        ear = EarAudiogram(air={500: 30.0, 1000: ThresholdPoint(40.0, masked=True)})
        restored = EarAudiogram.from_json(ear.to_json())
        assert restored.air[500].threshold_db == 30.0
        assert restored.air[1000].masked is True


# ---------------------------------------------------------------------------
# BinauralAudiogram
# ---------------------------------------------------------------------------

class TestBinauralAudiogram:
    def test_metadata_stored(self, full_ba):
        assert full_ba.audiogram_id == "full-001"
        assert full_ba.subject_id == "pt-123"
        assert full_ba.performed_at == "2024-01-15"
        assert full_ba.source == "clinic"

    def test_get_threshold_air(self, full_ba):
        pt = full_ba.get_threshold(500, "left")
        assert pt.threshold_db == 25.0

    def test_get_threshold_bone(self, full_ba):
        pt = full_ba.get_threshold(1000, "right", pathway="bone")
        assert pt.threshold_db == 15.0
        assert pt.masked is True

    def test_get_threshold_missing(self, full_ba):
        assert full_ba.get_threshold(250, "left") is None

    def test_get_threshold_invalid_side(self, full_ba):
        with pytest.raises(ValueError):
            full_ba.get_threshold(500, "center")

    def test_pta_returns_both_ears(self, symmetric_ba):
        result = symmetric_ba.pta()
        assert "left" in result
        assert "right" in result

    def test_pta_known_values(self):
        left = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0})
        right = EarAudiogram(air={500: 10.0, 1000: 10.0, 2000: 10.0})
        ba = BinauralAudiogram(left, right)
        assert ba.pta()["left"] == pytest.approx(30.0)
        assert ba.pta()["right"] == pytest.approx(10.0)

    def test_better_ear_pta(self):
        left = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0})
        right = EarAudiogram(air={500: 10.0, 1000: 10.0, 2000: 10.0})
        ba = BinauralAudiogram(left, right)
        assert ba.better_ear_pta() == pytest.approx(10.0)

    def test_worse_ear_pta(self):
        left = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0})
        right = EarAudiogram(air={500: 10.0, 1000: 10.0, 2000: 10.0})
        ba = BinauralAudiogram(left, right)
        assert ba.worse_ear_pta() == pytest.approx(30.0)

    def test_symmetry_known_values(self):
        left = EarAudiogram(air={500: 30.0, 1000: 40.0})
        right = EarAudiogram(air={500: 20.0, 1000: 40.0})
        ba = BinauralAudiogram(left, right)
        sym = ba.symmetry()
        assert sym[500] == pytest.approx(10.0)
        assert sym[1000] == pytest.approx(0.0)

    def test_symmetry_only_shared_freqs(self):
        left = EarAudiogram(air={500: 20.0, 1000: 30.0})
        right = EarAudiogram(air={1000: 30.0, 2000: 40.0})
        ba = BinauralAudiogram(left, right)
        assert set(ba.symmetry().keys()) == {1000}

    # --- Long-row round-trips ---

    def test_long_rows_count(self, full_ba):
        rows = full_ba.to_long_rows()
        # 4 air left + 4 air right + 2 bone left + 2 bone right = 12
        assert len(rows) == 12

    def test_long_rows_pathways(self, full_ba):
        rows = full_ba.to_long_rows()
        pathways = {r["pathway"] for r in rows}
        assert pathways == {"air", "bone"}

    def test_long_rows_masked_preserved(self, full_ba):
        rows = full_ba.to_long_rows()
        bone_rows = [r for r in rows if r["pathway"] == "bone"]
        assert all(r["masked"] is True for r in bone_rows)

    def test_long_rows_nr_preserved(self, full_ba):
        rows = full_ba.to_long_rows()
        nr_rows = [r for r in rows if r["nr"]]
        assert len(nr_rows) == 1
        assert nr_rows[0]["freq_hz"] == 4000
        assert nr_rows[0]["ear"] == "right"

    def test_long_rows_round_trip(self, full_ba):
        rows = full_ba.to_long_rows()
        restored = BinauralAudiogram.from_long_rows(rows)
        # Air thresholds
        assert restored.left.air[500].threshold_db == 25.0
        assert restored.right.air[4000].nr is True
        assert restored.right.air[4000].threshold_db == 120.0
        # Bone thresholds
        assert restored.left.bone[1000].masked is True
        assert restored.left.bone[1000].threshold_db == 20.0

    def test_long_rows_metadata_round_trip(self, full_ba):
        rows = full_ba.to_long_rows()
        restored = BinauralAudiogram.from_long_rows(rows)
        assert restored.audiogram_id == "full-001"
        assert restored.subject_id == "pt-123"

    def test_from_long_rows_empty_raises(self):
        with pytest.raises(ValueError, match="at least one row"):
            BinauralAudiogram.from_long_rows([])

    def test_from_long_rows_mixed_ids_raises(self, full_ba):
        rows = full_ba.to_long_rows()
        rows[0] = {**rows[0], "audiogram_id": "different-id"}
        with pytest.raises(ValueError, match="audiogram_id"):
            BinauralAudiogram.from_long_rows(rows)

    # --- Table rows ---

    def test_table_rows_structure(self, full_ba):
        test_row, obs_rows = full_ba.to_table_rows()
        assert test_row["audiogram_id"] == "full-001"
        assert "schema_version" in test_row
        assert len(obs_rows) == 12

    def test_table_rows_obs_no_subject_id(self, full_ba):
        _, obs_rows = full_ba.to_table_rows()
        assert "subject_id" not in obs_rows[0]

    # --- Dict / JSON round-trips ---

    def test_dict_round_trip(self, full_ba):
        restored = BinauralAudiogram.from_dict(full_ba.to_dict())
        assert restored.audiogram_id == full_ba.audiogram_id
        assert restored.left.air[1000].threshold_db == full_ba.left.air[1000].threshold_db
        assert restored.right.bone[2000].masked == full_ba.right.bone[2000].masked

    def test_json_round_trip(self, full_ba):
        restored = BinauralAudiogram.from_json(full_ba.to_json())
        assert restored.subject_id == "pt-123"
        assert restored.right.air[4000].nr is True

