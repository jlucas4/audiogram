"""Tests for schema validation helpers."""
import pytest

from audiogram_object import validate_long_row, validate_long_rows
from audiogram_object.schema import (
    validate_observations_row,
    validate_tests_row,
    validate_ear,
    validate_pathway,
    validate_frequency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_long_row(**overrides):
    base = {
        "audiogram_id": "test-001",
        "subject_id": "pt-1",
        "performed_at": "2024-01-01",
        "source": "clinic",
        "ear": "left",
        "freq_hz": 1000,
        "threshold_db": 25.0,
        "pathway": "air",
        "masked": False,
        "nr": False,
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# validate_long_row
# ---------------------------------------------------------------------------

class TestValidateLongRow:
    def test_valid_row_passes(self):
        validate_long_row(_valid_long_row())

    def test_missing_audiogram_id_raises(self):
        row = _valid_long_row()
        del row["audiogram_id"]
        with pytest.raises(ValueError, match="audiogram_id"):
            validate_long_row(row)

    def test_missing_ear_raises(self):
        row = _valid_long_row()
        del row["ear"]
        with pytest.raises(ValueError, match="ear"):
            validate_long_row(row)

    def test_invalid_ear_raises(self):
        with pytest.raises(ValueError, match="Invalid ear"):
            validate_long_row(_valid_long_row(ear="center"))

    def test_invalid_pathway_raises(self):
        with pytest.raises(ValueError, match="Invalid pathway"):
            validate_long_row(_valid_long_row(pathway="skin"))

    def test_bone_pathway_valid(self):
        validate_long_row(_valid_long_row(pathway="bone"))

    def test_right_ear_valid(self):
        validate_long_row(_valid_long_row(ear="right"))

    def test_non_bool_masked_raises(self):
        with pytest.raises(ValueError, match="masked must be bool"):
            validate_long_row(_valid_long_row(masked="yes"))

    def test_non_bool_nr_raises(self):
        with pytest.raises(ValueError, match="nr must be bool"):
            validate_long_row(_valid_long_row(nr=1))

    def test_non_numeric_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold_db"):
            validate_long_row(_valid_long_row(threshold_db="loud"))

    def test_strict_freqs_standard_passes(self):
        validate_long_row(_valid_long_row(freq_hz=4000), strict_freqs=True)

    def test_strict_freqs_nonstandard_raises(self):
        with pytest.raises(ValueError, match="Unexpected frequency"):
            validate_long_row(_valid_long_row(freq_hz=999), strict_freqs=True)

    def test_nonstandard_freq_permissive_passes(self):
        validate_long_row(_valid_long_row(freq_hz=999), strict_freqs=False)


# ---------------------------------------------------------------------------
# validate_long_rows
# ---------------------------------------------------------------------------

class TestValidateLongRows:
    def test_multiple_valid_rows(self):
        rows = [_valid_long_row(freq_hz=f) for f in [500, 1000, 2000]]
        validate_long_rows(rows)

    def test_one_invalid_row_raises(self):
        rows = [_valid_long_row(freq_hz=500), _valid_long_row(ear="bad")]
        with pytest.raises(ValueError):
            validate_long_rows(rows)


# ---------------------------------------------------------------------------
# validate_tests_row
# ---------------------------------------------------------------------------

class TestValidateTestsRow:
    def test_valid(self):
        validate_tests_row({"audiogram_id": "x", "subject_id": "p"})

    def test_missing_audiogram_id(self):
        with pytest.raises(ValueError, match="audiogram_id"):
            validate_tests_row({"subject_id": "p"})


# ---------------------------------------------------------------------------
# validate_observations_row
# ---------------------------------------------------------------------------

class TestValidateObservationsRow:
    def test_valid(self):
        validate_observations_row({
            "audiogram_id": "x",
            "ear": "left",
            "freq_hz": 1000,
            "threshold_db": 25.0,
            "pathway": "air",
            "masked": False,
            "nr": False,
        })

    def test_missing_field_raises(self):
        with pytest.raises(ValueError):
            validate_observations_row({"audiogram_id": "x"})

