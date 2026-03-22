from __future__ import annotations

"""Canonical dataframe / table schema for audiogram data.

This module defines the opinionated v1 schema used to move between:
- Audiogram / BinauralAudiogram objects
- flat long-form dataframes for analysis / plotting
- normalized tables for database-style storage

Design goals
------------
- Keep the core schema small and stable.
- Prefer long/tidy format as the canonical interchange format.
- Support a normalized two-table representation for storage pipelines.
- Make validation explicit and easy to reuse.
"""

from dataclasses import dataclass
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Canonical column names
# ---------------------------------------------------------------------------

AUDIOGRAM_ID = "audiogram_id"
SUBJECT_ID = "subject_id"
PERFORMED_AT = "performed_at"
SOURCE = "source"
META = "meta"
SCHEMA_VERSION_COL = "schema_version"

EAR = "ear"
FREQ_HZ = "freq_hz"
THRESHOLD_DB = "threshold_db"
PATHWAY = "pathway"
MASKED = "masked"
NR = "nr"


# ---------------------------------------------------------------------------
# Allowed values / controlled vocab (v1)
# ---------------------------------------------------------------------------

EAR_LEFT = "left"
EAR_RIGHT = "right"
VALID_EARS = {EAR_LEFT, EAR_RIGHT}

PATHWAY_AIR = "air"
PATHWAY_BONE = "bone"
VALID_PATHWAYS = {PATHWAY_AIR, PATHWAY_BONE}

# Common clinical / research frequencies for v1 validation.
# Keep this structured-but-extensible: validators may choose to be strict or permissive.
STANDARD_FREQUENCIES_HZ = {
    125,
    250,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    4000,
    6000,
    8000,
    10000,
    12500,
    14000,
    16000,
}


# ---------------------------------------------------------------------------
# Flat long-form dataframe schema (default export for most users)
# One row = one threshold observation, with lightweight metadata repeated.
# ---------------------------------------------------------------------------

LONG_COLUMNS = [
    AUDIOGRAM_ID,
    SUBJECT_ID,
    PERFORMED_AT,
    SOURCE,
    EAR,
    FREQ_HZ,
    THRESHOLD_DB,
    PATHWAY,
    MASKED,
    NR,
]


# ---------------------------------------------------------------------------
# Normalized table schema (advanced / database-oriented export)
# ---------------------------------------------------------------------------

TESTS_COLUMNS = [
    AUDIOGRAM_ID,
    SUBJECT_ID,
    PERFORMED_AT,
    SOURCE,
    SCHEMA_VERSION_COL,
    META,
]

OBSERVATIONS_COLUMNS = [
    AUDIOGRAM_ID,
    EAR,
    FREQ_HZ,
    THRESHOLD_DB,
    PATHWAY,
    MASKED,
    NR,
]


# ---------------------------------------------------------------------------
# Wide-format naming grammar (convenience export/import only; not canonical)
#
# Pattern: {ear}_{pathway}_{measure}_{freq}
# Examples:
#   L_air_thr_500
#   R_air_nr_8000
#   L_bone_masked_2000
# ---------------------------------------------------------------------------

WIDE_EARS = {"L", "R"}
WIDE_MEASURES_CORE = {"thr", "nr", "masked"}


@dataclass(frozen=True)
class WideColumnParts:
    ear: str
    pathway: str
    measure: str
    freq_hz: int


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_ear(value: str) -> None:
    if value not in VALID_EARS:
        raise ValueError(f"Invalid ear '{value}'. Expected one of: {sorted(VALID_EARS)}")



def validate_pathway(value: str) -> None:
    if value not in VALID_PATHWAYS:
        raise ValueError(f"Invalid pathway '{value}'. Expected one of: {sorted(VALID_PATHWAYS)}")



def validate_frequency(freq_hz: int, *, strict: bool = False) -> None:
    freq_hz = int(freq_hz)
    if strict and freq_hz not in STANDARD_FREQUENCIES_HZ:
        raise ValueError(
            f"Unexpected frequency {freq_hz} Hz for strict validation. "
            f"Expected one of: {sorted(STANDARD_FREQUENCIES_HZ)}"
        )



def validate_long_row(row: dict[str, Any], *, strict_freqs: bool = False) -> None:
    """Validate one long-form observation row.

    Required fields for v1:
    - audiogram_id
    - ear
    - freq_hz
    - threshold_db
    - pathway
    - masked
    - nr
    """
    required = {
        AUDIOGRAM_ID,
        EAR,
        FREQ_HZ,
        THRESHOLD_DB,
        PATHWAY,
        MASKED,
        NR,
    }
    missing = sorted(required - set(row.keys()))
    if missing:
        raise ValueError(f"Long row missing required columns: {missing}")

    validate_ear(str(row[EAR]))
    validate_pathway(str(row[PATHWAY]))
    validate_frequency(int(row[FREQ_HZ]), strict=strict_freqs)

    # Threshold may be int/float-like; convertibility is enough for v1.
    try:
        float(row[THRESHOLD_DB])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid threshold_db value: {row[THRESHOLD_DB]!r}") from e

    if not isinstance(row[MASKED], bool):
        raise ValueError(f"masked must be bool, got {type(row[MASKED]).__name__}")
    if not isinstance(row[NR], bool):
        raise ValueError(f"nr must be bool, got {type(row[NR]).__name__}")



def validate_long_rows(rows: Iterable[dict[str, Any]], *, strict_freqs: bool = False) -> None:
    for row in rows:
        validate_long_row(row, strict_freqs=strict_freqs)



def validate_tests_row(row: dict[str, Any]) -> None:
    required = {AUDIOGRAM_ID}
    missing = sorted(required - set(row.keys()))
    if missing:
        raise ValueError(f"Tests row missing required columns: {missing}")



def validate_observations_row(row: dict[str, Any], *, strict_freqs: bool = False) -> None:
    required = {
        AUDIOGRAM_ID,
        EAR,
        FREQ_HZ,
        THRESHOLD_DB,
        PATHWAY,
        MASKED,
        NR,
    }
    missing = sorted(required - set(row.keys()))
    if missing:
        raise ValueError(f"Observations row missing required columns: {missing}")

    validate_ear(str(row[EAR]))
    validate_pathway(str(row[PATHWAY]))
    validate_frequency(int(row[FREQ_HZ]), strict=strict_freqs)

    try:
        float(row[THRESHOLD_DB])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid threshold_db value: {row[THRESHOLD_DB]!r}") from e

    if not isinstance(row[MASKED], bool):
        raise ValueError(f"masked must be bool, got {type(row[MASKED]).__name__}")
    if not isinstance(row[NR], bool):
        raise ValueError(f"nr must be bool, got {type(row[NR]).__name__}")


# ---------------------------------------------------------------------------
# Wide-column helpers
# ---------------------------------------------------------------------------


def parse_wide_column(name: str) -> WideColumnParts | None:
    """Parse a wide-format measurement column.

    Returns None for columns that do not match the measurement grammar.
    This allows metadata columns (e.g. audiogram_id, subject_id) to coexist.
    """
    parts = name.split("_")
    if len(parts) != 4:
        return None

    ear, pathway, measure, freq = parts
    if ear not in WIDE_EARS:
        return None
    if pathway not in VALID_PATHWAYS:
        return None
    if measure not in WIDE_MEASURES_CORE:
        # Extensible grammar: allow parser to reject unknown core measures for now.
        return None

    try:
        freq_hz = int(freq)
    except ValueError:
        return None

    return WideColumnParts(ear=ear, pathway=pathway, measure=measure, freq_hz=freq_hz)



def make_wide_column(*, ear: str, pathway: str, measure: str, freq_hz: int) -> str:
    """Build a canonical wide-format column name.

    Parameters
    ----------
    ear
        'L' or 'R'
    pathway
        'air' or 'bone'
    measure
        v1 core: 'thr', 'nr', or 'masked'
    freq_hz
        Frequency in Hz
    """
    if ear not in WIDE_EARS:
        raise ValueError(f"Invalid wide-format ear '{ear}'. Expected one of: {sorted(WIDE_EARS)}")
    validate_pathway(pathway)
    if measure not in WIDE_MEASURES_CORE:
        raise ValueError(
            f"Invalid wide-format measure '{measure}'. Expected one of: {sorted(WIDE_MEASURES_CORE)}"
        )
    return f"{ear}_{pathway}_{measure}_{int(freq_hz)}"
