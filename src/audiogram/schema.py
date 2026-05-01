from __future__ import annotations

"""Canonical dataframe / table schema for audiogram data.

This module defines the opinionated v1 schema used to move between:
- Audiogram / BinauralAudiogram objects
- flat long-form dataframes for analysis / plotting
- normalized tables for database-style storage

Design goals
------------
- Keep the core schema small and stable.
- Dict/JSON is the canonical lossless interchange format.
- Long-form tidy schemas for thresholds and speech separately.
- Wide-format as a flat, full-fidelity tabular representation.
- Support a normalized two-table representation for storage pipelines.
- Make validation explicit and easy to reuse.
"""

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

# Shared
EAR = "ear"
MASKED = "masked"

# Threshold columns
FREQ_HZ = "freq_hz"
THRESHOLD_DB = "threshold_db"
PATHWAY = "pathway"
NR = "nr"

# Speech columns
MEASURE = "measure"
VALUE = "value"
LEVEL_DB = "level_db"
WORD_LIST = "word_list"

# Combined long-form discriminator
MEASURE_TYPE = "measure_type"
MEASURE_TYPE_THRESHOLD = "threshold"
MEASURE_TYPE_SPEECH = "speech"


# ---------------------------------------------------------------------------
# Allowed values / controlled vocab (v1)
# ---------------------------------------------------------------------------

EAR_LEFT = "left"
EAR_RIGHT = "right"
VALID_EARS = {EAR_LEFT, EAR_RIGHT}

PATHWAY_AIR = "air"
PATHWAY_BONE = "bone"
PATHWAY_SOUNDFIELD = "soundfield"
PATHWAY_CI = "ci"
VALID_PATHWAYS = {PATHWAY_AIR, PATHWAY_BONE, PATHWAY_SOUNDFIELD, PATHWAY_CI}

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
    11200,
    12500,
    14000,
    16000,
    18000,
    20000,
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
# Speech long-form schema
# One row = one speech measurement per ear.
# ---------------------------------------------------------------------------

SPEECH_COLUMNS = [
    AUDIOGRAM_ID,
    SUBJECT_ID,
    PERFORMED_AT,
    SOURCE,
    EAR,
    MEASURE,
    VALUE,
    LEVEL_DB,
    MASKED,
    WORD_LIST,
]

VALID_SPEECH_MEASURES = {"srt", "sat", "wrs"}


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



_THRESHOLD_ROW_REQUIRED = {
    AUDIOGRAM_ID,
    EAR,
    FREQ_HZ,
    THRESHOLD_DB,
    PATHWAY,
    MASKED,
    NR,
}


def _validate_threshold_row(row: dict[str, Any], *, strict_freqs: bool = False, label: str = "Row") -> None:
    """Shared validation for long-form and observations rows."""
    missing = sorted(_THRESHOLD_ROW_REQUIRED - set(row.keys()))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")

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


def validate_long_row(row: dict[str, Any], *, strict_freqs: bool = False) -> None:
    """Validate one long-form observation row."""
    _validate_threshold_row(row, strict_freqs=strict_freqs, label="Long row")



def validate_long_rows(rows: Iterable[dict[str, Any]], *, strict_freqs: bool = False) -> None:
    for row in rows:
        validate_long_row(row, strict_freqs=strict_freqs)



def validate_tests_row(row: dict[str, Any]) -> None:
    required = {AUDIOGRAM_ID}
    missing = sorted(required - set(row.keys()))
    if missing:
        raise ValueError(f"Tests row missing required columns: {missing}")



# ---------------------------------------------------------------------------
# Wide-format column convention
# ---------------------------------------------------------------------------
# Canonical pattern: {ear}_{pathway}_{frequency}
# Optional boolean suffixes: {ear}_{pathway}_{frequency}_masked
#                             {ear}_{pathway}_{frequency}_nr
#
# Examples: r_air_500, l_bone_1000, r_air_4000_masked, l_air_250_nr

WIDE_EARS = {"r": EAR_RIGHT, "l": EAR_LEFT}
WIDE_PATHWAYS = {
    "air": PATHWAY_AIR,
    "bone": PATHWAY_BONE,
    "sf": PATHWAY_SOUNDFIELD,
    "ci": PATHWAY_CI,
}
WIDE_EARS_INV = {v: k for k, v in WIDE_EARS.items()}
WIDE_PATHWAYS_INV = {v: k for k, v in WIDE_PATHWAYS.items()}

WIDE_META_COLUMNS = [AUDIOGRAM_ID, SUBJECT_ID, PERFORMED_AT, SOURCE]


def wide_column_name(ear: str, pathway: str, freq_hz: int, suffix: str | None = None) -> str:
    """Build a canonical wide column name.

    >>> wide_column_name('right', 'air', 500)
    'r_air_500'
    >>> wide_column_name('left', 'bone', 1000, 'masked')
    'l_bone_1000_masked'
    """
    e = WIDE_EARS_INV.get(ear, ear)
    p = WIDE_PATHWAYS_INV.get(pathway, pathway)
    col = f"{e}_{p}_{freq_hz}"
    if suffix:
        col = f"{col}_{suffix}"
    return col


def parse_wide_column(col: str) -> dict[str, Any] | None:
    """Parse a canonical wide column name into its components.

    Returns None if the column does not match the convention.

    >>> parse_wide_column('r_air_500')
    {'ear': 'right', 'pathway': 'air', 'freq_hz': 500, 'field': 'threshold'}
    >>> parse_wide_column('l_bone_1000_masked')
    {'ear': 'left', 'pathway': 'bone', 'freq_hz': 1000, 'field': 'masked'}
    """
    parts = col.split("_")
    if len(parts) < 3:
        return None

    ear_abbr = parts[0]
    pathway_abbr = parts[1]

    if ear_abbr not in WIDE_EARS or pathway_abbr not in WIDE_PATHWAYS:
        return None

    try:
        freq_hz = int(parts[2])
    except ValueError:
        return None

    field = "threshold"
    if len(parts) == 4:
        if parts[3] in ("masked", "nr"):
            field = parts[3]
        else:
            return None
    elif len(parts) > 4:
        return None

    return {
        "ear": WIDE_EARS[ear_abbr],
        "pathway": WIDE_PATHWAYS[pathway_abbr],
        "freq_hz": freq_hz,
        "field": field,
    }


def canonical_wide_columns(
    frequencies: Iterable[int] | None = None,
    pathways: Iterable[str] = ("air",),
    include_masked: bool = False,
    include_nr: bool = False,
) -> list[str]:
    """Generate the list of canonical wide column names for a given configuration.

    >>> canonical_wide_columns([500, 1000], pathways=['air'])
    ['r_air_500', 'r_air_1000', 'l_air_500', 'l_air_1000']
    """
    if frequencies is None:
        frequencies = sorted(STANDARD_FREQUENCIES_HZ)
    else:
        frequencies = sorted(frequencies)

    cols: list[str] = []
    for ear in ("r", "l"):
        for pw in pathways:
            pw_abbr = WIDE_PATHWAYS_INV.get(pw, pw)
            for f in frequencies:
                cols.append(f"{ear}_{pw_abbr}_{f}")
                if include_masked:
                    cols.append(f"{ear}_{pw_abbr}_{f}_masked")
                if include_nr:
                    cols.append(f"{ear}_{pw_abbr}_{f}_nr")
    return cols


def apply_column_map(row: dict[str, Any], column_map: dict[str, str]) -> dict[str, Any]:
    """Rename keys in *row* according to *column_map*.

    Keys not in the map are passed through unchanged. This is the user's
    one-time bridge from their dataset's naming convention to the canonical
    wide schema.

    >>> apply_column_map({'R_AC_500': 25.0, 'patient': 'A'}, {'R_AC_500': 'r_air_500'})
    {'r_air_500': 25.0, 'patient': 'A'}
    """
    return {column_map.get(k, k): v for k, v in row.items()}


def validate_observations_row(row: dict[str, Any], *, strict_freqs: bool = False) -> None:
    """Validate one normalized observations-table row."""
    _validate_threshold_row(row, strict_freqs=strict_freqs, label="Observations row")

