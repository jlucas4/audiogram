"""audiogram: typed Python objects for clinical audiometric data."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("audiogram")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from .audiogram import ThresholdPoint, WordRecognitionScore, EarAudiogram, BinauralAudiogram
from .asymmetry import InterauralDifference, ASYMMETRY_CRITERIA, compute_interaural_differences
from .metrics import (
    pta_from_thresholds,
    PTA_STANDARDS,
    symmetry_from_thresholds,
    better_ear_from_values,
    worse_ear_from_values,
    best_wrs,
    srt_pta_agreement,
    severity_from_pta,
    severity_from_thresholds,
    abg_from_thresholds,
    abg_pta,
    loss_type,
    LOSS_TYPES,
    compute_summary,
    register_summary_metric,
    unregister_summary_metric,
    SUMMARY_METRICS,
    SEVERITY_GRADES,
    VALID_STANDARDS,
)
from .serialization import enrich_wide_rows
from .schema import (
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
    MEASURE,
    VALUE,
    LEVEL_DB,
    WORD_LIST,
    MEASURE_TYPE,
    LONG_COLUMNS,
    SPEECH_COLUMNS,
    OBSERVATIONS_COLUMNS,
    TESTS_COLUMNS,
    SCHEMA_VERSION,
    validate_long_row,
    validate_long_rows,
    wide_column_name,
    parse_wide_column,
    canonical_wide_columns,
    apply_column_map,
)

__all__ = [
    "__version__",
    # Core objects
    "ThresholdPoint",
    "WordRecognitionScore",
    "EarAudiogram",
    "BinauralAudiogram",
    # Schema column names
    "AUDIOGRAM_ID",
    "SUBJECT_ID",
    "PERFORMED_AT",
    "SOURCE",
    "EAR",
    "FREQ_HZ",
    "THRESHOLD_DB",
    "PATHWAY",
    "MASKED",
    "NR",
    # Schema column names (speech)
    "MEASURE",
    "VALUE",
    "LEVEL_DB",
    "WORD_LIST",
    "MEASURE_TYPE",
    # Schema structure
    "LONG_COLUMNS",
    "SPEECH_COLUMNS",
    "OBSERVATIONS_COLUMNS",
    "TESTS_COLUMNS",
    "SCHEMA_VERSION",
    # Validation
    "validate_long_row",
    "validate_long_rows",
    # Wide-format helpers
    "wide_column_name",
    "parse_wide_column",
    "canonical_wide_columns",
    "apply_column_map",
    # Batch enrichment
    "enrich_wide_rows",
    # Asymmetry
    "InterauralDifference",
    "ASYMMETRY_CRITERIA",
    "compute_interaural_differences",
    # Metrics (standalone functions)
    "pta_from_thresholds",
    "PTA_STANDARDS",
    "symmetry_from_thresholds",
    "better_ear_from_values",
    "worse_ear_from_values",
    "best_wrs",
    "srt_pta_agreement",
    # ABG / loss type
    "abg_from_thresholds",
    "abg_pta",
    "loss_type",
    "LOSS_TYPES",
    "compute_summary",
    # Summary registry
    "register_summary_metric",
    "unregister_summary_metric",
    "SUMMARY_METRICS",
    # Severity
    "SEVERITY_GRADES",
    "VALID_STANDARDS",
    "severity_from_pta",
    "severity_from_thresholds",
]
