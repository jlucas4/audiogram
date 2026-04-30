"""Pure-function metrics: PTA, severity, ABG, loss type, and summary registry."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .audiogram import BinauralAudiogram

SummaryMetric = Callable[..., dict[str, Any]]


def pta_from_thresholds(
    thresholds: dict[int, float],
    freqs: Iterable[int] = (500, 1000, 2000),
    *,
    require_all: bool = False,
) -> float | None:
    """
    Compute pure-tone average from a frequency->threshold mapping.

    Parameters
    ----------
    thresholds
        Mapping of frequency (Hz) -> threshold (dB HL).
    freqs
        Frequencies to include in the average.
    require_all
        If True, return None unless all requested frequencies are present.

    Returns
    -------
    float | None
        Mean threshold across requested frequencies, or None if unavailable.
    """
    req_freqs = tuple(int(f) for f in freqs)
    vals = [float(thresholds[f]) for f in req_freqs if f in thresholds]

    if require_all and len(vals) != len(req_freqs):
        return None
    if not vals:
        return None

    return float(sum(vals) / len(vals))


def symmetry_from_thresholds(
    left: dict[int, float],
    right: dict[int, float],
) -> dict[int, float]:
    """
    Compute left-right asymmetry by frequency.

    Returns
    -------
    dict[int, float]
        Mapping of frequency -> (left - right) threshold difference.
        Positive values mean the left ear is worse at that frequency.
    """
    common_freqs = set(left) & set(right)
    return {int(f): float(left[f] - right[f]) for f in sorted(common_freqs)}


def better_ear_from_values(
    values: Iterable[float | None],
) -> float | None:
    """
    Return the lower (better) non-missing value.
    """
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return min(cleaned)


def worse_ear_from_values(
    values: Iterable[float | None],
) -> float | None:
    """
    Return the higher (worse) non-missing value.
    """
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return max(cleaned)


# ---------------------------------------------------------------------------
# Speech metrics
# ---------------------------------------------------------------------------


def best_wrs(wrs_list: list) -> Any | None:
    """Return the WRS with the highest score_pct, or None if the list is empty."""
    if not wrs_list:
        return None
    return max(wrs_list, key=lambda w: w.score_pct)


def srt_pta_agreement(
    srt: float | None,
    thresholds: dict[int, float],
    freqs: Iterable[int] = (500, 1000, 2000),
) -> float | None:
    """Compute SRT minus PTA.

    Clinically, SRT should be within ~10 dB of the 3-frequency PTA.
    A large positive value suggests the SRT is worse than expected.
    Returns None if SRT or PTA is unavailable.
    """
    if srt is None:
        return None
    pta = pta_from_thresholds(thresholds, freqs=freqs)
    if pta is None:
        return None
    return srt - pta


# ---------------------------------------------------------------------------
# Air-bone gap
# ---------------------------------------------------------------------------


def abg_from_thresholds(
    air: dict[int, float],
    bone: dict[int, float],
    *,
    mask_warning: bool = False,
    bone_masked: dict[int, bool] | None = None,
) -> dict[int, float]:
    """Compute air-bone gap at each frequency where both air and bone exist.

    Parameters
    ----------
    air / bone
        Frequency -> threshold (dB HL) mappings.
    mask_warning
        If True, emit a warning when any bone threshold used is unmasked.
    bone_masked
        Frequency -> masked flag for bone thresholds. Only used when
        mask_warning is True.
    """
    common = set(air) & set(bone)
    if mask_warning and bone_masked is not None:
        unmasked = [f for f in sorted(common) if not bone_masked.get(f, False)]
        if unmasked:
            warnings.warn(
                f"Bone conduction unmasked at {unmasked} Hz — "
                f"ABG may not reflect true conductive component.",
                stacklevel=2,
            )
    return {int(f): float(air[f] - bone[f]) for f in sorted(common)}


def abg_pta(
    air: dict[int, float],
    bone: dict[int, float],
    *,
    standard: str = "who2021",
    freqs: Iterable[int] = (500, 1000, 2000, 4000),
    require_all: bool = False,
) -> float | None:
    """Compute PTA of the air-bone gap.

    Parameters
    ----------
    standard
        'who2021' uses *freqs* (default 500/1000/2000/4000).
        'aao_hns' uses 500/1000/2000/3000 with avg(2000, 4000) fallback.
    """
    if standard not in VALID_STANDARDS:
        raise ValueError(f"Unknown standard {standard!r}. Expected one of: {sorted(VALID_STANDARDS)}")

    gaps = abg_from_thresholds(air, bone)
    if not gaps:
        return None

    if standard == "aao_hns":
        return aao_hns_pta(gaps, require_all=require_all)
    return pta_from_thresholds(gaps, freqs=freqs, require_all=require_all)


LOSS_TYPES = ("normal", "sensorineural", "conductive", "mixed")


def loss_type(
    air_pta: float | None,
    bone_pta: float | None,
    abg: float | None,
    *,
    abg_cutoff: float = 10.0,
    normal_cutoff: float = 25.0,
) -> str | None:
    """Classify hearing loss type from PTA values.

    Returns 'normal', 'sensorineural', 'conductive', or 'mixed'.
    Returns None if air or bone PTA is unavailable.
    """
    if air_pta is None or bone_pta is None or abg is None:
        return None
    if air_pta <= normal_cutoff:
        return "normal"
    if abg < abg_cutoff:
        return "sensorineural"
    if bone_pta <= normal_cutoff:
        return "conductive"
    return "mixed"


# ---------------------------------------------------------------------------
# Hearing loss severity (WHO 2021)
# ---------------------------------------------------------------------------

SEVERITY_GRADES: list[tuple[float, str]] = [
    (25.0, "normal"),
    (40.0, "mild"),
    (55.0, "moderate"),
    (70.0, "moderately_severe"),
    (90.0, "severe"),
    (float("inf"), "profound"),
]

VALID_STANDARDS = {"who2021", "aao_hns"}


def severity_from_pta(pta: float | None) -> str | None:
    """Classify hearing loss severity from a PTA value.

    Uses WHO 2021 grading: normal (<=25), mild (26-40), moderate (41-55),
    moderately severe (56-70), severe (71-90), profound (>90).
    """
    if pta is None:
        return None
    for cutoff, grade in SEVERITY_GRADES:
        if pta <= cutoff:
            return grade
    return None


def aao_hns_pta(
    thresholds: dict[int, float],
    *,
    require_all: bool = False,
) -> float | None:
    """Compute PTA using the AAO-HNS method (500/1000/2000/3000 Hz).

    If 3000 Hz is not present, uses the average of 2000 and 4000 Hz as
    the 3000 Hz value. If that fallback is also not possible, the 3000 Hz
    contribution is omitted (unless require_all is True).
    """
    vals: list[float] = []
    base_freqs = (500, 1000, 2000)
    expected = 4

    for f in base_freqs:
        if f in thresholds:
            vals.append(float(thresholds[f]))
        elif require_all:
            return None

    if 3000 in thresholds:
        vals.append(float(thresholds[3000]))
    elif 2000 in thresholds and 4000 in thresholds:
        vals.append((float(thresholds[2000]) + float(thresholds[4000])) / 2.0)
    elif require_all:
        return None

    if not vals:
        return None
    if require_all and len(vals) != expected:
        return None

    return sum(vals) / len(vals)


def severity_from_thresholds(
    thresholds: dict[int, float],
    freqs: Iterable[int] = (500, 1000, 2000, 4000),
    *,
    standard: str = "who2021",
    require_all: bool = False,
) -> str | None:
    """Classify hearing loss severity from raw thresholds.

    Parameters
    ----------
    thresholds
        Mapping of frequency (Hz) -> threshold (dB HL).
    freqs
        Frequencies for the PTA calculation (WHO 2021 only).
        Ignored when standard='aao_hns'.
    standard
        'who2021' (default) or 'aao_hns'. AAO-HNS uses 500/1000/2000/3000
        with a fallback of avg(2000, 4000) when 3000 is absent.
    require_all
        If True, return None unless all required frequencies are present.
    """
    if standard not in VALID_STANDARDS:
        raise ValueError(f"Unknown standard {standard!r}. Expected one of: {sorted(VALID_STANDARDS)}")

    if standard == "aao_hns":
        pta = aao_hns_pta(thresholds, require_all=require_all)
    else:
        pta = pta_from_thresholds(thresholds, freqs=freqs, require_all=require_all)
    return severity_from_pta(pta)


# ---------------------------------------------------------------------------
# Summary metric registry
# ---------------------------------------------------------------------------

SUMMARY_METRICS: dict[str, SummaryMetric] = {}


def register_summary_metric(name: str, fn: SummaryMetric) -> None:
    """Register a metric function for inclusion in summary() output."""
    SUMMARY_METRICS[name] = fn


def unregister_summary_metric(name: str) -> None:
    """Remove a previously registered summary metric."""
    SUMMARY_METRICS.pop(name, None)


def compute_summary(
    ba: BinauralAudiogram,
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run registered summary metrics and return a flat dict of results.

    Parameters
    ----------
    ba
        The audiogram to summarize.
    include
        If provided, only run these metric categories.
    exclude
        If provided, skip these metric categories.
    **kwargs
        Forwarded to each metric function (e.g. ``standard="aao_hns"``).
    """
    include_set = set(include) if include is not None else None
    exclude_set = set(exclude) if exclude is not None else set()

    result: dict[str, Any] = {}
    for name, fn in SUMMARY_METRICS.items():
        if include_set is not None and name not in include_set:
            continue
        if name in exclude_set:
            continue
        result.update(fn(ba, **kwargs))
    return result


# ---------------------------------------------------------------------------
# Built-in summary metrics
# ---------------------------------------------------------------------------

def _pta_summary(ba: BinauralAudiogram, **kwargs: Any) -> dict[str, Any]:
    ptas = ba.pta()
    return {
        "pta_right": ptas["right"],
        "pta_left": ptas["left"],
        "better_ear_pta": ba.better_ear_pta(),
        "worse_ear_pta": ba.worse_ear_pta(),
    }


def _asymmetry_summary(ba: BinauralAudiogram, **kwargs: Any) -> dict[str, Any]:
    from .asymmetry import ASYMMETRY_CRITERIA
    return {
        f"asymmetric_{name}": ba.is_asymmetric(name)
        for name in ASYMMETRY_CRITERIA
    }


def _frequency_count_summary(ba: BinauralAudiogram, **kwargs: Any) -> dict[str, Any]:
    result = {
        "right_air_freq_count": len(ba.right.air),
        "left_air_freq_count": len(ba.left.air),
        "right_bone_freq_count": len(ba.right.bone),
        "left_bone_freq_count": len(ba.left.bone),
    }
    if ba.right.soundfield or ba.left.soundfield:
        result["right_soundfield_freq_count"] = len(ba.right.soundfield)
        result["left_soundfield_freq_count"] = len(ba.left.soundfield)
    if ba.right.ci or ba.left.ci:
        result["right_ci_freq_count"] = len(ba.right.ci)
        result["left_ci_freq_count"] = len(ba.left.ci)
    return result


def _severity_summary(ba: BinauralAudiogram, **kwargs: Any) -> dict[str, Any]:
    standard = kwargs.get("standard", "who2021")
    return {
        "right_severity": severity_from_thresholds(
            {f: p.threshold_db for f, p in ba.right.air.items()},
            standard=standard,
        ),
        "left_severity": severity_from_thresholds(
            {f: p.threshold_db for f, p in ba.left.air.items()},
            standard=standard,
        ),
    }


def _speech_summary(ba: BinauralAudiogram, **kwargs: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for ear_name, ear_obj in (("right", ba.right), ("left", ba.left)):
        result[f"{ear_name}_srt"] = ear_obj.srt
        result[f"{ear_name}_sat"] = ear_obj.sat
        best = ear_obj.best_wrs()
        result[f"{ear_name}_wrs"] = best.score_pct if best else None
        result[f"{ear_name}_wrs_level"] = best.level_db if best else None
        result[f"{ear_name}_srt_pta_agreement"] = ear_obj.srt_pta_agreement()
    return result


def _abg_summary(ba: BinauralAudiogram, **kwargs: Any) -> dict[str, Any]:
    standard = kwargs.get("standard", "who2021")
    result: dict[str, Any] = {}
    for ear_name, ear_obj in (("right", ba.right), ("left", ba.left)):
        air = {f: p.threshold_db for f, p in ear_obj.air.items()}
        bone = {f: p.threshold_db for f, p in ear_obj.bone.items()}
        ear_abg_pta = abg_pta(air, bone, standard=standard)
        air_pta_val = pta_from_thresholds(air) if standard == "who2021" else aao_hns_pta(air)
        bone_pta_val = pta_from_thresholds(bone) if standard == "who2021" else aao_hns_pta(bone)
        result[f"{ear_name}_abg_pta"] = ear_abg_pta
        result[f"{ear_name}_loss_type"] = loss_type(air_pta_val, bone_pta_val, ear_abg_pta)
    return result


register_summary_metric("pta", _pta_summary)
register_summary_metric("severity", _severity_summary)
register_summary_metric("speech", _speech_summary)
register_summary_metric("asymmetry", _asymmetry_summary)
register_summary_metric("frequency_count", _frequency_count_summary)
register_summary_metric("abg", _abg_summary)
