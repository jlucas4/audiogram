from __future__ import annotations

from typing import Iterable


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


def available_frequencies_from_thresholds(
    thresholds: dict[int, float],
) -> list[int]:
    """
    Return sorted available frequencies from a threshold mapping.
    """
    return sorted(int(f) for f in thresholds.keys())


def has_required_frequencies(
    thresholds: dict[int, float],
    freqs: Iterable[int],
) -> bool:
    """
    Return True if all requested frequencies are present.
    """
    req_freqs = {int(f) for f in freqs}
    return req_freqs.issubset(thresholds.keys())