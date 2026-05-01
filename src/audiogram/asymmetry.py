from __future__ import annotations

"""Interaural difference computation and asymmetry criteria.

Criterion functions
-------------------
A criterion is any callable with the signature:

    (ba: BinauralAudiogram) -> bool | None

- True  : audiogram meets the asymmetry definition
- False : audiogram does not meet the definition
- None  : cannot be determined from available data (e.g. missing frequencies)

Named built-in criteria are stored in ASYMMETRY_CRITERIA and can be passed
by name to is_asymmetric(). Custom criteria can be passed as callables:

    ba.is_asymmetric(lambda ba: abs(ba.pta()["left"] - ba.pta()["right"]) > 20)

Note: the clinical thresholds used in built-in criteria are reasonable defaults
but not universally standardized. Verify against your target guideline before
using for clinical decision support.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .audiogram import BinauralAudiogram

AsymmetryCriterion = Callable["BinauralAudiogram", "bool | None"]


@dataclass
class InterauralDifference:
    """Interaural threshold difference at one frequency.

    difference_db = left_threshold - right_threshold
      Positive → left ear is worse at this frequency.
      Negative → right ear is worse at this frequency.

    left_nr / right_nr indicate that the contributing threshold was a
    no-response measurement. The numeric value is equipment-limited in
    that case, not a true threshold — downstream code should account for
    this when interpreting the magnitude.
    """

    freq_hz: int
    difference_db: float
    left_threshold: float
    right_threshold: float
    left_nr: bool = False
    right_nr: bool = False

    @property
    def nr_involved(self) -> bool:
        """True if either contributing threshold was a no-response."""
        return self.left_nr or self.right_nr

    @property
    def better_ear(self) -> str | None:
        """The ear with the lower (better) threshold. None if equal."""
        if self.difference_db > 0:
            return "right"
        if self.difference_db < 0:
            return "left"
        return None


def compute_interaural_differences(
    ba: BinauralAudiogram,
    pathway: str = "air",
) -> list[InterauralDifference]:
    """Compute interaural differences for all frequencies present in both ears.

    Frequencies present in only one ear are excluded. The result is sorted
    by frequency ascending.
    """
    left_data = ba.left._pathway_data(pathway)
    right_data = ba.right._pathway_data(pathway)
    common_freqs = sorted(set(left_data) & set(right_data))

    result = []
    for freq in common_freqs:
        lp = left_data[freq]
        rp = right_data[freq]
        result.append(
            InterauralDifference(
                freq_hz=freq,
                difference_db=lp.threshold_db - rp.threshold_db,
                left_threshold=lp.threshold_db,
                right_threshold=rp.threshold_db,
                left_nr=lp.nr,
                right_nr=rp.nr,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Built-in criteria
# ---------------------------------------------------------------------------

def _n_consecutive(diffs: list[InterauralDifference], threshold: float, n: int) -> bool:
    run = 0
    for d in diffs:
        if abs(d.difference_db) >= threshold:
            run += 1
            if run >= n:
                return True
        else:
            run = 0
    return False


def _any_15db(ba: BinauralAudiogram) -> bool | None:
    """Any shared frequency with an interaural difference >= 15 dB."""
    diffs = compute_interaural_differences(ba)
    if not diffs:
        return None
    return any(abs(d.difference_db) >= 15 for d in diffs)


def _two_consecutive_10db(ba: BinauralAudiogram) -> bool | None:
    """Two or more consecutive test frequencies with an interaural difference >= 10 dB.

    'Consecutive' means adjacent in the sorted list of frequencies tested in
    both ears — not adjacent integers. If only one shared frequency exists,
    returns None.
    """
    diffs = sorted(compute_interaural_differences(ba), key=lambda d: d.freq_hz)
    if len(diffs) < 2:
        return None
    return _n_consecutive(diffs, 10, 2)


def _pta_15db(ba: BinauralAudiogram) -> bool | None:
    """Difference in air-conduction PTA (500/1000/2000 Hz) between ears >= 15 dB."""
    ptas = ba.pta()
    left = ptas.get("left")
    right = ptas.get("right")
    if left is None or right is None:
        return None
    return abs(left - right) >= 15


def _nr_one_side(ba: BinauralAudiogram) -> bool | None:
    """Any frequency where one ear has a measurable threshold and the other is NR.

    This treats one-sided NR as inherently asymmetric regardless of the
    numeric difference.
    """
    diffs = compute_interaural_differences(ba)
    if not diffs:
        return None
    one_sided_nr = any(
        (d.left_nr and not d.right_nr) or (d.right_nr and not d.left_nr)
        for d in diffs
    )
    return one_sided_nr


def _any_30db(ba: BinauralAudiogram) -> bool | None:
    """Any shared frequency with an interaural difference >= 30 dB."""
    diffs = compute_interaural_differences(ba)
    if not diffs:
        return None
    return any(abs(d.difference_db) >= 30 for d in diffs)


def _3k_rule(ba: BinauralAudiogram) -> bool | None:
    """Interaural difference >= 15 dB at 3000 Hz."""
    diffs = compute_interaural_differences(ba)
    by_freq = {d.freq_hz: d for d in diffs}
    if 3000 not in by_freq:
        return None
    return abs(by_freq[3000].difference_db) >= 15


def _two_consecutive_15db(ba: BinauralAudiogram) -> bool | None:
    """Two or more consecutive test frequencies with an interaural difference >= 15 dB."""
    diffs = sorted(compute_interaural_differences(ba), key=lambda d: d.freq_hz)
    if len(diffs) < 2:
        return None
    return _n_consecutive(diffs, 15, 2)


def _three_consecutive_10db(ba: BinauralAudiogram) -> bool | None:
    """Three or more consecutive test frequencies with an interaural difference >= 10 dB."""
    diffs = sorted(compute_interaural_differences(ba), key=lambda d: d.freq_hz)
    if len(diffs) < 3:
        return None
    return _n_consecutive(diffs, 10, 3)


def _wrs_15pct(ba: BinauralAudiogram) -> bool | None:
    """Best word recognition score differs by > 15% between ears."""
    from .metrics import best_wrs
    l_best = best_wrs(ba.left.wrs)
    r_best = best_wrs(ba.right.wrs)
    if l_best is None or r_best is None:
        return None
    return abs(l_best.score_pct - r_best.score_pct) > 15


ASYMMETRY_CRITERIA: dict[str, AsymmetryCriterion] = {
    "any_15db": _any_15db,
    "any_30db": _any_30db,
    "two_consecutive_10db": _two_consecutive_10db,
    "pta_15db": _pta_15db,
    "nr_one_side": _nr_one_side,
    "3k_rule": _3k_rule,
    "two_consecutive_15db": _two_consecutive_15db,
    "three_consecutive_10db": _three_consecutive_10db,
    "wrs_15pct": _wrs_15pct,
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def is_asymmetric(
    ba: BinauralAudiogram,
    criterion: str | AsymmetryCriterion = "any_15db",
) -> bool | None:
    """Evaluate whether a BinauralAudiogram meets an asymmetry criterion.

    Parameters
    ----------
    ba
        The audiogram to evaluate.
    criterion
        A named built-in criterion string or any callable that accepts a
        BinauralAudiogram and returns bool | None.

    Returns
    -------
    bool | None
        True if asymmetric, False if not, None if indeterminate.
    """
    if callable(criterion):
        return criterion(ba)
    if criterion not in ASYMMETRY_CRITERIA:
        raise ValueError(
            f"Unknown criterion {criterion!r}. "
            f"Built-in options: {sorted(ASYMMETRY_CRITERIA)}"
        )
    return ASYMMETRY_CRITERIA[criterion](ba)
