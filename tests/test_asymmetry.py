"""Tests for InterauralDifference and asymmetry criteria."""
import pytest

from audiogram_object import ThresholdPoint, WordRecognitionScore, EarAudiogram, BinauralAudiogram, ASYMMETRY_CRITERIA
from audiogram_object.asymmetry import compute_interaural_differences, InterauralDifference


# ---------------------------------------------------------------------------
# InterauralDifference properties
# ---------------------------------------------------------------------------

class TestInterauralDifference:
    def test_better_ear_left(self):
        d = InterauralDifference(freq_hz=1000, difference_db=-10.0,
                                 left_threshold=20.0, right_threshold=30.0)
        assert d.better_ear == "left"

    def test_better_ear_right(self):
        d = InterauralDifference(freq_hz=1000, difference_db=10.0,
                                 left_threshold=30.0, right_threshold=20.0)
        assert d.better_ear == "right"

    def test_better_ear_equal(self):
        d = InterauralDifference(freq_hz=1000, difference_db=0.0,
                                 left_threshold=25.0, right_threshold=25.0)
        assert d.better_ear is None

    def test_nr_involved_neither(self):
        d = InterauralDifference(freq_hz=1000, difference_db=5.0,
                                 left_threshold=30.0, right_threshold=25.0)
        assert d.nr_involved is False

    def test_nr_involved_left(self):
        d = InterauralDifference(freq_hz=4000, difference_db=90.0,
                                 left_threshold=120.0, right_threshold=30.0,
                                 left_nr=True)
        assert d.nr_involved is True

    def test_nr_involved_right(self):
        d = InterauralDifference(freq_hz=4000, difference_db=-90.0,
                                 left_threshold=30.0, right_threshold=120.0,
                                 right_nr=True)
        assert d.nr_involved is True


# ---------------------------------------------------------------------------
# compute_interaural_differences
# ---------------------------------------------------------------------------

class TestComputeInterauralDifferences:
    def test_common_freqs_only(self):
        left = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0})
        right = EarAudiogram(air={1000: 20.0, 2000: 25.0, 4000: 50.0})
        ba = BinauralAudiogram(left, right)
        diffs = compute_interaural_differences(ba)
        assert [d.freq_hz for d in diffs] == [1000, 2000]

    def test_difference_db_direction(self):
        """Positive = left worse, negative = right worse."""
        left = EarAudiogram(air={1000: 40.0})
        right = EarAudiogram(air={1000: 25.0})
        ba = BinauralAudiogram(left, right)
        diffs = compute_interaural_differences(ba)
        assert diffs[0].difference_db == pytest.approx(15.0)
        assert diffs[0].better_ear == "right"

    def test_sorted_by_frequency(self):
        left = EarAudiogram(air={4000: 50.0, 500: 20.0, 2000: 35.0})
        right = EarAudiogram(air={4000: 45.0, 500: 25.0, 2000: 30.0})
        ba = BinauralAudiogram(left, right)
        diffs = compute_interaural_differences(ba)
        freqs = [d.freq_hz for d in diffs]
        assert freqs == sorted(freqs)

    def test_nr_flags_propagated(self, asymmetric_ba):
        diffs = compute_interaural_differences(asymmetric_ba)
        nr_diffs = [d for d in diffs if d.nr_involved]
        assert len(nr_diffs) == 1
        assert nr_diffs[0].freq_hz == 4000
        assert nr_diffs[0].right_nr is True
        assert nr_diffs[0].left_nr is False

    def test_no_common_freqs(self):
        left = EarAudiogram(air={500: 20.0})
        right = EarAudiogram(air={1000: 30.0})
        ba = BinauralAudiogram(left, right)
        assert compute_interaural_differences(ba) == []

    def test_bone_pathway(self):
        left = EarAudiogram(bone={1000: ThresholdPoint(20.0), 2000: ThresholdPoint(25.0)})
        right = EarAudiogram(bone={1000: ThresholdPoint(10.0), 2000: ThresholdPoint(15.0)})
        ba = BinauralAudiogram(left, right)
        diffs = compute_interaural_differences(ba, pathway="bone")
        assert len(diffs) == 2
        assert diffs[0].difference_db == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# is_asymmetric / named criteria
# ---------------------------------------------------------------------------

class TestIsAsymmetric:
    def test_any_15db_symmetric(self, symmetric_ba):
        assert symmetric_ba.is_asymmetric("any_15db") is False

    def test_any_15db_asymmetric(self, asymmetric_ba):
        assert asymmetric_ba.is_asymmetric("any_15db") is True

    def test_any_15db_none_no_data(self):
        ba = BinauralAudiogram(EarAudiogram(), EarAudiogram())
        assert ba.is_asymmetric("any_15db") is None

    def test_two_consecutive_10db_symmetric(self, symmetric_ba):
        assert symmetric_ba.is_asymmetric("two_consecutive_10db") is False

    def test_two_consecutive_10db_asymmetric(self, asymmetric_ba):
        assert asymmetric_ba.is_asymmetric("two_consecutive_10db") is True

    def test_two_consecutive_10db_one_freq_none(self):
        """Only one shared frequency — cannot evaluate consecutiveness."""
        left = EarAudiogram(air={1000: 20.0})
        right = EarAudiogram(air={1000: 50.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("two_consecutive_10db") is None

    def test_two_consecutive_10db_nonconsecutive(self):
        """Large diff at 500 and 4000 but not at 1000/2000 — not consecutive."""
        left = EarAudiogram(air={500: 10.0, 1000: 25.0, 2000: 25.0, 4000: 10.0})
        right = EarAudiogram(air={500: 30.0, 1000: 20.0, 2000: 20.0, 4000: 30.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("two_consecutive_10db") is False

    def test_pta_15db_symmetric(self, symmetric_ba):
        assert symmetric_ba.is_asymmetric("pta_15db") is False

    def test_pta_15db_asymmetric(self, asymmetric_ba):
        assert asymmetric_ba.is_asymmetric("pta_15db") is True

    def test_pta_15db_none_missing_freqs(self):
        """PTA returns None when no speech frequencies are present."""
        left = EarAudiogram(air={8000: 20.0})
        right = EarAudiogram(air={8000: 50.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("pta_15db") is None

    def test_nr_one_side_catches_asymmetry(self, asymmetric_ba):
        assert asymmetric_ba.is_asymmetric("nr_one_side") is True

    def test_nr_one_side_symmetric_no_nr(self, symmetric_ba):
        assert symmetric_ba.is_asymmetric("nr_one_side") is False

    def test_nr_one_side_both_nr_not_triggered(self):
        left = EarAudiogram(air={4000: ThresholdPoint(120.0, nr=True)})
        right = EarAudiogram(air={4000: ThresholdPoint(120.0, nr=True)})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("nr_one_side") is False

    def test_custom_callable(self, asymmetric_ba):
        custom = lambda ba: abs(ba.pta()["left"] - ba.pta()["right"]) > 20
        assert asymmetric_ba.is_asymmetric(custom) is True

    def test_custom_callable_symmetric(self, symmetric_ba):
        custom = lambda ba: abs(ba.pta()["left"] - ba.pta()["right"]) > 20
        assert symmetric_ba.is_asymmetric(custom) is False

    def test_unknown_criterion_raises(self, symmetric_ba):
        with pytest.raises(ValueError, match="Unknown criterion"):
            symmetric_ba.is_asymmetric("made_up_criterion")

    def test_all_criteria_in_registry(self):
        expected = {"any_15db", "any_30db", "two_consecutive_10db", "two_consecutive_15db", "three_consecutive_10db", "pta_15db", "nr_one_side", "3k_rule", "wrs_15pct"}
        assert set(ASYMMETRY_CRITERIA.keys()) == expected

    def test_crossing_pta_blind_spot(self):
        """PTAs cancel when asymmetry crosses frequencies — pta_15db misses it."""
        left = EarAudiogram(air={500: 15.0, 1000: 20.0, 2000: 50.0, 4000: 60.0})
        right = EarAudiogram(air={500: 40.0, 1000: 45.0, 2000: 20.0, 4000: 25.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("pta_15db") is False
        assert ba.is_asymmetric("any_15db") is True


class TestAny30db:
    def test_meets_30db(self):
        left = EarAudiogram(air={500: 10.0, 1000: 20.0})
        right = EarAudiogram(air={500: 10.0, 1000: 50.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("any_30db") is True

    def test_below_30db(self):
        left = EarAudiogram(air={500: 10.0, 1000: 20.0})
        right = EarAudiogram(air={500: 10.0, 1000: 45.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("any_30db") is False

    def test_exactly_30db(self):
        left = EarAudiogram(air={1000: 10.0})
        right = EarAudiogram(air={1000: 40.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("any_30db") is True

    def test_no_shared_freqs(self):
        left = EarAudiogram(air={500: 10.0})
        right = EarAudiogram(air={1000: 40.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("any_30db") is None


class Test3kRule:
    def test_meets_15db_at_3k(self):
        left = EarAudiogram(air={3000: 10.0})
        right = EarAudiogram(air={3000: 30.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("3k_rule") is True

    def test_below_15db_at_3k(self):
        left = EarAudiogram(air={3000: 20.0})
        right = EarAudiogram(air={3000: 30.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("3k_rule") is False

    def test_exactly_15db_at_3k(self):
        left = EarAudiogram(air={3000: 10.0})
        right = EarAudiogram(air={3000: 25.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("3k_rule") is True

    def test_no_3k_data(self):
        left = EarAudiogram(air={500: 10.0, 1000: 20.0})
        right = EarAudiogram(air={500: 10.0, 1000: 50.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("3k_rule") is None

    def test_asymmetry_at_other_freqs_ignored(self):
        left = EarAudiogram(air={1000: 10.0, 3000: 20.0})
        right = EarAudiogram(air={1000: 60.0, 3000: 30.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("3k_rule") is False


class TestTwoConsecutive15db:
    def test_meets(self):
        left = EarAudiogram(air={500: 10.0, 1000: 10.0, 2000: 10.0, 4000: 10.0})
        right = EarAudiogram(air={500: 10.0, 1000: 25.0, 2000: 25.0, 4000: 10.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("two_consecutive_15db") is True

    def test_one_only_not_enough(self):
        left = EarAudiogram(air={500: 10.0, 1000: 10.0, 2000: 10.0, 4000: 10.0})
        right = EarAudiogram(air={500: 10.0, 1000: 25.0, 2000: 10.0, 4000: 10.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("two_consecutive_15db") is False

    def test_non_consecutive(self):
        left = EarAudiogram(air={500: 10.0, 1000: 10.0, 2000: 10.0, 4000: 10.0})
        right = EarAudiogram(air={500: 25.0, 1000: 10.0, 2000: 25.0, 4000: 10.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("two_consecutive_15db") is False

    def test_single_shared_freq(self):
        left = EarAudiogram(air={1000: 10.0})
        right = EarAudiogram(air={1000: 30.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("two_consecutive_15db") is None


class TestThreeConsecutive10db:
    def test_meets(self):
        left = EarAudiogram(air={500: 10.0, 1000: 10.0, 2000: 10.0, 4000: 10.0})
        right = EarAudiogram(air={500: 10.0, 1000: 20.0, 2000: 20.0, 4000: 20.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("three_consecutive_10db") is True

    def test_two_consecutive_not_enough(self):
        left = EarAudiogram(air={500: 10.0, 1000: 10.0, 2000: 10.0, 4000: 10.0})
        right = EarAudiogram(air={500: 10.0, 1000: 20.0, 2000: 20.0, 4000: 10.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("three_consecutive_10db") is False

    def test_non_consecutive(self):
        left = EarAudiogram(air={500: 10.0, 1000: 10.0, 2000: 10.0, 4000: 10.0, 8000: 10.0})
        right = EarAudiogram(air={500: 20.0, 1000: 10.0, 2000: 20.0, 4000: 10.0, 8000: 20.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("three_consecutive_10db") is False

    def test_fewer_than_three_shared(self):
        left = EarAudiogram(air={500: 10.0, 1000: 10.0})
        right = EarAudiogram(air={500: 20.0, 1000: 20.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("three_consecutive_10db") is None


class TestWrs15pct:
    def test_exceeds_15pct(self):
        left = EarAudiogram(air={1000: 20.0}, wrs=[WordRecognitionScore(92.0, 70.0)])
        right = EarAudiogram(air={1000: 20.0}, wrs=[WordRecognitionScore(72.0, 70.0)])
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("wrs_15pct") is True

    def test_within_15pct(self):
        left = EarAudiogram(air={1000: 20.0}, wrs=[WordRecognitionScore(92.0, 70.0)])
        right = EarAudiogram(air={1000: 20.0}, wrs=[WordRecognitionScore(80.0, 70.0)])
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("wrs_15pct") is False

    def test_exactly_15pct_not_asymmetric(self):
        left = EarAudiogram(air={1000: 20.0}, wrs=[WordRecognitionScore(90.0, 70.0)])
        right = EarAudiogram(air={1000: 20.0}, wrs=[WordRecognitionScore(75.0, 70.0)])
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("wrs_15pct") is False

    def test_no_wrs_one_ear(self):
        left = EarAudiogram(air={1000: 20.0}, wrs=[WordRecognitionScore(92.0, 70.0)])
        right = EarAudiogram(air={1000: 20.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("wrs_15pct") is None

    def test_no_wrs_either_ear(self):
        left = EarAudiogram(air={1000: 20.0})
        right = EarAudiogram(air={1000: 20.0})
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("wrs_15pct") is None

    def test_uses_best_wrs(self):
        left = EarAudiogram(air={1000: 20.0}, wrs=[
            WordRecognitionScore(80.0, 60.0),
            WordRecognitionScore(96.0, 70.0),
        ])
        right = EarAudiogram(air={1000: 20.0}, wrs=[WordRecognitionScore(96.0, 70.0)])
        ba = BinauralAudiogram(left, right)
        assert ba.is_asymmetric("wrs_15pct") is False
