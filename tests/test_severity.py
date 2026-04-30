"""Tests for hearing loss severity grading (WHO 2021, AAO-HNS)."""
import pytest

from audiogram_object import (
    EarAudiogram, BinauralAudiogram, ThresholdPoint,
    severity_from_pta, severity_from_thresholds, pta_from_thresholds,
    PTA_STANDARDS, SEVERITY_GRADES, VALID_STANDARDS,
)


class TestSeverityFromPta:
    def test_normal(self):
        assert severity_from_pta(20.0) == "normal"
        assert severity_from_pta(25.0) == "normal"

    def test_mild(self):
        assert severity_from_pta(26.0) == "mild"
        assert severity_from_pta(40.0) == "mild"

    def test_moderate(self):
        assert severity_from_pta(41.0) == "moderate"
        assert severity_from_pta(55.0) == "moderate"

    def test_moderately_severe(self):
        assert severity_from_pta(56.0) == "moderately_severe"
        assert severity_from_pta(70.0) == "moderately_severe"

    def test_severe(self):
        assert severity_from_pta(71.0) == "severe"
        assert severity_from_pta(90.0) == "severe"

    def test_profound(self):
        assert severity_from_pta(91.0) == "profound"
        assert severity_from_pta(120.0) == "profound"

    def test_none_input(self):
        assert severity_from_pta(None) is None

    def test_boundary_zero(self):
        assert severity_from_pta(0.0) == "normal"


class TestSeverityFromThresholds:
    def test_normal_hearing(self):
        thresholds = {500: 10.0, 1000: 15.0, 2000: 20.0, 4000: 20.0}
        assert severity_from_thresholds(thresholds) == "normal"

    def test_moderate_4freq(self):
        thresholds = {500: 40.0, 1000: 45.0, 2000: 50.0, 4000: 55.0}
        assert severity_from_thresholds(thresholds) == "moderate"

    def test_custom_freqs(self):
        thresholds = {500: 30.0, 1000: 35.0, 2000: 40.0}
        assert severity_from_thresholds(thresholds, freqs=(500, 1000, 2000)) == "mild"

    def test_require_all_missing(self):
        thresholds = {500: 20.0, 1000: 25.0}
        assert severity_from_thresholds(thresholds, require_all=True) is None

    def test_require_all_satisfied(self):
        thresholds = {500: 20.0, 1000: 25.0, 2000: 30.0, 4000: 35.0}
        assert severity_from_thresholds(thresholds, require_all=True) == "mild"

    def test_empty_thresholds(self):
        assert severity_from_thresholds({}) is None


class TestEarSeverity:
    def test_normal(self):
        ear = EarAudiogram(air={500: 10.0, 1000: 15.0, 2000: 20.0, 4000: 20.0})
        assert ear.severity() == "normal"

    def test_moderate(self):
        ear = EarAudiogram(air={500: 40.0, 1000: 45.0, 2000: 50.0, 4000: 55.0})
        assert ear.severity() == "moderate"

    def test_3freq(self):
        ear = EarAudiogram(air={500: 30.0, 1000: 35.0, 2000: 40.0})
        assert ear.severity(freqs=(500, 1000, 2000)) == "mild"

    def test_empty_air(self):
        ear = EarAudiogram()
        assert ear.severity() is None

    def test_bone_pathway(self):
        ear = EarAudiogram(
            air={500: 50.0},
            bone={500: ThresholdPoint(15.0), 1000: ThresholdPoint(20.0), 2000: ThresholdPoint(25.0)},
        )
        assert ear.severity(freqs=(500, 1000, 2000), pathway="bone") == "normal"


class TestBinauralSeverity:
    def test_both_ears(self):
        left = EarAudiogram(air={500: 10.0, 1000: 15.0, 2000: 20.0, 4000: 20.0})
        right = EarAudiogram(air={500: 40.0, 1000: 50.0, 2000: 55.0, 4000: 60.0})
        ba = BinauralAudiogram(left, right)
        sev = ba.severity()
        assert sev["left"] == "normal"
        assert sev["right"] == "moderate"

    def test_one_ear_empty(self):
        left = EarAudiogram(air={500: 30.0, 1000: 35.0, 2000: 40.0, 4000: 45.0})
        right = EarAudiogram()
        ba = BinauralAudiogram(left, right)
        sev = ba.severity()
        assert sev["left"] == "mild"
        assert sev["right"] is None


class TestPtaStandards:
    def test_3tone_default(self):
        thresholds = {500: 20.0, 1000: 30.0, 2000: 40.0, 4000: 100.0}
        assert pta_from_thresholds(thresholds) == pytest.approx(30.0)

    def test_4tone(self):
        thresholds = {500: 20.0, 1000: 30.0, 2000: 40.0, 4000: 50.0}
        assert pta_from_thresholds(thresholds, standard="4tone") == pytest.approx(35.0)

    def test_aao_hns_with_3000(self):
        thresholds = {500: 20.0, 1000: 30.0, 2000: 40.0, 3000: 50.0}
        assert pta_from_thresholds(thresholds, standard="aao_hns") == pytest.approx(35.0)

    def test_aao_hns_fallback_avg_2000_4000(self):
        thresholds = {500: 20.0, 1000: 30.0, 2000: 40.0, 4000: 60.0}
        assert pta_from_thresholds(thresholds, standard="aao_hns") == pytest.approx(35.0)

    def test_aao_hns_3000_present_ignores_4000(self):
        thresholds = {500: 20.0, 1000: 30.0, 2000: 40.0, 3000: 50.0, 4000: 100.0}
        assert pta_from_thresholds(thresholds, standard="aao_hns") == pytest.approx(35.0)

    def test_aao_hns_no_3000_no_4000_omits(self):
        thresholds = {500: 20.0, 1000: 30.0, 2000: 40.0}
        assert pta_from_thresholds(thresholds, standard="aao_hns") == pytest.approx(30.0)

    def test_aao_hns_require_all_with_3000(self):
        thresholds = {500: 20.0, 1000: 30.0, 2000: 40.0, 3000: 50.0}
        assert pta_from_thresholds(thresholds, standard="aao_hns", require_all=True) == pytest.approx(35.0)

    def test_aao_hns_require_all_with_fallback(self):
        thresholds = {500: 20.0, 1000: 30.0, 2000: 40.0, 4000: 60.0}
        assert pta_from_thresholds(thresholds, standard="aao_hns", require_all=True) == pytest.approx(35.0)

    def test_aao_hns_require_all_missing_3000_and_4000(self):
        thresholds = {500: 20.0, 1000: 30.0, 2000: 40.0}
        assert pta_from_thresholds(thresholds, standard="aao_hns", require_all=True) is None

    def test_aao_hns_require_all_missing_base_freq(self):
        thresholds = {500: 20.0, 2000: 40.0, 3000: 50.0}
        assert pta_from_thresholds(thresholds, standard="aao_hns", require_all=True) is None

    def test_aao_hns_empty(self):
        assert pta_from_thresholds({}, standard="aao_hns") is None

    def test_freqs_overrides_standard(self):
        thresholds = {500: 10.0, 1000: 30.0, 2000: 50.0, 3000: 70.0}
        result = pta_from_thresholds(thresholds, standard="aao_hns", freqs=(500,))
        assert result == pytest.approx(10.0)

    def test_invalid_standard_raises(self):
        with pytest.raises(ValueError, match="Unknown PTA standard"):
            pta_from_thresholds({500: 20.0}, standard="bogus")

    def test_pta_standards_dict(self):
        assert "3tone" in PTA_STANDARDS
        assert "4tone" in PTA_STANDARDS
        assert PTA_STANDARDS["3tone"] == (500, 1000, 2000)
        assert PTA_STANDARDS["4tone"] == (500, 1000, 2000, 4000)


class TestSeverityAAOHNS:
    def test_from_thresholds_with_3000(self):
        thresholds = {500: 30.0, 1000: 35.0, 2000: 40.0, 3000: 45.0}
        # PTA = 37.5 → mild
        assert severity_from_thresholds(thresholds, standard="aao_hns") == "mild"

    def test_from_thresholds_fallback(self):
        thresholds = {500: 30.0, 1000: 35.0, 2000: 40.0, 4000: 60.0}
        # 3000 fallback = avg(40, 60) = 50; PTA = (30+35+40+50)/4 = 38.75 → mild
        assert severity_from_thresholds(thresholds, standard="aao_hns") == "mild"

    def test_freqs_overrides_aao_hns(self):
        thresholds = {500: 10.0, 1000: 35.0, 2000: 40.0, 3000: 55.0}
        result_freqs = severity_from_thresholds(
            thresholds, freqs=(500,), standard="aao_hns",
        )
        assert result_freqs == "normal"
        result_aao = severity_from_thresholds(thresholds, standard="aao_hns")
        assert result_aao == "mild"

    def test_invalid_standard_raises(self):
        with pytest.raises(ValueError, match="Unknown standard"):
            severity_from_thresholds({500: 20.0}, standard="bogus")


class TestEarSeverityAAOHNS:
    def test_with_3000(self):
        ear = EarAudiogram(air={500: 30.0, 1000: 35.0, 2000: 40.0, 3000: 45.0})
        assert ear.severity(standard="aao_hns") == "mild"

    def test_fallback(self):
        ear = EarAudiogram(air={500: 30.0, 1000: 35.0, 2000: 40.0, 4000: 60.0})
        assert ear.severity(standard="aao_hns") == "mild"

    def test_who2021_default_unchanged(self):
        ear = EarAudiogram(air={500: 10.0, 1000: 15.0, 2000: 20.0, 4000: 20.0})
        assert ear.severity() == "normal"
        assert ear.severity(standard="who2021") == "normal"


class TestBinauralSeverityAAOHNS:
    def test_both_ears(self):
        left = EarAudiogram(air={500: 30.0, 1000: 35.0, 2000: 40.0, 3000: 45.0})
        right = EarAudiogram(air={500: 50.0, 1000: 55.0, 2000: 60.0, 4000: 80.0})
        ba = BinauralAudiogram(left, right)
        sev = ba.severity(standard="aao_hns")
        assert sev["left"] == "mild"
        # right: 3000 fallback = avg(60, 80) = 70; PTA = (50+55+60+70)/4 = 58.75
        assert sev["right"] == "moderately_severe"


class TestSeverityGrades:
    def test_grades_ordered(self):
        cutoffs = [g[0] for g in SEVERITY_GRADES]
        assert cutoffs == sorted(cutoffs)

    def test_all_grades_present(self):
        grades = {g[1] for g in SEVERITY_GRADES}
        expected = {"normal", "mild", "moderate", "moderately_severe", "severe", "profound"}
        assert grades == expected

    def test_valid_standards(self):
        assert VALID_STANDARDS == {"who2021", "aao_hns"}
