"""Tests for speech audiometry: WordRecognitionScore, SRT, SAT, WRS on EarAudiogram."""
import pytest

from audiogram import (
    ThresholdPoint, WordRecognitionScore, EarAudiogram, BinauralAudiogram,
)


class TestWordRecognitionScore:
    def test_defaults(self):
        w = WordRecognitionScore(92.0, 70.0)
        assert w.score_pct == 92.0
        assert w.level_db == 70.0
        assert w.masked is False
        assert w.word_list is None

    def test_explicit_fields(self):
        w = WordRecognitionScore(68.0, 90.0, masked=True, word_list="CNC")
        assert w.masked is True
        assert w.word_list == "CNC"

    def test_to_dict(self):
        w = WordRecognitionScore(88.0, 70.0, word_list="NU-6")
        d = w.to_dict()
        assert d["score_pct"] == 88.0
        assert d["level_db"] == 70.0
        assert d["word_list"] == "NU-6"
        assert d["masked"] is False

    def test_to_dict_omits_none_word_list(self):
        w = WordRecognitionScore(88.0, 70.0)
        assert "word_list" not in w.to_dict()

    def test_from_dict_round_trip(self):
        w = WordRecognitionScore(92.0, 70.0, masked=True, word_list="CNC")
        assert WordRecognitionScore.from_dict(w.to_dict()) == w

    def test_from_dict_defaults(self):
        w = WordRecognitionScore.from_dict({"score_pct": 80.0, "level_db": 60.0})
        assert w.masked is False
        assert w.word_list is None

    def test_equality(self):
        a = WordRecognitionScore(92.0, 70.0, word_list="CNC")
        b = WordRecognitionScore(92.0, 70.0, word_list="CNC")
        assert a == b

    def test_inequality(self):
        a = WordRecognitionScore(92.0, 70.0)
        b = WordRecognitionScore(88.0, 70.0)
        assert a != b

    def test_score_pct_too_high(self):
        with pytest.raises(ValueError, match="score_pct must be 0–100"):
            WordRecognitionScore(101.0, 70.0)

    def test_score_pct_negative(self):
        with pytest.raises(ValueError, match="score_pct must be 0–100"):
            WordRecognitionScore(-1.0, 70.0)

    def test_score_pct_boundary_zero(self):
        w = WordRecognitionScore(0.0, 70.0)
        assert w.score_pct == 0.0

    def test_score_pct_boundary_hundred(self):
        w = WordRecognitionScore(100.0, 70.0)
        assert w.score_pct == 100.0


class TestEarAudiogramSpeech:
    def test_speech_defaults(self):
        ear = EarAudiogram(air={500: 20.0})
        assert ear.srt is None
        assert ear.sat is None
        assert ear.wrs == []

    def test_speech_fields_stored(self):
        wrs = [WordRecognitionScore(92.0, 70.0)]
        ear = EarAudiogram(air={500: 20.0}, srt=25.0, sat=20.0, wrs=wrs)
        assert ear.srt == 25.0
        assert ear.sat == 20.0
        assert len(ear.wrs) == 1
        assert ear.wrs[0].score_pct == 92.0

    def test_best_wrs_single(self):
        ear = EarAudiogram(wrs=[WordRecognitionScore(88.0, 70.0)])
        assert ear.best_wrs().score_pct == 88.0

    def test_best_wrs_multiple(self):
        ear = EarAudiogram(wrs=[
            WordRecognitionScore(80.0, 50.0),
            WordRecognitionScore(92.0, 70.0),
            WordRecognitionScore(84.0, 60.0),
        ])
        assert ear.best_wrs().score_pct == 92.0
        assert ear.best_wrs().level_db == 70.0

    def test_best_wrs_none_when_empty(self):
        ear = EarAudiogram(air={500: 20.0})
        assert ear.best_wrs() is None

    def test_srt_pta_agreement(self):
        ear = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0}, srt=30.0)
        assert ear.srt_pta_agreement() == pytest.approx(0.0)

    def test_srt_pta_agreement_positive(self):
        ear = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0}, srt=40.0)
        assert ear.srt_pta_agreement() == pytest.approx(10.0)

    def test_srt_pta_agreement_none_no_srt(self):
        ear = EarAudiogram(air={500: 20.0, 1000: 30.0, 2000: 40.0})
        assert ear.srt_pta_agreement() is None

    def test_srt_pta_agreement_none_no_air(self):
        ear = EarAudiogram(srt=30.0)
        assert ear.srt_pta_agreement() is None

    def test_eq_with_speech(self):
        a = EarAudiogram(air={500: 20.0}, srt=25.0, wrs=[WordRecognitionScore(92.0, 70.0)])
        b = EarAudiogram(air={500: 20.0}, srt=25.0, wrs=[WordRecognitionScore(92.0, 70.0)])
        assert a == b

    def test_neq_different_srt(self):
        a = EarAudiogram(air={500: 20.0}, srt=25.0)
        b = EarAudiogram(air={500: 20.0}, srt=30.0)
        assert a != b

    def test_neq_different_wrs(self):
        a = EarAudiogram(air={500: 20.0}, wrs=[WordRecognitionScore(92.0, 70.0)])
        b = EarAudiogram(air={500: 20.0}, wrs=[WordRecognitionScore(88.0, 70.0)])
        assert a != b

    def test_repr_includes_speech(self):
        ear = EarAudiogram(air={500: 20.0}, srt=25.0)
        assert "srt=25.0" in repr(ear)

    def test_repr_excludes_speech_when_none(self):
        ear = EarAudiogram(air={500: 20.0})
        assert "srt" not in repr(ear)


class TestEarAudiogramSpeechSerialization:
    def test_to_dict_includes_speech(self):
        ear = EarAudiogram(
            air={500: 20.0}, srt=25.0, sat=20.0,
            wrs=[WordRecognitionScore(92.0, 70.0, word_list="CNC")],
        )
        d = ear.to_dict()
        assert d["srt"] == 25.0
        assert d["sat"] == 20.0
        assert len(d["wrs"]) == 1
        assert d["wrs"][0]["score_pct"] == 92.0

    def test_to_dict_omits_speech_when_empty(self):
        ear = EarAudiogram(air={500: 20.0})
        d = ear.to_dict()
        assert "srt" not in d
        assert "sat" not in d
        assert "wrs" not in d

    def test_dict_round_trip_with_speech(self):
        ear = EarAudiogram(
            air={500: 20.0}, srt=25.0,
            wrs=[WordRecognitionScore(92.0, 70.0, word_list="CNC")],
        )
        restored = EarAudiogram.from_dict(ear.to_dict())
        assert ear == restored

    def test_dict_round_trip_without_speech(self):
        ear = EarAudiogram(air={500: 20.0})
        restored = EarAudiogram.from_dict(ear.to_dict())
        assert ear == restored


class TestBinauralSpeechRoundTrips:
    def test_dict_round_trip(self, speech_ba):
        restored = BinauralAudiogram.from_dict(speech_ba.to_dict())
        assert speech_ba == restored

    def test_json_round_trip(self, speech_ba):
        restored = BinauralAudiogram.from_json(speech_ba.to_json())
        assert speech_ba == restored

    def test_speech_preserved_in_dict(self, speech_ba):
        restored = BinauralAudiogram.from_dict(speech_ba.to_dict())
        assert restored.left.srt == 28.0
        assert len(restored.left.wrs) == 2
        assert restored.right.sat == 45.0
        assert restored.right.wrs[0].masked is True
