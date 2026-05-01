import pytest

from audiogram import ThresholdPoint, WordRecognitionScore, EarAudiogram, BinauralAudiogram


@pytest.fixture
def symmetric_ba():
    """Both ears within 5 dB at all frequencies — not clinically asymmetric."""
    left = EarAudiogram(air={500: 20.0, 1000: 25.0, 2000: 30.0, 4000: 35.0})
    right = EarAudiogram(air={500: 25.0, 1000: 20.0, 2000: 25.0, 4000: 30.0})
    return BinauralAudiogram(left, right, audiogram_id="sym-001", subject_id="pt-sym")


@pytest.fixture
def asymmetric_ba():
    """Right ear clearly worse; NR at 4000 Hz right."""
    left = EarAudiogram(air={500: 20.0, 1000: 20.0, 2000: 25.0, 4000: 30.0})
    right = EarAudiogram(
        air={
            500: ThresholdPoint(40.0),
            1000: ThresholdPoint(50.0),
            2000: ThresholdPoint(55.0),
            4000: ThresholdPoint(120.0, nr=True),
        }
    )
    return BinauralAudiogram(left, right, audiogram_id="asym-001")


@pytest.fixture
def full_ba():
    """Air + bone, masked bone, NR — exercises all ThresholdPoint fields."""
    left = EarAudiogram(
        air={
            500: ThresholdPoint(25.0),
            1000: ThresholdPoint(30.0),
            2000: ThresholdPoint(35.0),
            4000: ThresholdPoint(50.0),
        },
        bone={
            1000: ThresholdPoint(20.0, masked=True),
            2000: ThresholdPoint(25.0, masked=True),
        },
    )
    right = EarAudiogram(
        air={
            500: ThresholdPoint(20.0),
            1000: ThresholdPoint(25.0),
            2000: ThresholdPoint(40.0),
            4000: ThresholdPoint(120.0, nr=True),
        },
        bone={
            1000: ThresholdPoint(15.0, masked=True),
            2000: ThresholdPoint(30.0, masked=True),
        },
    )
    return BinauralAudiogram(
        left,
        right,
        audiogram_id="full-001",
        subject_id="pt-123",
        performed_at="2024-01-15",
        source="clinic",
    )


@pytest.fixture
def speech_ba():
    """BinauralAudiogram with full speech audiometry data."""
    left = EarAudiogram(
        air={500: 25.0, 1000: 30.0, 2000: 35.0, 4000: 50.0},
        srt=28.0,
        wrs=[
            WordRecognitionScore(92.0, 70.0, word_list="CNC"),
            WordRecognitionScore(80.0, 50.0, word_list="CNC"),
        ],
    )
    right = EarAudiogram(
        air={500: 40.0, 1000: 55.0, 2000: 60.0, 4000: 75.0},
        srt=52.0,
        sat=45.0,
        wrs=[WordRecognitionScore(68.0, 90.0, masked=True)],
    )
    return BinauralAudiogram(
        left, right,
        audiogram_id="speech-001",
        subject_id="pt-speech",
    )
