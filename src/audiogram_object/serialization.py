"""Serialization: dict, JSON, long-form rows, wide-format rows, speech rows, table rows."""

from __future__ import annotations

import json
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .audiogram import EarAudiogram, BinauralAudiogram


def _load_schema_module():
    from . import schema as s
    return s


def _float_or_none(val: Any) -> float | None:
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return None
    return float(val)


# ---------------------------------------------------------------------------
# EarAudiogram serialization
# ---------------------------------------------------------------------------

def ear_to_dict(ear: EarAudiogram) -> dict[str, Any]:
    d: dict[str, Any] = {
        "air": {int(f): p.to_dict() for f, p in ear.air.items()},
        "bone": {int(f): p.to_dict() for f, p in ear.bone.items()},
    }
    if ear.soundfield:
        d["soundfield"] = {int(f): p.to_dict() for f, p in ear.soundfield.items()}
    if ear.ci:
        d["ci"] = {int(f): p.to_dict() for f, p in ear.ci.items()}
    if ear.srt is not None:
        d["srt"] = ear.srt
    if ear.sat is not None:
        d["sat"] = ear.sat
    if ear.wrs:
        d["wrs"] = [w.to_dict() for w in ear.wrs]
    return d


def ear_from_dict(data: dict[str, Any]) -> EarAudiogram:
    from .audiogram import ThresholdPoint, WordRecognitionScore, EarAudiogram

    def _parse(d: dict) -> dict[int, ThresholdPoint]:
        out = {}
        for f, v in d.items():
            if isinstance(v, dict):
                out[int(f)] = ThresholdPoint.from_dict(v)
            else:
                out[int(f)] = ThresholdPoint(threshold_db=float(v))
        return out

    wrs_raw = data.get("wrs", [])
    wrs = [WordRecognitionScore.from_dict(w) for w in wrs_raw] if wrs_raw else None

    return EarAudiogram(
        air=_parse(data.get("air", {})),
        bone=_parse(data.get("bone", {})),
        soundfield=_parse(data.get("soundfield", {})),
        ci=_parse(data.get("ci", {})),
        srt=data.get("srt"),
        sat=data.get("sat"),
        wrs=wrs,
    )


def ear_to_json(ear: EarAudiogram) -> str:
    return json.dumps(ear_to_dict(ear))


def ear_from_json(json_str: str) -> EarAudiogram:
    return ear_from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# BinauralAudiogram serialization
# ---------------------------------------------------------------------------

def binaural_to_dict(ba: BinauralAudiogram) -> dict[str, Any]:
    return {
        "audiogram_id": ba.audiogram_id,
        "subject_id": ba.subject_id,
        "performed_at": ba.performed_at,
        "source": ba.source,
        "meta": dict(ba.meta),
        "left": ear_to_dict(ba.left),
        "right": ear_to_dict(ba.right),
    }


def binaural_from_dict(data: dict[str, Any]) -> BinauralAudiogram:
    from .audiogram import BinauralAudiogram

    left = ear_from_dict(data.get("left", {}))
    right = ear_from_dict(data.get("right", {}))
    return BinauralAudiogram(
        left,
        right,
        audiogram_id=data.get("audiogram_id"),
        subject_id=data.get("subject_id"),
        performed_at=data.get("performed_at"),
        source=data.get("source"),
        meta=data.get("meta") or {},
    )


def binaural_to_json(ba: BinauralAudiogram) -> str:
    return json.dumps(binaural_to_dict(ba))


def binaural_from_json(json_str: str) -> BinauralAudiogram:
    return binaural_from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# Internal row iterators
# ---------------------------------------------------------------------------

def _iter_threshold_rows(ba: BinauralAudiogram):
    for ear_name, ear_obj in (("left", ba.left), ("right", ba.right)):
        for pathway, data in (
            ("air", ear_obj.air),
            ("bone", ear_obj.bone),
            ("soundfield", ear_obj.soundfield),
            ("ci", ear_obj.ci),
        ):
            for freq_hz, point in sorted(data.items()):
                yield {
                    "ear": ear_name,
                    "freq_hz": int(freq_hz),
                    "threshold_db": float(point.threshold_db),
                    "pathway": pathway,
                    "masked": point.masked,
                    "nr": point.nr,
                }


def _iter_speech_rows(ba: BinauralAudiogram):
    from .audiogram import WordRecognitionScore

    for ear_name, ear_obj in (("left", ba.left), ("right", ba.right)):
        if ear_obj.srt is not None:
            yield {
                "ear": ear_name,
                "measure": "srt",
                "value": ear_obj.srt,
                "level_db": None,
                "masked": False,
                "word_list": None,
            }
        if ear_obj.sat is not None:
            yield {
                "ear": ear_name,
                "measure": "sat",
                "value": ear_obj.sat,
                "level_db": None,
                "masked": False,
                "word_list": None,
            }
        for w in ear_obj.wrs:
            yield {
                "ear": ear_name,
                "measure": "wrs",
                "value": w.score_pct,
                "level_db": w.level_db,
                "masked": w.masked,
                "word_list": w.word_list,
            }


# ---------------------------------------------------------------------------
# Long-form rows
# ---------------------------------------------------------------------------

def to_long_rows(
    ba: BinauralAudiogram,
    *,
    include_meta: bool = True,
    include_speech: bool = False,
    strict_freqs: bool = False,
) -> list[dict[str, Any]]:
    s = _load_schema_module()
    rows: list[dict[str, Any]] = []

    meta = {}
    if include_meta:
        meta = {
            s.AUDIOGRAM_ID: ba.audiogram_id,
            s.SUBJECT_ID: ba.subject_id,
            s.PERFORMED_AT: ba.performed_at,
            s.SOURCE: ba.source,
        }
    else:
        meta = {
            s.AUDIOGRAM_ID: ba.audiogram_id,
            s.SUBJECT_ID: None,
            s.PERFORMED_AT: None,
            s.SOURCE: None,
        }

    for obs in _iter_threshold_rows(ba):
        row = {
            **meta,
            s.EAR: obs["ear"],
            s.FREQ_HZ: obs["freq_hz"],
            s.THRESHOLD_DB: obs["threshold_db"],
            s.PATHWAY: obs["pathway"],
            s.MASKED: obs["masked"],
            s.NR: obs["nr"],
        }
        if include_speech:
            row[s.MEASURE_TYPE] = s.MEASURE_TYPE_THRESHOLD
            row[s.MEASURE] = None
            row[s.VALUE] = None
            row[s.LEVEL_DB] = None
            row[s.WORD_LIST] = None
        s.validate_long_row(row, strict_freqs=strict_freqs)
        rows.append(row)

    if include_speech:
        for obs in _iter_speech_rows(ba):
            row = {
                **meta,
                s.MEASURE_TYPE: s.MEASURE_TYPE_SPEECH,
                s.EAR: obs["ear"],
                s.FREQ_HZ: None,
                s.THRESHOLD_DB: None,
                s.PATHWAY: None,
                s.MASKED: obs["masked"],
                s.NR: None,
                s.MEASURE: obs["measure"],
                s.VALUE: obs["value"],
                s.LEVEL_DB: obs["level_db"],
                s.WORD_LIST: obs["word_list"],
            }
            rows.append(row)

    return rows


def from_long_rows(
    rows: Iterable[dict[str, Any]],
    *,
    strict_freqs: bool = False,
) -> BinauralAudiogram:
    from .audiogram import ThresholdPoint, WordRecognitionScore, EarAudiogram, BinauralAudiogram
    s = _load_schema_module()

    rows = list(rows)
    if not rows:
        raise ValueError("from_long_rows() requires at least one row")

    audiogram_ids = {row.get(s.AUDIOGRAM_ID) for row in rows}
    if len(audiogram_ids) != 1:
        raise ValueError(
            f"from_long_rows() expected rows for exactly one audiogram_id, "
            f"got: {sorted(audiogram_ids, key=str)}"
        )

    first = rows[0]
    meta = {
        "audiogram_id": first.get(s.AUDIOGRAM_ID),
        "subject_id": first.get(s.SUBJECT_ID),
        "performed_at": first.get(s.PERFORMED_AT),
        "source": first.get(s.SOURCE),
    }

    ear_data: dict[str, dict[str, dict[int, ThresholdPoint]]] = {
        "left": {"air": {}, "bone": {}, "soundfield": {}, "ci": {}},
        "right": {"air": {}, "bone": {}, "soundfield": {}, "ci": {}},
    }

    speech: dict[str, dict[str, Any]] = {
        "left": {"srt": None, "sat": None, "wrs": []},
        "right": {"srt": None, "sat": None, "wrs": []},
    }

    threshold_rows = []
    for row in rows:
        is_speech = (
            row.get(s.MEASURE_TYPE) == s.MEASURE_TYPE_SPEECH
            or (s.MEASURE in row and row.get(s.MEASURE) is not None
                and row.get(s.FREQ_HZ) is None)
        )

        if is_speech:
            ear = row[s.EAR]
            measure = row[s.MEASURE]
            value = row.get(s.VALUE)
            if value is None:
                continue
            if measure == "srt":
                speech[ear]["srt"] = float(value)
            elif measure == "sat":
                speech[ear]["sat"] = float(value)
            elif measure == "wrs":
                speech[ear]["wrs"].append(WordRecognitionScore(
                    score_pct=float(value),
                    level_db=float(row.get(s.LEVEL_DB) or 0),
                    masked=bool(row.get(s.MASKED, False)),
                    word_list=row.get(s.WORD_LIST),
                ))
        else:
            threshold_rows.append(row)

    if threshold_rows:
        s.validate_long_rows(threshold_rows, strict_freqs=strict_freqs)

    for row in threshold_rows:
        ear = row[s.EAR]
        freq_hz = int(row[s.FREQ_HZ])
        point = ThresholdPoint(
            threshold_db=float(row[s.THRESHOLD_DB]),
            masked=bool(row[s.MASKED]),
            nr=bool(row[s.NR]),
        )
        pathway = row[s.PATHWAY]

        if ear not in ear_data:
            raise ValueError(f"Unexpected ear value during import: {ear!r}")
        if pathway not in ear_data[ear]:
            raise ValueError(f"Unexpected pathway value during import: {pathway!r}")

        ear_data[ear][pathway][freq_hz] = point

    left = EarAudiogram(
        air=ear_data["left"]["air"], bone=ear_data["left"]["bone"],
        soundfield=ear_data["left"]["soundfield"], ci=ear_data["left"]["ci"],
        srt=speech["left"]["srt"],
        sat=speech["left"]["sat"],
        wrs=speech["left"]["wrs"] or None,
    )
    right = EarAudiogram(
        air=ear_data["right"]["air"], bone=ear_data["right"]["bone"],
        soundfield=ear_data["right"]["soundfield"], ci=ear_data["right"]["ci"],
        srt=speech["right"]["srt"],
        sat=speech["right"]["sat"],
        wrs=speech["right"]["wrs"] or None,
    )
    return BinauralAudiogram(left, right, meta={}, **meta)


# ---------------------------------------------------------------------------
# Speech rows
# ---------------------------------------------------------------------------

def to_speech_rows(
    ba: BinauralAudiogram,
    *,
    include_meta: bool = True,
) -> list[dict[str, Any]]:
    s = _load_schema_module()
    rows: list[dict[str, Any]] = []
    for obs in _iter_speech_rows(ba):
        row: dict[str, Any] = {}
        if include_meta:
            row[s.AUDIOGRAM_ID] = ba.audiogram_id
            row[s.SUBJECT_ID] = ba.subject_id
            row[s.PERFORMED_AT] = ba.performed_at
            row[s.SOURCE] = ba.source
        else:
            row[s.AUDIOGRAM_ID] = ba.audiogram_id
        row[s.EAR] = obs["ear"]
        row[s.MEASURE] = obs["measure"]
        row[s.VALUE] = obs["value"]
        row[s.LEVEL_DB] = obs["level_db"]
        row[s.MASKED] = obs["masked"]
        row[s.WORD_LIST] = obs["word_list"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Normalized table rows
# ---------------------------------------------------------------------------

def to_table_rows(
    ba: BinauralAudiogram,
    *,
    strict_freqs: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    s = _load_schema_module()

    test_row = {
        s.AUDIOGRAM_ID: ba.audiogram_id,
        s.SUBJECT_ID: ba.subject_id,
        s.PERFORMED_AT: ba.performed_at,
        s.SOURCE: ba.source,
        s.SCHEMA_VERSION_COL: s.SCHEMA_VERSION,
        s.META: dict(ba.meta),
    }
    s.validate_tests_row(test_row)

    obs_rows: list[dict[str, Any]] = []
    for obs in _iter_threshold_rows(ba):
        row = {
            s.AUDIOGRAM_ID: ba.audiogram_id,
            s.EAR: obs["ear"],
            s.FREQ_HZ: obs["freq_hz"],
            s.THRESHOLD_DB: obs["threshold_db"],
            s.PATHWAY: obs["pathway"],
            s.MASKED: obs["masked"],
            s.NR: obs["nr"],
        }
        s.validate_observations_row(row, strict_freqs=strict_freqs)
        obs_rows.append(row)

    return test_row, obs_rows


# ---------------------------------------------------------------------------
# Wide-format I/O
# ---------------------------------------------------------------------------

def to_wide_row(
    ba: BinauralAudiogram,
    *,
    include_meta: bool = True,
) -> dict[str, Any]:
    s = _load_schema_module()
    row: dict[str, Any] = {}

    if include_meta:
        row[s.AUDIOGRAM_ID] = ba.audiogram_id
        row[s.SUBJECT_ID] = ba.subject_id
        row[s.PERFORMED_AT] = ba.performed_at
        row[s.SOURCE] = ba.source

    for ear_name, ear_obj in (("right", ba.right), ("left", ba.left)):
        e = s.WIDE_EARS_INV[ear_name]
        for pathway, data in (
            ("air", ear_obj.air),
            ("bone", ear_obj.bone),
            ("soundfield", ear_obj.soundfield),
            ("ci", ear_obj.ci),
        ):
            for freq_hz, point in sorted(data.items()):
                col = s.wide_column_name(ear_name, pathway, freq_hz)
                row[col] = point.threshold_db
                if point.masked:
                    row[s.wide_column_name(ear_name, pathway, freq_hz, "masked")] = True
                if point.nr:
                    row[s.wide_column_name(ear_name, pathway, freq_hz, "nr")] = True

        if ear_obj.srt is not None:
            row[f"{e}_srt"] = ear_obj.srt
        if ear_obj.sat is not None:
            row[f"{e}_sat"] = ear_obj.sat
        for i, w in enumerate(ear_obj.wrs):
            suffix = f"_{i + 1}" if len(ear_obj.wrs) > 1 else ""
            row[f"{e}_wrs{suffix}"] = w.score_pct
            row[f"{e}_wrs{suffix}_level"] = w.level_db
            if w.masked:
                row[f"{e}_wrs{suffix}_masked"] = True
            if w.word_list is not None:
                row[f"{e}_wrs{suffix}_list"] = w.word_list

    return row


def from_wide_row(
    row: dict[str, Any],
    column_map: dict[str, str] | None = None,
    *,
    audiogram_id: str | None = None,
    subject_id: str | None = None,
    performed_at: str | None = None,
    source: str | None = None,
) -> BinauralAudiogram:
    from .audiogram import ThresholdPoint, WordRecognitionScore, EarAudiogram, BinauralAudiogram
    s = _load_schema_module()

    if column_map is not None:
        row = s.apply_column_map(row, column_map)

    meta_id = audiogram_id or row.get(s.AUDIOGRAM_ID)
    meta_subj = subject_id or row.get(s.SUBJECT_ID)
    meta_date = performed_at or row.get(s.PERFORMED_AT)
    meta_src = source or row.get(s.SOURCE)

    ear_data: dict[str, dict[str, dict[int, ThresholdPoint]]] = {
        "left": {"air": {}, "bone": {}, "soundfield": {}, "ci": {}},
        "right": {"air": {}, "bone": {}, "soundfield": {}, "ci": {}},
    }

    for col, value in row.items():
        parsed = s.parse_wide_column(str(col))
        if parsed is None or parsed["field"] != "threshold":
            continue
        if value is None or (isinstance(value, str) and value.strip() == ""):
            continue

        ear = parsed["ear"]
        pathway = parsed["pathway"]
        freq_hz = parsed["freq_hz"]

        masked_col = s.wide_column_name(ear, pathway, freq_hz, "masked")
        nr_col = s.wide_column_name(ear, pathway, freq_hz, "nr")

        point = ThresholdPoint(
            threshold_db=float(value),
            masked=bool(row.get(masked_col, False)),
            nr=bool(row.get(nr_col, False)),
        )
        ear_data[ear][pathway][freq_hz] = point

    speech: dict[str, dict[str, Any]] = {"left": {}, "right": {}}
    for ear_abbr, ear_name in s.WIDE_EARS.items():
        speech[ear_name]["srt"] = _float_or_none(row.get(f"{ear_abbr}_srt"))
        speech[ear_name]["sat"] = _float_or_none(row.get(f"{ear_abbr}_sat"))
        wrs_list: list[WordRecognitionScore] = []
        single = row.get(f"{ear_abbr}_wrs")
        if single is not None and single != "":
            wrs_list.append(WordRecognitionScore(
                score_pct=float(single),
                level_db=float(row.get(f"{ear_abbr}_wrs_level", 0)),
                masked=bool(row.get(f"{ear_abbr}_wrs_masked", False)),
                word_list=row.get(f"{ear_abbr}_wrs_list"),
            ))
        i = 1
        while f"{ear_abbr}_wrs_{i}" in row:
            val = row[f"{ear_abbr}_wrs_{i}"]
            if val is not None and val != "":
                wrs_list.append(WordRecognitionScore(
                    score_pct=float(val),
                    level_db=float(row.get(f"{ear_abbr}_wrs_{i}_level", 0)),
                    masked=bool(row.get(f"{ear_abbr}_wrs_{i}_masked", False)),
                    word_list=row.get(f"{ear_abbr}_wrs_{i}_list"),
                ))
            i += 1
        speech[ear_name]["wrs"] = wrs_list or None

    left = EarAudiogram(
        air=ear_data["left"]["air"], bone=ear_data["left"]["bone"],
        soundfield=ear_data["left"]["soundfield"], ci=ear_data["left"]["ci"],
        srt=speech["left"]["srt"], sat=speech["left"]["sat"],
        wrs=speech["left"]["wrs"],
    )
    right = EarAudiogram(
        air=ear_data["right"]["air"], bone=ear_data["right"]["bone"],
        soundfield=ear_data["right"]["soundfield"], ci=ear_data["right"]["ci"],
        srt=speech["right"]["srt"], sat=speech["right"]["sat"],
        wrs=speech["right"]["wrs"],
    )
    return BinauralAudiogram(
        left,
        right,
        audiogram_id=meta_id,
        subject_id=meta_subj,
        performed_at=meta_date,
        source=meta_src,
    )
