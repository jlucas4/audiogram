from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple


def _load_plot_module():
    """Lazy-load the plotting module so core objects do not require plotting deps at import time."""
    try:
        from . import plot_mpl as pm
    except Exception as e:
        raise ImportError(
            "Plotting requires optional dependencies. Install with: pip install audiogram-object[plot]"
        ) from e
    return pm


def _load_metrics_module():
    """Lazy-load the metrics module so core objects stay lightweight and reusable."""
    from . import metrics as m
    return m


def _load_schema_module():
    """Lazy-load the schema module so the core object layer stays modular."""
    from . import schema as s
    return s


@dataclass
class ThresholdPoint:
    """A single threshold measurement at one frequency.

    For NR (no-response) measurements, threshold_db stores the maximum
    output level tested and nr=True.
    """

    threshold_db: float
    masked: bool = False
    nr: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_db": self.threshold_db,
            "masked": self.masked,
            "nr": self.nr,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThresholdPoint:
        return cls(
            threshold_db=float(data["threshold_db"]),
            masked=bool(data.get("masked", False)),
            nr=bool(data.get("nr", False)),
        )


@dataclass
class WordRecognitionScore:
    """A single word recognition score measurement.

    Pairs the percent-correct score with the presentation level at which
    it was obtained.  Multiple instances per ear represent a
    performance-intensity (PI) function.
    """

    score_pct: float
    level_db: float
    masked: bool = False
    word_list: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "score_pct": self.score_pct,
            "level_db": self.level_db,
            "masked": self.masked,
        }
        if self.word_list is not None:
            d["word_list"] = self.word_list
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WordRecognitionScore:
        return cls(
            score_pct=float(data["score_pct"]),
            level_db=float(data["level_db"]),
            masked=bool(data.get("masked", False)),
            word_list=data.get("word_list"),
        )


def _float_or_none(val: Any) -> float | None:
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return None
    return float(val)


def _coerce_thresholds(
    data: dict[int, ThresholdPoint | float] | None,
) -> dict[int, ThresholdPoint]:
    """Normalize threshold input: accept raw floats or ThresholdPoints."""
    if not data:
        return {}
    out: dict[int, ThresholdPoint] = {}
    for k, v in data.items():
        if isinstance(v, ThresholdPoint):
            out[int(k)] = v
        else:
            out[int(k)] = ThresholdPoint(threshold_db=float(v))
    return out


class EarAudiogram:
    """Air and bone conduction thresholds for one ear, plus optional
    soundfield and cochlear-implant aided thresholds."""

    def __init__(
        self,
        air: dict[int, ThresholdPoint | float] | None = None,
        bone: dict[int, ThresholdPoint | float] | None = None,
        *,
        soundfield: dict[int, ThresholdPoint | float] | None = None,
        ci: dict[int, ThresholdPoint | float] | None = None,
        srt: float | None = None,
        sat: float | None = None,
        wrs: list[WordRecognitionScore] | None = None,
    ):
        self.air: dict[int, ThresholdPoint] = _coerce_thresholds(air)
        self.bone: dict[int, ThresholdPoint] = _coerce_thresholds(bone)
        self.soundfield: dict[int, ThresholdPoint] = _coerce_thresholds(soundfield)
        self.ci: dict[int, ThresholdPoint] = _coerce_thresholds(ci)
        self.srt = srt
        self.sat = sat
        self.wrs = list(wrs) if wrs else []

    def __repr__(self) -> str:
        air_freqs = sorted(self.air)
        bone_freqs = sorted(self.bone)
        parts = [f"air={air_freqs}", f"bone={bone_freqs}"]
        if self.soundfield:
            parts.append(f"soundfield={sorted(self.soundfield)}")
        if self.ci:
            parts.append(f"ci={sorted(self.ci)}")
        if self.srt is not None:
            parts.append(f"srt={self.srt}")
        if self.sat is not None:
            parts.append(f"sat={self.sat}")
        if self.wrs:
            parts.append(f"wrs={self.wrs}")
        return f"EarAudiogram({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EarAudiogram):
            return NotImplemented
        return (
            self.air == other.air
            and self.bone == other.bone
            and self.soundfield == other.soundfield
            and self.ci == other.ci
            and self.srt == other.srt
            and self.sat == other.sat
            and self.wrs == other.wrs
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def available_frequencies(self, pathway: str = "air") -> list[int]:
        """Return sorted frequencies for the given pathway."""
        return sorted(self._pathway_data(pathway))

    def pta(
        self,
        freqs: Iterable[int] = (500, 1000, 2000),
        *,
        pathway: str = "air",
        require_all: bool = False,
    ) -> float | None:
        """Pure-tone average for this ear.

        Parameters
        ----------
        freqs
            Frequencies to include in the average.
        pathway
            'air', 'bone', 'soundfield', or 'ci'.
        require_all
            If True, return None unless all requested frequencies are present.
        """
        m = _load_metrics_module()
        data = self._pathway_data(pathway)
        thresholds = {f: p.threshold_db for f, p in data.items()}
        return m.pta_from_thresholds(thresholds, freqs=freqs, require_all=require_all)

    def severity(
        self,
        freqs: Iterable[int] = (500, 1000, 2000, 4000),
        *,
        standard: str = "who2021",
        pathway: str = "air",
        require_all: bool = False,
    ) -> str | None:
        """Hearing loss severity grade for this ear.

        Parameters
        ----------
        freqs
            Frequencies for the PTA calculation (WHO 2021 only).
            Ignored when standard='aao_hns'.
        standard
            'who2021' (default) or 'aao_hns'. AAO-HNS uses
            500/1000/2000/3000 with avg(2000, 4000) fallback for 3000.
        pathway
            'air', 'bone', 'soundfield', or 'ci'.
        require_all
            If True, return None unless all requested frequencies are present.
        """
        m = _load_metrics_module()
        data = self._pathway_data(pathway)
        thresholds = {f: p.threshold_db for f, p in data.items()}
        return m.severity_from_thresholds(
            thresholds, freqs=freqs, standard=standard, require_all=require_all,
        )

    def abg(self, *, mask_warning: bool = False) -> dict[int, float]:
        """Air-bone gap at each frequency where both air and bone exist.

        Returns a dict of frequency -> gap (dB). Positive values indicate
        a conductive component.
        """
        m = _load_metrics_module()
        air = {f: p.threshold_db for f, p in self.air.items()}
        bone = {f: p.threshold_db for f, p in self.bone.items()}
        bone_masked = {f: p.masked for f, p in self.bone.items()} if mask_warning else None
        return m.abg_from_thresholds(air, bone, mask_warning=mask_warning, bone_masked=bone_masked)

    def abg_pta(
        self,
        *,
        standard: str = "who2021",
        freqs: Iterable[int] = (500, 1000, 2000, 4000),
        require_all: bool = False,
    ) -> float | None:
        """PTA of the air-bone gap."""
        m = _load_metrics_module()
        air = {f: p.threshold_db for f, p in self.air.items()}
        bone = {f: p.threshold_db for f, p in self.bone.items()}
        return m.abg_pta(air, bone, standard=standard, freqs=freqs, require_all=require_all)

    def loss_type(self, *, standard: str = "who2021") -> str | None:
        """Classify hearing loss type: normal, sensorineural, conductive, or mixed."""
        m = _load_metrics_module()
        air = {f: p.threshold_db for f, p in self.air.items()}
        bone = {f: p.threshold_db for f, p in self.bone.items()}
        if standard == "aao_hns":
            air_pta_val = m.aao_hns_pta(air)
            bone_pta_val = m.aao_hns_pta(bone)
        else:
            air_pta_val = m.pta_from_thresholds(air)
            bone_pta_val = m.pta_from_thresholds(bone)
        abg_val = m.abg_pta(air, bone, standard=standard)
        return m.loss_type(air_pta_val, bone_pta_val, abg_val)

    def best_wrs(self) -> WordRecognitionScore | None:
        """Return the WRS with the highest score_pct, or None if no WRS data."""
        m = _load_metrics_module()
        return m.best_wrs(self.wrs)

    def srt_pta_agreement(self, freqs: Iterable[int] = (500, 1000, 2000)) -> float | None:
        """Difference between SRT and PTA (SRT - PTA).

        Clinically, SRT should be within ~10 dB of the 3-frequency PTA.
        A large positive value suggests the SRT is worse than expected.
        Returns None if SRT or PTA is unavailable.
        """
        m = _load_metrics_module()
        thresholds = {f: p.threshold_db for f, p in self.air.items()}
        return m.srt_pta_agreement(self.srt, thresholds, freqs=freqs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pathway_data(self, pathway: str) -> dict[int, ThresholdPoint]:
        if pathway == "air":
            return self.air
        if pathway == "bone":
            return self.bone
        if pathway == "soundfield":
            return self.soundfield
        if pathway == "ci":
            return self.ci
        raise ValueError(f"pathway must be 'air', 'bone', 'soundfield', or 'ci', got {pathway!r}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        ear: str = "left",
        cfg: Any = None,
        style: Any = None,
        preset: Any = None,
        ax=None,
        title: str | None = None,
        show_air: bool = True,
        show_bone: bool = False,
        show_soundfield: bool = False,
        show_ci: bool = False,
        figsize: Tuple[float, float] | None = None,
        dpi: int | None = None,
        constrained_layout: bool = True,
        show: bool = True,
    ):
        """Plot this ear using the Matplotlib plotting helpers.

        Parameters
        ----------
        ear
            'left' or 'right' — controls color and symbol convention.
        cfg
            AudiogramRenderConfig controlling symbol geometry and style.
            Defaults to DEFAULT_RENDER_CONFIG if not provided.
        show_air
            If True (default), plot air conduction thresholds.
        show_bone
            If True, overlay bone conduction thresholds.
        show_soundfield
            If True, overlay soundfield thresholds (plotted as 'S').
        show_ci
            If True, overlay cochlear implant aided thresholds (plotted as 'CI').
        """
        pm = _load_plot_module()
        import matplotlib.pyplot as plt
        from . import symbols as sym
        if cfg is None:
            cfg = sym.DEFAULT_RENDER_CONFIG

        st = pm.get_plot_style(style)

        if ax is None:
            fig, ax = pm.new_audiogram_canvas(
                style=st,
                figsize=figsize,
                dpi=dpi,
                preset=preset,
                title=title,
                constrained_layout=constrained_layout,
            )
        else:
            fig = ax.figure
            if preset is not None:
                pm.setup_audiogram_axes(ax, preset=preset, title=title)

        sf_kwargs = {}
        if show_soundfield and self.soundfield:
            sf_kwargs["soundfield_thresholds"] = {f: p.threshold_db for f, p in self.soundfield.items()}

        ci_kwargs = {}
        if show_ci and self.ci:
            ci_kwargs["ci_thresholds"] = {f: p.threshold_db for f, p in self.ci.items()}

        pm.plot_ear(
            ax,
            {f: p.threshold_db for f, p in self.air.items()},
            ear=ear,
            cfg=cfg,
            show_air=show_air,
            masked_freqs={f for f, p in self.air.items() if p.masked},
            nr_freqs={f for f, p in self.air.items() if p.nr},
            text_marker_fontsize=st.text_marker_fontsize,
            text_marker_fontweight=st.text_marker_fontweight,
            **sf_kwargs,
            **ci_kwargs,
        )

        if show_bone and self.bone:
            pm.plot_ear(
                ax,
                {},
                ear=ear,
                cfg=cfg,
                show_air=False,
                show_bone=True,
                bone_thresholds={f: p.threshold_db for f, p in self.bone.items()},
                bone_masked_freqs={f for f, p in self.bone.items() if p.masked},
                bone_nr_freqs={f for f, p in self.bone.items() if p.nr},
            )

        if show:
            plt.show()

        return fig, ax

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "air": {int(f): p.to_dict() for f, p in self.air.items()},
            "bone": {int(f): p.to_dict() for f, p in self.bone.items()},
        }
        if self.soundfield:
            d["soundfield"] = {int(f): p.to_dict() for f, p in self.soundfield.items()}
        if self.ci:
            d["ci"] = {int(f): p.to_dict() for f, p in self.ci.items()}
        if self.srt is not None:
            d["srt"] = self.srt
        if self.sat is not None:
            d["sat"] = self.sat
        if self.wrs:
            d["wrs"] = [w.to_dict() for w in self.wrs]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EarAudiogram:
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

        return cls(
            air=_parse(data.get("air", {})),
            bone=_parse(data.get("bone", {})),
            soundfield=_parse(data.get("soundfield", {})),
            ci=_parse(data.get("ci", {})),
            srt=data.get("srt"),
            sat=data.get("sat"),
            wrs=wrs,
        )

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> EarAudiogram:
        import json
        return cls.from_dict(json.loads(json_str))


class BinauralAudiogram:
    def __init__(
        self,
        left: EarAudiogram,
        right: EarAudiogram,
        *,
        audiogram_id: str | None = None,
        subject_id: str | None = None,
        performed_at: str | None = None,
        source: str | None = None,
        meta: dict[str, Any] | None = None,
    ):
        self.left = left
        self.right = right
        self.L = self.left
        self.R = self.right

        self.audiogram_id = audiogram_id
        self.subject_id = subject_id
        self.performed_at = performed_at
        self.source = source
        self.meta = dict(meta or {})

    def __repr__(self) -> str:
        parts = [f"id={self.audiogram_id!r}"] if self.audiogram_id else []
        parts.append(f"left={self.left!r}")
        parts.append(f"right={self.right!r}")
        return f"BinauralAudiogram({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinauralAudiogram):
            return NotImplemented
        return (
            self.left == other.left
            and self.right == other.right
            and self.audiogram_id == other.audiogram_id
            and self.subject_id == other.subject_id
            and self.performed_at == other.performed_at
            and self.source == other.source
            and self.meta == other.meta
        )

    # ------------------------------------------------------------------
    # Threshold access
    # ------------------------------------------------------------------

    def get_threshold(self, freq: int, side: str, pathway: str = "air") -> ThresholdPoint | None:
        """Return the ThresholdPoint for a given frequency, side, and pathway."""
        ear = {"left": self.left, "right": self.right}.get(side.lower())
        if ear is None:
            raise ValueError("side must be 'left' or 'right'")
        return ear._pathway_data(pathway).get(freq)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def symmetry(self, pathway: str = "air") -> dict[int, float]:
        """Frequency-to-difference map (left - right). Positive = left is worse."""
        m = _load_metrics_module()
        left_data = {f: p.threshold_db for f, p in self.left._pathway_data(pathway).items()}
        right_data = {f: p.threshold_db for f, p in self.right._pathway_data(pathway).items()}
        return m.symmetry_from_thresholds(left_data, right_data)

    def pta(
        self,
        freqs: Iterable[int] = (500, 1000, 2000),
        *,
        pathway: str = "air",
        require_all: bool = False,
    ) -> dict[str, float | None]:
        """PTA for each ear as {'left': ..., 'right': ...}."""
        return {
            "left": self.left.pta(freqs=freqs, pathway=pathway, require_all=require_all),
            "right": self.right.pta(freqs=freqs, pathway=pathway, require_all=require_all),
        }

    def better_ear_pta(
        self,
        freqs: Iterable[int] = (500, 1000, 2000),
        *,
        pathway: str = "air",
        require_all: bool = False,
    ) -> float | None:
        """Return the lower (better) PTA across ears."""
        m = _load_metrics_module()
        vals = self.pta(freqs=freqs, pathway=pathway, require_all=require_all).values()
        return m.better_ear_from_values(vals)

    def worse_ear_pta(
        self,
        freqs: Iterable[int] = (500, 1000, 2000),
        *,
        pathway: str = "air",
        require_all: bool = False,
    ) -> float | None:
        """Return the higher (worse) PTA across ears."""
        m = _load_metrics_module()
        vals = self.pta(freqs=freqs, pathway=pathway, require_all=require_all).values()
        return m.worse_ear_from_values(vals)

    def severity(
        self,
        freqs: Iterable[int] = (500, 1000, 2000, 4000),
        *,
        standard: str = "who2021",
        pathway: str = "air",
        require_all: bool = False,
    ) -> dict[str, str | None]:
        """Hearing loss severity grade for each ear.

        Parameters
        ----------
        freqs
            Frequencies for the PTA calculation (WHO 2021 only).
            Ignored when standard='aao_hns'.
        standard
            'who2021' (default) or 'aao_hns'. AAO-HNS uses
            500/1000/2000/3000 with avg(2000, 4000) fallback for 3000.
        pathway
            'air', 'bone', 'soundfield', or 'ci'.
        require_all
            If True, return None unless all requested frequencies are present.

        Returns
        -------
        dict
            {'left': 'moderate', 'right': 'mild', ...}
        """
        return {
            "left": self.left.severity(freqs=freqs, standard=standard, pathway=pathway, require_all=require_all),
            "right": self.right.severity(freqs=freqs, standard=standard, pathway=pathway, require_all=require_all),
        }

    def interaural_differences(self, pathway: str = "air"):
        """Compute interaural threshold differences for all shared frequencies.

        Returns a list of InterauralDifference objects sorted by frequency.
        difference_db = left - right (positive = left worse, negative = right worse).
        """
        from . import asymmetry as asym
        return asym.compute_interaural_differences(self, pathway=pathway)

    def is_asymmetric(
        self,
        criterion: str = "any_15db",
    ) -> bool | None:
        """Evaluate whether this audiogram meets an asymmetry criterion.

        Parameters
        ----------
        criterion
            A named built-in criterion (e.g. 'any_15db', 'pta_15db',
            'two_consecutive_10db', 'nr_one_side') or any callable that
            accepts a BinauralAudiogram and returns bool | None.

        Returns
        -------
        bool | None
            True if asymmetric, False if not, None if indeterminate.
        """
        from . import asymmetry as asym
        return asym.is_asymmetric(self, criterion=criterion)

    def summary(
        self,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a flat dict of computed metrics, ready to merge into a DataFrame row.

        Parameters
        ----------
        include
            If provided, only run these metric categories (e.g. ['pta', 'asymmetry']).
        exclude
            If provided, skip these metric categories.
        **kwargs
            Forwarded to each metric function (e.g. ``standard="aao_hns"``).
        """
        m = _load_metrics_module()
        return m.compute_summary(self, include=include, exclude=exclude, **kwargs)

    # ------------------------------------------------------------------
    # Schema row I/O
    # ------------------------------------------------------------------

    def _iter_threshold_rows(self):
        """Yield canonical threshold observation dicts for both ears and pathways."""
        for ear_name, ear_obj in (("left", self.left), ("right", self.right)):
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

    def _iter_speech_rows(self):
        """Yield speech observation dicts for both ears."""
        for ear_name, ear_obj in (("left", self.left), ("right", self.right)):
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

    @staticmethod
    def _meta_from_first_row(row: dict[str, Any]) -> dict[str, Any]:
        s = _load_schema_module()
        return {
            "audiogram_id": row.get(s.AUDIOGRAM_ID),
            "subject_id": row.get(s.SUBJECT_ID),
            "performed_at": row.get(s.PERFORMED_AT),
            "source": row.get(s.SOURCE),
        }

    def to_long_rows(
        self,
        *,
        include_meta: bool = True,
        include_speech: bool = False,
        strict_freqs: bool = False,
    ) -> list[dict[str, Any]]:
        """Export as flat long-form schema rows.

        Parameters
        ----------
        include_meta
            If True, include metadata columns on each row.
        include_speech
            If True, append speech rows with a measure_type discriminator.
            Threshold rows get measure_type='threshold', speech rows get
            measure_type='speech'. When False (default), returns threshold
            rows only with no measure_type column.
        strict_freqs
            If True, validate frequencies against the standard set.
        """
        s = _load_schema_module()
        rows: list[dict[str, Any]] = []

        meta = {}
        if include_meta:
            meta = {
                s.AUDIOGRAM_ID: self.audiogram_id,
                s.SUBJECT_ID: self.subject_id,
                s.PERFORMED_AT: self.performed_at,
                s.SOURCE: self.source,
            }
        else:
            meta = {
                s.AUDIOGRAM_ID: self.audiogram_id,
                s.SUBJECT_ID: None,
                s.PERFORMED_AT: None,
                s.SOURCE: None,
            }

        for obs in self._iter_threshold_rows():
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
            for obs in self._iter_speech_rows():
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

    def to_speech_rows(self, *, include_meta: bool = True) -> list[dict[str, Any]]:
        """Export speech data as tidy long-form rows.

        One row per speech measurement (SRT, SAT, or WRS) per ear.
        """
        s = _load_schema_module()
        rows: list[dict[str, Any]] = []
        for obs in self._iter_speech_rows():
            row: dict[str, Any] = {}
            if include_meta:
                row[s.AUDIOGRAM_ID] = self.audiogram_id
                row[s.SUBJECT_ID] = self.subject_id
                row[s.PERFORMED_AT] = self.performed_at
                row[s.SOURCE] = self.source
            else:
                row[s.AUDIOGRAM_ID] = self.audiogram_id
            row[s.EAR] = obs["ear"]
            row[s.MEASURE] = obs["measure"]
            row[s.VALUE] = obs["value"]
            row[s.LEVEL_DB] = obs["level_db"]
            row[s.MASKED] = obs["masked"]
            row[s.WORD_LIST] = obs["word_list"]
            rows.append(row)
        return rows

    def to_table_rows(self, *, strict_freqs: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Export as normalized table rows.

        Returns
        -------
        tuple[test_row, observation_rows]
        """
        s = _load_schema_module()

        test_row = {
            s.AUDIOGRAM_ID: self.audiogram_id,
            s.SUBJECT_ID: self.subject_id,
            s.PERFORMED_AT: self.performed_at,
            s.SOURCE: self.source,
            s.SCHEMA_VERSION_COL: s.SCHEMA_VERSION,
            s.META: dict(self.meta),
        }
        s.validate_tests_row(test_row)

        obs_rows: list[dict[str, Any]] = []
        for obs in self._iter_threshold_rows():
            row = {
                s.AUDIOGRAM_ID: self.audiogram_id,
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

    def to_rows(self, *, include_meta: bool = True, strict_freqs: bool = False) -> list[dict[str, Any]]:
        """Convenience alias for to_long_rows()."""
        return self.to_long_rows(include_meta=include_meta, strict_freqs=strict_freqs)

    @classmethod
    def from_long_rows(cls, rows: Iterable[dict[str, Any]], *, strict_freqs: bool = False) -> BinauralAudiogram:
        """Construct a BinauralAudiogram from canonical long-form schema rows.

        Accepts threshold-only rows, speech-only rows, or combined rows
        (with a measure_type discriminator column). Speech rows are
        identified by the presence of a 'measure' column or
        measure_type='speech'.
        """
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
        meta = cls._meta_from_first_row(first)

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
        return cls(left, right, meta={}, **meta)

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]], *, strict_freqs: bool = False) -> BinauralAudiogram:
        """Convenience alias for from_long_rows()."""
        return cls.from_long_rows(rows, strict_freqs=strict_freqs)

    # ------------------------------------------------------------------
    # Wide-format I/O
    # ------------------------------------------------------------------

    def to_wide_row(self, *, include_meta: bool = True) -> dict[str, Any]:
        """Export as a single flat dict in canonical wide format.

        Column convention: {ear}_{pathway}_{frequency}
        Boolean flags:     {ear}_{pathway}_{frequency}_masked
                           {ear}_{pathway}_{frequency}_nr

        Masked/NR columns are only included when True for at least one
        threshold, keeping the common (sparse) case clean.
        """
        s = _load_schema_module()
        row: dict[str, Any] = {}

        if include_meta:
            row[s.AUDIOGRAM_ID] = self.audiogram_id
            row[s.SUBJECT_ID] = self.subject_id
            row[s.PERFORMED_AT] = self.performed_at
            row[s.SOURCE] = self.source

        for ear_name, ear_obj in (("right", self.right), ("left", self.left)):
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

    @classmethod
    def from_wide_row(
        cls,
        row: dict[str, Any],
        column_map: dict[str, str] | None = None,
        *,
        audiogram_id: str | None = None,
        subject_id: str | None = None,
        performed_at: str | None = None,
        source: str | None = None,
    ) -> BinauralAudiogram:
        """Construct from a single wide-format row.

        Parameters
        ----------
        row
            A flat dict where keys follow the canonical wide convention
            ({ear}_{pathway}_{freq}) or are mapped to it via *column_map*.
        column_map
            Optional dict mapping the dataset's column names to canonical
            wide names.  Applied before parsing.  Keys not in the map are
            passed through unchanged.
        audiogram_id, subject_id, performed_at, source
            Metadata overrides.  If not provided, the function looks for
            canonical metadata columns in the row.
        """
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
        return cls(
            left,
            right,
            audiogram_id=meta_id,
            subject_id=meta_subj,
            performed_at=meta_date,
            source=meta_src,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        *,
        cfg: Any = None,
        style: Any = None,
        preset: Any = None,
        two_panel: bool = True,
        show_air: bool = True,
        show_bone: bool = False,
        show_soundfield: bool = False,
        show_ci: bool = False,
        title: str | None = None,
        figsize: Tuple[float, float] | None = None,
        dpi: int | None = None,
        constrained_layout: bool = True,
        show: bool = True,
    ):
        """Plot both ears.

        Parameters
        ----------
        cfg
            AudiogramRenderConfig controlling symbol geometry and style.
            Defaults to DEFAULT_RENDER_CONFIG if not provided.
        two_panel
            True: clinic-style two-panel layout (right ear on left panel).
            False: single combined axes.
        show_air
            If True (default), plot air conduction thresholds.
        show_bone
            If True, overlay bone conduction thresholds.
        show_soundfield
            If True, overlay soundfield thresholds (plotted as 'S').
            On two-panel layout, soundfield is mirrored on both panels.
        show_ci
            If True, overlay cochlear implant aided thresholds (plotted as 'CI').
        """
        import matplotlib.pyplot as plt
        pm = _load_plot_module()
        from . import symbols as sym
        if cfg is None:
            cfg = sym.DEFAULT_RENDER_CONFIG

        left_air = {f: p.threshold_db for f, p in self.left.air.items()}
        right_air = {f: p.threshold_db for f, p in self.right.air.items()}
        left_masked = {f for f, p in self.left.air.items() if p.masked}
        right_masked = {f for f, p in self.right.air.items() if p.masked}
        left_nr = {f for f, p in self.left.air.items() if p.nr}
        right_nr = {f for f, p in self.right.air.items() if p.nr}

        bone_kwargs = {}
        if show_bone:
            bone_kwargs = dict(
                show_bone=True,
                left_bone={f: p.threshold_db for f, p in self.left.bone.items()},
                right_bone={f: p.threshold_db for f, p in self.right.bone.items()},
                left_bone_masked={f for f, p in self.left.bone.items() if p.masked},
                right_bone_masked={f for f, p in self.right.bone.items() if p.masked},
                left_bone_nr={f for f, p in self.left.bone.items() if p.nr},
                right_bone_nr={f for f, p in self.right.bone.items() if p.nr},
            )

        sf_kwargs = {}
        if show_soundfield:
            left_sf = {f: p.threshold_db for f, p in self.left.soundfield.items()} or None
            right_sf = {f: p.threshold_db for f, p in self.right.soundfield.items()} or None
            if two_panel:
                # Soundfield is binaural — merge and mirror on both panels
                merged_sf: dict[int, float] = {}
                if left_sf:
                    merged_sf.update(left_sf)
                if right_sf:
                    merged_sf.update(right_sf)
                if merged_sf:
                    sf_kwargs = dict(
                        left_soundfield=merged_sf,
                        right_soundfield=merged_sf,
                    )
            else:
                # Single panel — combine into one series
                merged_sf = {}
                if left_sf:
                    merged_sf.update(left_sf)
                if right_sf:
                    merged_sf.update(right_sf)
                if merged_sf:
                    sf_kwargs = dict(
                        left_soundfield=merged_sf,
                    )

        ci_kwargs = {}
        if show_ci:
            left_ci = {f: p.threshold_db for f, p in self.left.ci.items()} or None
            right_ci = {f: p.threshold_db for f, p in self.right.ci.items()} or None
            ci_kwargs = dict(left_ci=left_ci, right_ci=right_ci)

        if two_panel:
            fig, axes = pm.plot_binaural_two_panel(
                left=left_air,
                right=right_air,
                cfg=cfg,
                style=style,
                preset=preset,
                title=title,
                figsize=figsize,
                dpi=dpi,
                constrained_layout=constrained_layout,
                show_air=show_air,
                left_masked=left_masked,
                right_masked=right_masked,
                left_nr=left_nr,
                right_nr=right_nr,
                **bone_kwargs,
                **sf_kwargs,
                **ci_kwargs,
            )
            if show:
                plt.show()
            return fig, axes

        fig, ax = pm.plot_binaural(
            left=left_air,
            right=right_air,
            cfg=cfg,
            style=style,
            preset=preset,
            title=title,
            figsize=figsize,
            dpi=dpi,
            constrained_layout=constrained_layout,
            show_air=show_air,
            left_masked=left_masked,
            right_masked=right_masked,
            left_nr=left_nr,
            right_nr=right_nr,
            **bone_kwargs,
            **sf_kwargs,
            **ci_kwargs,
        )
        if show:
            plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "audiogram_id": self.audiogram_id,
            "subject_id": self.subject_id,
            "performed_at": self.performed_at,
            "source": self.source,
            "meta": dict(self.meta),
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinauralAudiogram:
        left = EarAudiogram.from_dict(data.get("left", {}))
        right = EarAudiogram.from_dict(data.get("right", {}))
        return cls(
            left,
            right,
            audiogram_id=data.get("audiogram_id"),
            subject_id=data.get("subject_id"),
            performed_at=data.get("performed_at"),
            source=data.get("source"),
            meta=data.get("meta") or {},
        )

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> BinauralAudiogram:
        import json
        return cls.from_dict(json.loads(json_str))

