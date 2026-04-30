"""Core object model: ThresholdPoint, WordRecognitionScore, EarAudiogram, BinauralAudiogram."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def _load_plot_module():
    """Lazy-load the plotting module so core objects do not require plotting deps at import time."""
    try:
        from . import plot_mpl as pm
    except Exception as e:
        raise ImportError(
            "Plotting requires optional dependencies. Install with: pip install audiogram-object[plot]"
        ) from e
    return pm


from . import metrics as _metrics
from . import serialization as _serialization


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

    def __post_init__(self) -> None:
        if not (0 <= self.score_pct <= 100):
            raise ValueError(f"score_pct must be 0–100, got {self.score_pct}")

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
        if srt is not None:
            srt = float(srt)
        if sat is not None:
            sat = float(sat)
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
        data = self._pathway_data(pathway)
        thresholds = {f: p.threshold_db for f, p in data.items()}
        return _metrics.pta_from_thresholds(thresholds, freqs=freqs, require_all=require_all)

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
        data = self._pathway_data(pathway)
        thresholds = {f: p.threshold_db for f, p in data.items()}
        return _metrics.severity_from_thresholds(
            thresholds, freqs=freqs, standard=standard, require_all=require_all,
        )

    def abg(self, *, mask_warning: bool = False) -> dict[int, float]:
        """Air-bone gap at each frequency where both air and bone exist.

        Returns a dict of frequency -> gap (dB). Positive values indicate
        a conductive component.
        """
        air = {f: p.threshold_db for f, p in self.air.items()}
        bone = {f: p.threshold_db for f, p in self.bone.items()}
        bone_masked = {f: p.masked for f, p in self.bone.items()} if mask_warning else None
        return _metrics.abg_from_thresholds(air, bone, mask_warning=mask_warning, bone_masked=bone_masked)

    def abg_pta(
        self,
        *,
        standard: str = "who2021",
        freqs: Iterable[int] = (500, 1000, 2000, 4000),
        require_all: bool = False,
    ) -> float | None:
        """PTA of the air-bone gap."""
        air = {f: p.threshold_db for f, p in self.air.items()}
        bone = {f: p.threshold_db for f, p in self.bone.items()}
        return _metrics.abg_pta(air, bone, standard=standard, freqs=freqs, require_all=require_all)

    def loss_type(self, *, standard: str = "who2021") -> str | None:
        """Classify hearing loss type: normal, sensorineural, conductive, or mixed."""
        air = {f: p.threshold_db for f, p in self.air.items()}
        bone = {f: p.threshold_db for f, p in self.bone.items()}
        if standard == "aao_hns":
            air_pta_val = _metrics.aao_hns_pta(air)
            bone_pta_val = _metrics.aao_hns_pta(bone)
        else:
            air_pta_val = _metrics.pta_from_thresholds(air)
            bone_pta_val = _metrics.pta_from_thresholds(bone)
        abg_val = _metrics.abg_pta(air, bone, standard=standard)
        return _metrics.loss_type(air_pta_val, bone_pta_val, abg_val)

    def best_wrs(self) -> WordRecognitionScore | None:
        """Return the WRS with the highest score_pct, or None if no WRS data."""
        return _metrics.best_wrs(self.wrs)

    def srt_pta_agreement(self, freqs: Iterable[int] = (500, 1000, 2000)) -> float | None:
        """Difference between SRT and PTA (SRT - PTA).

        Clinically, SRT should be within ~10 dB of the 3-frequency PTA.
        A large positive value suggests the SRT is worse than expected.
        Returns None if SRT or PTA is unavailable.
        """
        thresholds = {f: p.threshold_db for f, p in self.air.items()}
        return _metrics.srt_pta_agreement(self.srt, thresholds, freqs=freqs)

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
        figsize: tuple[float, float] | None = None,
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
        return _serialization.ear_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EarAudiogram:
        return _serialization.ear_from_dict(data)

    def to_json(self) -> str:
        return _serialization.ear_to_json(self)

    @classmethod
    def from_json(cls, json_str: str) -> EarAudiogram:
        return _serialization.ear_from_json(json_str)


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
        left_data = {f: p.threshold_db for f, p in self.left._pathway_data(pathway).items()}
        right_data = {f: p.threshold_db for f, p in self.right._pathway_data(pathway).items()}
        return _metrics.symmetry_from_thresholds(left_data, right_data)

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
        vals = self.pta(freqs=freqs, pathway=pathway, require_all=require_all).values()
        return _metrics.better_ear_from_values(vals)

    def worse_ear_pta(
        self,
        freqs: Iterable[int] = (500, 1000, 2000),
        *,
        pathway: str = "air",
        require_all: bool = False,
    ) -> float | None:
        """Return the higher (worse) PTA across ears."""
        vals = self.pta(freqs=freqs, pathway=pathway, require_all=require_all).values()
        return _metrics.worse_ear_from_values(vals)

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
        return _metrics.compute_summary(self, include=include, exclude=exclude, **kwargs)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_long_rows(self, *, include_meta: bool = True, include_speech: bool = False, strict_freqs: bool = False) -> list[dict[str, Any]]:
        """Export as flat long-form schema rows."""
        return _serialization.to_long_rows(self, include_meta=include_meta, include_speech=include_speech, strict_freqs=strict_freqs)

    @classmethod
    def from_long_rows(cls, rows: Iterable[dict[str, Any]], *, strict_freqs: bool = False) -> BinauralAudiogram:
        """Construct from canonical long-form schema rows."""
        return _serialization.from_long_rows(rows, strict_freqs=strict_freqs)

    def to_speech_rows(self, *, include_meta: bool = True) -> list[dict[str, Any]]:
        """Export speech data as tidy long-form rows."""
        return _serialization.to_speech_rows(self, include_meta=include_meta)

    def to_table_rows(self, *, strict_freqs: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Export as normalized table rows (test_row, observation_rows)."""
        return _serialization.to_table_rows(self, strict_freqs=strict_freqs)

    def to_wide_row(self, *, include_meta: bool = True) -> dict[str, Any]:
        """Export as a single flat dict in canonical wide format."""
        return _serialization.to_wide_row(self, include_meta=include_meta)

    @classmethod
    def from_wide_row(cls, row: dict[str, Any], column_map: dict[str, str] | None = None, *, audiogram_id: str | None = None, subject_id: str | None = None, performed_at: str | None = None, source: str | None = None) -> BinauralAudiogram:
        """Construct from a single wide-format row."""
        return _serialization.from_wide_row(row, column_map, audiogram_id=audiogram_id, subject_id=subject_id, performed_at=performed_at, source=source)

    def to_dict(self) -> dict[str, Any]:
        return _serialization.binaural_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BinauralAudiogram:
        return _serialization.binaural_from_dict(data)

    def to_json(self) -> str:
        return _serialization.binaural_to_json(self)

    @classmethod
    def from_json(cls, json_str: str) -> BinauralAudiogram:
        return _serialization.binaural_from_json(json_str)

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
        figsize: tuple[float, float] | None = None,
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

