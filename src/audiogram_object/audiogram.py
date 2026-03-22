from __future__ import annotations
import csv
from io import StringIO
from typing import Tuple, Any, Iterable

from . import symbols as sym


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


class BaseAudiogram:
    def __init__(self, thresholds=None):
        cleaned = thresholds or {}
        self.thresholds: dict[int, float] = {int(k): float(v) for k, v in cleaned.items()}

    def available_frequencies(self) -> list[int]:
        """Return sorted available test frequencies for this ear."""
        m = _load_metrics_module()
        return m.available_frequencies_from_thresholds(self.thresholds)

    def pta(self, freqs: Iterable[int] = (500, 1000, 2000), *, require_all: bool = False) -> float | None:
        """Compute pure-tone average for this ear.

        Parameters
        - freqs: frequencies to include in the average
        - require_all: if True, return None unless all requested frequencies are present
        """
        m = _load_metrics_module()
        return m.pta_from_thresholds(self.thresholds, freqs=freqs, require_all=require_all)

    def plot(
        self,
        *,
        ear: str = "left",
        cfg: sym.AudiogramRenderConfig = sym.DEFAULT_RENDER_CONFIG,
        style: Any = None,
        preset: Any = None,
        ax=None,
        title: str | None = None,
        masked_freqs=(),
        nr_freqs=(),
        figsize: Tuple[float, float] | None = None,
        dpi: int | None = None,
        constrained_layout: bool = True,
        show: bool = True,
    ):
        """Plot this audiogram (single ear) using the Matplotlib plotting helpers.

        Parameters
        - ear: 'left' or 'right' (controls color + symbol convention)
        - cfg: render config (symbols + style)
        - style: high-level preset controlling axes style, default figsize, palette, etc.
        - preset: optional axes-only override (wins over style for axis configuration)
        - ax: optional Matplotlib axes to draw into
        - masked_freqs / nr_freqs: iterables of frequencies toggling masked/NR glyphs
        """
        pm = _load_plot_module()

        # Local import to avoid hard matplotlib dependency at import time
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = pm.new_audiogram_canvas(
                style=style,
                figsize=figsize,
                dpi=dpi,
                preset=preset,
                title=title,
                constrained_layout=constrained_layout,
            )
        else:
            fig = ax.figure
            # If the caller gave us an axes, we assume they will configure it,
            # but we can still apply the preset if they want.
            if preset is not None:
                pm.setup_audiogram_axes(ax, preset=preset, title=title)

        pm.plot_ear(
            ax,
            self.thresholds,
            ear=ear,
            cfg=cfg,
            masked_freqs=masked_freqs,
            nr_freqs=nr_freqs,
        )

        if show:
            plt.show()

        return fig, ax

    def to_dict(self) -> dict[int, float]:
        """Return a dictionary of frequency -> threshold."""
        return dict(self.thresholds)

    @classmethod
    def from_dict(cls, data: dict[int, float]):
        """Create an Audiogram object from a dictionary."""
        # Coerce keys to int so JSON and CSV inputs work cleanly
        cleaned = {int(k): v for k, v in data.items()}
        return cls(thresholds=cleaned)

    def to_json(self) -> str:
        """Serialize the audiogram to a JSON string."""
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """Create an Audiogram from a JSON string."""
        import json
        data = json.loads(json_str)
        return cls.from_dict(data)

def make_threshold_property(freq):
    def getter(self):
        return self.thresholds.get(freq)

    def setter(self, value):
        self.thresholds[freq] = value

    return property(getter, setter)

# Class generator to allow iteration through threshold property creation
def make_audiogram_class(default_freqs=None):
    default_freqs = default_freqs or [250, 500, 1000, 2000, 4000, 8000]

    class Audiogram(BaseAudiogram):
        STANDARD_FREQS = default_freqs

        def __init__(self, thresholds=None, frequencies=None):
            super().__init__(thresholds)
            self._frequencies = frequencies or self.STANDARD_FREQS

            for freq in self._frequencies:
                attr = f"thres{freq}"
                if not hasattr(self.__class__, attr):
                    setattr(self.__class__, attr, make_threshold_property(freq))

    for freq in default_freqs:
        setattr(Audiogram, f"thres{freq}", make_threshold_property(freq))

    return Audiogram


# This is what users will import:
Audiogram = make_audiogram_class()


class BinauralAudiogram:
    def __init__(
        self,
        left: BaseAudiogram,
        right: BaseAudiogram,
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

        # Test-level metadata belongs on the binaural object, not per-ear.
        self.audiogram_id = audiogram_id
        self.subject_id = subject_id
        self.performed_at = performed_at
        self.source = source
        self.meta = dict(meta or {})

    def get_threshold(self, freq: int, side: str):
        ear = {"left": self.left, "right": self.right}.get(side.lower())
        if not ear:
            raise ValueError("Side must be 'left' or 'right'")
        return ear.thresholds.get(freq)

    def _iter_threshold_rows(self):
        """Yield canonical threshold observation tuples for both ears.

        v1 currently exports air-conduction thresholds only. Masking / NR are not yet
        first-class object fields, so they default to False for schema exports.
        """
        for ear_name, ear_obj in (("left", self.left), ("right", self.right)):
            for freq_hz, threshold_db in sorted(ear_obj.to_dict().items()):
                yield {
                    "ear": ear_name,
                    "freq_hz": int(freq_hz),
                    "threshold_db": float(threshold_db),
                    "pathway": "air",
                    "masked": False,
                    "nr": False,
                }

    @staticmethod
    def _meta_from_first_row(row: dict[str, Any]) -> dict[str, Any]:
        """Extract test-level metadata from one long-form schema row."""
        s = _load_schema_module()
        return {
            "audiogram_id": row.get(s.AUDIOGRAM_ID),
            "subject_id": row.get(s.SUBJECT_ID),
            "performed_at": row.get(s.PERFORMED_AT),
            "source": row.get(s.SOURCE),
        }

    def symmetry(self) -> dict[int, float]:
        """Return frequency-to-difference map (left - right)"""
        m = _load_metrics_module()
        return m.symmetry_from_thresholds(self.left.thresholds, self.right.thresholds)

    def pta(self, freqs: Iterable[int] = (500, 1000, 2000), *, require_all: bool = False) -> dict[str, float | None]:
        """Return PTA for each ear as {'left': ..., 'right': ...}."""
        return {
            "left": self.left.pta(freqs=freqs, require_all=require_all),
            "right": self.right.pta(freqs=freqs, require_all=require_all),
        }

    def better_ear_pta(self, freqs: Iterable[int] = (500, 1000, 2000), *, require_all: bool = False) -> float | None:
        """Return the lower (better) PTA across ears, ignoring missing sides when possible."""
        m = _load_metrics_module()
        vals = self.pta(freqs=freqs, require_all=require_all).values()
        return m.better_ear_from_values(vals)

    def worse_ear_pta(self, freqs: Iterable[int] = (500, 1000, 2000), *, require_all: bool = False) -> float | None:
        """Return the higher (worse) PTA across ears, ignoring missing sides when possible."""
        m = _load_metrics_module()
        vals = self.pta(freqs=freqs, require_all=require_all).values()
        return m.worse_ear_from_values(vals)

    def to_long_rows(self, *, include_meta: bool = True, strict_freqs: bool = False) -> list[dict[str, Any]]:
        """Export this binaural audiogram as flat long-form schema rows.

        One row = one threshold observation.
        This is the beginner-friendly / analysis-friendly export shape.
        """
        s = _load_schema_module()

        rows: list[dict[str, Any]] = []
        for obs in self._iter_threshold_rows():
            row = {
                s.AUDIOGRAM_ID: self.audiogram_id,
                s.SUBJECT_ID: self.subject_id if include_meta else None,
                s.PERFORMED_AT: self.performed_at if include_meta else None,
                s.SOURCE: self.source if include_meta else None,
                s.EAR: obs["ear"],
                s.FREQ_HZ: obs["freq_hz"],
                s.THRESHOLD_DB: obs["threshold_db"],
                s.PATHWAY: obs["pathway"],
                s.MASKED: obs["masked"],
                s.NR: obs["nr"],
            }
            s.validate_long_row(row, strict_freqs=strict_freqs)
            rows.append(row)

        return rows

    def to_table_rows(self, *, strict_freqs: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Export this binaural audiogram as normalized table rows.

        Returns
        -------
        tuple[test_row, observation_rows]
            - test_row: one row for the `tests` table
            - observation_rows: many rows for the `observations` table
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
        """Convenience alias for the default flat long-form export."""
        return self.to_long_rows(include_meta=include_meta, strict_freqs=strict_freqs)

    @classmethod
    def from_long_rows(cls, rows: Iterable[dict[str, Any]], *, strict_freqs: bool = False):
        """Construct a BinauralAudiogram from canonical long-form schema rows.

        Notes
        -----
        - v1 imports only air-conduction thresholds into the current object model.
        - Rows with pathway != 'air' are ignored for now.
        - If multiple audiogram_ids are present, this raises ValueError.
        """
        s = _load_schema_module()

        rows = list(rows)
        if not rows:
            raise ValueError("from_long_rows() requires at least one row")

        # Validate input rows first.
        s.validate_long_rows(rows, strict_freqs=strict_freqs)

        # Require a single audiogram_id in this import call.
        audiogram_ids = {row.get(s.AUDIOGRAM_ID) for row in rows}
        if len(audiogram_ids) != 1:
            raise ValueError(
                f"from_long_rows() expected rows for exactly one audiogram_id, got: {sorted(audiogram_ids, key=str)}"
            )

        first = rows[0]
        meta = cls._meta_from_first_row(first)

        left_data: dict[int, float] = {}
        right_data: dict[int, float] = {}

        for row in rows:
            # v1 object model only stores air thresholds directly.
            if row.get(s.PATHWAY) != s.PATHWAY_AIR:
                continue

            ear = row[s.EAR]
            freq_hz = int(row[s.FREQ_HZ])
            threshold_db = float(row[s.THRESHOLD_DB])

            if ear == s.EAR_LEFT:
                left_data[freq_hz] = threshold_db
            elif ear == s.EAR_RIGHT:
                right_data[freq_hz] = threshold_db
            else:
                raise ValueError(f"Unexpected ear value during import: {ear!r}")

        left = Audiogram.from_dict(left_data)
        right = Audiogram.from_dict(right_data)
        return cls(left, right, meta={}, **meta)

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]], *, strict_freqs: bool = False):
        """Convenience alias for canonical long-form row import."""
        return cls.from_long_rows(rows, strict_freqs=strict_freqs)

    def plot(
        self,
        *,
        cfg: sym.AudiogramRenderConfig = sym.DEFAULT_RENDER_CONFIG,
        style: Any = None,
        preset: Any = None,
        two_panel: bool = True,
        title: str | None = None,
        figsize: Tuple[float, float] | None = None,
        dpi: int | None = None,
        constrained_layout: bool = True,
        left_masked=(),
        right_masked=(),
        left_nr=(),
        right_nr=(),
        show: bool = True,
    ):
        """Plot both ears.

        - two_panel=True: clinic-style two-panel layout (Right ear on LEFT panel)
        - two_panel=False: single combined axes
        - style: high-level preset controlling axes style, default figsize, palette, etc.
        - preset: optional axes-only override (wins over style for axis configuration)
        """
        import matplotlib.pyplot as plt
        pm = _load_plot_module()

        if two_panel:
            fig, (ax_r, ax_l) = pm.plot_binaural_two_panel(
                left=self.left.thresholds,
                right=self.right.thresholds,
                cfg=cfg,
                style=style,
                preset=preset,
                title=title,
                figsize=figsize,
                dpi=dpi,
                constrained_layout=constrained_layout,
                left_masked=left_masked,
                right_masked=right_masked,
                left_nr=left_nr,
                right_nr=right_nr,
            )
            if show:
                plt.show()
            return fig, (ax_r, ax_l)

        # Single-panel overlay
        fig, ax = pm.plot_binaural(
            left=self.left.thresholds,
            right=self.right.thresholds,
            cfg=cfg,
            style=style,
            preset=preset,
            title=title,
            figsize=figsize,
            dpi=dpi,
            constrained_layout=constrained_layout,
            left_masked=left_masked,
            right_masked=right_masked,
            left_nr=left_nr,
            right_nr=right_nr,
        )
        if show:
            plt.show()
        return fig, ax

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation including metadata and ear thresholds.

        Structure
        ---------
        {
            'audiogram_id': ...,
            'subject_id': ...,
            'performed_at': ...,
            'source': ...,
            'meta': {...},
            'left': {freq: threshold, ...},
            'right': {freq: threshold, ...}
        }
        """
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
    def from_dict(cls, data: dict[str, Any]):
        """Construct a BinauralAudiogram from a dictionary produced by to_dict().

        Expected keys include:
        - left, right (threshold dicts)
        - optional metadata: audiogram_id, subject_id, performed_at, source, meta
        """
        left = Audiogram.from_dict(data.get("left", {}))
        right = Audiogram.from_dict(data.get("right", {}))
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
        """Serialize the binaural audiogram to a JSON string."""
        import json
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """Construct a BinauralAudiogram from a JSON string."""
        import json
        data = json.loads(json_str)
        return cls.from_dict(data)


    # --- Legacy wide-format helpers (pre-schema v1) ---
    @classmethod
    def from_wide_dict(cls, data: dict[str, float]):
        """Parse wide-format dict with keys like 'L_500', 'R_500' into a BinauralAudiogram."""
        left_data = {}
        right_data = {}

        for key, value in data.items():
            if key.startswith("L_"):
                try:
                    freq = int(key[2:])
                    left_data[freq] = value
                except ValueError:
                    continue
            elif key.startswith("R_"):
                try:
                    freq = int(key[2:])
                    right_data[freq] = value
                except ValueError:
                    continue

        left = Audiogram.from_dict(left_data)
        right = Audiogram.from_dict(right_data)
        return cls(left, right)

    @classmethod
    def from_csv(cls, csv_str: str):
        """
        Create a BinauralAudiogram from a CSV string.
        Assumes a single row with columns like L_250, R_250, etc.
        """
        reader = csv.DictReader(StringIO(csv_str))
        rows = list(reader)
        if not rows:
            raise ValueError("CSV must contain at least one row.")
        return cls.from_wide_dict(rows[0])

    def to_csv(self) -> str:
        """
        Export the BinauralAudiogram as a one-row CSV string.
        """
        wide_dict = {}
        for freq, val in self.left.to_dict().items():
            wide_dict[f"L_{freq}"] = val
        for freq, val in self.right.to_dict().items():
            wide_dict[f"R_{freq}"] = val

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=sorted(wide_dict.keys()))
        writer.writeheader()
        writer.writerow(wide_dict)
        return output.getvalue()