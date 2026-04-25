# audiogram-object

A Python library for representing clinical audiometric data as typed objects.

Audiograms are structured data — frequencies, thresholds, air and bone conduction, masking,
no-response flags — but they're routinely stored as flat dictionaries, wide-format CSVs, or
strings in EMR free-text fields. `audiogram-object` gives them a proper object model with
metrics, asymmetry analysis, serialization, and optional matplotlib plotting.

## Install

```bash
pip install audiogram-object
```

With matplotlib plotting support:

```bash
pip install audiogram-object[plot]
```

Requires Python 3.11+. No required dependencies for the core library.

## Quick start

```python
from audiogram_object import ThresholdPoint, EarAudiogram, BinauralAudiogram

left = EarAudiogram(
    air={
        500:  ThresholdPoint(25.0),
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
        500:  ThresholdPoint(20.0),
        1000: ThresholdPoint(25.0),
        2000: ThresholdPoint(40.0),
        4000: ThresholdPoint(120.0, nr=True),  # no response — store max output + nr=True
    },
)

ba = BinauralAudiogram(
    left=left,
    right=right,
    audiogram_id="aud-001",
    subject_id="pt-042",
    performed_at="2024-06-15",
    source="clinic",
)
```

## Key concepts

### ThresholdPoint

The atomic unit — one threshold measurement at one frequency.

```python
ThresholdPoint(threshold_db=30.0)                    # unmasked, response present
ThresholdPoint(30.0, masked=True)                    # contralateral ear was masked
ThresholdPoint(120.0, nr=True)                       # no response; value = max output tested
```

Plain floats are accepted anywhere a `ThresholdPoint` is expected and are coerced automatically.

### WordRecognitionScore

Pairs a percent-correct score with its presentation level.

```python
WordRecognitionScore(92.0, 70.0)                     # 92% at 70 dB HL
WordRecognitionScore(88.0, 90.0, masked=True)         # masked
WordRecognitionScore(92.0, 70.0, word_list="CNC")     # with word list
```

### EarAudiogram

One ear's air and bone conduction thresholds, plus optional speech audiometry.

```python
ear = EarAudiogram(
    air={500: 20.0, 1000: 25.0, 2000: 30.0},
    srt=25.0,
    wrs=[WordRecognitionScore(92.0, 70.0, word_list="CNC")],
)

ear.pta()                                            # 25.0 — standard 3-freq PTA
ear.pta(freqs=(500, 1000, 2000, 4000))              # 4-frequency PTA
ear.pta(pathway="bone")                              # bone conduction PTA
ear.pta(require_all=True)                            # None if any freq is missing
ear.available_frequencies()                          # [500, 1000, 2000]
ear.available_frequencies("bone")                    # bone frequencies

# Speech audiometry
ear.srt                                              # 25.0
ear.best_wrs()                                       # WordRecognitionScore(92.0, 70.0, ...)
ear.srt_pta_agreement()                              # 0.0 (SRT - PTA; should be within ±10 dB)
```

### BinauralAudiogram

Both ears with test-level metadata.

```python
ba.L is ba.left                                      # True — shorthand aliases
ba.R is ba.right

ba.get_threshold(1000, "left")                       # ThresholdPoint
ba.get_threshold(1000, "left", pathway="bone")

ba.pta()                                             # {"left": 30.0, "right": 28.3}
ba.better_ear_pta()                                  # 28.3
ba.worse_ear_pta()                                   # 30.0
ba.symmetry()                                        # {500: 5.0, 1000: 5.0, ...} (left - right)
```

## Air-bone gap & loss type

```python
# Per-frequency air-bone gap
left.abg()                                           # {500: 5.0, 1000: 10.0, ...}
left.abg(mask_warning=True)                          # warns if any bone thresholds are unmasked

# ABG PTA — average of the gaps
left.abg_pta()                                       # WHO 2021 default: 500/1000/2000/4000
left.abg_pta(standard="aao_hns")                     # AAO-HNS: 500/1000/2000/3000 (with fallback)

# Loss type classification
left.loss_type()                                     # "sensorineural", "conductive", "mixed", or "normal"
left.loss_type(standard="aao_hns")
```

Classification: air PTA ≤ 25 → `normal`; ABG-PTA < 10 → `sensorineural`;
ABG-PTA ≥ 10 with normal bone → `conductive`; ABG-PTA ≥ 10 with elevated bone → `mixed`.
Returns `None` when air or bone data is insufficient to classify.

## Severity

WHO 2021 hearing loss grading based on pure-tone average. Also supports AAO-HNS
(500/1000/2000/3000 Hz with avg(2000, 4000) fallback when 3000 is absent).

```python
left.severity()                                      # "moderate" (4-freq PTA default)
left.severity(standard="aao_hns")                    # AAO-HNS method
left.severity(freqs=(500, 1000, 2000))               # custom frequencies (WHO only)
ba.severity()                                        # {"left": "moderate", "right": "mild"}
```

Grades: `normal` (≤25), `mild` (26–40), `moderate` (41–55), `moderately_severe` (56–70),
`severe` (71–90), `profound` (>90).

## Asymmetry

```python
from audiogram_object import ASYMMETRY_CRITERIA

# Named built-in criteria
ba.is_asymmetric("any_15db")             # any frequency >= 15 dB interaural difference
ba.is_asymmetric("two_consecutive_10db") # two consecutive freqs >= 10 dB apart
ba.is_asymmetric("pta_15db")             # PTA difference between ears >= 15 dB
ba.is_asymmetric("nr_one_side")          # one ear NR, other has a threshold

# Custom criterion — any callable (BinauralAudiogram) -> bool | None
ba.is_asymmetric(lambda ba: abs(ba.pta()["left"] - ba.pta()["right"]) > 20)

# Interaural differences per frequency
for d in ba.interaural_differences():
    print(d.freq_hz, d.difference_db, d.better_ear, d.nr_involved)
```

`is_asymmetric` returns `True`, `False`, or `None` (indeterminate — e.g. missing frequencies).
Clinical criteria in the built-ins are reasonable defaults; verify against your target guideline
before using for clinical decision support.

## Serialization

Dict/JSON is the canonical lossless interchange format — it round-trips all data
including thresholds, bone conduction, masking, NR, and speech audiometry.

```python
# Dict / JSON — full fidelity
ba2 = BinauralAudiogram.from_dict(ba.to_dict())
ba2 = BinauralAudiogram.from_json(ba.to_json())

# Long-form threshold rows — one dict per threshold observation
rows = ba.to_long_rows()
ba2  = BinauralAudiogram.from_long_rows(rows)

# Long-form speech rows — one dict per speech measurement
speech_rows = ba.to_speech_rows()

# Combined long-form — thresholds + speech with measure_type discriminator
all_rows = ba.to_long_rows(include_speech=True)
ba2 = BinauralAudiogram.from_long_rows(all_rows)

# With pandas or polars
import polars as pl
df = pl.DataFrame(ba.to_long_rows())

# Normalized two-table output (mirrors a relational DB schema)
test_row, obs_rows = ba.to_table_rows()
```

| Format | Thresholds | Speech | Lossless |
|---|---|---|---|
| Dict / JSON | yes | yes | yes |
| Wide row | yes | yes | yes |
| Long rows (default) | yes | no | thresholds only |
| Long rows (`include_speech=True`) | yes | yes | yes |
| Speech rows | no | yes | speech only |

## Wide-format ingest

Research datasets and EMR exports typically store audiograms in wide format — one row
per audiogram, columns like `R_AC_500`, `L_BC_1K`, etc. Every dataset uses different
column names.

`audiogram-object` defines a canonical wide column convention and a simple column-mapping
workflow so you can ingest any wide dataset with a one-time mapping dict.

### Canonical wide column convention

```
{ear}_{pathway}_{frequency}            # threshold value
{ear}_{pathway}_{frequency}_masked     # boolean (default False if absent)
{ear}_{pathway}_{frequency}_nr         # boolean (default False if absent)
```

Examples: `r_air_500`, `l_bone_1000`, `r_air_4000_masked`, `l_air_250_nr`

Ears: `r` / `l`. Pathways: `air` / `bone`. Frequencies: raw integers (Hz).

### Import from wide format

```python
# Your dataset has columns like R_AC_500, L_AC_1K, etc.
# Define a column map once:
column_map = {
    "PatientID":  "subject_id",
    "TestDate":   "performed_at",
    "R_AC_500":   "r_air_500",
    "R_AC_1K":    "r_air_1000",
    "R_AC_2K":    "r_air_2000",
    "L_AC_500":   "l_air_500",
    "L_AC_1K":    "l_air_1000",
    "L_AC_2K":    "l_air_2000",
}

# Ingest a single row
ba = BinauralAudiogram.from_wide_row(row, column_map=column_map)

# Ingest a whole DataFrame
audiograms = [BinauralAudiogram.from_wide_row(row, column_map=column_map)
              for row in df.to_dict(orient="records")]
```

Missing or empty values are silently skipped. Masked and NR flags default to `False`
when absent — which is the common case for research datasets that only contain air
thresholds.

### Export to wide format

```python
wide_row = ba.to_wide_row()
# {'audiogram_id': 'aud-001', 'r_air_500': 20.0, 'r_air_1000': 25.0, ...}
```

### Helper functions

```python
from audiogram_object import wide_column_name, parse_wide_column, canonical_wide_columns

wide_column_name("right", "air", 500)           # "r_air_500"
parse_wide_column("l_bone_1000_masked")          # {"ear": "left", "pathway": "bone", ...}
canonical_wide_columns([500, 1000, 2000])        # all column names for those frequencies
```

## Summary metrics

`summary()` returns a flat dict of all computed metrics — ready to merge into a DataFrame row.

```python
ba.summary()
# {
#     'pta_right': 35.0, 'pta_left': 28.3,
#     'better_ear_pta': 28.3, 'worse_ear_pta': 35.0,
#     'right_severity': 'mild', 'left_severity': 'mild',
#     'right_abg_pta': 15.0, 'left_abg_pta': 5.0,
#     'right_loss_type': 'conductive', 'left_loss_type': 'sensorineural',
#     'asymmetric_any_15db': False, ...
# }

# Pass standard through to severity, ABG, and loss type
ba.summary(standard="aao_hns")

# Filter by category
ba.summary(include=["pta", "abg"])
ba.summary(exclude=["frequency_count"])
```

### Enrichment workflow

```python
# Ingest a wide dataset and enrich with computed metrics
for i, row in df.iterrows():
    ba = BinauralAudiogram.from_wide_row(row, column_map=col_map)
    for k, v in ba.summary().items():
        df.loc[i, k] = v
```

### Custom metrics

Register your own metric functions — they automatically appear in `summary()` output.

```python
from audiogram_object import register_summary_metric

def hearing_loss_degree(ba):
    ptas = ba.pta()
    result = {}
    for ear in ("right", "left"):
        pta = ptas[ear]
        if pta is None:
            result[f"{ear}_hl_degree"] = None
        elif pta <= 25:
            result[f"{ear}_hl_degree"] = "normal"
        elif pta <= 40:
            result[f"{ear}_hl_degree"] = "mild"
        # ... etc
    return result

register_summary_metric("hl_degree", hearing_loss_degree)
ba.summary()  # now includes right_hl_degree, left_hl_degree
```

## Plotting

Requires `pip install audiogram-object[plot]`.

```python
# Both ears — clinic-style two-panel layout
ba.plot()

# Single ear with bone conduction overlay
ba.left.plot(ear="left", show_bone=True)
```

Symbols follow ASHA convention: O (right air), X (left air), [ / ] (bone).
Connecting lines join response-present air conduction points (including masked).
NR points are plotted but not connected.

## License

MIT
