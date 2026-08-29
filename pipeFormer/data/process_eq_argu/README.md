# Compressor-curve embeddings

This package turns compressor performance curves into fixed-size numeric
features. It is used by the parent `pipeFormer/data/process_eq_argu.py`
processor when static equipment features are prepared.

## What it does

1. Normalizes flow and head with the affinity-law approximation
   (`Q / RPM` and `H / RPM²`).
2. Resamples each compressor's normalized head and efficiency curves onto a
   fixed grid (32 points by default).
3. Adds interpretable descriptors such as peak efficiency, best-efficiency
   point, efficiency-band width, and local head slope.
4. Standardizes the resulting vectors and applies PCA when processing a whole
   workbook.

The default per-compressor shape vector has `2 × 32 + 4 = 68` values. Workbook
embeddings contain the requested PCA components (8 by default) plus an `id`
column.

## Input workbook

`embed_from_excel()` reads sheets whose names start with `C_`. Each usable sheet
needs columns whose headers identify:

- head in metres (`水头` plus `m`/`M`),
- flow in `m³/h`, `m3/h`, `m3h`, or `/h`,
- speed in `RPM`/`rpm`, and
- optionally efficiency (`效率`).

Efficiency percentages are converted to the `0..1` range. At least one valid
`C_` sheet is required.

## Usage

Run from `pipeFormer/`:

```python
from data.process_eq_argu import embed_from_excel

# Replace this path with your workbook.
embeddings = embed_from_excel("path/to/C_arguments.xlsx")
embeddings.to_csv("data/equipment_arguments/compressor_embeddings.csv", index=False)
```

For one already-loaded compressor table, use the lower-level helpers:

```python
import pandas as pd

from data.process_eq_argu.process_c_argument import (
    embed_from_dataframe,
    physics_normalize,
)

frame = pd.read_excel("path/to/C_arguments.xlsx", sheet_name="C_001")
normalized = physics_normalize(frame)
shape_vector = embed_from_dataframe(frame, grid_size=32)
```

`embed_from_dataframe()` returns the 68-value shape vector. `embed_from_excel()`
returns a pandas `DataFrame` with one row per compressor and columns `id`,
`pc1`, `pc2`, ... . The helper does not write files by itself.

## Limitations

The normalization uses RPM because no impeller diameter is supplied. It is a
baseline representation for retrieval or downstream modeling, not a validated
hydraulic simulator. Add a domain-specific normalization or embedding method
in a separate module if stricter physical comparability is required.
