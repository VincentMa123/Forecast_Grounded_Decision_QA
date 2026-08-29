# Data tokenizer

`tokenizer_save/` contains the `DataTokenizer` implementation and the files it
writes after fitting a shared vocabulary across boundary and equipment
variables. It is used by `build_cache.py`, the data loader, and training code.

## Public API

```python
from data.tokenizer_save import DataTokenizer, load_tokenizer
import numpy as np

# Replace these examples with your aligned feature matrix and names.
values = np.array([[0.0, 1.0], [0.5, 1.2]], dtype=np.float32)
names = ["B_001:FR", "C_001:SP_"]

tokenizer = DataTokenizer("data/mock_lifecycle")
tokenizer.fit(values, variable_names=names)
tokenizer.save_stats()

loaded = load_tokenizer("data/mock_lifecycle")
if loaded is None:
    raise RuntimeError("tokenizer statistics are missing")
tokens = loaded.transform_to_tokens(values)
values_again = loaded.tokens_to_values(tokens)
```

`load_tokenizer()` looks for `data_tokenizer_config.json` and
`token_stats.csv` under `<data_dir>/tokenizer_save/` and returns `None` when the
statistics are unavailable.

## Fit from the command line

Run from `pipeFormer/` after the sequence/cache inputs exist:

```bash
python data/compute_tokenizer_stats.py \
  --data_dir data/mock_lifecycle \
  --static-dir data/mock_lifecycle/static/mock_lifecycle \
  --force
```

The command writes tokenizer statistics into the selected static directory's
`tokenizer_save/` folder. Run it again whenever the input data or tokenizer
parameters change.

## Main parameters

These constructor defaults can be overridden by the CLI or a saved config:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `constant_freq_threshold` | `0.02` | Frequent values that receive dedicated constant tokens. |
| `constant_variable_threshold` | `0.999` | Coverage required to treat a whole variable as constant. |
| `quantile_step` | `0.02` | Approximate coverage of each continuous range token. |
| `quantile_method` | `linear` | Method passed to `numpy.quantile`. |
| `range_gap_epsilon` | `1e-9` | Gap used to keep adjacent ranges disjoint. |
| `round_gap` | `0.3` | Rounding granularity used before token lookup. |

The fitted values are saved in `data_tokenizer_config.json`; use that file as
the source of truth for a particular dataset because generated fixtures may
override the constructor defaults.

## Files

- `core.py` — vocabulary fitting, encoding, decoding, and persistence.
- `types.py` — token configuration data structures.
- `range_utils.py` — continuous-range and binary-token construction.
- `node_utils.py` — variable/node grouping and static-name loading.
- `deduplication.py` — shared token detection for identical equipment columns.
- `metadata.py`, `array_utils.py` — metadata and array helpers.
- `token_stats.csv` — fitted token metadata (generated).
- `data_tokenizer_config.json` — fitted parameter snapshot (generated).
- `tokenizer_output.txt` — optional fitting log (generated).
