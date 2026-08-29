# Mock lifecycle PipeFormer fixture

This directory contains a deterministic, synthetic fixture for exercising the
PipeFormer data, tokenizer, graph, and causal-training paths. It combines the
control vocabularies from the v4 and v7 lifecycle datasets. `manifest.json`
records the source and derived-variable counts for the generated fixture. The
signals are not physically validated and must not be used as production
evidence.

## Contents

- `dataset/{train,valid,test}/case_*/` — generated CSV time series for each case.
- `static/mock_lifecycle/` — graph, variable mapping, attention, tokenizer, and
  normalization artifacts used by the active configuration.
- `process_eq_argu/` — generated static equipment features.
- `manifest.json` — source counts and fixture metadata.
- `intervention_manifest.json` — causal intervention cases used by the mock
  training configuration.

## Regenerate and train

Run these commands from `pipeFormer/`:

```powershell
python scripts/create_mock_pipeformer_data.py --force
python build_cache.py --data-dir data/mock_lifecycle --static-dir data/mock_lifecycle/static/mock_lifecycle --skip-tokens --force
python data/compute_tokenizer_stats.py --data_dir data/mock_lifecycle --static-dir data/mock_lifecycle/static/mock_lifecycle --force
python build_cache.py --data-dir data/mock_lifecycle --static-dir data/mock_lifecycle/static/mock_lifecycle --force
python data/compute_normalization_stats.py --static_dir data/mock_lifecycle/static/mock_lifecycle --method standard --force
python scripts/train_mock_causal.py --config configs/mock_decoder.json
```

The first command recreates the fixture and its metadata. The next commands
build the cache, fit tokenizer and normalization statistics, rebuild tokenized
cache files, and launch the small causal-training configuration. Use
`--output-dir` or repeated `--dataset` options on the generator when testing a
different destination or source dataset pair.
