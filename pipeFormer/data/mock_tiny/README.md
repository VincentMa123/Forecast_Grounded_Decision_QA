# Mock Tiny PipeFormer Data

This folder contains synthetic data for PipeFormer smoke tests only. It is not
real pipeline telemetry and should not be used for scientific results.

Generated contents:

- `dataset/train` and `dataset/test`: tiny case folders with `Boundary.csv` and
  equipment CSVs.
- `static/mock_tiny`: active variable mapping, prediction mask, graph cache, and
  topology attention files.
- `static/full`: full 6,712-variable mapping used by `DataProcessor.combine_all_data`.
- `process_eq_argu`: placeholder static feature files.

Regenerate from the project root:

```powershell
python scripts/create_mock_pipeformer_data.py --force
```

Smoke-test commands:

```powershell
python build_cache.py --data-dir data/mock_tiny --static-dir data/mock_tiny/static/mock_tiny --skip-tokens --force
python data/compute_tokenizer_stats.py --data_dir data/mock_tiny --static-dir data/mock_tiny/static/mock_tiny --force
python build_cache.py --data-dir data/mock_tiny --static-dir data/mock_tiny/static/mock_tiny --force
python data/compute_normalization_stats.py --static_dir data/mock_tiny/static/mock_tiny --method standard --force
python train.py --config configs/mock_tiny_decoder.json
```
