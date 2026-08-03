# Mock lifecycle PipeFormer fixture

This dataset covers the union of explicit control-variable vocabularies in the
v4 and v7 lifecycle datasets and adds equipment-specific derived forecast
states for the engineering constraint library. All signals are synthetic and
the checkpoint is not physically validated.

Regenerate and train from `pipeFormer/`:

```powershell
python scripts/create_mock_pipeformer_data.py --force
python build_cache.py --data-dir data/mock_lifecycle --static-dir data/mock_lifecycle/static/mock_lifecycle --skip-tokens --force
python data/compute_tokenizer_stats.py --data_dir data/mock_lifecycle --static-dir data/mock_lifecycle/static/mock_lifecycle --force
python build_cache.py --data-dir data/mock_lifecycle --static-dir data/mock_lifecycle/static/mock_lifecycle --force
python data/compute_normalization_stats.py --static_dir data/mock_lifecycle/static/mock_lifecycle --method standard --force
python scripts/train_mock_causal.py --config configs/mock_decoder.json
```
