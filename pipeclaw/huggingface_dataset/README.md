---
language:
- zh
pretty_name: PipeClaw Open Datasets
size_categories:
- n<1K
configs:
- config_name: llm-question-all
  data_files:
  - split: train
    path: data/llm_data/question_all.jsonl
- config_name: llm-question-template-80
  data_files:
  - split: train
    path: data/llm_data/question_template_80.jsonl
- config_name: llm-question-template-all
  data_files:
  - split: train
    path: data/llm_data/question_template_all.jsonl
- config_name: pipeclaw-v2
  data_files:
  - split: train
    path: data/pipeclaw_data/pipeclaw_dataset_v2.jsonl
- config_name: pipeformer-v4
  data_files:
  - split: train
    path: data/pipeclaw_data/Pipeline_Full_Life_Cycle_Test_Dataset-v4.jsonl
- config_name: pipeformer-v7
  data_files:
  - split: train
    path: data/pipeclaw_data/Pipeline_Full_Life_Cycle_Test_Dataset-v7.jsonl
---

# PipeClaw Open Datasets

This directory is a push-ready Hugging Face dataset repository generated from the public `backend/llm_data` and `backend/pipeclaw_data` files in PipeClaw. It keeps the original JSON snapshots under `raw/` and adds line-delimited JSON files under `data/` so the Hub dataset viewer and `datasets.load_dataset(...)` can work without a custom builder script.

## Included Configs

- `llm-question-all`: `80` rows. Expanded LLM question instances with concrete parameters and final prompts. Source snapshot: `backend/llm_data/question_all.json`.
- `llm-question-template-80`: `80` rows. The 80 public LLM prompt templates curated for the released package. Source snapshot: `backend/llm_data/question_template_80.json`.
- `llm-question-template-all`: `80` rows. Full public LLM template set bundled with the open-source package. Source snapshot: `backend/llm_data/question_template_all.json`.
- `pipeclaw-v2`: `40` rows. Long-lifecycle PipeClaw evaluation scenarios with multi-session dialogues. Source snapshot: `backend/pipeclaw_data/pipeclaw_dataset_v2.json`.
- `pipeformer-v4`: `70` rows. Released PipeFormer/PipeClaw evaluation scenarios from the v4 dataset snapshot. Source snapshot: `backend/pipeclaw_data/Pipeline_Full_Life_Cycle_Test_Dataset-v4.json`.
- `pipeformer-v7`: `40` rows. Released PipeFormer/PipeClaw evaluation scenarios from the v7 dataset snapshot. Source snapshot: `backend/pipeclaw_data/Pipeline_Full_Life_Cycle_Test_Dataset-v7.json`.

## One-Click Publish

```bash
pip install -r requirements.txt
python scripts/upload_to_hf.py --repo-id zly7/pipeclaw-open-datasets
```

If `HF_TOKEN` is already set in the environment, the upload script uses it directly. Otherwise it prompts securely for the token and then creates or updates the dataset repo.

## Quickstart

```python
from datasets import load_dataset

repo_id = "zly7/pipeclaw-open-datasets"
ds = load_dataset(repo_id, "pipeclaw-v2", split="train")
print(ds.num_rows)
print(ds[0]["scenario_id"])
```

## Download The Entire Repository

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="zly7/pipeclaw-open-datasets", repo_type="dataset")
print(local_dir)
```

## Repository Layout

- `data/`: JSONL files wired into the dataset configs in the YAML header above.
- `raw/`: original JSON files preserved with their source filenames for direct download.
- `DATASET_SUMMARY.json`: row counts, top-level fields, and light structural metadata for every config.
- `examples/load_dataset_example.py`: copy-paste examples for `datasets` and `huggingface_hub`.
- `scripts/validate_repo.py`: local integrity check that compares each raw JSON file with its generated JSONL file.
- `scripts/upload_to_hf.py`: one-click uploader that creates or updates the dataset repo on the Hub.
- `requirements.txt`: minimal helper dependencies for upload and example usage.
- `PUBLISH_TO_HF.md`: step-by-step instructions for publishing this folder as a dataset repo.

## Notes

- Every config exposes a single `train` split because these files are released as fixed public artifacts, not randomized ML train/test benchmarks.
- Nested fields such as `params`, `sessions`, and `dialogue` are intentionally preserved instead of flattened, so downstream users can choose their own transformation strategy.
- Replace the placeholder repo id `zly7/pipeclaw-open-datasets` before publishing if you want the code snippets to be immediately runnable from the dataset card.
