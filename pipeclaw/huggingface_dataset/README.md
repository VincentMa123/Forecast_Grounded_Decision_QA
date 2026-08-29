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

# PipeClaw open datasets

This folder is a ready-to-upload Hugging Face dataset repository. It contains
JSON snapshots under `raw/` and equivalent JSONL files under `data/` so the Hub
viewer and `datasets.load_dataset()` can read them without a custom builder.

## Configurations

| Config | Records | Contents |
| --- | ---: | --- |
| `llm-question-all` | 80 | Expanded questions with concrete parameters. |
| `llm-question-template-80` | 80 | Public question templates. |
| `llm-question-template-all` | 80 | Full public template snapshot. |
| `pipeclaw-v2` | 40 | OpenClaw lifecycle scenarios. |
| `pipeformer-v4` | 70 | v4 PipeFormer/OpenClaw lifecycle scenarios. |
| `pipeformer-v7` | 40 | v7 PipeFormer lifecycle scenarios. |

Each configuration exposes a fixed `train` split. Nested fields such as
`params`, `sessions`, and `dialogue` are intentionally preserved.

## Validate and publish

Run from this directory:

```bash
python -m pip install -r requirements.txt
python scripts/validate_repo.py
python scripts/upload_to_hf.py --repo-id your-org/your-dataset
```

The validator compares every raw JSON file with its generated JSONL file. The
uploader creates or updates the dataset repository and prompts for `HF_TOKEN`
when it is not already set. See [PUBLISH_TO_HF.md](PUBLISH_TO_HF.md) for the
manual upload path.

## Load a dataset

```python
from datasets import load_dataset

dataset = load_dataset("your-org/your-dataset", "pipeclaw-v2", split="train")
print(dataset.num_rows)
print(dataset[0]["scenario_id"])
```

`DATASET_SUMMARY.json` records the source paths, row counts, fields, and sample
identifiers for each configuration. The data are released snapshots, not
randomized benchmark splits.
