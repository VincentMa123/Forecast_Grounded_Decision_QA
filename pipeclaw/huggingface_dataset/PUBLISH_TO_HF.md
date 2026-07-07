# Publish To Hugging Face

This folder is already organized as a Hugging Face dataset repository. The fastest release flow is:

1. Install helper dependencies: `pip install -r requirements.txt`
2. Run `python scripts/upload_to_hf.py --repo-id zly7/pipeclaw-open-datasets`
3. If `HF_TOKEN` is not set, paste it when the script prompts.
4. Wait for the upload to finish.
5. Verify that these configs load from the Hub: llm-question-all, llm-question-template-80, llm-question-template-all, pipeclaw-v2, pipeformer-v4, pipeformer-v7.

The upload script will:

- create the dataset repo if it does not exist
- reuse the repo if it already exists
- upload the whole folder, including `README.md`, `data/`, `raw/`, examples, and helper scripts

Recommended verification after upload:

```python
from datasets import load_dataset

repo_id = "zly7/pipeclaw-open-datasets"
ds = load_dataset(repo_id, "llm-question-all", split="train")
print(ds.num_rows, ds.column_names)
```

If you prefer manual control, you can still create the repo yourself and push this folder with `git` or the Hugging Face web UI.
