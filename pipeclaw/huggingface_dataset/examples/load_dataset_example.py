from __future__ import annotations

from datasets import load_dataset
from huggingface_hub import snapshot_download


REPO_ID = "zly7/pipeclaw-open-datasets"
CONFIG_NAMES = ['llm-question-all', 'llm-question-template-80', 'llm-question-template-all', 'pipeclaw-v2', 'pipeformer-v4', 'pipeformer-v7']


def main() -> None:
    for config_name in CONFIG_NAMES:
        dataset = load_dataset(REPO_ID, config_name, split="train")
        print(f"{config_name}: rows={dataset.num_rows} columns={dataset.column_names}")

    local_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
    print(f"Downloaded full dataset repo to: {local_dir}")


if __name__ == "__main__":
    main()
