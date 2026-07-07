from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from huggingface_hub import HfApi


def resolve_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token.strip()
    return getpass.getpass("HF_TOKEN: ").strip()


def resolve_repo_id(repo_id: str | None) -> str:
    if repo_id:
        return repo_id.strip()
    return input("Dataset repo id (for example your-org/pipeclaw-open-datasets): ").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update the Hugging Face dataset repo from this local folder.")
    parser.add_argument("--repo-id", help="Target dataset repo id, for example org/name.")
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private if it does not already exist.")
    parser.add_argument("--message", default="Upload PipeClaw open datasets", help="Commit message used for the upload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    repo_id = resolve_repo_id(args.repo_id)
    token = resolve_token()
    if not repo_id:
        raise SystemExit("Dataset repo id is required.")
    if not token:
        raise SystemExit("HF_TOKEN is required.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(repo_root),
        path_in_repo="",
        commit_message=args.message,
        ignore_patterns=["__pycache__", "*.pyc", ".DS_Store"],
    )
    print(f"Uploaded dataset repo: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
