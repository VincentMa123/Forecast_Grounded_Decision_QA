from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    summary = json.loads((repo_root / "DATASET_SUMMARY.json").read_text(encoding="utf-8"))
    for config in summary["configs"]:
        raw_path = repo_root / config["raw_path"]
        jsonl_path = repo_root / config["jsonl_path"]
        raw_rows = json.loads(raw_path.read_text(encoding="utf-8"))
        jsonl_rows = load_jsonl(jsonl_path)
        if raw_rows != jsonl_rows:
            raise SystemExit(f"Mismatch between {raw_path} and {jsonl_path}")
        print(f"OK {config['config_name']}: {len(raw_rows)} rows")


if __name__ == "__main__":
    main()
