from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def write_json(path: Path, payload: Any, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
