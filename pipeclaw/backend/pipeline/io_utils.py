from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, TextIO


def load_records(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [
            value
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
            for value in [json.loads(line)]
            if isinstance(value, dict)
        ]
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    raise TypeError(
        f"Teacher trace must contain a JSON object, list, or JSONL records: {path}"
    )


def _atomic_write(
    path: Path,
    force: bool,
    writer: Callable[[TextIO], None],
    *,
    newline: str | None = None,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=newline,
            dir=path.parent,
            prefix=".tmp.",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer(handle)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_json(path: Path, payload: Any, force: bool) -> None:
    _atomic_write(
        Path(path),
        force,
        lambda handle: handle.write(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        ),
    )


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]], force: bool) -> None:
    def write_records(handle: TextIO) -> None:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    _atomic_write(Path(path), force, write_records, newline="\n")
