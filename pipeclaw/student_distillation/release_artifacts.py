from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


class JsonlArtifactError(ValueError):
    """A JSONL transport failure with stable path and line diagnostics."""

    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = Path(path)
        self.line_number = line_number
        self.message = message
        super().__init__(f"{self.path}:{self.line_number}: {self.message}")


def stable_json(value: Any) -> str:
    """Serialize deterministic UTF-8 JSON without escaping Unicode text."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path, *, skip_blank_lines: bool = False) -> list[dict[str, Any]]:
    """Read JSONL records with path/line diagnostics and UTF-8 BOM support."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                if skip_blank_lines:
                    continue
                raise JsonlArtifactError(path, line_number, "blank JSONL row")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise JsonlArtifactError(
                    path,
                    line_number,
                    f"invalid JSON: {exc.msg}",
                ) from exc
            if not isinstance(record, Mapping):
                raise JsonlArtifactError(
                    path, line_number, "JSONL row must be an object"
                )
            records.append(dict(record))
    return records


def read_jsonl_domain(
    path: Path,
    *,
    error_factory: Callable[[str], BaseException] = ValueError,
    skip_blank_lines: bool = False,
) -> list[dict[str, Any]]:
    """Read JSONL, mapping transport diagnostics into a caller's error type."""

    try:
        return read_jsonl(path, skip_blank_lines=skip_blank_lines)
    except JsonlArtifactError as exc:
        raise error_factory(str(exc)) from exc


def required_text(
    record: Mapping[str, Any],
    field: str,
    location: str,
    *,
    error_factory: Callable[[str], BaseException] = ValueError,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise error_factory(f"{location}: {field} must be nonempty text")
    return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@contextmanager
def _atomic_text_writer(path: Path) -> Iterator[Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            yield handle
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace UTF-8 text, leaving no temporary artifact on failure."""

    with _atomic_text_writer(path) as handle:
        handle.write(content)


@contextmanager
def atomic_jsonl_writer(
    path: Path,
    *,
    default: Any | None = None,
) -> Iterator[Callable[[Mapping[str, Any]], None]]:
    """Yield a streaming JSONL writer and commit it atomically on success."""

    with _atomic_text_writer(path) as handle:

        def write_record(record: Mapping[str, Any]) -> None:
            serialized = (
                stable_json(record)
                if default is None
                else json.dumps(record, ensure_ascii=False, default=default)
            )
            handle.write(serialized + "\n")

        yield write_record


def utc_now() -> str:
    """Return a second-precision UTC timestamp in the release-file format."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
