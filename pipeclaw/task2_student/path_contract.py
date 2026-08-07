"""Platform-neutral path predicates shared by dataset and rollout code."""

from __future__ import annotations

import ntpath
import re
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any


def is_host_absolute_path(value: Any) -> bool:
    """Recognize POSIX, drive-letter, and UNC paths regardless of host OS."""

    if not isinstance(value, str) or not value.strip():
        return False
    raw = value.strip()
    normalized = raw.replace("\\", "/")
    return bool(
        Path(normalized).is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or ntpath.splitdrive(raw)[0]
        or normalized.startswith("//")
    )


def normalize_relative_path(value: str) -> str:
    """Normalize a model-facing relative path to POSIX separators."""

    return str(value).replace("\\", "/")


def redact_host_paths(value: Any) -> Any:
    """Return a copy with host-specific absolute paths removed recursively."""

    if isinstance(value, Mapping):
        return {key: redact_host_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_host_paths(item) for item in value]
    if not isinstance(value, str):
        return value
    redacted = re.sub(
        r"(?i)(?<![A-Za-z0-9_])(?:"
        r"[A-Z]:[\\/]+(?:[^\\/\s\"'<>]+[\\/]+)+[^\\/\s\"'<>]+|"
        r"\\\\[A-Za-z0-9._-]{2,}[\\/]+[A-Za-z0-9$._-]{2,}"
        r"(?:[\\/]+[^\\/\s\"'<>]+)*"
        r")",
        "<host-path>",
        value,
    )
    return re.sub(
        r"(?<![\w.])/(?:root|home|Users|var|tmp)/"
        r"(?:[^\\/\s\"'<>]+/)*[^\\/\s\"'<>]+",
        "<host-path>",
        redacted,
    )
