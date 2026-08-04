"""Platform-neutral path predicates shared by dataset and rollout code."""

from __future__ import annotations

import ntpath
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
