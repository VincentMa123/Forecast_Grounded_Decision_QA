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


def canonicalize_recorded_tool_arguments(
    tool_name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Make recorded workspace-tool arguments portable without changing execution.

    Saved training and rollout calls retain logical relative paths, while
    host-specific values are omitted or redacted. Execution-time workspace
    rebasing remains a separate concern in the rollout scenario adapter.
    """

    canonical = dict(arguments)
    if tool_name != "run_command":
        return canonical

    cwd = canonical.get("cwd")
    if cwd is None or cwd == "<host-path>" or is_host_absolute_path(cwd):
        canonical.pop("cwd", None)
    elif isinstance(cwd, str):
        canonical["cwd"] = str(cwd).replace("\\", "/")

    command = canonical.get("cmd")
    if isinstance(command, list):
        canonical["cmd"] = [
            (
                "<host-path>"
                if isinstance(item, str) and is_host_absolute_path(item)
                else str(item).replace("\\", "/")
                if isinstance(item, str) and not item.startswith("-")
                else item
            )
            for item in command
        ]
    return canonical


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
