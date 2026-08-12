"""
Workspace Tools - minimal toolset for the agent.
"""
from __future__ import annotations

import logging
import ntpath
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional

from .registry import create_json_schema_from_params, register_tool
from pipeclaw.backend.executor.runner import get_runner
from pipeclaw.backend.executor.workspace_models import (
    EditFileResult,
    ReadFileResult,
    RunCommandResult,
    WriteFileResult,
)

logger = logging.getLogger(__name__)
_REGISTERED = False


def _public_model_dump(result: Any) -> Dict[str, Any]:
    """Return logical workspace evidence without host/audit metadata."""

    payload = dict(result.model_dump())
    for key in ("abs_path", "workspace", "session_id", "timestamp", "run_dir", "output_dir"):
        payload.pop(key, None)
    for key in ("path", "cwd"):
        value = payload.get(key)
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            absolute = bool(
                Path(normalized).is_absolute()
                or PureWindowsPath(value).is_absolute()
                or ntpath.splitdrive(value)[0]
                or normalized.startswith("//")
            )
            payload[key] = "<host-path>" if absolute else normalized
    if isinstance(payload.get("cmd"), list):
        payload["cmd"] = [
            (
                "<host-path>"
                if isinstance(item, str)
                and (
                    Path(item.replace("\\", "/")).is_absolute()
                    or PureWindowsPath(item).is_absolute()
                    or ntpath.splitdrive(item)[0]
                )
                else item.replace("\\", "/") if isinstance(item, str) else item
            )
            for item in payload["cmd"]
        ]
    if isinstance(payload.get("error"), str):
        payload["error"] = re.sub(
            r"(?i)(?:[A-Z]:[\\/]|\\\\)[^\s\"'<>]+|(?<![\w.])/(?:root|home|Users|var|tmp)/[^\s\"'<>]+",
            "<host-path>",
            payload["error"],
        )
    return payload


class WorkspaceTools:
    def __init__(self, session_id: Optional[str] = None):
        self.runner = get_runner()
        self._register_tools()

    def _register_tools(self) -> None:
        global _REGISTERED
        if _REGISTERED:
            return
        _REGISTERED = True
        runner = self.runner

        @register_tool(
            name="write_file",
            description="Write or overwrite exactly one file. Each tool call must provide one complete JSON arguments object with only path and content.",
            parameters=create_json_schema_from_params(
                properties={
                    "path": {"type": "string", "description": "Path in the active configured workspace (absolute or workspace-relative; POSIX separators are preferred)."},
                    "content": {"type": "string", "description": "File content (UTF-8 text)"},
                },
                required=["path", "content"],
            ),
        )
        def write_file(path: str, content: str, session_id: str, agent_id: str = "default") -> Dict[str, Any]:
            result: WriteFileResult = runner.write_file(session_id=session_id, agent_id=agent_id, path=path, content=content)
            return _public_model_dump(result)

        @register_tool(
            name="edit_file",
            description="Edit exactly one file by exact string replacement. Each tool call must provide one complete JSON arguments object; do not reuse arguments from another tool call.",
            parameters=create_json_schema_from_params(
                properties={
                    "path": {"type": "string", "description": "Path in the active configured workspace (absolute or workspace-relative)."},
                    "old_string": {"type": "string", "description": "Exact string to replace"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                    "replace_all": {"type": "boolean", "description": "Replace all matches when true"},
                },
                required=["path", "old_string", "new_string"],
            ),
        )
        def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False, session_id: str = "", agent_id: str = "default") -> Dict[str, Any]:
            result: EditFileResult = runner.edit_file(session_id=session_id, agent_id=agent_id, path=path, old_string=old_string, new_string=new_string, replace_all=replace_all)
            return _public_model_dump(result)

        @register_tool(
            name="run_command",
            description="Run exactly one command. cmd must be a JSON array of strings, not a shell string. Script paths may use the active configured workspace path or a workspace-relative path; the runner chooses the host shell/interpreter.",
            parameters=create_json_schema_from_params(
                properties={
                    "cmd": {"type": "array", "items": {"type": "string"}, "description": "Command list; script paths must resolve inside the active configured workspace."},
                    "timeout_s": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                    "cwd": {"type": "string", "description": "Optional absolute or workspace-relative working directory inside the active workspace; omit it to use the active workspace."},
                },
                required=["cmd"],
            ),
        )
        def run_command(cmd: List[str], timeout_s: int = 30, cwd: Optional[str] = None, session_id: str = "", agent_id: str = "default") -> Dict[str, Any]:
            result: RunCommandResult = runner.run_command(session_id=session_id, agent_id=agent_id, cmd=cmd, timeout_s=timeout_s, cwd=cwd)
            return _public_model_dump(result)

        @register_tool(
            name="read_file",
            description="Read one workspace file, approved skill file, or read-only pipeline_data/... file. Provide one complete JSON arguments object with path and optional offset or limit only.",
            parameters=create_json_schema_from_params(
                properties={
                    "path": {"type": "string", "description": "Path in the active configured workspace, logical pipeline_data/... path, or an approved runtime skill path."},
                    "offset": {"type": "integer", "description": "1-based line number to start reading from."},
                    "limit": {"type": "integer", "description": "How many lines to return."},
                },
                required=["path"],
            ),
        )
        def read_file(path: str, offset: Optional[int] = None, limit: Optional[int] = None, session_id: str = "", agent_id: str = "default") -> Dict[str, Any]:
            result: ReadFileResult = runner.read_file(session_id=session_id, agent_id=agent_id, path=path, offset=offset, limit=limit)
            return _public_model_dump(result)
