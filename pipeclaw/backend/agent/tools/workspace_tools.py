"""
Workspace Tools - minimal toolset for the agent.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .registry import create_json_schema_from_params, register_tool
from executor.runner import get_runner
from executor.workspace_models import EditFileResult, ReadFileResult, RunCommandResult, WriteFileResult

logger = logging.getLogger(__name__)
_REGISTERED = False


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
                    "path": {"type": "string", "description": "Path relative to WORKSPACE_ROOT, or an absolute path under D:/ml_pro_master/chroes/fluid_model. Example: 'plan.md'"},
                    "content": {"type": "string", "description": "File content (UTF-8 text)"},
                },
                required=["path", "content"],
            ),
        )
        def write_file(path: str, content: str, session_id: str, agent_id: str = "default") -> Dict[str, Any]:
            result: WriteFileResult = runner.write_file(session_id=session_id, agent_id=agent_id, path=path, content=content)
            return result.model_dump()

        @register_tool(
            name="edit_file",
            description="Edit exactly one file by exact string replacement. Each tool call must provide one complete JSON arguments object; do not reuse arguments from another tool call.",
            parameters=create_json_schema_from_params(
                properties={
                    "path": {"type": "string", "description": "Path relative to WORKSPACE_ROOT, or an absolute path under D:/ml_pro_master/chroes/fluid_model."},
                    "old_string": {"type": "string", "description": "Exact string to replace"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                    "replace_all": {"type": "boolean", "description": "Replace all matches when true"},
                },
                required=["path", "old_string", "new_string"],
            ),
        )
        def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False, session_id: str = "", agent_id: str = "default") -> Dict[str, Any]:
            result: EditFileResult = runner.edit_file(session_id=session_id, agent_id=agent_id, path=path, old_string=old_string, new_string=new_string, replace_all=replace_all)
            return result.model_dump()

        @register_tool(
            name="run_command",
            description="Run exactly one command. cmd must be a JSON array of strings, not a shell string. On Windows prefer cmd or powershell; do not default to bash. Each tool call must provide one complete JSON arguments object only for this command.",
            parameters=create_json_schema_from_params(
                properties={
                    "cmd": {"type": "array", "items": {"type": "string"}, "description": "Command list"},
                    "timeout_s": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                    "cwd": {"type": "string", "description": "Optional absolute working directory"},
                },
                required=["cmd"],
            ),
        )
        def run_command(cmd: List[str], timeout_s: int = 30, cwd: Optional[str] = None, session_id: str = "", agent_id: str = "default") -> Dict[str, Any]:
            result: RunCommandResult = runner.run_command(session_id=session_id, agent_id=agent_id, cmd=cmd, timeout_s=timeout_s, cwd=cwd)
            return result.model_dump()

        @register_tool(
            name="read_file",
            description="Read exactly one file. Each tool call must provide one complete JSON arguments object with path and optional offset or limit only; never concatenate JSON objects from multiple reads.",
            parameters=create_json_schema_from_params(
                properties={
                    "path": {"type": "string", "description": "File path relative to WORKSPACE_ROOT, an absolute path under D:/ml_pro_master/chroes/fluid_model, or an absolute path under backend/agent/skills."},
                    "offset": {"type": "integer", "description": "1-based line number to start reading from."},
                    "limit": {"type": "integer", "description": "How many lines to return."},
                },
                required=["path"],
            ),
        )
        def read_file(path: str, offset: Optional[int] = None, limit: Optional[int] = None, session_id: str = "", agent_id: str = "default") -> Dict[str, Any]:
            result: ReadFileResult = runner.read_file(session_id=session_id, agent_id=agent_id, path=path, offset=offset, limit=limit)
            return result.model_dump()