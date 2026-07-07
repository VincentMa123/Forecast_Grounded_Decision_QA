"""
Workspace tool result models (Pydantic).

These models are returned by the workspace tools:
- write_file
- edit_file
- run_command
- read_file
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


def now_iso() -> str:
    """Current time as ISO string (seconds precision)."""
    return datetime.now().isoformat(timespec="seconds")


class FileMeta(BaseModel):
    """Minimal metadata for a file in the workspace."""

    name: str = Field(
        ..., description="Path relative to WORKSPACE_DIR (POSIX style)"
    )


class WorkspaceSnapshot(BaseModel):
    """A snapshot of files currently in the workspace."""

    root: str = Field(..., description="Absolute WORKSPACE_DIR on server")
    files: List[FileMeta] = Field(
        default_factory=list,
        description="Relative file paths under workspace (recursive)",
    )


class ToolResultBase(BaseModel):
    """Common fields for all tool results."""

    success: bool = Field(..., description="Whether the tool succeeded")
    tool: str = Field(..., description="Tool name")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(
        default_factory=now_iso,
        description="When the tool was executed (ISO)",
    )
    error: Optional[str] = Field(
        None, description="Error message if success=False"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Non-fatal warnings"
    )
    workspace: Optional[WorkspaceSnapshot] = Field(
        None,
        description=(
            "Workspace snapshot after tool execution "
            "(may be None if too early or failed)"
        ),
    )


class WriteFileResult(ToolResultBase):
    tool: Literal["write_file"] = "write_file"
    path: str = Field(
        ..., description="Path relative to WORKSPACE_DIR that was written"
    )
    abs_path: Optional[str] = Field(
        None, description="Absolute path on server"
    )
    bytes_written: Optional[int] = Field(
        None, ge=0, description="Bytes written (UTF-8)"
    )
    overwritten: Optional[bool] = Field(
        None, description="Whether the file existed before write"
    )


class EditFileResult(ToolResultBase):
    tool: Literal["edit_file"] = "edit_file"
    path: str = Field(
        ..., description="Path relative to WORKSPACE_DIR that was edited"
    )
    abs_path: Optional[str] = Field(
        None, description="Absolute path on server"
    )
    old_string: Optional[str] = Field(
        None, description="The old string that was matched"
    )
    new_string: Optional[str] = Field(
        None, description="The new string that replaced old_string"
    )
    replace_all: Optional[bool] = Field(
        None, description="Whether replace_all mode was used"
    )
    replaced_count: Optional[int] = Field(
        None, ge=0, description="Number of replacements performed"
    )


class ReadFileResult(ToolResultBase):
    tool: Literal["read_file"] = "read_file"
    path: str = Field(
        ..., description="Path relative to WORKSPACE_DIR that was read"
    )
    abs_path: Optional[str] = Field(
        None, description="Absolute path on server"
    )
    size_bytes: Optional[int] = Field(
        None, ge=0, description="File size"
    )
    content: Optional[str] = Field(
        None, description="File content (may be truncated)"
    )
    truncated: bool = Field(
        False, description="Whether content was truncated"
    )
    parsed_json: Optional[Any] = Field(
        None, description="If JSON, parsed value (optional)"
    )
    start_line: Optional[int] = Field(
        None, ge=1, description="First line (1-based) included in content"
    )
    end_line: Optional[int] = Field(
        None, ge=1, description="Last line (1-based) included in content"
    )


class RunCommandResult(ToolResultBase):
    tool: Literal["run_command"] = "run_command"
    cmd: List[str] = Field(
        ..., description="Executed command list (e.g. ['python','task.py'])"
    )
    cwd: Optional[str] = Field(
        None, description="Working directory (WORKSPACE_DIR)"
    )
    exit_code: Optional[int] = Field(
        None, description="Process exit code"
    )
    duration_s: Optional[float] = Field(
        None, ge=0, description="Duration in seconds"
    )
    stdout: str = Field("", description="Captured stdout (truncated)")
    stderr: str = Field("", description="Captured stderr (truncated)")
    run_dir: Optional[str] = Field(
        None,
        description="Directory storing run logs (stdout.txt, stderr.txt, meta.json)",
    )
    output_dir: Optional[str] = Field(
        None, description="OUTPUT_DIR used for artifacts"
    )
    output_files: List[FileMeta] = Field(
        default_factory=list,
        description="Files under OUTPUT_DIR after execution",
    )
