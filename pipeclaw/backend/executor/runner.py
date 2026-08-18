"""
Workspace Runner - agent-root workspace execution.
"""
from __future__ import annotations

import json
import logging
import ntpath
import os
import subprocess
import time
from datetime import datetime
import shlex
import locale
from dataclasses import dataclass
from pathlib import Path
from pathlib import PureWindowsPath
from typing import List, Optional, Tuple

from .workspace_models import (
    EditFileResult,
    FileMeta,
    ReadFileResult,
    RunCommandResult,
    WorkspaceSnapshot,
    WriteFileResult,
)

logger = logging.getLogger(__name__)

MAX_READ_CHARS = 200_000
MAX_OUTPUT_CHARS = 50_000
DEFAULT_READ_LINE_LIMIT = 400
CSV_DEFAULT_READ_LINE_LIMIT = 400
PIPELINE_DATA_FILE_SUFFIXES = {
    "node_flow": "_node.csv",
    "pipeline_flow": "_pipeline.csv",
    "consumer_flow": "_consumer.csv",
}
PIPELINE_DATA_ENV_TEMPLATE = {
    "node_flow": "NODE_FLOW_DIR",
    "pipeline_flow": "PIPELINE_FLOW_DIR",
    "consumer_flow": "CONSUMER_FLOW_DIR",
}
ALLOWED_EXTERNAL_TOOL_PATH = Path("D:/ml_pro_master/chroes/fluid_model").resolve()
ALLOWED_COMMANDS = {
    "python", "python3", "py", "pip", "pip3", "powershell", "powershell.exe", "pwsh",
    "cmd", "cmd.exe", "bash", "sh", "node", "npm", "npx", "mkdir", "rmdir", "dir",
    "ls", "cp", "copy", "xcopy", "robocopy", "mv", "rm", "del", "echo",
}
SHELL_WRAPPED_COMMANDS = {"dir", "ls", "cp", "mv", "rm", "del", "mkdir", "rmdir", "copy", "xcopy", "robocopy", "echo"}
NON_ERROR_EXIT_CODES = {"robocopy": set(range(0, 8))}


@dataclass(frozen=True)
class WorkspaceCwdResolution:
    """Cross-platform resolution of a model-supplied working directory."""

    valid: bool
    path: Optional[Path]
    logical_path: Optional[str]
    error_code: Optional[str] = None
    error: Optional[str] = None
    rebased: bool = False


def _is_absolute_path_any_platform(value: str) -> bool:
    """Recognize POSIX, Windows drive-letter, and UNC paths on every host."""

    text = str(value).strip()
    normalized = text.replace("\\", "/")
    return bool(
        Path(normalized).is_absolute()
        or PureWindowsPath(text).is_absolute()
        or ntpath.splitdrive(text)[0]
        or normalized.startswith("//")
    )


def resolve_workspace_cwd(
    workspace_dir: Path | str,
    cwd: Optional[str],
) -> WorkspaceCwdResolution:
    """Resolve a relative model cwd without ever nesting a foreign absolute path."""

    root = Path(workspace_dir).resolve()
    if cwd is None or not str(cwd).strip() or str(cwd).strip() == ".":
        return WorkspaceCwdResolution(True, root, ".")
    raw = str(cwd).strip()
    normalized = raw.replace("\\", "/")
    if _is_absolute_path_any_platform(raw):
        # A path absolute to this host can be accepted only when it is already
        # inside the active workspace. Foreign drive/UNC paths are rejected.
        candidate = Path(normalized)
        if candidate.is_absolute():
            target = candidate.resolve()
            try:
                relative = target.relative_to(root)
            except (ValueError, RuntimeError):
                relative = None
            if relative is not None:
                logical = relative.as_posix() or "."
                return WorkspaceCwdResolution(True, target, logical, rebased=True)
        return WorkspaceCwdResolution(
            False,
            None,
            None,
            error_code="invalid_cwd",
            error="cwd must be omitted or a workspace-relative subdirectory",
        )
    parts = tuple(part for part in Path(normalized).parts if part not in ("", "."))
    if ".." in parts:
        return WorkspaceCwdResolution(
            False,
            None,
            None,
            error_code="invalid_cwd",
            error="cwd cannot traverse outside the active workspace",
        )
    target = (root / normalized).resolve()
    try:
        relative = target.relative_to(root)
    except (ValueError, RuntimeError):
        return WorkspaceCwdResolution(
            False,
            None,
            None,
            error_code="invalid_cwd",
            error="cwd must remain inside the active workspace",
        )
    return WorkspaceCwdResolution(True, target, relative.as_posix() or ".")


def _decode_output(data: Optional[bytes]) -> str:
    if not data:
        return ""
    preferred = locale.getpreferredencoding(False) or "utf-8"
    candidates = ["utf-8"]
    if preferred and preferred.lower() not in {enc.lower() for enc in candidates}:
        candidates.append(preferred)
    candidates.append("latin-1")
    for encoding in candidates:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class WorkspaceRunner:
    """Manage a long-lived workspace per agent."""

    def __init__(self, workspace_root: Optional[str] = None):
        backend_dir = Path(__file__).resolve().parents[1]
        root = Path(workspace_root) if workspace_root else (backend_dir / ".openclaw")
        if not root.is_absolute():
            root = (backend_dir / root).resolve()
        self.backend_dir = backend_dir
        self.pipeline_data_root = (self.backend_dir / "pipeline_data").resolve()
        self.workspace_root = root
        self.allowed_external_tool_path = ALLOWED_EXTERNAL_TOOL_PATH
        self.allowed_skill_read_path = (self.backend_dir / "agent" / "skills").resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.active_agent_id = "default"

    def set_workspace_root(self, workspace_root: str | Path) -> Path:
        root = Path(workspace_root)
        if not root.is_absolute():
            root = (self.backend_dir / root).resolve()
        self.workspace_root = root
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        return self.workspace_root

    def set_active_agent(self, agent_id: str) -> Path:
        self.active_agent_id = agent_id or "default"
        workspace_dir, _ = self._ensure_agent_dirs(self.active_agent_id)
        return workspace_dir

    def _ensure_agent_dirs(self, agent_id: Optional[str] = None) -> Tuple[Path, Path]:
        agent = agent_id or self.active_agent_id or "default"
        workspace_dir = self.workspace_root / f"workspace-{agent}"
        assets_dir = workspace_dir / "assets" / "command_runs"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        assets_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir, assets_dir

    def get_session_workspace(self, session_id: str, agent_id: Optional[str] = None) -> Path:
        workspace_dir, _ = self._ensure_agent_dirs(agent_id)
        return workspace_dir

    def get_workspace_snapshot(self, session_id: str, agent_id: Optional[str] = None) -> WorkspaceSnapshot:
        workspace_dir = self.get_session_workspace(session_id, agent_id=agent_id)
        return self._snapshot_workspace(workspace_dir)

    def _safe_resolve(self, base_dir: Path, path: str) -> Optional[Path]:
        try:
            target = (base_dir / path).resolve()
            target.relative_to(base_dir.resolve())
        except (ValueError, RuntimeError):
            return None
        return target

    def _is_within_root(self, target: Path, root: Path) -> bool:
        try:
            target.resolve().relative_to(root.resolve())
        except (ValueError, RuntimeError):
            return False
        return True

    def _is_allowed_absolute_tool_path(self, target: Path) -> bool:
        return self._is_within_root(target, self.allowed_external_tool_path)

    def _is_allowed_skill_read_path(self, target: Path) -> bool:
        return self._is_within_root(target, self.allowed_skill_read_path)

    def _resolve_tool_path(self, workspace_dir: Path, path: str) -> Optional[Path]:
        if self._is_pipeline_data_logical_path(path):
            return None
        raw_path = Path(str(path).replace("\\", "/"))
        if _is_absolute_path_any_platform(str(path)):
            if not raw_path.is_absolute():
                return None
            target = raw_path.resolve()
            if self._is_within_root(target, workspace_dir) or self._is_allowed_absolute_tool_path(target):
                return target
            return None
        return self._safe_resolve(workspace_dir, raw_path.as_posix())

    def _resolve_read_path(self, workspace_dir: Path, path: str) -> Optional[Path]:
        if self._is_pipeline_data_logical_path(path):
            relative = path.replace("\\", "/")[len("pipeline_data/") :]
            if not self._valid_pipeline_data_relative(relative):
                return None
            return self._safe_resolve(self.pipeline_data_root, relative)
        raw_path = Path(str(path).replace("\\", "/"))
        if _is_absolute_path_any_platform(str(path)):
            if not raw_path.is_absolute():
                return None
            target = raw_path.resolve()
            if (
                self._is_within_root(target, workspace_dir)
                or self._is_allowed_absolute_tool_path(target)
                or self._is_allowed_skill_read_path(target)
            ):
                return target
            return None
        return self._safe_resolve(workspace_dir, raw_path.as_posix())

    def _env_template_path(self, target: Path) -> Optional[str]:
        try:
            rel = target.resolve().relative_to(self.pipeline_data_root)
        except ValueError:
            return None
        parts = rel.parts
        if len(parts) >= 2 and parts[0] in PIPELINE_DATA_ENV_TEMPLATE:
            return f"{PIPELINE_DATA_ENV_TEMPLATE[parts[0]]}/{'/'.join(parts[1:])}"
        return None

    @staticmethod
    def _is_pipeline_data_logical_path(path: str) -> bool:
        normalized = str(path).replace("\\", "/")
        return normalized == "pipeline_data" or normalized.startswith("pipeline_data/")

    @staticmethod
    def _valid_pipeline_data_relative(relative: str) -> bool:
        parts = Path(relative).parts
        if len(parts) != 2:
            return False
        suffix = PIPELINE_DATA_FILE_SUFFIXES.get(parts[0])
        if not suffix or not parts[1].endswith(suffix):
            return False
        date = parts[1][: -len(suffix)]
        return len(date) == 8 and date.isdigit()

    def _file_meta(self, file_path: Path, base_dir: Path) -> FileMeta:
        rel_path = file_path.relative_to(base_dir).as_posix()
        return FileMeta(name=rel_path)

    def _list_files(self, root: Path, base_dir: Path) -> List[FileMeta]:
        files = [p for p in root.rglob("*") if p.is_file()]
        files.sort(key=lambda p: p.as_posix())
        return [self._file_meta(path, base_dir) for path in files]

    def _wrap_shell_command(self, cmd: List[str]) -> List[str]:
        if os.name == "nt":
            command_str = subprocess.list2cmdline(cmd)
            return ["powershell", "-NoProfile", "-Command", command_str]
        shell_cmd = " ".join(shlex.quote(part) for part in cmd)
        return ["bash", "-lc", shell_cmd]

    def _prepare_command(self, cmd: List[str]) -> List[str]:
        command = str(cmd[0]).lower()
        if os.name == "nt" and command in SHELL_WRAPPED_COMMANDS:
            return self._wrap_shell_command(cmd)
        return cmd

    def _python_syntax_precheck(
        self, cmd: List[str], workspace_dir: Path, cwd: Path
    ) -> Optional[RunCommandResult]:
        """py_compile pre-check for `python <script.py>` calls.

        Returns a clean first-class SyntaxError result when the target script does not
        compile, so the model gets a precise signal without burning a real subprocess run.
        Returns None when the check does not apply (non-python command, `-c`, `-m`, or a
        script path that does not resolve inside the workspace) or when compilation passes.
        """
        command = str(cmd[0]).lower()
        if command not in {"python", "python3", "py"} or len(cmd) < 2:
            return None
        script_arg = str(cmd[1])
        if script_arg.startswith("-"):
            return None  # -c / -m / flags: no workspace script to pre-check
        if not script_arg.lower().endswith(".py"):
            return None
        raw = Path(script_arg.replace("\\", "/"))
        if raw.is_absolute():
            target = raw.resolve()
            if not self._is_within_root(target, workspace_dir):
                return None
        else:
            # Runner default cwd is the workspace; honor an explicit cwd for resolution.
            target = self._safe_resolve(cwd, raw.as_posix())
            if target is None or not self._is_within_root(target, workspace_dir):
                return None
        if not target.exists() or not target.is_file():
            return None  # let the real run surface "can't open file" naturally
        try:
            compile(target.read_text(encoding="utf-8"), str(target), "exec")
        except SyntaxError as exc:
            line = (exc.text or "").rstrip("\n")
            location = f"{target.name}:{exc.lineno}"
            detail = f"SyntaxError at {location}, offset {exc.offset}: {exc.msg}"
            if line:
                caret = " " * max((exc.offset or 1) - 1, 0) + "^"
                detail = f"{detail}\n{line}\n{caret}"
            return RunCommandResult(
                success=False,
                session_id="",
                cmd=list(cmd),
                cwd=None,
                exit_code=1,
                duration_s=0.0,
                stdout="",
                stderr=detail,
                error=f"Python file did not compile (pre-execution check). {detail}",
                error_code="python_syntax_error",
                workspace=None,
            )
        return None

    def _success_exit_codes(self, command: str) -> set:
        codes = NON_ERROR_EXIT_CODES.get(command)
        return codes if codes is not None else {0}

    def _snapshot_workspace(self, workspace_dir: Path) -> WorkspaceSnapshot:
        return WorkspaceSnapshot(root=str(workspace_dir), files=self._list_files(workspace_dir, workspace_dir))

    def write_file(self, session_id: str, path: str, content: str, agent_id: Optional[str] = None) -> WriteFileResult:
        workspace_dir, _ = self._ensure_agent_dirs(agent_id)
        normalized_path = Path(path).as_posix()
        target = self._resolve_tool_path(workspace_dir, path)
        if not target:
            return WriteFileResult(success=False, session_id=session_id, path=normalized_path, error=f"Path is not under WORKSPACE_DIR or allowed external root: {path}", workspace=None)
        try:
            overwritten = target.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return WriteFileResult(success=True, session_id=session_id, path=normalized_path, abs_path=str(target), bytes_written=len(content.encode("utf-8")), overwritten=overwritten, workspace=None)
        except Exception as exc:
            return WriteFileResult(success=False, session_id=session_id, path=normalized_path, error=str(exc), workspace=None)

    def edit_file(self, session_id: str, path: str, old_string: str, new_string: str, replace_all: bool = False, agent_id: Optional[str] = None) -> EditFileResult:
        workspace_dir, _ = self._ensure_agent_dirs(agent_id)
        normalized_path = Path(path).as_posix()
        if not old_string:
            return EditFileResult(success=False, session_id=session_id, path=normalized_path, old_string=old_string, new_string=new_string, replace_all=replace_all, error="old_string must not be empty", workspace=None)
        target = self._resolve_tool_path(workspace_dir, path)
        if not target:
            return EditFileResult(success=False, session_id=session_id, path=normalized_path, old_string=old_string, new_string=new_string, replace_all=replace_all, error=f"Path is not under WORKSPACE_DIR or allowed external root: {path}", workspace=None)
        if not target.exists():
            return EditFileResult(success=False, session_id=session_id, path=normalized_path, old_string=old_string, new_string=new_string, replace_all=replace_all, error=f"File not found: {path}", workspace=None)
        try:
            content = target.read_text(encoding="utf-8")
            count = content.count(old_string)
            if count == 0:
                return EditFileResult(success=False, session_id=session_id, path=normalized_path, old_string=old_string, new_string=new_string, replace_all=replace_all, error="old_string not found", workspace=None)
            if count > 1 and not replace_all:
                return EditFileResult(success=False, session_id=session_id, path=normalized_path, old_string=old_string, new_string=new_string, replace_all=replace_all, error="old_string matched multiple times; use replace_all", workspace=None)
            new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
            target.write_text(new_content, encoding="utf-8")
            return EditFileResult(success=True, session_id=session_id, path=normalized_path, abs_path=str(target), old_string=old_string, new_string=new_string, replace_all=replace_all, replaced_count=count if replace_all else 1, workspace=None)
        except Exception as exc:
            return EditFileResult(success=False, session_id=session_id, path=normalized_path, old_string=old_string, new_string=new_string, replace_all=replace_all, error=str(exc), workspace=None)

    def read_file(self, session_id: str, path: str, offset: Optional[int] = None, limit: Optional[int] = None, agent_id: Optional[str] = None) -> ReadFileResult:
        workspace_dir, _ = self._ensure_agent_dirs(agent_id)
        normalized_path = Path(path).as_posix()
        invalid_path_error = (
            "Allowed read_file paths: relative to WORKSPACE_DIR, absolute under "
            "D:/ml_pro_master/chroes/fluid_model, absolute under backend/agent/skills, "
            "or read-only under pipeline_data/. "
            f"Received: {path}"
        )
        target = self._resolve_read_path(workspace_dir, path)
        if not target:
            return ReadFileResult(success=False, session_id=session_id, path=normalized_path, error=invalid_path_error, workspace=None)
        if not target.exists() or not target.is_file():
            return ReadFileResult(success=False, session_id=session_id, path=normalized_path, error=f"File not found: {path}", workspace=None)
        try:
            start_line = offset if offset and offset > 0 else 1
            user_limit = limit if limit is not None and limit > 0 else None
            suffix = target.suffix.lower()
            effective_limit = user_limit if user_limit is not None else (CSV_DEFAULT_READ_LINE_LIMIT if suffix == ".csv" and limit is None else DEFAULT_READ_LINE_LIMIT if limit is None else None)
            used_default_limit = limit is None
            lines: List[str] = []
            line_truncated = False
            with target.open("r", encoding="utf-8", errors="replace") as fh:
                current_line = 0
                while True:
                    line = fh.readline()
                    if not line:
                        break
                    current_line += 1
                    if current_line < start_line:
                        continue
                    lines.append(line)
                    if effective_limit is not None and len(lines) >= effective_limit:
                        if fh.readline():
                            line_truncated = True
                        break
            content = "".join(lines)
            char_truncated = False
            if len(content) > MAX_READ_CHARS:
                content = content[:MAX_READ_CHARS]
                char_truncated = True
            truncated = line_truncated or char_truncated
            parsed_json = None
            if suffix == ".json" and not truncated:
                try:
                    parsed_json = json.loads(content)
                except json.JSONDecodeError:
                    parsed_json = None
            lines_returned = len(lines)
            end_line = start_line + lines_returned - 1 if lines_returned else start_line - 1
            warnings: List[str] = []
            if used_default_limit:
                warnings.append(f"Default chunk returned lines {start_line}-{end_line}. Set offset/limit to read other sections.")
            return ReadFileResult(success=True, session_id=session_id, path=normalized_path, abs_path=str(target), script_path=self._env_template_path(target), size_bytes=target.stat().st_size, content=content, truncated=truncated, parsed_json=parsed_json, start_line=start_line if lines_returned else None, end_line=end_line if lines_returned else None, warnings=warnings, workspace=None)
        except Exception as exc:
            return ReadFileResult(success=False, session_id=session_id, path=normalized_path, error=str(exc), workspace=None)

    def run_command(self, session_id: str, cmd: List[str], timeout_s: int = 30, cwd: Optional[str] = None, agent_id: Optional[str] = None) -> RunCommandResult:
        workspace_dir, runs_dir = self._ensure_agent_dirs(agent_id)
        if not cmd:
            return RunCommandResult(success=False, session_id=session_id, cmd=[], error="cmd must not be empty", workspace=None)
        command = str(cmd[0]).lower()
        if command not in ALLOWED_COMMANDS:
            return RunCommandResult(success=False, session_id=session_id, cmd=cmd, error=f"Command not allowed: {cmd[0]}. Allowed: {sorted(ALLOWED_COMMANDS)}", workspace=None)
        cwd_resolution = resolve_workspace_cwd(workspace_dir, cwd)
        if not cwd_resolution.valid or cwd_resolution.path is None:
            return RunCommandResult(
                success=False,
                session_id=session_id,
                cmd=cmd,
                cwd=None,
                error_code=cwd_resolution.error_code or "invalid_cwd",
                error=cwd_resolution.error or "Invalid working directory",
                workspace=None,
            )
        precheck = self._python_syntax_precheck(cmd, workspace_dir, cwd_resolution.path)
        if precheck is not None:
            precheck.session_id = session_id
            precheck.cwd = cwd_resolution.logical_path
            return precheck
        effective_cmd = self._prepare_command(cmd)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / f"run_{run_id}"
        counter = 1
        while run_dir.exists():
            run_dir = runs_dir / f"run_{run_id}_{counter:02d}"
            counter += 1
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir = run_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        data_root = self.backend_dir / "pipeline_data"
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "BACKEND_DIR": str(self.backend_dir), "WORKSPACE_DIR": str(workspace_dir), "NODE_FLOW_DIR": str(data_root / "node_flow"), "PIPELINE_FLOW_DIR": str(data_root / "pipeline_flow"), "CONSUMER_FLOW_DIR": str(data_root / "consumer_flow"), "OUTPUT_DIR": str(output_dir), "RUN_DIR": str(run_dir)})
        start = time.perf_counter()
        stdout_data: bytes = b""
        stderr_data: bytes = b""
        exit_code = None
        error = None
        effective_cwd = str(cwd_resolution.path)
        try:
            result = subprocess.run(effective_cmd, cwd=effective_cwd, env=env, capture_output=True, text=False, timeout=timeout_s)
            exit_code = result.returncode
            stdout_data = result.stdout or b""
            stderr_data = result.stderr or b""
        except subprocess.TimeoutExpired as exc:
            stdout_data = exc.stdout or b""
            stderr_data = exc.stderr or b""
            error = f"Command timed out after {timeout_s}s"
        except Exception as exc:
            error = str(exc)
        stdout = _decode_output(stdout_data)
        stderr = _decode_output(stderr_data)
        success_exit_codes = self._success_exit_codes(command)
        if error is None and exit_code not in success_exit_codes:
            error = f"Command failed with exit code {exit_code}"
        duration = time.perf_counter() - start
        truncated_stdout = stdout[:MAX_OUTPUT_CHARS]
        truncated_stderr = stderr[:MAX_OUTPUT_CHARS]
        (run_dir / "stdout.txt").write_text(truncated_stdout, encoding="utf-8")
        (run_dir / "stderr.txt").write_text(truncated_stderr, encoding="utf-8")
        (run_dir / "meta.json").write_text(json.dumps({"cmd": cmd, "effective_cmd": effective_cmd, "exit_code": exit_code, "duration_s": round(duration, 3), "timestamp": datetime.now().isoformat(timespec="seconds"), "timeout_s": timeout_s, "error": error}, ensure_ascii=False, indent=2), encoding="utf-8")
        output_files = self._list_files(output_dir, workspace_dir)
        success = error is None and exit_code in success_exit_codes
        return RunCommandResult(
            success=success,
            session_id=session_id,
            cmd=cmd,
            cwd=cwd_resolution.logical_path,
            exit_code=exit_code,
            duration_s=round(duration, 3),
            stdout=truncated_stdout,
            stderr=truncated_stderr,
            run_dir=run_dir.relative_to(workspace_dir).as_posix(),
            output_dir=output_dir.relative_to(workspace_dir).as_posix(),
            output_files=output_files,
            error=error,
            workspace=None,
        )


_runner: Optional[WorkspaceRunner] = None


def get_runner() -> WorkspaceRunner:
    global _runner
    if _runner is None:
        _runner = WorkspaceRunner()
    return _runner


