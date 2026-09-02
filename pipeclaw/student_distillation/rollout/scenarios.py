from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeclaw.backend.pipeline.forecast.result import ForecastResult

from ..path_contract import (
    canonicalize_recorded_tool_arguments,
    is_host_absolute_path,
    redact_host_paths,
)
from .models import PromptCase
from .tools import ToolDispatcher
from pipeclaw.protocols.tool_calls import ToolCall


OPENCLAW_PYTHON_COMMANDS = {"python", "python3", "py"}
OPENCLAW_WORKSPACE_TOOLS = {"read_file", "write_file", "edit_file", "run_command"}
# OpenClaw traces use the same deterministic topology/registry tools as the
# PipeFormer traces. Forecast execution remains excluded: it is expensive and
# would turn a general agent rollout into a nested model evaluation.
OPENCLAW_READONLY_PIPEFORMER_TOOLS = {
    "analyze_pipeline_topology",
    "search_pipeformer_registry",
}
OPENCLAW_ALLOWED_TOOLS = OPENCLAW_WORKSPACE_TOOLS | OPENCLAW_READONLY_PIPEFORMER_TOOLS

# These are read-only/forecast operations.  Workspace mutation and shell
# execution are intentionally excluded from PipeFormer autonomous evaluation.
PIPEFORMER_ALLOWED_TOOLS = {
    "analyze_pipeline_topology",
    "search_pipeformer_registry",
    "set_decision_policy",
    "run_pipeformer_forecast",
    "read_file",
}

_TOPOLOGY_FILE_RULES = (
    (
        "node_file",
        r"\d{8}_node\.csv",
        "analyze_pipeline_topology node_file must be a daily YYYYMMDD_node.csv filename",
    ),
    (
        "pipeline_file",
        r"\d{8}_pipeline\.csv",
        "analyze_pipeline_topology pipeline_file must be a daily YYYYMMDD_pipeline.csv filename",
    ),
)


def is_openclaw_scenario(scenario_type: Any) -> bool:
    """Accept the dataset's ``openclaw`` label and the user-facing alias."""

    return str(scenario_type or "").casefold() in {"openclaw", "pipeclaw"}


def scenario_key(scenario_type: Any) -> str:
    """Return the dispatcher family for a scenario type."""

    return "openclaw" if is_openclaw_scenario(scenario_type) else "pipeformer"


def workspace_for(output_dir: Path, sample_id: str) -> Path:
    """Return the isolated workspace directory used by one evaluation scope."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8]
    base = output_dir / "workspaces" / f"{safe}-{digest}"
    # Match production AgentWorkspaceManager: PromptBuilder and the runner both
    # operate inside <workspace_root_base>/workspace-<agent>.
    return base / "workspace-autonomous-evaluation"


def evaluation_workspace_key(source: Mapping[str, Any]) -> str:
    """Return the workspace scope used by an autonomous evaluation case.

    OpenClaw scenarios span several simulated sessions and intentionally carry
    memory files between them.  Strip only the turn/session suffix so distinct
    source datasets and scenario IDs still receive isolated workspaces.
    """

    sample_id = str(source.get("sample_id") or source.get("example_id") or "sample")
    if not is_openclaw_scenario(source.get("scenario_type")):
        return sample_id
    scenario_scope = sample_id.split("::turn_", 1)[0]
    scenario_scope = re.sub(r"_session_[^:]+$", "", scenario_scope)
    if scenario_scope == sample_id:
        scenario_scope = str(source.get("scenario_id") or scenario_scope)
    return f"openclaw-{scenario_scope}"


def compact_openclaw_tool_result(
    value: Mapping[str, Any],
    *,
    portability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project workspace results into logical, bounded model-visible fields."""

    compact: dict[str, Any] = {
        key: value[key]
        for key in ("success", "error_code", "exit_code")
        if key in value
    }
    if "error" in value and value.get("error"):
        compact["error"] = redact_host_paths(value.get("error"))
    for key in ("stdout", "stderr"):
        if key in value and value.get(key) is not None:
            compact[key] = redact_host_paths(str(value[key])[:8_000])
    if (
        "path" in value
        and isinstance(value.get("path"), str)
        and not is_host_absolute_path(value["path"])
    ):
        compact["path"] = value["path"].replace("\\", "/")
    if "content" in value and value.get("content") is not None:
        compact["content"] = str(value["content"])[:12_000]
    for key in (
        "parsed_json", "truncated", "start_line", "end_line",
        "bytes_written", "overwritten", "replaced_count",
    ):
        if key in value:
            compact[key] = value[key]
    if "cmd" in value and isinstance(value.get("cmd"), list):
        compact["cmd"] = [
            item.replace("\\", "/")
            if isinstance(item, str) and not is_host_absolute_path(item)
            else "<host-path>"
            for item in value["cmd"]
        ]
    cwd = value.get("cwd")
    if isinstance(cwd, str) and not is_host_absolute_path(cwd):
        compact["cwd"] = cwd.replace("\\", "/")
    output_files = [
        {"name": name.replace("\\", "/")}
        for item in value.get("output_files") or []
        for name in [item.get("name") if isinstance(item, Mapping) else item]
        if isinstance(name, str) and not is_host_absolute_path(name)
    ]
    if output_files:
        compact["output_files"] = output_files
    if value.get("warnings"):
        compact["warnings"] = [redact_host_paths(str(item)) for item in value["warnings"]]
    if portability:
        compact.update(
            {key: True for key in ("cwd_rebased", "portable_path_normalization") if portability.get(key)}
        )
    return compact


def openclaw_portability_metadata(call: ToolCall) -> dict[str, Any]:
    """Report how one call's paths had to be normalized to stay portable."""

    if call.name != "run_command":
        return {}
    cwd = call.arguments.get("cwd")
    if cwd is None:
        return {"portable_path_normalization": True}
    if is_host_absolute_path(cwd):
        return {
            "cwd_rebased": True,
            "portable_path_normalization": True,
            "original_cwd": str(cwd),
        }
    return {
        "portable_path_normalization": str(cwd).replace("\\", "/") != str(cwd)
    }


def compact_model_tool_result(
    tool_name: str,
    value: Any,
    *,
    portability: Mapping[str, Any] | None = None,
) -> Any:
    """Return the bounded tool result that the student sees and we save by default.

    The dispatcher keeps the complete result for authorization while this
    projection matches the canonical PipeFormer training projection.  The raw
    payload is intentionally not placed in the model context or rollout JSON;
    callers can request it separately for debugging.
    """

    if not isinstance(value, Mapping):
        return value
    if tool_name in OPENCLAW_WORKSPACE_TOOLS:
        return compact_openclaw_tool_result(value, portability=portability)
    if tool_name != "run_pipeformer_forecast" or value.get("success") is False:
        return value

    if {"task_resolution", "prediction", "verification", "provenance"} & value.keys():
        forecast = ForecastResult.model_validate(value)
    else:
        forecast = ForecastResult.from_execution(value)
    return forecast.model_dump()


class ScenarioPolicy:
    """The ``RolloutPolicy`` used by both PipeFormer and OpenClaw rollouts.

    Portability metadata is only meaningful for the OpenClaw workspace tools, so
    the scenario type of the case decides whether it is collected; compaction is
    keyed on the tool name and therefore stays correct for both families.
    """

    def portability_metadata(
        self, call: ToolCall, case: PromptCase
    ) -> Mapping[str, Any]:
        if not is_openclaw_scenario(case.scenario_type):
            return {}
        return openclaw_portability_metadata(call)

    def recorded_arguments(self, call: ToolCall, case: PromptCase) -> Mapping[str, Any]:
        del case
        return canonicalize_recorded_tool_arguments(call.name, call.arguments)

    def compact_tool_result(
        self,
        call: ToolCall,
        result: Any,
        *,
        portability: Mapping[str, Any] | None = None,
    ) -> Any:
        return compact_model_tool_result(call.name, result, portability=portability)


def path_within(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
    except (ValueError, RuntimeError):
        return False
    return True


def safe_openclaw_path(
    value: Any,
    workspace_root: Path,
    *,
    allow_pipeline_data: bool,
) -> bool:
    """Return True when a model-supplied path stays inside the allowed roots."""

    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.replace("\\", "/")
    parts = Path(normalized).parts
    if ".." in parts:
        return False
    if allow_pipeline_data and (
        normalized == "pipeline_data" or normalized.startswith("pipeline_data/")
    ):
        return True
    raw_path = Path(normalized)
    if is_host_absolute_path(value) and not raw_path.is_absolute():
        # A Windows drive/UNC path must never be interpreted as a POSIX
        # relative path such as ``<workspace>/C:/...``.
        return False
    target = raw_path.resolve() if raw_path.is_absolute() else (workspace_root / raw_path).resolve()
    return path_within(workspace_root, target)


def sandbox_error(message: str) -> dict[str, Any]:
    """Return the rejection payload used for a sandbox policy violation."""

    return {
        "success": False,
        "record_in_teacher_trace": False,
        "error_code": "sandbox_violation",
        "error": message,
    }


def openclaw_policy_error(
    call: ToolCall, workspace_root: Path
) -> Mapping[str, Any] | None:
    """Return a sandbox rejection for a disallowed OpenClaw call, else None."""

    arguments = dict(call.arguments)
    if call.name == "analyze_pipeline_topology":
        for key, pattern, message in _TOPOLOGY_FILE_RULES:
            value = arguments.get(key)
            if not isinstance(value, str) or not re.fullmatch(
                pattern, value, re.IGNORECASE
            ):
                return sandbox_error(message)
        scope = arguments.get("pipeline_scope")
        if scope is not None and (
            not isinstance(scope, list)
            or any(not isinstance(item, str) for item in scope)
        ):
            return sandbox_error(
                "analyze_pipeline_topology pipeline_scope must be a list of strings"
            )
        return None
    if call.name == "search_pipeformer_registry":
        limit = arguments.get("limit", 12)
        offset = arguments.get("offset", 0)
        if not isinstance(limit, int) or not 1 <= limit <= 50:
            return sandbox_error(
                "search_pipeformer_registry limit must be an integer from 1 through 50"
            )
        if not isinstance(offset, int) or offset < 0:
            return sandbox_error(
                "search_pipeformer_registry offset must be a non-negative integer"
            )
        return None
    if call.name in {"read_file", "write_file", "edit_file"}:
        allow_pipeline_data = call.name == "read_file"
        if not safe_openclaw_path(
            arguments.get("path"), workspace_root, allow_pipeline_data=allow_pipeline_data
        ):
            return sandbox_error(
                "read_file path is outside the evaluation workspace or allowed pipeline_data root"
                if allow_pipeline_data
                else f"{call.name} path is outside the evaluation workspace"
            )
        return None
    if call.name == "run_command":
        command = arguments.get("cmd")
        if not isinstance(command, list) or not command:
            return sandbox_error("run_command requires a non-empty command array")
        if str(command[0]).casefold() not in OPENCLAW_PYTHON_COMMANDS:
            return sandbox_error(
                "only Python workspace scripts are allowed during OpenClaw evaluation"
            )
        if len(command) < 2 or str(command[1]).startswith("-"):
            return sandbox_error(
                "run_command must execute a workspace-relative Python script"
            )
        if not safe_openclaw_path(
            command[1], workspace_root, allow_pipeline_data=False
        ):
            return sandbox_error("run_command script is outside the evaluation workspace")
        if any(str(item) in {"-c", "-m"} for item in command[1:]):
            return sandbox_error(
                "inline and module Python execution are disabled during evaluation"
            )
        timeout = arguments.get("timeout_s", 30)
        if not isinstance(timeout, int) or not 1 <= timeout <= 60:
            return sandbox_error(
                "run_command timeout_s must be an integer from 1 through 60"
            )
        cwd = arguments.get("cwd")
        if cwd is not None and not is_host_absolute_path(cwd) and not safe_openclaw_path(
            cwd, workspace_root, allow_pipeline_data=False
        ):
            return sandbox_error("run_command cwd is outside the evaluation workspace")
    return None


def normalize_openclaw_execution_arguments(
    call: ToolCall,
    workspace_root: Path,
) -> Mapping[str, Any]:
    """Rebase a host or Windows ``cwd`` onto the active evaluation workspace."""

    del workspace_root
    arguments = dict(call.arguments)
    if call.name == "run_command":
        cwd = arguments.get("cwd")
        if cwd is None or is_host_absolute_path(cwd):
            arguments["cwd"] = "."
        else:
            arguments["cwd"] = str(cwd).replace("\\", "/")
    return arguments


def _build_dispatcher(
    family: str, schemas: Sequence[Mapping[str, Any]], repo_root: Path
) -> ToolDispatcher:
    """Build one family dispatcher while sharing registry/workspace plumbing."""

    openclaw = family == "openclaw"
    backend_root = Path(repo_root).resolve() / "pipeclaw" / "backend"
    from pipeclaw.backend.agent.tools.registry import tool_registry
    from pipeclaw.backend.agent.tools.pipeformer_tools import register_pipeformer_tools

    register_pipeformer_tools(backend_root)
    schema_names = {
        str(function.get("name"))
        for schema in schemas
        if isinstance(schema, Mapping)
        and isinstance(function := schema.get("function"), Mapping)
    }
    workspace_runner = None
    if openclaw or "read_file" in schema_names:
        try:
            from pipeclaw.backend.agent.tools.workspace_tools import WorkspaceTools

            workspace_runner = WorkspaceTools(session_id="autonomous-evaluation").runner
        except Exception:
            if openclaw:
                raise
    allowed = set(OPENCLAW_ALLOWED_TOOLS if openclaw else PIPEFORMER_ALLOWED_TOOLS)
    if workspace_runner is None:
        allowed.discard("read_file")
    allowed &= schema_names
    workspace_state: dict[str, Path | None] = {"root": None}
    dispatcher: ToolDispatcher | None = None

    def setup_workspace(workspace_root: Path) -> None:
        if workspace_runner is None:
            return
        root = Path(workspace_root).resolve() if openclaw else Path(workspace_root)
        if openclaw:
            workspace_state["root"] = root
        workspace_runner.set_workspace_root(root.parent)
        workspace_runner.set_active_agent(
            root.name.removeprefix("workspace-") or "autonomous-evaluation"
        )

    def authorize(
        call: ToolCall, completed: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any] | None:
        if openclaw:
            del completed
            root = workspace_state["root"]
            return (
                sandbox_error("OpenClaw workspace has not been initialized")
                if root is None
                else openclaw_policy_error(call, root)
            )
        if call.name not in ("run_pipeformer_forecast", "set_decision_policy"):
            return None
        from pipeclaw.backend.agent.orchestrator import forecast_preexecution_failure

        return forecast_preexecution_failure(
            call.name,
            dict(call.arguments),
            list(completed),
            current_user_request=dispatcher.current_user_request if dispatcher else "",
        )

    def transform(call: ToolCall) -> Mapping[str, Any]:
        root = workspace_state["root"]
        if root is None:
            raise RuntimeError("OpenClaw workspace has not been initialized")
        return normalize_openclaw_execution_arguments(call, root)

    dispatcher = ToolDispatcher(
        tool_registry,
        schemas=schemas,
        allowed_names=allowed,
        authorization_callback=authorize,
        execution_arguments_callback=transform if openclaw else None,
        execution_context={
            "session_id": "autonomous-evaluation",
            "agent_id": "autonomous-evaluation",
        },
        workspace_setup=setup_workspace if workspace_runner is not None else None,
    )
    return dispatcher


def build_openclaw_dispatcher(
    schemas: Sequence[Mapping[str, Any]], repo_root: Path
) -> ToolDispatcher:
    """Build the sandboxed workspace dispatcher used by OpenClaw rollouts."""

    return _build_dispatcher("openclaw", schemas, repo_root)


def build_dispatcher(
    scenario_type: Any,
    schemas: Sequence[Mapping[str, Any]],
    repo_root: Path,
) -> ToolDispatcher:
    """Build the dispatcher for one scenario family."""

    return _build_dispatcher(scenario_key(scenario_type), schemas, repo_root)
