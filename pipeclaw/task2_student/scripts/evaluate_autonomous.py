"""Run prompt-only autonomous rollouts and score them against teacher oracles.

Examples::

    python -m pipeclaw.task2_student.scripts.evaluate_autonomous \
        --source pipeclaw/task2_student/data/trace_level/test.jsonl \
        --adapters pipeclaw/task2_student/outputs/qwen35-9b \
        --output-dir pipeclaw/task2_student/outputs/evaluation/autonomous

Use ``--dry-run`` to inspect the exact PromptBuilder messages and tool schemas
without loading a model or executing tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - MS-SWIFT normally installs tqdm
    def tqdm(iterable=None, *args, **kwargs):
        del args, kwargs
        return iterable if iterable is not None else []

if __package__ in {None, ""}:  # support both ``-m`` and direct script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pipeclaw.task2_student.scripts.autonomous_rollout import (
        Generator,
        PromptCase,
        PromptCaseBuilder,
        ToolCall,
        ToolDispatcher,
        _jsonable,
        append_tool_exchange,
        parse_tool_calls,
    )
    from pipeclaw.task2_student.scripts.oracle_metrics import aggregate_results, evaluate_rollout
else:
    from .autonomous_rollout import (
        Generator,
        PromptCase,
        PromptCaseBuilder,
        ToolCall,
        ToolDispatcher,
        _jsonable,
        append_tool_exchange,
        parse_tool_calls,
    )
    from .oracle_metrics import aggregate_results, evaluate_rollout


def run_case(
    case: PromptCase,
    generator: Generator,
    dispatcher: ToolDispatcher,
    *,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    capture_raw_response: bool = False,
) -> dict[str, Any]:
    """Run one bounded model/tool conversation and preserve all partial state.

    ``capture_raw_response`` is intentionally opt-in because SDK response objects
    can be large.  When enabled, each generator response is serialized before
    parsing so diagnostics can distinguish model output from normalized rollout
    messages.
    """

    reset_history = getattr(dispatcher, "reset_history", None)
    if callable(reset_history):
        reset_history()
    set_current_user_request = getattr(dispatcher, "set_current_user_request", None)
    if callable(set_current_user_request):
        user_messages = [
            message.get("content")
            for message in case.messages
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ]
        set_current_user_request(user_messages[-1] if user_messages else "")
    set_case_workspace = getattr(dispatcher, "set_case_workspace", None)
    if callable(set_case_workspace) and case.workspace_root is not None:
        set_case_workspace(case.workspace_root)
    messages = [dict(message) for message in case.messages]
    result: dict[str, Any] = {
        "sample_id": case.sample_id,
        "scenario_id": case.scenario_id,
        "scenario_type": case.scenario_type,
        "tool_calls": [],
        "tool_outputs": [],
        "final_answer": "",
        "trace_status": "",
        "json_errors": [],
        "messages": messages,
        "turns": 0,
    }
    if capture_raw_response:
        result["raw_responses"] = []

    for turn in range(max(0, max_turns)):
        result["turns"] = turn + 1
        try:
            response = generator.generate(
                messages,
                case.tools,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # keep one model/runtime failure from dropping the test suite
            result["generation_error"] = str(exc)
            result["trace_status"] = "generation_error"
            break
        if capture_raw_response:
            result["raw_responses"].append(_jsonable(response))
        text, calls, errors = parse_tool_calls(response)
        result["json_errors"].extend(errors)

        if calls:
            for index, call in enumerate(calls):
                tool_result = dispatcher.dispatch(call)
                schema_valid = tool_result.get("error_code") not in {"unknown_tool", "invalid_arguments"}
                execution_success = (
                    tool_result.get("success", True) is not False
                    and not tool_result.get("error")
                    and tool_result.get("error_code") not in {
                        "tool_execution_error",
                        "forecast_registry_precondition_failed",
                    }
                    and tool_result.get("exit_code") in (None, 0)
                )
                result["tool_calls"].append(
                    {
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "arguments": dict(call.arguments),
                        "schema_valid": schema_valid,
                        "execution_success": execution_success,
                    }
                )
                result["tool_outputs"].append(
                    {
                        "tool_call_id": call.call_id,
                        "name": call.name,
                        "output": tool_result,
                    }
                )
                append_tool_exchange(
                    messages,
                    call,
                    tool_result,
                    assistant_content=text if index == 0 else "",
                )
            result["messages"] = messages
            continue

        if text:
            messages.append({"role": "assistant", "content": text})
            result["messages"] = messages
            result["final_answer"] = text
            result["trace_status"] = "completed"
            break

        # A malformed tagged call is a completed-but-invalid response.  Recording
        # it lets JSON-validity metrics report the failure without losing the case.
        if errors:
            result["trace_status"] = "completed"
            break
        result["trace_status"] = "empty_response"
        break
    else:
        result["trace_status"] = "max_turns_exceeded"

    if not result["trace_status"]:
        result["trace_status"] = "max_turns_exceeded"
    return result


class SwiftGenerator:
    """Small adapter around MS-SWIFT's TransformersEngine."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    @classmethod
    def from_args(cls, *, model: str, adapters: str, device: str | None = None) -> "SwiftGenerator":
        if device:
            os.environ["CUDA_VISIBLE_DEVICES"] = device
        try:
            from peft import PeftModel
            from swift import get_model_processor, get_template
            from swift.infer_engine import TransformersEngine
        except ImportError as exc:  # pragma: no cover - depends on the training environment
            raise RuntimeError("MS-SWIFT and PEFT are required for non-dry-run evaluation") from exc

        model_obj, processor = get_model_processor(model)
        model_obj = PeftModel.from_pretrained(model_obj, adapters)
        template = get_template(processor, enable_thinking=False)
        engine = TransformersEngine(model_obj, template=template)
        return cls(engine)

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        from swift.infer_engine import InferRequest, RequestConfig

        request = InferRequest(messages=[dict(message) for message in messages], tools=list(tools) or None)
        config = RequestConfig(max_tokens=max_tokens, temperature=temperature, stream=False)
        responses = self.engine.infer([request], request_config=config)
        if not responses:
            return ""
        return responses[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(dict(record))
    return records


def _progress_cases(
    cases: Sequence[tuple[dict[str, Any], PromptCase]],
    *,
    description: str,
):
    """Wrap case iterations with a terminal progress bar."""

    return tqdm(cases, total=len(cases), desc=description, unit="case")


def _sample_keys(record: Mapping[str, Any]) -> set[str]:
    keys = set()
    for key in ("sample_id", "example_id", "source_sample_id", "source_id"):
        value = record.get(key)
        if value is not None:
            keys.add(str(value))
    return keys


def _tool_schema_index(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        schemas = record.get("tools")
        if isinstance(schemas, str):
            try:
                schemas = json.loads(schemas)
            except json.JSONDecodeError:
                schemas = []
        if isinstance(schemas, Mapping):
            schemas = [schemas]
        if not isinstance(schemas, Sequence) or isinstance(schemas, (str, bytes)):
            schemas = []
        normalised = [dict(schema) for schema in schemas if isinstance(schema, Mapping)]
        for key in _sample_keys(record):
            index[key] = normalised
    return index


def _generic_schemas(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fallback schemas for dry-runs when a projection file is unavailable."""

    names: set[str] = set()
    for item in source.get("tool_calls", []) if isinstance(source.get("tool_calls"), Sequence) else []:
        if isinstance(item, Mapping) and item.get("name"):
            names.add(str(item["name"]))
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Tool schema unavailable in the source projection.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
            },
        }
        for name in sorted(names)
    ]


def _workspace_for(output_dir: Path, sample_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8]
    base = output_dir / "workspaces" / f"{safe}-{digest}"
    # Match production AgentWorkspaceManager: PromptBuilder and the runner both
    # operate inside <workspace_root_base>/workspace-<agent>.
    return base / "workspace-autonomous-evaluation"


def _is_openclaw_scenario(scenario_type: Any) -> bool:
    """Accept the dataset's ``openclaw`` label and the user-facing alias."""

    return str(scenario_type or "").casefold() in {"openclaw", "pipeclaw"}


def _evaluation_workspace_key(source: Mapping[str, Any]) -> str:
    """Return the workspace scope used by an autonomous evaluation case.

    OpenClaw scenarios span several simulated sessions and intentionally carry
    memory files between them.  Strip only the turn/session suffix so distinct
    source datasets and scenario IDs still receive isolated workspaces.
    """

    sample_id = str(source.get("sample_id") or source.get("example_id") or "sample")
    if not _is_openclaw_scenario(source.get("scenario_type")):
        return sample_id
    scenario_scope = sample_id.split("::turn_", 1)[0]
    scenario_scope = re.sub(r"_session_[^:]+$", "", scenario_scope)
    if scenario_scope == sample_id:
        scenario_scope = str(source.get("scenario_id") or scenario_scope)
    return f"openclaw-{scenario_scope}"


_OPENCLAW_PYTHON_COMMANDS = {"python", "python3", "py"}
_OPENCLAW_WORKSPACE_TOOLS = {"read_file", "write_file", "edit_file", "run_command"}
# OpenClaw traces use the same deterministic topology/registry tools as the
# PipeFormer traces. Forecast execution remains excluded: it is expensive and
# would turn a general agent rollout into a nested model evaluation.
_OPENCLAW_READONLY_PIPEFORMER_TOOLS = {
    "analyze_pipeline_topology",
    "search_pipeformer_registry",
}
_OPENCLAW_ALLOWED_TOOLS = _OPENCLAW_WORKSPACE_TOOLS | _OPENCLAW_READONLY_PIPEFORMER_TOOLS


def _path_within(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
    except (ValueError, RuntimeError):
        return False
    return True


def _safe_openclaw_path(
    value: Any,
    workspace_root: Path,
    *,
    allow_pipeline_data: bool,
) -> bool:
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
    raw_path = Path(value)
    target = raw_path.resolve() if raw_path.is_absolute() else (workspace_root / raw_path).resolve()
    return _path_within(workspace_root, target)


def _sandbox_error(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "record_in_teacher_trace": False,
        "error_code": "sandbox_violation",
        "error": message,
    }


def _openclaw_policy_error(call: ToolCall, workspace_root: Path) -> Mapping[str, Any] | None:
    arguments = dict(call.arguments)
    if call.name == "analyze_pipeline_topology":
        node_file = arguments.get("node_file")
        pipeline_file = arguments.get("pipeline_file")
        if not isinstance(node_file, str) or not re.fullmatch(r"\d{8}_node\.csv", node_file, re.IGNORECASE):
            return _sandbox_error("analyze_pipeline_topology node_file must be a daily YYYYMMDD_node.csv filename")
        if not isinstance(pipeline_file, str) or not re.fullmatch(
            r"\d{8}_pipeline\.csv", pipeline_file, re.IGNORECASE
        ):
            return _sandbox_error(
                "analyze_pipeline_topology pipeline_file must be a daily YYYYMMDD_pipeline.csv filename"
            )
        scope = arguments.get("pipeline_scope")
        if scope is not None and (
            not isinstance(scope, list) or any(not isinstance(item, str) for item in scope)
        ):
            return _sandbox_error("analyze_pipeline_topology pipeline_scope must be a list of strings")
        return None
    if call.name == "search_pipeformer_registry":
        limit = arguments.get("limit", 12)
        offset = arguments.get("offset", 0)
        if not isinstance(limit, int) or not 1 <= limit <= 50:
            return _sandbox_error("search_pipeformer_registry limit must be an integer from 1 through 50")
        if not isinstance(offset, int) or offset < 0:
            return _sandbox_error("search_pipeformer_registry offset must be a non-negative integer")
        return None
    if call.name == "read_file":
        if not _safe_openclaw_path(arguments.get("path"), workspace_root, allow_pipeline_data=True):
            return _sandbox_error("read_file path is outside the evaluation workspace or allowed pipeline_data root")
        return None
    if call.name in {"write_file", "edit_file"}:
        if not _safe_openclaw_path(arguments.get("path"), workspace_root, allow_pipeline_data=False):
            return _sandbox_error(f"{call.name} path is outside the evaluation workspace")
        return None
    if call.name == "run_command":
        command = arguments.get("cmd")
        if not isinstance(command, list) or not command:
            return _sandbox_error("run_command requires a non-empty command array")
        executable = str(command[0]).casefold()
        if executable not in _OPENCLAW_PYTHON_COMMANDS:
            return _sandbox_error("only Python workspace scripts are allowed during OpenClaw evaluation")
        if len(command) < 2 or str(command[1]).startswith("-"):
            return _sandbox_error("run_command must execute a workspace-relative Python script")
        if not _safe_openclaw_path(command[1], workspace_root, allow_pipeline_data=False):
            return _sandbox_error("run_command script is outside the evaluation workspace")
        if any(str(item) in {"-c", "-m"} for item in command[1:]):
            return _sandbox_error("inline and module Python execution are disabled during evaluation")
        timeout = arguments.get("timeout_s", 30)
        if not isinstance(timeout, int) or not 1 <= timeout <= 60:
            return _sandbox_error("run_command timeout_s must be an integer from 1 through 60")
        cwd = arguments.get("cwd")
        if cwd is not None and not _safe_openclaw_path(cwd, workspace_root, allow_pipeline_data=False):
            return _sandbox_error("run_command cwd is outside the evaluation workspace")
    return None


def _normalize_openclaw_execution_arguments(
    call: ToolCall,
    workspace_root: Path,
) -> Mapping[str, Any]:
    arguments = dict(call.arguments)
    if call.name == "run_command":
        cwd = arguments.get("cwd")
        if cwd is None:
            arguments["cwd"] = str(workspace_root)
        else:
            raw_path = Path(str(cwd))
            arguments["cwd"] = str(
                raw_path.resolve() if raw_path.is_absolute() else (workspace_root / raw_path).resolve()
            )
    return arguments


def _build_pipeformer_dispatcher(schemas: Sequence[Mapping[str, Any]], repo_root: Path) -> ToolDispatcher:
    backend_root = repo_root / "pipeclaw" / "backend"
    for import_root in (repo_root, backend_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from pipeclaw.backend.agent.tools.registry import tool_registry
    from pipeclaw.backend.agent.tools.pipeformer_tools import register_pipeformer_tools

    register_pipeformer_tools(repo_root / "pipeclaw" / "backend")
    workspace_ready = True
    workspace_runner = None
    if any(
        isinstance(schema, Mapping)
        and isinstance(schema.get("function"), Mapping)
        and schema["function"].get("name") == "read_file"
        for schema in schemas
    ):
        try:
            from pipeclaw.backend.agent.tools.workspace_tools import WorkspaceTools

            workspace_runner = WorkspaceTools(session_id="autonomous-evaluation").runner
        except Exception:
            # Read-only forecast evaluation remains usable if the workspace runner
            # is unavailable; the unavailable tool is removed from the allowlist.
            workspace_ready = False
    # These are read-only/forecast operations.  Workspace mutation and shell
    # execution are intentionally excluded from autonomous evaluation.
    allowed = {
        "analyze_pipeline_topology",
        "search_pipeformer_registry",
        "set_decision_policy",
        "run_pipeformer_forecast",
        "read_file",
    }
    if not workspace_ready:
        allowed.discard("read_file")
    schema_names = {
        str(schema.get("function", {}).get("name"))
        for schema in schemas
        if isinstance(schema, Mapping) and isinstance(schema.get("function"), Mapping)
    }
    dispatcher_ref: dict[str, ToolDispatcher] = {}

    def setup_workspace(workspace_root: Path) -> None:
        if workspace_runner is None:
            return
        agent_name = workspace_root.name
        if agent_name.startswith("workspace-"):
            agent_name = agent_name[len("workspace-") :]
        workspace_runner.set_workspace_root(workspace_root.parent)
        workspace_runner.set_active_agent(agent_name or "autonomous-evaluation")

    def authorize(call: ToolCall, completed: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        dispatcher = dispatcher_ref.get("dispatcher")
        current_request = dispatcher.current_user_request if dispatcher else ""
        if call.name == "run_pipeformer_forecast":
            from pipeline.forecast_registry_contract import forecast_registry_failure_result

            return forecast_registry_failure_result(dict(call.arguments), completed)
        if call.name == "set_decision_policy":
            objectives = call.arguments.get("objectives") or []
            invalid = [
                str(item.get("metric") or "missing")
                for item in objectives
                if not isinstance(item, Mapping)
                or not str(item.get("source_excerpt") or "").strip()
                or str(item.get("source_excerpt") or "") not in current_request
            ]
            if invalid:
                return {
                    "success": False,
                    "record_in_teacher_trace": False,
                    "error_code": "decision_policy_source_not_in_current_user_request",
                    "error": "Each decision objective must quote an exact phrase from the current user request.",
                    "invalid_objectives": invalid,
                }
        return None

    dispatcher = ToolDispatcher(
        tool_registry,
        schemas=schemas,
        allowed_names=allowed & schema_names,
        authorization_callback=authorize,
        execution_context={
            "session_id": "autonomous-evaluation",
            "agent_id": "autonomous-evaluation",
        },
        workspace_setup=setup_workspace if workspace_runner is not None else None,
    )
    dispatcher_ref["dispatcher"] = dispatcher
    return dispatcher


def _build_openclaw_dispatcher(schemas: Sequence[Mapping[str, Any]], repo_root: Path) -> ToolDispatcher:
    backend_root = repo_root / "pipeclaw" / "backend"
    for import_root in (repo_root, backend_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    from pipeclaw.backend.agent.tools.registry import tool_registry
    from pipeclaw.backend.agent.tools.pipeformer_tools import register_pipeformer_tools
    from pipeclaw.backend.agent.tools.workspace_tools import WorkspaceTools

    register_pipeformer_tools(backend_root)
    workspace_runner = WorkspaceTools(session_id="autonomous-evaluation").runner
    workspace_state: dict[str, Path | None] = {"root": None}

    def setup_workspace(workspace_root: Path) -> None:
        root = Path(workspace_root).resolve()
        workspace_state["root"] = root
        workspace_runner.set_workspace_root(root.parent)
        agent_name = root.name
        if agent_name.startswith("workspace-"):
            agent_name = agent_name[len("workspace-") :]
        workspace_runner.set_active_agent(agent_name or "autonomous-evaluation")

    def authorize(call: ToolCall, completed: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        del completed
        root = workspace_state.get("root")
        if root is None:
            return _sandbox_error("OpenClaw workspace has not been initialized")
        return _openclaw_policy_error(call, root)

    def transform(call: ToolCall) -> Mapping[str, Any]:
        root = workspace_state.get("root")
        if root is None:
            raise RuntimeError("OpenClaw workspace has not been initialized")
        return _normalize_openclaw_execution_arguments(call, root)

    schema_names = {
        str(schema.get("function", {}).get("name"))
        for schema in schemas
        if isinstance(schema, Mapping) and isinstance(schema.get("function"), Mapping)
    }
    dispatcher = ToolDispatcher(
        tool_registry,
        schemas=schemas,
        allowed_names=_OPENCLAW_ALLOWED_TOOLS & schema_names,
        authorization_callback=authorize,
        execution_arguments_callback=transform,
        execution_context={
            "session_id": "autonomous-evaluation",
            "agent_id": "autonomous-evaluation",
        },
        workspace_setup=setup_workspace,
    )
    return dispatcher


def evaluate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_records = _read_jsonl(Path(args.source))
    schema_records = _read_jsonl(Path(args.tool_schema_source)) if args.tool_schema_source else source_records
    schemas_by_key = _tool_schema_index(schema_records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = PromptCaseBuilder()
    cases: list[tuple[dict[str, Any], PromptCase]] = []
    for source in source_records:
        requested_scenario = str(args.scenario_type or "").casefold()
        source_scenario = str(source.get("scenario_type") or "").casefold()
        if requested_scenario:
            if requested_scenario in {"openclaw", "pipeclaw"}:
                requested_scenario = "openclaw"
            if source_scenario in {"openclaw", "pipeclaw"}:
                source_scenario = "openclaw"
            if source_scenario != requested_scenario:
                continue
        keys = _sample_keys(source)
        schemas = next((schemas_by_key[key] for key in keys if key in schemas_by_key), _generic_schemas(source))
        case = builder.build(
            source,
            workspace_root=_workspace_for(output_dir, _evaluation_workspace_key(source)),
            tool_schemas=schemas,
        )
        cases.append((source, case))
        if args.limit is not None and len(cases) >= args.limit:
            break

    rollouts_path = output_dir / "rollouts.jsonl"
    results: list[dict[str, Any]] = []
    with rollouts_path.open("w", encoding="utf-8") as handle:
        if args.dry_run:
            progress = _progress_cases(cases, description="Preparing evaluation")
            for source, case in progress:
                item = {
                    "sample_id": case.sample_id,
                    "scenario_id": case.scenario_id,
                    "scenario_type": case.scenario_type,
                    "prompt_messages": case.messages,
                    "tools": case.tools,
                    "teacher_future_hidden": True,
                    "status": "dry_run",
                }
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
                results.append(item)
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(scenario=case.scenario_type, status="dry_run", refresh=False)
            summary = {"mode": "dry_run", "record_count": len(results)}
        else:
            if not args.adapters:
                raise ValueError("--adapters is required unless --dry-run is used")
            model = args.model or _discover_base_model(Path(args.adapters))
            generator = SwiftGenerator.from_args(model=model, adapters=args.adapters, device=args.device)
            schemas_by_scenario: dict[str, list[Mapping[str, Any]]] = {}
            for _, case in cases:
                scenario_key = "openclaw" if _is_openclaw_scenario(case.scenario_type) else "pipeformer"
                scenario_schemas = schemas_by_scenario.setdefault(scenario_key, [])
                existing_names = {
                    str(item.get("function", {}).get("name"))
                    for item in scenario_schemas
                    if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)
                }
                for schema in case.tools:
                    function = schema.get("function") if isinstance(schema, Mapping) else None
                    name = function.get("name") if isinstance(function, Mapping) else None
                    if name and str(name) not in existing_names:
                        scenario_schemas.append(schema)
                        existing_names.add(str(name))
            dispatchers: dict[str, ToolDispatcher] = {}
            repo_root = Path(args.repo_root)
            if "pipeformer" in schemas_by_scenario:
                dispatchers["pipeformer"] = _build_pipeformer_dispatcher(
                    schemas_by_scenario["pipeformer"], repo_root
                )
            if "openclaw" in schemas_by_scenario:
                dispatchers["openclaw"] = _build_openclaw_dispatcher(
                    schemas_by_scenario["openclaw"], repo_root
                )
            progress = _progress_cases(cases, description="Evaluating")
            for source, case in progress:
                scenario_key = "openclaw" if _is_openclaw_scenario(case.scenario_type) else "pipeformer"
                dispatcher = dispatchers[scenario_key]
                rollout = run_case(
                    case,
                    generator,
                    dispatcher,
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    capture_raw_response=args.save_raw_responses,
                )
                rollout["metrics"] = evaluate_rollout(source, rollout)
                handle.write(json.dumps(rollout, ensure_ascii=False, default=str) + "\n")
                results.append(rollout)
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(
                        scenario=case.scenario_type,
                        status=rollout.get("trace_status", "unknown"),
                        refresh=False,
                    )
            summary = aggregate_results(results)
            summary["mode"] = "autonomous"

    # Keep the combined score convenient while also exposing separate
    # denominators for PipeFormer and OpenClaw cases.  This prevents a metric
    # that is inapplicable to one scenario family from hiding the other family's
    # performance.
    summary["by_scenario_type"] = {}
    for scenario_type in sorted({str(case.scenario_type or "unknown") for _, case in cases}):
        scenario_results = [
            result
            for result, (_, case) in zip(results, cases)
            if str(case.scenario_type or "unknown") == scenario_type
        ]
        if summary.get("mode") == "autonomous":
            scenario_summary = aggregate_results(scenario_results)
        else:
            scenario_summary = {"record_count": len(scenario_results)}
        scenario_summary["mode"] = summary.get("mode", "dry_run")
        summary["by_scenario_type"][scenario_type] = scenario_summary

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    return summary


def _discover_base_model(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        model = config.get("base_model_name_or_path")
        if model:
            return str(model)
    raise ValueError("--model is required when adapter_config.json has no base_model_name_or_path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Teacher-source JSONL used for prompts and oracle metrics")
    parser.add_argument("--tool-schema-source", help="Projection JSONL containing OpenAI tool schemas")
    parser.add_argument("--adapters", help="LoRA adapter directory")
    parser.add_argument("--model", help="Base model name/path; inferred from adapter_config.json when omitted")
    parser.add_argument("--output-dir", required=True, help="Directory for rollouts.jsonl and summary.json")
    parser.add_argument("--repo-root", default=".", help="Repository root used to import PipeClaw tools")
    parser.add_argument(
        "--scenario-type",
        help="Evaluate one scenario type: pipeformer, openclaw (or the pipeclaw alias); omit for both",
    )
    parser.add_argument("--limit", type=int, help="Limit the number of cases")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--save-raw-responses",
        action="store_true",
        help="Save serialized generator responses before parsing them",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build prompts and schemas without loading a model")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
