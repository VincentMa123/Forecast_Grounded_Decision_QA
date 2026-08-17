"""The bounded model/tool conversation loop.

``RolloutRunner`` performs execution only.  It has no teacher-oracle, metric, or
score dependency: scoring happens afterwards through
``pipeclaw.backend.evaluator``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from .models import (
    Generator,
    PromptCase,
    RolloutConfig,
    RolloutResult,
    ToolCall,
    jsonable,
)
from .tools import ToolDispatcher, append_tool_exchange, parse_tool_calls


class RolloutPolicy(Protocol):
    """Scenario-specific normalization applied to recorded calls and outputs."""

    def portability_metadata(
        self, call: ToolCall, case: PromptCase
    ) -> Mapping[str, Any]:
        """Return path-portability diagnostics for one call."""

    def recorded_arguments(self, call: ToolCall, case: PromptCase) -> Mapping[str, Any]:
        """Return the arguments saved in the rollout record."""

    def compact_tool_result(
        self,
        call: ToolCall,
        result: Any,
        *,
        portability: Mapping[str, Any] | None = None,
    ) -> Any:
        """Return the bounded tool result shown to the model and saved."""


class PassthroughPolicy:
    """Record calls and results verbatim; used by tests and simple harnesses."""

    def portability_metadata(
        self, call: ToolCall, case: PromptCase
    ) -> Mapping[str, Any]:
        del call, case
        return {}

    def recorded_arguments(self, call: ToolCall, case: PromptCase) -> Mapping[str, Any]:
        del case
        return dict(call.arguments)

    def compact_tool_result(
        self,
        call: ToolCall,
        result: Any,
        *,
        portability: Mapping[str, Any] | None = None,
    ) -> Any:
        del call, portability
        return result


_SCHEMA_ERROR_CODES = {"unknown_tool", "invalid_arguments", "tool_not_allowed"}
_EXECUTION_ERROR_CODES = {
    "tool_execution_error",
    "forecast_registry_precondition_failed",
}
_PORTABILITY_RECORD_KEYS = ("cwd_rebased", "portable_path_normalization")


def _schema_valid(result: Mapping[str, Any]) -> bool:
    return result.get("error_code") not in _SCHEMA_ERROR_CODES


def _execution_success(result: Mapping[str, Any]) -> bool:
    return (
        result.get("success", True) is not False
        and not result.get("error")
        and result.get("error_code") not in _EXECUTION_ERROR_CODES
        and result.get("exit_code") in (None, 0)
    )


class RolloutRunner:
    """Run one bounded model/tool conversation and preserve all partial state."""

    def __init__(
        self,
        generator: Generator,
        dispatcher: ToolDispatcher,
        *,
        policy: RolloutPolicy | None = None,
    ) -> None:
        self.generator = generator
        self.dispatcher = dispatcher
        self.policy: RolloutPolicy = policy or PassthroughPolicy()

    def run(self, case: PromptCase, config: RolloutConfig) -> RolloutResult:
        """Execute ``case`` under ``config`` and return its full trajectory."""

        self._prepare_dispatcher(case)
        messages = [dict(message) for message in case.messages]
        result = RolloutResult(
            sample_id=case.sample_id,
            scenario_id=case.scenario_id,
            scenario_type=case.scenario_type,
            messages=messages,
        )
        if config.capture_raw_responses:
            result.raw_responses = []
        if config.capture_raw_tool_outputs:
            result.raw_tool_outputs = []

        for turn in range(max(0, config.max_turns)):
            result.turns = turn + 1
            try:
                response = self.generator.generate(
                    messages,
                    case.tools,
                    max_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                )
            except Exception as exc:
                # Keep one model/runtime failure from dropping the whole suite.
                result.generation_error = str(exc)
                result.trace_status = "generation_error"
                break
            if result.raw_responses is not None:
                result.raw_responses.append(jsonable(response))
            text, calls, errors = parse_tool_calls(response)
            result.json_errors.extend(errors)

            if calls:
                self._dispatch_calls(case, result, messages, calls, text)
                continue

            if text:
                messages.append({"role": "assistant", "content": text})
                result.final_answer = text
                result.trace_status = "completed"
                break

            # A malformed tagged call is a completed-but-invalid response.
            # Recording it lets JSON-validity metrics report the failure without
            # losing the case.
            if errors:
                result.trace_status = "completed"
                break
            result.trace_status = "empty_response"
            break
        else:
            result.trace_status = "max_turns_exceeded"

        return result

    def _prepare_dispatcher(self, case: PromptCase) -> None:
        reset_history = getattr(self.dispatcher, "reset_history", None)
        if callable(reset_history):
            reset_history()
        set_current_user_request = getattr(
            self.dispatcher, "set_current_user_request", None
        )
        if callable(set_current_user_request):
            user_messages = [
                message.get("content")
                for message in case.messages
                if message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ]
            set_current_user_request(user_messages[-1] if user_messages else "")
        set_case_workspace = getattr(self.dispatcher, "set_case_workspace", None)
        if callable(set_case_workspace) and case.workspace_root is not None:
            set_case_workspace(Path(case.workspace_root))

    def _dispatch_calls(
        self,
        case: PromptCase,
        result: RolloutResult,
        messages: list[dict[str, Any]],
        calls: list[ToolCall],
        text: str,
    ) -> None:
        for index, call in enumerate(calls):
            # Record the schema-normalized form: identical logical calls stay one
            # thrash signature even when the model stringifies typed values.
            normalized = self.dispatcher.schema_normalized_call(call)
            portability = dict(self.policy.portability_metadata(normalized, case))
            tool_result = self.dispatcher.dispatch(normalized)
            call_record: dict[str, Any] = {
                "tool_call_id": call.call_id,
                "name": call.name,
                "arguments": dict(self.policy.recorded_arguments(normalized, case)),
                "schema_valid": _schema_valid(tool_result),
                "execution_success": _execution_success(tool_result),
            }
            call_record.update(
                {
                    key: portability[key]
                    for key in _PORTABILITY_RECORD_KEYS
                    if portability.get(key)
                }
            )
            result.tool_calls.append(call_record)
            compact_tool_result = self.policy.compact_tool_result(
                call, tool_result, portability=portability
            )
            result.tool_outputs.append(
                {
                    "tool_call_id": call.call_id,
                    "name": call.name,
                    "output": compact_tool_result,
                }
            )
            if result.raw_tool_outputs is not None:
                raw_entry: dict[str, Any] = {
                    "tool_call_id": call.call_id,
                    "name": call.name,
                    "output": jsonable(tool_result),
                }
                if portability:
                    raw_entry["diagnostics"] = dict(portability)
                result.raw_tool_outputs.append(raw_entry)
            append_tool_exchange(
                messages,
                call,
                compact_tool_result,
                assistant_content=text if index == 0 else "",
            )
