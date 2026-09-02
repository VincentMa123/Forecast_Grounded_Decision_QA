from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from .models import (
    Generator,
    PromptCase,
    RolloutConfig,
    RolloutResult,
)
from .episode import dispatch_and_record
from .tools import ToolDispatcher
from pipeclaw.protocols.tool_calls import ToolCall, jsonable, parse_tool_calls


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
            raw_responses=[] if config.capture_raw_responses else None,
            raw_tool_outputs=[] if config.capture_raw_tool_outputs else None,
        )

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
                self._dispatch_calls(case, result, calls, text)
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
            user_message = next(
                (
                    message["content"]
                    for message in reversed(case.messages)
                    if message.get("role") == "user"
                    and isinstance(message.get("content"), str)
                ),
                "",
            )
            set_current_user_request(user_message)
        set_case_workspace = getattr(self.dispatcher, "set_case_workspace", None)
        if callable(set_case_workspace) and case.workspace_root is not None:
            set_case_workspace(Path(case.workspace_root))

    def _dispatch_calls(
        self,
        case: PromptCase,
        result: RolloutResult,
        calls: list[ToolCall],
        text: str,
    ) -> None:
        dispatch_and_record(
            calls,
            dispatcher=self.dispatcher,
            result=result,
            portability_metadata=lambda call: self.policy.portability_metadata(
                call, case
            ),
            record_arguments=lambda call: self.policy.recorded_arguments(call, case),
            compact_result=lambda call, tool_result, portability: self.policy.compact_tool_result(
                call, tool_result, portability=portability
            ),
            assistant_content=text,
        )
