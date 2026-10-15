from __future__ import annotations

from pathlib import Path

from .models import (
    Generator,
    PromptCase,
    RolloutConfig,
    RolloutResult,
)
from .scenarios import (
    compact_model_tool_result,
    is_openclaw_scenario,
    openclaw_portability_metadata,
)
from .tools import ToolDispatcher, dispatch_and_record
from pipeclaw.protocols.path_contract import canonicalize_recorded_tool_arguments
from pipeclaw.protocols.tool_calls import jsonable, parse_tool_calls


class RolloutRunner:
    """Run one bounded model/tool conversation and preserve all partial state."""

    def __init__(self, generator: Generator, dispatcher: ToolDispatcher) -> None:
        self.generator = generator
        self.dispatcher = dispatcher

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
                # Portability diagnostics exist only for OpenClaw workspace tools;
                # argument canonicalization and compaction are keyed on the tool
                # name, so they are correct for both scenario families.
                dispatch_and_record(
                    calls,
                    dispatcher=self.dispatcher,
                    result=result,
                    portability_metadata=(
                        openclaw_portability_metadata
                        if is_openclaw_scenario(case.scenario_type)
                        else None
                    ),
                    record_arguments=lambda call: canonicalize_recorded_tool_arguments(
                        call.name, call.arguments
                    ),
                    compact_result=lambda call, tool_result, portability: compact_model_tool_result(
                        call.name, tool_result, portability=portability
                    ),
                    assistant_content=text,
                )
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
        self.dispatcher.reset_history()
        self.dispatcher.set_current_user_request(
            next(
                (
                    message["content"]
                    for message in reversed(case.messages)
                    if message.get("role") == "user"
                    and isinstance(message.get("content"), str)
                ),
                "",
            )
        )
        if case.workspace_root is not None:
            self.dispatcher.set_case_workspace(Path(case.workspace_root))
