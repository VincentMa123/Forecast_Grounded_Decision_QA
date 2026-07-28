"""
Trace-first agent orchestrator.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .schemas import AgentChatRequest, AgentChatResponse, OneTurnAfterRunAgent, TraceSummary
from .llm_provider import LLMProvider, LLMProviderSettings
from .services import MemoryAssembler, MemoryManager, VerifiedStateManager
from .agent_workspace_manager import AgentWorkspaceManager
from .prompt_builder import PromptBuilder
from .trace_writer import TraceWriter
from .tools.registry import tool_registry
from .skills.skill_manager import SkillManager
from executor.runner import get_runner
from evaluator.decision_policy import METRIC_CATALOG
from evaluator.decision_trace_state import (
    VerifiedDecisionState,
    bounded_recent_turns,
    serialize_verified_decision_state,
)
from evaluator.grounding_contract import (
    GroundingContractBuilder,
    candidate_contract_message,
    finalize_applied_disturbance_disclosure,
)
from pipeline.forecast_registry_contract import forecast_registry_failure_result

logger = logging.getLogger(__name__)

AnswerValidator = Callable[[str, List[Dict[str, Any]]], List[str]]


def finalize_runtime_answer(
    answer: str,
    question: str,
    completed_tool_calls: List[Dict[str, Any]],
    *,
    prior_state: Optional[VerifiedDecisionState] = None,
) -> str:
    """Apply the canonical disclosure from current and prior verified evidence."""
    state = prior_state or VerifiedDecisionState()
    contract = GroundingContractBuilder().build(
        question,
        completed_tool_calls,
        require_decision_policy=True,
        prior_candidate_results=state.candidates,
        prior_decision_policy=state.decision_policy,
        prior_decision_policy_source_question=(
            state.decision_policy_source_question
        ),
        prior_applied_disturbances=state.applied_disturbances,
    )
    if not contract.get("applied_disturbances") and state.applied_disturbances:
        contract["applied_disturbances"] = state.applied_disturbances
    return finalize_applied_disturbance_disclosure(answer, contract)


def _canonical_forecast_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonical_forecast_value(item)
            for key, item in sorted(value.items())
            if key not in {"question", "candidate_id", "candidate_role", "disturbance_assumption"}
        }
    if isinstance(value, list):
        items = [_canonical_forecast_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _forecast_call_signature(arguments: Dict[str, Any]) -> str:
    normalized = dict(arguments)
    if normalized.get("current_operating_condition_number") is None:
        match = re.search(r"(\d+)$", str(normalized.get("case_id") or ""))
        if match:
            normalized["current_operating_condition_number"] = int(match.group(1))
    return json.dumps(
        _canonical_forecast_value(normalized),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def equivalent_forecast_call(
    arguments: Dict[str, Any],
    completed_tool_calls: List[Dict[str, Any]],
) -> Optional[str]:
    """Return the prior call id when a successful forecast already evaluated this action."""
    signature = _forecast_call_signature(arguments)
    for item in completed_tool_calls:
        if item.get("name") != "run_pipeformer_forecast":
            continue
        output = item.get("output")
        if isinstance(output, dict) and (output.get("success") is False or output.get("error")):
            continue
        if _forecast_call_signature(dict(item.get("arguments") or {})) == signature:
            return str(item.get("tool_call_id") or "")
    return None


def forecast_preexecution_failure(
    tool_name: str,
    arguments: Dict[str, Any],
    completed_tool_calls: List[Dict[str, Any]],
    *,
    current_user_request: str = "",
) -> Optional[Dict[str, Any]]:
    """Return a structured forecast guard failure before any tool execution."""
    if tool_name == "set_decision_policy" and current_user_request.strip():
        normalized_request = " ".join(
            current_user_request.split()
        ).casefold()
        invalid_objectives = []
        for objective in arguments.get("objectives") or []:
            item = dict(objective or {})
            excerpt = " ".join(
                str(item.get("source_excerpt") or "").split()
            ).casefold()
            if len(excerpt) < 4 or excerpt not in normalized_request:
                invalid_objectives.append(
                    str(item.get("metric") or "missing")
                )
        if invalid_objectives:
            return {
                "success": False,
                "record_in_teacher_trace": False,
                "error_code": (
                    "decision_policy_source_not_in_current_user_request"
                ),
                "error": (
                    "Decision policy rejected before execution. Retry "
                    "set_decision_policy with one exact contiguous "
                    "source_excerpt from the current user request for each "
                    "objective; do not concatenate separate phrases."
                ),
                "validation_errors": [
                    f"objective_source_not_in_current_user_request:{metric}"
                    for metric in invalid_objectives
                ],
                "retry_tool": "set_decision_policy",
            }
        return None
    if tool_name != "run_pipeformer_forecast":
        return None
    invalid_decision_metrics = list(dict.fromkeys(
        str(value)
        for value in arguments.get("output_state_variables") or []
        if str(value) in METRIC_CATALOG
    ))
    if invalid_decision_metrics:
        return {
            "success": False,
            "record_in_teacher_trace": False,
            "error_code": "decision_metric_used_as_output_state_variable",
            "error": (
                "Decision-policy metrics are derived from forecast audit results "
                "and cannot be requested as PipeFormer state variables. Remove "
                "the listed metrics from output_state_variables, then use "
                "canonical registry output IDs/groups if additional raw states "
                "are needed. If only the decision policy changed, reuse prior "
                "verified candidate forecasts instead of rerunning them."
            ),
            "invalid_output_state_variables": invalid_decision_metrics,
            "retry_tool": "run_pipeformer_forecast",
        }
    registry_failure = forecast_registry_failure_result(
        arguments,
        completed_tool_calls,
    )
    if registry_failure:
        return registry_failure
    duplicate_of = equivalent_forecast_call(arguments, completed_tool_calls)
    if not duplicate_of:
        return None
    return {
        "success": False,
        "record_in_teacher_trace": False,
        "error_code": "duplicate_equivalent_forecast",
        "error": (
            "This forecast action was already evaluated. Reuse the prior result "
            f"from tool call {duplicate_of} instead of creating another candidate."
        ),
        "duplicate_of_tool_call_id": duplicate_of,
    }


def should_record_tool_result(tool_name: str, result: Any) -> bool:
    """Keep internal pre-execution planning corrections out of SFT traces."""
    if not isinstance(result, dict):
        return True
    if result.get("record_in_teacher_trace") is False:
        return False
    return True


class AgentOrchestrator:
    def __init__(
        self,
        data_loader=None,
        *,
        agent_id: str = "default",
        session_id: str = "default",
        enable_skills: Optional[bool] = None,
        request_timeout_seconds: Optional[float] = None,
        workspace_root_base: Optional[Path] = None,
    ):
        self.data_loader = data_loader
        self.agent_id = agent_id or "default"
        self.session_id = session_id
        provider_settings = LLMProviderSettings.from_env()
        self.max_steps = int(os.getenv("MAX_AGENT_STEPS", "30"))
        raw_timeout = request_timeout_seconds
        if raw_timeout not in (None, "", 0, "0"):
            provider_settings = replace(provider_settings, timeout_seconds=float(raw_timeout))
        self.llm_provider = LLMProvider(provider_settings)
        self.api_key = provider_settings.api_key
        self.api_base = provider_settings.base_url
        self.model = provider_settings.model
        self.request_timeout_seconds = provider_settings.timeout_seconds
        self.enable_skills = bool(enable_skills)
        self.backend_root = Path(__file__).resolve().parent.parent
        self.workspace_root_base = Path(workspace_root_base) if workspace_root_base is not None else (self.backend_root / ".openclaw")
        self.workspace_manager = AgentWorkspaceManager(self.backend_root, self.agent_id, workspace_root_base=self.workspace_root_base)
        self.workspace_dir = self.workspace_manager.workspace_root
        self.trace_writer = TraceWriter(self.workspace_manager.trace_root, self.agent_id, self.workspace_manager.plan_path)
        self.memory_manager = MemoryManager(self.workspace_dir)
        self.memory_assembler = MemoryAssembler(self.workspace_dir)
        self.verified_state_manager = VerifiedStateManager(self.workspace_dir)
        self.prompt_builder = PromptBuilder(self.workspace_dir)
        self.skill_manager = SkillManager(self.backend_root / "agent" / "skills") if self.enable_skills else None
        runner = get_runner()
        runner.set_workspace_root(self.workspace_root_base)
        runner.set_active_agent(self.agent_id)
        self._init_tools()

    def _init_tools(self) -> None:
        from .tools.workspace_tools import WorkspaceTools
        from .tools.pipeformer_tools import register_pipeformer_tools
        WorkspaceTools(self.session_id)
        register_pipeformer_tools(self.backend_root)

    def _recent_dialogue_turns(
        self,
        trace_messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        turns: List[Dict[str, Any]] = []
        pending_user: Optional[str] = None
        for item in trace_messages:
            role = str(item.get("role", "")).lower()
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                if pending_user:
                    turns.append({"user_input": pending_user})
                pending_user = content
            elif role == "assistant" and pending_user:
                turns.append(
                    {
                        "user_input": pending_user,
                        "assistant_output": content,
                    }
                )
                pending_user = None
        return bounded_recent_turns(
            turns,
            max_turns=2,
            max_chars=int(os.getenv("RECENT_TURNS_MAX_CHARS", "4000")),
        )

    def _build_user_context(self, request: AgentChatRequest) -> str:
        parts: List[str] = []
        if request.ui_context:
            parts.append("## UI Context")
            if request.ui_context.date:
                parts.append(f"- date: {request.ui_context.date}")
            if request.ui_context.selected:
                parts.append(f"- selected: {request.ui_context.selected.get('type')} {request.ui_context.selected.get('id')}")
        parts.append("\n## Current User Request")
        parts.append(request.message)
        return "\n".join(parts).strip()

    def _safe_parse_args(self, raw_args: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(raw_args) if raw_args else {}
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _chat_request_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.llm_provider.request_options(),
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return payload

    def _assistant_tool_history(
        self,
        message: Any,
        content: str,
        tool_calls: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
            **self.llm_provider.assistant_history_fields(message),
        }

    def _tool_loop_stream(
        self,
        system_prompt: str,
        user_context: str,
        *,
        event_time: Optional[str] = None,
        answer_validator: Optional[AnswerValidator] = None,
        current_user_request: str = "",
        prior_state: Optional[VerifiedDecisionState] = None,
    ):
        missing_key_message = self.llm_provider.missing_key_message()
        if missing_key_message:
            message = missing_key_message
            timestamp = event_time or datetime.now().isoformat()
            self.trace_writer.append_message(self.session_id, role="assistant", content=message, timestamp=timestamp)
            yield {"event": "assistant_message", "data": {"content": message, "timestamp": timestamp, "final": True}}
            return message, "error", []

        client = self.llm_provider.create_client()
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_context}]
        tools_schema = tool_registry.openai_tools_schema()
        completed_tool_calls: List[Dict[str, Any]] = []
        last_request_snapshot: Optional[Dict[str, Any]] = None
        logger.info("Agent tool loop ready: session_id=%s model=%s tools=%d", self.session_id, self.model, len(tools_schema))

        try:
            for step_index in range(self.max_steps):
                contract_message = candidate_contract_message(
                    user_context,
                    completed_tool_calls,
                    prior_candidate_results=(
                        prior_state.candidates if prior_state else None
                    ),
                    prior_decision_policy=(
                        prior_state.decision_policy if prior_state else None
                    ),
                    prior_decision_policy_source_question=(
                        prior_state.decision_policy_source_question
                        if prior_state
                        else None
                    ),
                    prior_applied_disturbances=(
                        prior_state.applied_disturbances
                        if prior_state
                        else None
                    ),
                )
                request_messages = list(messages)
                if contract_message:
                    request_messages.append({"role": "system", "content": contract_message})
                request_payload = self._chat_request_payload(
                    request_messages,
                    tools=tools_schema,
                    tool_choice="auto",
                )
                logger.info("LLM request started: session_id=%s step=%d model=%s", self.session_id, step_index + 1, self.model)
                last_request_snapshot = self.trace_writer.append_llm_call(
                    self.session_id,
                    step=step_index + 1,
                    phase="request",
                    payload_data=request_payload,
                    summary={"model": self.model},
                    timestamp=event_time,
                )
                response = client.chat.completions.create(**request_payload)
                choice = response.choices[0]
                message = choice.message
                finish_reason = choice.finish_reason
                tool_calls = message.tool_calls or []
                content = (message.content or "").strip()
                logger.info(
                    "LLM response received: session_id=%s step=%d finish_reason=%s tool_calls=%d content_chars=%d",
                    self.session_id,
                    step_index + 1,
                    finish_reason,
                    len(tool_calls),
                    len(content),
                )
                try:
                    response_trace = response.model_dump(mode="json")
                except Exception as exc:
                    response_trace = {
                        "serialization_error": str(exc),
                        "response_repr": repr(response),
                    }
                self.trace_writer.append_llm_call(
                    self.session_id,
                    step=step_index + 1,
                    phase="response",
                    payload_data={"finish_reason": finish_reason, "response": response_trace},
                    summary={"model": self.model, "finish_reason": finish_reason},
                    timestamp=event_time,
                )

                if finish_reason == "tool_calls" and not tool_calls:
                    logger.warning("Model returned finish_reason=tool_calls without tool_calls; retrying without tools")
                    fallback_payload = self._chat_request_payload(request_messages)
                    last_request_snapshot = self.trace_writer.append_llm_call(
                        self.session_id,
                        step=step_index + 1,
                        phase="fallback_request",
                        payload_data=fallback_payload,
                        summary={"model": self.model},
                        timestamp=event_time,
                    )
                    fallback_response = client.chat.completions.create(**fallback_payload)
                    fallback_choice = fallback_response.choices[0]
                    response = fallback_response
                    message = fallback_choice.message
                    finish_reason = fallback_choice.finish_reason
                    tool_calls = message.tool_calls or []
                    content = (message.content or "").strip()
                    try:
                        response_trace = response.model_dump(mode="json")
                    except Exception as exc:
                        response_trace = {
                            "serialization_error": str(exc),
                            "response_repr": repr(response),
                        }
                    self.trace_writer.append_llm_call(
                        self.session_id,
                        step=step_index + 1,
                        phase="fallback_response",
                        payload_data={"finish_reason": finish_reason, "response": response_trace},
                        summary={"model": self.model, "finish_reason": finish_reason},
                        timestamp=event_time,
                    )

                if content and tool_calls:
                    event_timestamp = event_time or datetime.now().isoformat()
                    yield {
                        "event": "assistant_message",
                        "data": {
                            "content": content,
                            "timestamp": event_timestamp,
                            "final": False,
                        },
                    }

                if not tool_calls:
                    final_text = content
                    if not final_text and finish_reason == "tool_calls":
                        final_text = "Model reported tool_calls without returning tool_calls. Check whether the current gateway supports OpenAI function calling."
                    if not final_text:
                        final_text = "Model did not return any displayable content."

                    final_text = finalize_runtime_answer(
                        final_text,
                        current_user_request or user_context,
                        completed_tool_calls,
                        prior_state=prior_state,
                    )
                    quality_issues = answer_validator(final_text, completed_tool_calls) if answer_validator else []
                    if quality_issues:
                        repair_messages = request_messages + [
                            {"role": "assistant", "content": final_text},
                            {
                                "role": "user",
                                "content": (
                                    "Rewrite only the final answer. The draft failed teacher-data validation: "
                                    + ", ".join(quality_issues)
                                    + ". Follow the applicable final-answer contract in the system prompt. "
                                    "Use only user-stated premises, explicitly retrieved source content, and "
                                    "structured forecast results; exit code 0 and prior assistant claims are not evidence. "
                                    "add no new facts and return no preface."
                                ),
                            },
                        ]
                        repair_payload = self._chat_request_payload(repair_messages)
                        self.trace_writer.append_llm_call(
                            self.session_id,
                            step=step_index + 1,
                            phase="quality_repair_request",
                            payload_data=repair_payload,
                            summary={"model": self.model, "quality_issues": quality_issues},
                            timestamp=event_time,
                        )
                        try:
                            repair_response = client.chat.completions.create(**repair_payload)
                            repair_choice = repair_response.choices[0]
                            repaired_text = (repair_choice.message.content or "").strip()
                            if repaired_text:
                                repaired_text = finalize_runtime_answer(
                                    repaired_text,
                                    current_user_request or user_context,
                                    completed_tool_calls,
                                    prior_state=prior_state,
                                )
                            repaired_issues = (
                                answer_validator(repaired_text, completed_tool_calls)
                                if repaired_text and answer_validator
                                else quality_issues
                            )
                            try:
                                repair_response_trace = repair_response.model_dump(mode="json")
                            except Exception as exc:
                                repair_response_trace = {
                                    "serialization_error": str(exc),
                                    "response_repr": repr(repair_response),
                                }
                            self.trace_writer.append_llm_call(
                                self.session_id,
                                step=step_index + 1,
                                phase="quality_repair_response",
                                payload_data={
                                    "finish_reason": repair_choice.finish_reason,
                                    "response": repair_response_trace,
                                    "remaining_quality_issues": repaired_issues,
                                },
                                summary={"model": self.model, "finish_reason": repair_choice.finish_reason},
                                timestamp=event_time,
                            )
                            if repaired_text and len(repaired_issues) < len(quality_issues):
                                final_text = repaired_text
                                quality_issues = repaired_issues
                            logger.info(
                                "Teacher answer repair finished: session_id=%s remaining_issues=%s",
                                self.session_id,
                                quality_issues,
                            )
                        except Exception:
                            logger.exception("Teacher answer repair failed: session_id=%s", self.session_id)

                    event_timestamp = event_time or datetime.now().isoformat()
                    yield {
                        "event": "assistant_message",
                        "data": {
                            "content": final_text,
                            "timestamp": event_timestamp,
                            "final": True,
                        },
                    }
                    self.trace_writer.append_message(self.session_id, role="assistant", content=final_text, timestamp=event_time)
                    return final_text, "completed", completed_tool_calls

                assistant_tool_calls = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]
                messages.append(
                    self._assistant_tool_history(message, content, assistant_tool_calls)
                )

                for call in tool_calls:
                    args = self._safe_parse_args(call.function.arguments)
                    tool_input: Dict[str, Any] = args if args is not None else {"_raw_arguments": call.function.arguments}
                    logger.info(
                        "Tool call started: session_id=%s tool=%s call_id=%s args=%s",
                        self.session_id,
                        call.function.name,
                        call.id,
                        json.dumps(tool_input, ensure_ascii=False, default=str)[:1200],
                    )
                    tool_timestamp = event_time or datetime.now().isoformat()
                    yield {
                        "event": "tool_start",
                        "data": {
                            "tool_call_id": call.id,
                            "tool": call.function.name,
                            "input": tool_input,
                            "timestamp": tool_timestamp,
                        },
                    }

                    if args is None:
                        result = {
                            "success": False,
                            "error": "Invalid JSON arguments",
                            "tool": call.function.name,
                            "raw_arguments": call.function.arguments,
                        }
                    else:
                        preexecution_failure = forecast_preexecution_failure(
                            call.function.name,
                            args,
                            completed_tool_calls,
                            current_user_request=current_user_request,
                        )
                        if preexecution_failure:
                            result = preexecution_failure
                        else:
                            try:
                                result = tool_registry.execute(
                                    call.function.name,
                                    session_id=self.session_id,
                                    agent_id=self.agent_id,
                                    **args,
                                )
                            except Exception as exc:
                                result = {
                                    "success": False,
                                    "error": str(exc),
                                    "tool": call.function.name,
                                    "params": args,
                                }

                    result_success = not isinstance(result, dict) or (
                        result.get("success") is not False
                        and not bool(result.get("error"))
                        and result.get("exit_code") in (None, 0)
                    )
                    record_result = should_record_tool_result(
                        call.function.name,
                        result,
                    )
                    if record_result:
                        completed_tool_calls.append(
                            {
                                "tool_call_id": call.id,
                                "name": call.function.name,
                                "arguments": args or {},
                                "output": result,
                            }
                        )
                    logger.info(
                        "Tool call finished: session_id=%s tool=%s call_id=%s success=%s",
                        self.session_id,
                        call.function.name,
                        call.id,
                        result_success,
                    )
                    result_summary = json.dumps(result, ensure_ascii=False)[:4000]
                    trace_extra = {"tool_call_id": call.id, "result": result}
                    if isinstance(result, dict) and result.get("error"):
                        trace_extra["failure_context"] = {
                            "finish_reason": finish_reason,
                            "assistant_message": {
                                "content": content,
                                "tool_calls": assistant_tool_calls,
                            },
                            "llm_response": response_trace,
                        }
                    if record_result:
                        self.trace_writer.append_tool_call(
                            self.session_id,
                            call.function.name,
                            args or {},
                            result_summary,
                            extra=trace_extra,
                            timestamp=event_time,
                        )
                    else:
                        self.trace_writer.append_audit_tool_call(
                            self.session_id,
                            call.function.name,
                            args or {},
                            result,
                            timestamp=event_time,
                        )
                        logger.info(
                            "Tool call retained as internal planning correction: "
                            "session_id=%s tool=%s call_id=%s error_code=%s",
                            self.session_id,
                            call.function.name,
                            call.id,
                            result.get("error_code")
                            if isinstance(result, dict)
                            else None,
                        )
                    tool_end_timestamp = event_time or datetime.now().isoformat()
                    yield {
                        "event": "tool_end",
                        "data": {
                            "tool_call_id": call.id,
                            "tool": call.function.name,
                            "output": result_summary,
                            "timestamp": tool_end_timestamp,
                            "success": result_success,
                        },
                    }
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
        except Exception as exc:
            last_request_path = last_request_snapshot.get("payload_path") if last_request_snapshot else None
            error_message = f"LLM call failed: {exc}" + (f" | last_request={last_request_path}" if last_request_path else "")
            logger.exception("LLM tool loop failed")
            self.trace_writer.append_message(self.session_id, role="assistant", content=error_message, timestamp=event_time)
            yield {"event": "error", "data": {"message": error_message, "timestamp": event_time or datetime.now().isoformat(), "last_request_path": last_request_path}}
            return error_message, "error", completed_tool_calls

        timeout_message = "Tool-call step limit reached; workflow stopped."
        timeout_timestamp = event_time or datetime.now().isoformat()
        self.trace_writer.append_message(self.session_id, role="assistant", content=timeout_message, timestamp=timeout_timestamp)
        yield {
            "event": "assistant_message",
            "data": {
                "content": timeout_message,
                "timestamp": timeout_timestamp,
                "final": True,
            },
        }
        return timeout_message, "timeout", completed_tool_calls

    def run_agent(
        self,
        request: AgentChatRequest,
        *,
        answer_validator: Optional[AnswerValidator] = None,
    ) -> OneTurnAfterRunAgent:
        stream = self.run_agent_stream(request, answer_validator=answer_validator)
        while True:
            try:
                next(stream)
            except StopIteration as stop:
                if stop.value is None:
                    raise RuntimeError("run_agent_stream did not return a result")
                return stop.value

    def run_agent_stream(
        self,
        request: AgentChatRequest,
        *,
        answer_validator: Optional[AnswerValidator] = None,
    ):
        event_time = request.event_time
        event_timestamp = event_time or datetime.now().isoformat()
        logger.info("Agent run started: agent_id=%s session_id=%s", self.agent_id, self.session_id)
        self.trace_writer.reset_turn()
        self.trace_writer.ensure_trace(self.session_id, created_at=event_time)
        self.trace_writer.set_status(self.session_id, "running", timestamp=event_time)
        self.trace_writer.append_message(self.session_id, role="user", content=request.message, timestamp=event_time)
        yield {
            "event": "session_created",
            "data": {
                "agent_id": self.agent_id,
                "session_id": self.session_id,
                "timestamp": event_timestamp,
            },
        }

        trace_payload = self.trace_writer.load_trace(self.session_id)
        memory_payload = self.memory_assembler.build(self.session_id)
        prior_state = self.verified_state_manager.load(self.session_id)
        state_payload = serialize_verified_decision_state(
            prior_state,
            max_chars=int(os.getenv("VERIFIED_STATE_MAX_CHARS", "16000")),
        )
        trace_messages = list(trace_payload.get("messages") or [])
        if (
            trace_messages
            and str(trace_messages[-1].get("role") or "").casefold() == "user"
        ):
            trace_messages = trace_messages[:-1]
        recent_turns = self._recent_dialogue_turns(trace_messages)
        self.trace_writer.set_context_injection(
            self.session_id,
            {
                "control_files": [item["name"] for item in memory_payload.get("control_files", [])],
                "assets": [item["path"] for item in memory_payload.get("assets", [])],
                "verified_state_schema": state_payload.get("schema_version"),
                "verified_state_snapshot": (
                    self.verified_state_manager.snapshot_path(
                        self.session_id
                    ).as_posix()
                ),
                "verified_state_chars": len(
                    json.dumps(
                        state_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                ),
                "recent_dialogue_turns": len(recent_turns),
                "history_policy": (
                    "one verified_decision_state_v1 snapshot plus at most two "
                    "bounded recent dialogue turns; timeline and context trace "
                    "are audit-only"
                ),
            },
            timestamp=event_time,
        )
        skills_section = self.skill_manager.render_skills_section() if self.skill_manager else ""
        system_prompt = self.prompt_builder.build(
            memory_payload=memory_payload,
            skills_section=skills_section,
            verified_state=state_payload,
            recent_turns=recent_turns,
        )
        user_context = self._build_user_context(request)
        assistant_message, final_status, completed_tool_calls = yield from self._tool_loop_stream(
            system_prompt,
            user_context,
            event_time=event_time,
            answer_validator=answer_validator,
            current_user_request=(
                request.message.rsplit("Current user request:", 1)[-1].strip()
                if "Current user request:" in request.message
                else request.message
            ),
            prior_state=prior_state,
        )
        turn_number = sum(
            1
            for item in trace_payload.get("messages") or []
            if str(item.get("role") or "").casefold() == "user"
        )
        next_state = prior_state.updated_from_tool_results(
            self.session_id,
            turn_number,
            (
                request.message.rsplit("Current user request:", 1)[-1].strip()
                if "Current user request:" in request.message
                else request.message
            ),
            completed_tool_calls,
        )
        state_commit = self.verified_state_manager.commit(
            self.session_id,
            next_state,
        )
        memory_commit = self.memory_manager.commit_turn(
            session_id=self.session_id,
            user_message=request.message,
            assistant_message=assistant_message,
            event_time=event_time,
        )
        memory_updates = [
            memory_commit["memory_commit_path"],
            state_commit["state_snapshot_path"],
            state_commit["state_event_path"],
        ]
        self.trace_writer.extend_memory_commits(self.session_id, memory_updates, timestamp=event_time)
        self.trace_writer.append_decision(self.session_id, summary=assistant_message[:500], artifact_paths=[], timestamp=event_time)
        self.trace_writer.set_status(self.session_id, final_status, timestamp=event_time)
        trace_summary = TraceSummary(**self.trace_writer.summarize(self.session_id))
        result = OneTurnAfterRunAgent(return_message=assistant_message, memory_updates=memory_updates, generated_artifacts=[], trace_summary=trace_summary)
        logger.info(
            "Agent run finished: agent_id=%s session_id=%s status=%s tool_calls=%d trace=%s",
            self.agent_id,
            self.session_id,
            final_status,
            trace_summary.tool_calls_count,
            trace_summary.trace_path,
        )
        yield {
            "event": "done",
            "data": AgentChatResponse(
                agent_id=self.agent_id,
                session_id=self.session_id,
                message_markdown=result.return_message,
                trace_summary=result.trace_summary,
                memory_updates=result.memory_updates,
                generated_artifacts=result.generated_artifacts,
                timestamp=event_timestamp,
            ).model_dump(),
        }
        return result


orchestrator: Optional[AgentOrchestrator] = None


def init_orchestrator(data_loader, session_id: str = "default", enable_skills: Optional[bool] = None, agent_id: str = "default"):
    global orchestrator
    orchestrator = AgentOrchestrator(data_loader, agent_id=agent_id, session_id=session_id, enable_skills=enable_skills)
    return orchestrator
