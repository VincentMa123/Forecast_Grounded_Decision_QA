from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pipeclaw.backend.grounding.decision_trace_state import VerifiedDecisionState

from .episode import execution_success, schema_valid
from .models import PromptCase, RolloutConfig, RolloutResult


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
_STATUS = {
    "completed": "completed",
    "timeout": "max_turns_exceeded",
    "error": "generation_error",
}


def _production_orchestrator(**kwargs: Any) -> Any:
    from pipeclaw.backend.agent.orchestrator import AgentOrchestrator

    return AgentOrchestrator(**kwargs)


def _seed_context(orchestrator: Any, case: PromptCase, session_id: str) -> None:
    source = case.source_record
    state = source.get("state_before")
    if isinstance(state, Mapping) and state:
        orchestrator.verified_state_manager.commit(
            session_id, VerifiedDecisionState.from_dict(dict(state))
        )
    orchestrator.trace_writer.ensure_trace(session_id)
    for turn in source.get("recent_turns") or []:
        if not isinstance(turn, Mapping):
            continue
        for role, key in (("user", "user_input"), ("assistant", "assistant_output")):
            content = turn.get(key)
            if isinstance(content, str) and content.strip():
                orchestrator.trace_writer.append_message(
                    session_id, role=role, content=content
                )


def _ordered_attempts(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    attempts = [
        item
        for key in ("tool_calls", "audit_tool_calls")
        for item in trace.get(key) or []
        if isinstance(item, Mapping)
    ]
    return sorted(
        attempts,
        key=lambda item: (
            int(item.get("sequence"))
            if isinstance(item.get("sequence"), int)
            else 2**31,
            str(item.get("timestamp") or ""),
        ),
    )


def _trace_rollout(
    case: PromptCase,
    trace: Mapping[str, Any],
    answer: str,
    *,
    capture_raw_responses: bool = False,
    capture_raw_tool_outputs: bool = False,
) -> RolloutResult:
    attempts = _ordered_attempts(trace)
    calls: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        result = attempt.get("result")
        call_id = str(attempt.get("tool_call_id") or f"production-call-{index}")
        name = str(attempt.get("tool_name") or attempt.get("name") or "")
        calls.append(
            {
                "tool_call_id": call_id,
                "name": name,
                "arguments": dict(attempt.get("args") or {}),
                "execution_success": execution_success(result),
                "schema_valid": schema_valid(result),
            }
        )
        outputs.append({"tool_call_id": call_id, "name": name, "output": result})
    turns = sum(
        item.get("phase") == "response"
        for item in trace.get("llm_calls") or []
        if isinstance(item, Mapping)
    )
    return RolloutResult(
        sample_id=case.sample_id,
        scenario_id=case.scenario_id,
        scenario_type=case.scenario_type,
        messages=[
            dict(item)
            for item in trace.get("messages") or []
            if isinstance(item, Mapping)
        ],
        tool_calls=calls,
        tool_outputs=outputs,
        final_answer=answer,
        trace_status=_STATUS.get(
            str(trace.get("status") or ""), str(trace.get("status") or "")
        ),
        turns=turns,
        raw_responses=(
            [
                dict(item)
                for item in trace.get("llm_calls") or []
                if isinstance(item, Mapping)
            ]
            if capture_raw_responses
            else None
        ),
        raw_tool_outputs=(
            [dict(item) for item in attempts] if capture_raw_tool_outputs else None
        ),
    )


class ProductionAgentRunner:
    """Run one case through the same orchestrator used by the FastAPI app."""

    def __init__(
        self, orchestrator_factory: Callable[..., Any] = _production_orchestrator
    ):
        self.orchestrator_factory = orchestrator_factory

    def run(self, case: PromptCase, config: RolloutConfig) -> RolloutResult:
        safe = _SAFE_ID.sub("_", case.sample_id).strip("._") or "episode"
        run_id = uuid.uuid4().hex[:10]
        agent_id = f"eval-{safe[-48:]}-{run_id}"
        session_id = f"{agent_id}-session"
        workspace = Path(case.workspace_root or ".")
        orchestrator = self.orchestrator_factory(
            data_loader=None,
            agent_id=agent_id,
            session_id=session_id,
            enable_skills=False,
            workspace_root_base=workspace,
            max_steps=config.max_turns,
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
        )
        _seed_context(orchestrator, case, session_id)

        from pipeclaw.backend.agent.schemas import AgentChatRequest

        response = orchestrator.run_agent(
            AgentChatRequest(
                agent_id=agent_id,
                session_id=session_id,
                message=str(case.messages[-1].get("content") or ""),
            )
        )
        trace = orchestrator.trace_writer.load_trace(session_id)
        return _trace_rollout(
            case,
            trace,
            response.return_message,
            capture_raw_responses=config.capture_raw_responses,
            capture_raw_tool_outputs=config.capture_raw_tool_outputs,
        )
