from __future__ import annotations

import re
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeclaw.student_distillation.path_contract import canonicalize_recorded_tool_arguments
from pipeclaw.student_distillation.rollout.episode import dispatch_and_record
from pipeclaw.student_distillation.rollout.models import RolloutResult
from pipeclaw.student_distillation.rollout.scenarios import ScenarioPolicy, build_dispatcher
from pipeclaw.student_distillation.rollout.tools import ToolDispatcher
from pipeclaw.protocols.tool_calls import ToolCall, parse_tool_calls
from pipeclaw.student_distillation.reward import composite_reward, episode_stats

from swift.rewards import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

_GRPO_WORKSPACES = _ROOT / "pipeclaw/student_distillation/outputs/grpo/workspaces"


@dataclass(frozen=True)
class _LastParse:
    """Parsed response fields shared by the scheduler lifecycle callbacks."""

    response_id: int | None
    text: str
    calls: list[ToolCall]
    errors: list[str]


@dataclass
class _SchedulerState:
    """Typed mutable state for one Swift episode."""

    result: RolloutResult
    dispatcher: ToolDispatcher
    workspace: Path
    last_parse: _LastParse = field(
        default_factory=lambda: _LastParse(None, "", [], [])
    )


class PythonScenarioScheduler(MultiTurnScheduler):
    """One episode per request: model turn → execute tool calls → feed results."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._policy = ScenarioPolicy()
        self._lock = threading.Lock()
        self._schemas: list[dict[str, Any]] | None = None
        self._states: dict[int, _SchedulerState] = {}

    def _tool_schemas(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._schemas is None:
                backend_root = (
                    Path(__file__).resolve().parents[2] / "pipeclaw" / "backend"
                )
                from pipeclaw.backend.agent.tools.pipeformer_tools import (
                    register_pipeformer_tools,
                )
                from pipeclaw.backend.agent.tools.workspace_tools import WorkspaceTools
                from pipeclaw.backend.agent.tools.registry import tool_registry

                register_pipeformer_tools(backend_root)  # idempotent; orders schema()
                WorkspaceTools(session_id="grpo-plugin")
                self._schemas = tool_registry.openai_tools_schema()
            return self._schemas

    def _new_dispatcher(self, scenario_type: Any) -> Any:
        """One dispatcher per request: concurrent generations in a GRPO group
        must never share completed-tool history (registry preconditions) or the
        current-user-request used to validate set_decision_policy excerpts."""
        return build_dispatcher(
            scenario_type or "openclaw", self._tool_schemas(), _ROOT
        )

    def _state(self, infer_request: Any) -> _SchedulerState:
        state = self._states.get(id(infer_request))
        if state is None:
            raise RuntimeError(
                "on_trajectory_start never registered this request (id collision "
                "or the request object flowed in through the wrong lifecycle)"
            )
        return state

    async def on_trajectory_start(self, requests: List[Any]) -> None:
        for request in requests:
            data = getattr(request, "data_dict", {}) or {}
            # Fresh state per episode: ids are object addresses and CPython
            # reuses them after swift frees the last generation's request.
            scenario_type = data.get("scenario_type") or "openclaw"
            scenario = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                str(data.get("scenario_id") or data.get("sample_id") or "scenario"),
            )
            workspace = (
                _GRPO_WORKSPACES
                / f"{scenario}-{uuid.uuid4().hex[:8]}"
                / "workspace-grpo"
            )
            workspace.mkdir(parents=True, exist_ok=True)
            user_message = next(
                (
                    str(message.get("content") or "")
                    for message in reversed(getattr(request, "messages", []) or [])
                    if isinstance(message, Mapping) and message.get("role") == "user"
                ),
                "",
            )
            dispatcher = self._new_dispatcher(scenario_type)
            dispatcher.set_current_user_request(user_message)
            dispatcher.set_case_workspace(workspace)
            result = RolloutResult(
                sample_id=str(data.get("sample_id") or data.get("scenario_id") or ""),
                scenario_id=str(data.get("scenario_id") or ""),
                scenario_type=str(scenario_type),
                # Keep Swift's live message list as the canonical transcript.
                messages=request.messages,
            )
            self._states[id(request)] = _SchedulerState(
                result=result,
                dispatcher=dispatcher,
                workspace=workspace,
            )

    def step(
        self, infer_request: Any, response_choice: Any, current_turn: int
    ) -> Dict[str, Any]:
        state = self._state(infer_request)
        text, calls, errors = parse_tool_calls(
            getattr(response_choice, "message", response_choice)
        )
        state.last_parse = _LastParse(id(response_choice), text, calls, errors)
        # ms-swift already appended this response as a raw assistant message;
        # replace it with its parsed text so history matches the offline runner.
        if (
            infer_request.messages
            and infer_request.messages[-1].get("role") == "assistant"
        ):
            infer_request.messages[-1]["content"] = text
        self._dispatch_calls(state, calls)
        return {"infer_request": infer_request}

    def _dispatch_calls(
        self, state: _SchedulerState, calls: Sequence[Any]
    ) -> None:
        dispatcher = state.dispatcher

        def dispatch_locked(call: Any) -> Mapping[str, Any]:
            with self._lock:
                return dispatcher.dispatch(call)

        dispatch_and_record(
            calls,
            dispatcher=dispatcher,
            result=state.result,
            record_arguments=lambda call: canonicalize_recorded_tool_arguments(
                call.name, call.arguments
            ),
            compact_result=lambda call, tool_result, portability: self._policy.compact_tool_result(
                call, tool_result, portability=portability
            ),
            dispatch=dispatch_locked,
        )

    @staticmethod
    def _cleanup_workspace(state: _SchedulerState) -> None:
        # Scoring reads only the recorded rollout_infos, never the workspace.
        shutil.rmtree(state.workspace.parent, ignore_errors=True)

    def check_finished(
        self, infer_request: Any, response_choice: Any, current_turn: int
    ) -> bool:
        state = self._state(infer_request)
        if super().check_finished(infer_request, response_choice, current_turn):
            state.result.trace_status = "max_turns_exceeded"
        elif state.last_parse.calls:
            return False
        self._cleanup_workspace(state)
        # swift snapshots rollout_infos before the finished branch runs; the
        # compact output records stay reachable through those snapshots.
        self._states.pop(id(infer_request), None)
        return True

    async def on_turn_end(
        self, infer_request: Any, response_choice: Any, current_turn: int
    ) -> Dict[str, Any]:
        data = getattr(infer_request, "data_dict", {}) or {}
        state = self._state(infer_request)
        # Single parse point per response: reuse step()'s parse, re-parsing only
        # when swift skipped step() (the capped turn).
        last = state.last_parse
        if last.response_id == id(response_choice):
            text, calls, errors = last.text, last.calls, last.errors
        else:
            text, calls, errors = parse_tool_calls(
                getattr(response_choice, "message", None) or response_choice
            )
        state.last_parse = _LastParse(id(response_choice), text, calls, errors)
        # dispatch the capped turn's calls BEFORE building the snapshot: swift
        # skips step() on the capped turn, but the frame merges AFTER this hook.
        capped = getattr(self, "max_turns", None) is not None and int(
            current_turn
        ) >= int(self.max_turns)
        if calls and capped:
            self._dispatch_calls(state, calls)
        state.result.json_errors.extend(errors)
        finish_reason = getattr(response_choice, "finish_reason", None)
        if not calls:
            # the answer arrived inside the turn budget: classify content first,
            # only the call-side aborts wear the cap labels.
            if finish_reason not in (None, "stop", "tool_calls"):
                state.result.trace_status = "max_completion_length"
            elif errors and not (text or "").strip():
                state.result.trace_status = "malformed_tool_json"
            else:
                state.result.final_answer = text or ""
                # an immediate-EOS completion (no text at all) is an abort, not a
                # completed answer — same label the Phase-0 harness assigns.
                state.result.trace_status = (
                    "empty_response" if not (text or "").strip() else "completed"
                )
        elif finish_reason not in (None, "stop", "tool_calls") or capped:
            state.result.trace_status = "max_turns_exceeded"
        return {
            "rollout_infos": {
                "sample_id": data.get("sample_id") or data.get("scenario_id") or "",
                "scenario_id": data.get("scenario_id") or "",
                "scenario_type": state.result.scenario_type,
                "tool_calls": state.result.tool_calls,
                "tool_outputs": state.result.tool_outputs,
                "json_errors": state.result.json_errors,
                "trace_status": state.result.trace_status,
                "final_answer": state.result.final_answer,
            }
        }


class PythonEpisodeReward(ORM):
    """Deterministic episode reward scored by the backend evaluator + rules."""

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> List[float]:
        infos = kwargs.get("rollout_infos") or []
        references = kwargs.get("reference") or []
        if len(infos) != len(completions):
            raise ValueError(
                f"rollout count mismatch: {len(infos)} infos for {len(completions)} completions"
            )
        rewards: list[float] = []
        for index, completion in enumerate(completions):
            reference = (
                references[index]
                if isinstance(references, Sequence) and index < len(references)
                else {}
            )
            if not reference:
                # An empty reference silently pays a vacuous evaluator pass —
                # corrupted batch beats any honest episode. Crash instead.
                raise ValueError(
                    f"empty or missing reference for episode {index}; "
                    "stop training and fix dataset integrity"
                )
            rewards.append(_score_episode(infos[index], reference))
        return rewards


def _score_episode(info: Mapping[str, Any], reference: Any) -> float:
    from pipeclaw.backend.evaluator import EvaluationProfile, evaluate

    rollout = RolloutResult(
        sample_id=info.get("sample_id") or "",
        scenario_id=info.get("scenario_id") or "",
        scenario_type=info.get("scenario_type") or "openclaw",
        tool_calls=list(info.get("tool_calls") or []),
        tool_outputs=list(info.get("tool_outputs") or []),
        final_answer=str(info.get("final_answer") or ""),
        trace_status=info.get("trace_status") or "",
        json_errors=list(info.get("json_errors") or []),
    ).to_dict()
    report = evaluate(
        rollout, profile=EvaluationProfile.AUTONOMOUS_ROLLOUT, reference=reference
    )
    return composite_reward(episode_stats(rollout), report.to_dict())


multi_turns["python_scenario_scheduler"] = PythonScenarioScheduler
orms["python_episode_reward"] = PythonEpisodeReward
