"""GRPO external plugin: pipeclaw python-scenario scheduler + rule reward.

Registered via ms-swift ``--external_plugins``:
- ``multi_turns['python_scenario_scheduler']``: drives one agent episode per
  request (read_file/write_file/edit_file/run_command through the real
  tools) in the backend workspace runner). Each request gets its own
  workspace key collapses a GRPO group into one directory.
- ``orms['python_episode_reward']``: deterministic reward = backend evaluator
  overall_score/hard gate + execution stats (compile, exit code, thrash,
  malformed JSON). Mirrors scripts/pass_at_k.composite_reward so the Phase-0
  gate measures exactly what the RL stage optimizes.

preflight (run BEFORE the first real run; upstream refs here are ms-swift 'main',
this env pins ms-swift==4.4.2):
  python -c "from swift.rewards import ORM, orms; from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns; print('ok')"
  swift rlhf --rlhf_type grpo <config> --max_steps 1   # fails fast if names differ
"""

from __future__ import annotations

import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeclaw.task2_student.rollout.scenarios import (
    ScenarioPolicy,
    build_openclaw_dispatcher,
)
from pipeclaw.task2_student.rollout.tools import append_tool_exchange, parse_tool_calls
from pipeclaw.task2_student.scripts.pass_at_k import composite_reward, episode_stats

from swift.rewards import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

_GRPO_WORKSPACES = Path("pipeclaw/task2_student/outputs/grpo/workspaces")


class PythonScenarioScheduler(MultiTurnScheduler):
    """One episode per request: model turn → execute tool calls → feed results."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dispatcher = build_openclaw_dispatcher([], Path("."))
        self._policy = ScenarioPolicy()
        self._lock = threading.Lock()
        self._states: dict[int, dict[str, Any]] = {}

    def _state(self, infer_request: Any) -> dict[str, Any]:
        return self._states.setdefault(
            id(infer_request),
            {
                "tool_calls": [],
                "tool_outputs": [],
                "json_errors": [],
                "final_answer": "",
                "trace_status": "",
            },
        )

    async def on_trajectory_start(self, requests: List[Any]) -> None:
        for request in requests:
            data = getattr(request, "data_dict", {}) or {}
            scenario = str(
                data.get("scenario_id") or data.get("sample_id") or "scenario"
            )
            workspace = (
                _GRPO_WORKSPACES / f"{scenario}-{uuid.uuid4().hex[:8]}" / "workspace-grpo"
            )
            workspace.mkdir(parents=True, exist_ok=True)
            self._state(request)["workspace"] = workspace
            with self._lock:
                self._dispatcher.set_case_workspace(workspace)

    def step(
        self, infer_request: Any, response_choice: Any, current_turn: int
    ) -> Dict[str, Any]:
        state = self._state(infer_request)
        text, calls, errors = parse_tool_calls(getattr(response_choice, "message", response_choice))
        state["json_errors"].extend(errors)
        for index, call in enumerate(calls):
            with self._lock:
                tool_result = self._dispatcher.dispatch(call)
            compact = self._policy.compact_tool_result(call, tool_result)
            state["tool_calls"].append(
                {
                    "tool_call_id": call.call_id,
                    "name": call.name,
                    "arguments": dict(call.arguments),
                    "execution_success": (
                        isinstance(tool_result, Mapping)
                        and tool_result.get("success", True) is not False
                        and not tool_result.get("error")
                        and tool_result.get("exit_code") in (None, 0)
                    ),
                }
            )
            state["tool_outputs"].append(
                {"tool_call_id": call.call_id, "name": call.name, "output": compact}
            )
            append_tool_exchange(
                infer_request.messages,
                call,
                compact,
                assistant_content=text if index == 0 else "",
            )
        return {"infer_request": infer_request}

    def check_finished(
        self, infer_request: Any, response_choice: Any, current_turn: int
    ) -> bool:
        state = self._state(infer_request)
        if super().check_finished(infer_request, response_choice, current_turn):
            state["trace_status"] = (
                "max_completion_length"
                if getattr(response_choice, "finish_reason", None) == "length"
                else "max_turns_exceeded"
            )
            return True
        message = getattr(response_choice, "message", None)
        text, calls, errors = parse_tool_calls(message if message is not None else response_choice)
        if calls:
            return False
        if errors and not (text or "").strip():
            state["trace_status"] = "malformed_tool_json"
            state["json_errors"].extend(errors)
            return True
        state["final_answer"] = text or ""
        state["trace_status"] = "completed"
        return True

    async def on_turn_end(
        self, infer_request: Any, response_choice: Any, current_turn: int
    ) -> Dict[str, Any]:
        data = getattr(infer_request, "data_dict", {}) or {}
        state = self._state(infer_request)
        return {
            "rollout_infos": {
                "sample_id": data.get("sample_id") or data.get("scenario_id") or "",
                "scenario_id": data.get("scenario_id") or "",
                "tool_calls": state["tool_calls"],
                "tool_outputs": state["tool_outputs"],
                "json_errors": state["json_errors"],
                "trace_status": state["trace_status"],
            }
        }


class PythonEpisodeReward(ORM):
    """Deterministic episode reward scored by the backend evaluator + rules."""

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> List[float]:
        infos = kwargs.get("rollout_infos") or []
        references = kwargs.get("reference") or []
        rewards: list[float] = []
        for index, completion in enumerate(completions):
            info = infos[index] if isinstance(infos, Sequence) and index < len(infos) else {}
            reference = (
                references[index]
                if isinstance(references, Sequence) and index < len(references)
                else {}
            )
            rewards.append(_score_episode(info, reference, completion))
        return rewards


def _score_episode(info: Mapping[str, Any], reference: Any, completion: str) -> float:
    from pipeclaw.backend.evaluator import EvaluationProfile, evaluate

    rollout = {
        "sample_id": info.get("sample_id") or "",
        "scenario_id": info.get("scenario_id") or "",
        "scenario_type": "openclaw",
        "tool_calls": list(info.get("tool_calls") or []),
        "tool_outputs": list(info.get("tool_outputs") or []),
        "final_answer": completion or "",
        "trace_status": info.get("trace_status") or "",
        "json_errors": list(info.get("json_errors") or []),
        "messages": [],
        "turns": 0,
    }
    report = evaluate(
        rollout, profile=EvaluationProfile.AUTONOMOUS_ROLLOUT, reference=reference
    )
    return composite_reward(episode_stats(rollout), report.to_dict())


multi_turns["python_scenario_scheduler"] = PythonScenarioScheduler
orms["python_episode_reward"] = PythonEpisodeReward
