"""Pure per-episode statistics and reward scoring shared by rollout callers."""

from __future__ import annotations

import json
from typing import Any, Mapping


_RUN_TOOL = "run_command"
_PRECONDITION_REJECTS = (
    "sandbox_violation",
    "forecast_registry_precondition_failed",
    "decision_metric_used_as_output_state_variable",
    "duplicate_equivalent_forecast",
    "decision_policy_source_not_in_current_user_request",
)
_ERROR_CODES = (
    "python_syntax_error",
    "sandbox_violation",
    "tool_arguments_schema_invalid",
    "duplicate_failed_tool_call",
    "invalid_arguments",
    "unknown_tool",
    "tool_not_allowed",
    *_PRECONDITION_REJECTS,
)
_ABORT_STATUSES = frozenset(
    {
        "max_turns_exceeded",
        "max_completion_length",
        "empty_response",
        "generation_error",
    }
)


def episode_stats(rollout: Mapping[str, Any]) -> dict[str, Any]:
    """Rule-based per-episode stats used by the gate and the GRPO reward."""
    errors = {code: 0 for code in _ERROR_CODES}
    first_run: Mapping[str, Any] | None = None
    tool_outputs = rollout.get("tool_outputs") or []
    for output in tool_outputs:
        payload = output.get("output")
        if not isinstance(payload, Mapping):
            continue
        code = payload.get("error_code")
        if code in errors:
            errors[code] += 1
        if output.get("name") == _RUN_TOOL and first_run is None:
            first_run = payload
    first_run_compile = bool(
        first_run
        and first_run.get("exit_code") is not None
        and first_run.get("error_code") != "python_syntax_error"
    )
    calls = rollout.get("tool_calls") or []
    failed_signatures: dict[str, int] = {}
    success_signatures: dict[str, int] = {}
    for call, output in zip(calls, tool_outputs):
        payload = output.get("output") if isinstance(output, Mapping) else None
        succeeded = call.get("execution_success") is not False and (
            not isinstance(payload, Mapping)
            or payload.get("success", True) is not False
        )
        signature = json.dumps(
            [call.get("name"), call.get("arguments")], sort_keys=True, default=str
        )
        pool = success_signatures if succeeded else failed_signatures
        pool[signature] = pool.get(signature, 0) + 1
    thrash = sum(count - 1 for count in failed_signatures.values() if count > 1)
    duplicate_success = sum(
        count - 1 for count in success_signatures.values() if count > 1
    )
    return {
        "turns": rollout.get("turns", 0),
        "trace_status": rollout.get("trace_status", ""),
        "first_run_compile": first_run_compile,
        "first_run_exit0": first_run is not None and first_run.get("exit_code") == 0,
        "error_counts": errors,
        "thrash_count": thrash,
        "duplicate_success_count": duplicate_success,
        # error_codes cover only sandbox/audit rejections; plain exit!=0 failures
        # arrive with error_code=None, so track failed calls directly.
        "failed_call_count": sum(failed_signatures.values()),
        "malformed_json": bool(rollout.get("json_errors")),
        "timeout_hit": rollout.get("trace_status") in _ABORT_STATUSES,
    }


def composite_reward(
    stats: Mapping[str, Any], report_fields: Mapping[str, Any]
) -> float:
    """Dense rule reward; mirrors the evaluator-facing GRPO reward."""
    reward = 0.0
    reward += 0.55 * float(report_fields.get("overall_score") or 0.0) / 100.0
    reward += 0.20 * (1.0 if report_fields.get("hard_gate_passed") else 0.0)
    reward += 0.10 * (1.0 if stats["first_run_compile"] else 0.0)
    reward += 0.05 * (1.0 if stats["first_run_exit0"] else 0.0)
    reward += 0.05 * (1.0 if report_fields.get("passed") else 0.0)
    reward -= 0.05 * (1.0 if stats["timeout_hit"] else 0.0)
    reward -= min(0.15, 0.05 * int(stats["thrash_count"]))
    reward -= 0.10 * (1.0 if stats["malformed_json"] else 0.0)
    # Repeated identical *successful* calls are free in rollout training but
    # rejected by the live audit layer; make them cost reward here too.
    reward -= min(0.06, 0.02 * int(stats.get("duplicate_success_count", 0)))
    # Rotated-arg retry loops dodge the identical-signature thrash signature;
    # count every failed call past the first (the first failure is free: an
    # honest one-shot error is information, not a loop).
    reward -= min(0.10, 0.05 * max(0, int(stats.get("failed_call_count", 0)) - 1))
    # A sandbox-policy attempt is the cheapest possible reject: the tool never
    # validates arguments, the failure costs only the free allowance. Its
    # precondition siblings (registry-less forecast, metric-guarded forecast,
    # duplicate forecast, decision-policy quote rejects) inherit the same
    # ladder — the bare attempt and the metric-padded dodge must BOTH lose to
    # stop-and-declare (AGENTS rule 3).
    reward -= 0.10 * float(
        any(stats.get("error_counts", {}).get(code) for code in _PRECONDITION_REJECTS)
    )
    # A recovery bonus (+0.05 for pass-with-any-failure) enabled the "1 junk
    # fail then pass outscores clean" exploit; ranking is already safe via the
    # penalty ladders above, so no bonus is emitted.
    return round(reward, 6)


__all__ = ["composite_reward", "episode_stats"]
