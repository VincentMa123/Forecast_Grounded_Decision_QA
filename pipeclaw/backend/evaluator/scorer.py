from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .teacher_quality import record_quality_issues
from .tool_evidence import attach_tool_arguments, classify_tool_evidence, requested_artifacts
from pipeline.forecast_registry_contract import authorize_forecast_registry


DEFAULT_MINIMUM_SCORE = 85.0
DEFAULT_MAX_RECORD_CHARS = 24_000
PIPEFORMER_TOOL = "run_pipeformer_forecast"
REQUIRED_RECORD_FIELDS = (
    "sample_id",
    "scenario_id",
    "scenario_type",
    "user_input",
    "parsed_task",
    "tool_calls",
    "tool_outputs",
    "prediction_summary",
    "constraint_check",
    "evidence",
    "risk_level",
    "manual_intervention_label",
    "dispatch_recommendation",
    "final_answer",
    "quality_flag",
)
CRITICAL_PIPEFORMER_CHECKS = {
    "parsed_task_correct",
    "forecast_tool_succeeded",
    "checkpoint_inference_used",
    "disturbance_applied_correctly",
    "forecast_horizon_consistent",
    "requested_constraints_executed",
    "verification_complete",
    "answer_grounded",
    "registry_search_precedes_forecast",
}


@dataclass(frozen=True)
class NativeEvaluationConfig:
    minimum_score: float = DEFAULT_MINIMUM_SCORE
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS


class NativeTraceEvaluator:
    """Evaluate records using one reusable native PipeClaw quality policy."""

    def __init__(self, config: Optional[NativeEvaluationConfig] = None) -> None:
        self.config = config or NativeEvaluationConfig()

    def evaluate(
        self,
        record: Dict[str, Any],
        *,
        hard_issues: Optional[Iterable[str]] = None,
        trace_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        return _evaluate_native_record(
            record,
            hard_issues=hard_issues,
            trace_status=trace_status,
            minimum_score=self.config.minimum_score,
            max_record_chars=self.config.max_record_chars,
        )

    @staticmethod
    def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return summarize_evaluations(results)

    @staticmethod
    def load(path: Path) -> List[Dict[str, Any]]:
        return load_records(path)


def evaluate_native_record(
    record: Dict[str, Any],
    *,
    hard_issues: Optional[Iterable[str]] = None,
    trace_status: Optional[str] = None,
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS,
) -> Dict[str, Any]:
    """Compatibility wrapper for callers that do not need a configured evaluator."""
    return _evaluate_native_record(
        record,
        hard_issues=hard_issues,
        trace_status=trace_status,
        minimum_score=minimum_score,
        max_record_chars=max_record_chars,
    )


def _evaluate_native_record(
    record: Dict[str, Any],
    *,
    hard_issues: Optional[Iterable[str]],
    trace_status: Optional[str],
    minimum_score: float,
    max_record_chars: int,
) -> Dict[str, Any]:
    if trace_status is None:
        trace_status = record.get("trace_status")
    if trace_status is None and {
        "trace_completed",
        "forecast_tool_succeeded",
    } & set(record.get("quality_failed_checks") or []):
        # Legacy records did not persist trace_status. Preserve an earlier
        # execution-status failure instead of silently upgrading it to pass.
        trace_status = "unknown"
    source_issues = (
        record_quality_issues(record) if hard_issues is None else list(hard_issues)
    )
    issues = list(dict.fromkeys(str(item) for item in source_issues))
    pipeformer_calls = [
        item
        for item in record.get("tool_calls") or []
        if item.get("name") == PIPEFORMER_TOOL
    ]
    if pipeformer_calls:
        checks = _pipeformer_checks(record, issues, trace_status, max_record_chars)
        critical_names = CRITICAL_PIPEFORMER_CHECKS
        profile = "native_pipeclaw_pipeformer"
    else:
        checks = _generic_checks(record, issues, trace_status, max_record_chars)
        critical_names = {"trace_completed", "answer_present", "tool_trajectory_valid", "answer_grounded"}
        profile = "native_pipeclaw_generic"

    score = round(sum(check["weight"] for check in checks if check["status"] == "pass"), 6)
    failed_critical = [
        check["name"]
        for check in checks
        if check["name"] in critical_names and check["status"] != "pass"
    ]
    failed_checks = [check["name"] for check in checks if check["status"] != "pass"]
    quality_flag = (
        "pass"
        if score >= float(minimum_score) and not issues and not failed_critical
        else "needs_review"
    )
    return {
        "profile": profile,
        "quality_flag": quality_flag,
        "quality_score": score,
        "minimum_pass_score": float(minimum_score),
        "failed_checks": failed_checks,
        "failed_critical_checks": failed_critical,
        "quality_issues": issues,
        "checks": checks,
    }


def summarize_evaluations(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [float(item["quality_score"]) for item in results]
    flags = Counter(str(item["quality_flag"]) for item in results)
    profiles = Counter(str(item["profile"]) for item in results)
    issues = Counter(
        str(issue)
        for item in results
        for issue in item.get("quality_issues") or []
    )
    return {
        "record_count": len(results),
        "pass_count": flags.get("pass", 0),
        "needs_review_count": flags.get("needs_review", 0),
        "average_quality_score": round(sum(scores) / len(scores), 6) if scores else None,
        "minimum_quality_score": min(scores) if scores else None,
        "maximum_quality_score": max(scores) if scores else None,
        "profile_counts": dict(sorted(profiles.items())),
        "quality_issue_counts": dict(sorted(issues.items())),
    }


def load_records(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [
            value
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
            for value in [json.loads(line)]
            if isinstance(value, dict)
        ]
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    raise TypeError(f"Teacher trace must contain a JSON object, list, or JSONL records: {path}")


def _pipeformer_checks(
    record: Dict[str, Any],
    issues: List[str],
    trace_status: Optional[str],
    max_record_chars: int,
) -> List[Dict[str, Any]]:
    tasks = _task_views(record.get("parsed_task") or {})
    referenced_call_ids = [
        str(task.get("tool_call_id") or "")
        for task in tasks
        if task.get("tool_call_id")
    ]
    all_tool_outputs = _pipeformer_outputs(
        record,
        referenced_call_ids=referenced_call_ids,
    )
    tool_outputs = [item for item in all_tool_outputs if item.get("success") is True]
    predictions = [dict(item.get("prediction") or {}) for item in tool_outputs]
    verifications = [dict(item.get("verification") or {}) for item in tool_outputs]

    parsed_ok = bool(tasks) and all(_task_is_complete(task) for task in tasks)
    successful = (
        trace_status in (None, "completed")
        and bool(tool_outputs)
        and len(tasks) == len(tool_outputs)
    )
    checkpoint_ok = successful and all(
        prediction.get("forecast_mode") == "checkpoint_inference"
        and bool((output.get("provenance") or {}).get("checkpoint_id"))
        for output, prediction in zip(tool_outputs, predictions)
    )
    disturbance_ok = successful and len(tasks) == len(tool_outputs) and all(
        _disturbance_was_applied(output, prediction, task)
        for output, prediction, task in zip(tool_outputs, predictions, tasks)
    )
    horizon_ok = successful and all(_horizon_is_consistent(prediction) for prediction in predictions)
    constraints_ok = bool(verifications) and all(
        _requested_constraints_executed(verification)
        for verification in verifications
    )
    complete_ok = bool(verifications) and all(
        verification.get("verification_complete") is True
        and not (verification.get("not_evaluated_rules") or [])
        for verification in verifications
    )
    record_chars, compact_ok, missing_fields = _record_contract(record, max_record_chars)
    registry_search_ok, registry_search_required = _registry_search_precedes_forecast(record, len(tool_outputs))
    return [
        _check("parsed_task_correct", 10.0, parsed_ok, task_count=len(tasks)),
        _check(
            "forecast_tool_succeeded",
            15.0,
            successful,
            successful_output_count=len(tool_outputs),
            trace_status=trace_status,
        ),
        _check("checkpoint_inference_used", 10.0, checkpoint_ok),
        _check("disturbance_applied_correctly", 15.0, disturbance_ok),
        _check("forecast_horizon_consistent", 10.0, horizon_ok),
        _check("requested_constraints_executed", 10.0, constraints_ok),
        _check("verification_complete", 10.0, complete_ok),
        _check(
            "registry_search_precedes_forecast",
            5.0,
            registry_search_ok,
            required=registry_search_required,
        ),
        _check("answer_grounded", 10.0, not issues, issues=issues),
        _check(
            "compact_record_contract",
            5.0,
            compact_ok,
            record_chars=record_chars,
            maximum_chars=max_record_chars,
            missing_fields=missing_fields,
        ),
    ]


def _registry_search_precedes_forecast(
    record: Dict[str, Any], successful_forecast_count: int
) -> tuple[bool, bool]:
    del successful_forecast_count
    calls = list(record.get("tool_calls") or [])
    forecast_indices = [
        index for index, call in enumerate(calls) if call.get("name") == PIPEFORMER_TOOL
    ]
    if not forecast_indices:
        return False, True
    outputs_by_call_id = {
        str(item.get("tool_call_id") or ""): item.get("output")
        for item in record.get("tool_outputs") or []
    }
    for forecast_index in forecast_indices:
        preceding_calls = []
        for call in calls[:forecast_index]:
            call_id = str(call.get("tool_call_id") or "")
            preceding_calls.append(
                {
                    "tool_call_id": call_id,
                    "name": call.get("name"),
                    "arguments": dict(call.get("arguments") or {}),
                    "output": outputs_by_call_id.get(call_id),
                }
            )
        authorization = authorize_forecast_registry(
            dict(calls[forecast_index].get("arguments") or {}),
            preceding_calls,
        )
        if not authorization["authorized"]:
            return False, True
    return True, True


def _generic_checks(
    record: Dict[str, Any],
    issues: List[str],
    trace_status: Optional[str],
    max_record_chars: int,
) -> List[Dict[str, Any]]:
    outputs = attach_tool_arguments(
        record.get("tool_outputs") or [],
        record.get("tool_calls") or [],
    )
    requested = requested_artifacts(str(record.get("user_input") or ""))
    assessments = [
        classify_tool_evidence(item, requested=requested)
        for item in outputs
    ]
    failed_indices = [
        index for index, assessment in enumerate(assessments)
        if not assessment.evidence_found
    ]
    successful_output_count = sum(assessment.evidence_found for assessment in assessments)
    requested_evidence_ok = "requested_evidence_not_retrieved" not in issues
    unresolved_failure = bool(failed_indices and not successful_output_count) or not requested_evidence_ok
    record_chars, compact_ok, missing_fields = _record_contract(record, max_record_chars)
    completed = trace_status in (None, "completed")
    return [
        _check("trace_completed", 25.0, completed, trace_status=trace_status),
        _check("answer_present", 25.0, bool(str(record.get("final_answer") or "").strip())),
        _check(
            "tool_trajectory_valid",
            20.0,
            not unresolved_failure,
            failed_tool_count=len(failed_indices),
            successful_tool_count=successful_output_count,
            requested_artifacts=list(requested),
            evidence_states=[item.state.value for item in assessments],
            evidence_reasons=[item.reason for item in assessments],
            recovered=bool(failed_indices) and not unresolved_failure,
        ),
        _check("answer_grounded", 20.0, not issues, issues=issues),
        _check(
            "compact_record_contract",
            10.0,
            compact_ok,
            record_chars=record_chars,
            maximum_chars=max_record_chars,
            missing_fields=missing_fields,
        ),
    ]


def _pipeformer_outputs(
    record: Dict[str, Any],
    *,
    referenced_call_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    call_ids = {
        str(item.get("tool_call_id") or "")
        for item in record.get("tool_calls") or []
        if item.get("name") == PIPEFORMER_TOOL
    }
    matching_outputs = [
        item
        for item in record.get("tool_outputs") or []
        if item.get("name") == PIPEFORMER_TOOL
        or str(item.get("tool_call_id") or "") in call_ids
    ]
    if referenced_call_ids:
        outputs_by_call_id = {
            str(item.get("tool_call_id") or ""): dict(item.get("output") or {})
            for item in matching_outputs
        }
        return [
            outputs_by_call_id[call_id]
            for call_id in referenced_call_ids
            if call_id in outputs_by_call_id
        ]
    return [dict(item.get("output") or {}) for item in matching_outputs]


def _task_views(parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = parsed_task.get("candidate_forecasts")
    if isinstance(candidates, list):
        return [dict(item) for item in candidates if isinstance(item, dict)]
    return [dict(parsed_task)] if parsed_task else []


def _task_is_complete(task: Dict[str, Any]) -> bool:
    variable = str(task.get("disturbance_variable") or "")
    boundary = dict(task.get("boundary_conditions") or {})
    has_change = (
        task.get("disturbance_magnitude_percent") is not None
        or variable in dict(boundary.get("setpoints") or {})
        or variable in dict(boundary.get("percentage_changes") or {})
    )
    resolved_output_count = task.get("resolved_output_variable_count")
    unresolved_attention = list(task.get("unresolved_attention_targets") or [])
    unresolved_outputs = list(task.get("unresolved_output_state_variables") or [])
    invalid_normalized_variables = list(task.get("invalid_normalized_variables") or [])
    return bool(
        task.get("case_id")
        and variable
        and has_change
        and task.get("forecast_horizon_minutes")
        and task.get("constraint_verification_types")
        and not unresolved_attention
        and not unresolved_outputs
        and not invalid_normalized_variables
        and (
            resolved_output_count is None
            or int(resolved_output_count or 0) > 0
        )
    )


def _disturbance_was_applied(
    output: Dict[str, Any],
    prediction: Dict[str, Any],
    task: Dict[str, Any],
) -> bool:
    variable = str(
        task.get("disturbance_variable")
        or prediction.get("disturbance_variable")
        or ""
    )
    if not variable or variable != str(prediction.get("disturbance_variable") or ""):
        return False

    expected = _expected_applied_disturbance(task, prediction, variable)
    if expected is None:
        return False
    applied = list((output.get("task_resolution") or {}).get("applied_boundary_conditions") or [])
    return any(
        str(item.get("variable") or "") == variable
        and item.get("mode") == expected["mode"]
        and _numbers_match(item.get("value"), expected["value"])
        for item in applied
    )


def _expected_applied_disturbance(
    task: Dict[str, Any],
    prediction: Dict[str, Any],
    variable: str,
) -> Optional[Dict[str, Any]]:
    boundary = dict(task.get("boundary_conditions") or {})
    setpoints = dict(boundary.get("setpoints") or {})
    percentage_changes = dict(boundary.get("percentage_changes") or {})
    has_setpoint = variable in setpoints
    has_percentage_change = variable in percentage_changes

    magnitude = task.get("disturbance_magnitude_percent")
    if magnitude is None:
        magnitude = prediction.get("disturbance_magnitude_percent")
    direction = str(
        task.get("disturbance_direction")
        or prediction.get("disturbance_direction")
        or ""
    )
    expected_percent = None
    if magnitude is not None:
        if direction not in {"up", "down"}:
            return None
        expected_percent = abs(float(magnitude)) * (1.0 if direction == "up" else -1.0)

    # An explicit setpoint is authoritative. The parser may also expose a
    # derived direction/magnitude (for example, closing a binary state is
    # represented as setpoint=0 and down 100%); that redundant description
    # must not invalidate the boundary condition that was actually applied.
    if has_setpoint:
        if has_percentage_change:
            return None
        return {"mode": "setpoint", "value": setpoints[variable]}
    if has_percentage_change:
        explicit_percent = percentage_changes[variable]
        if expected_percent is not None and not _numbers_match(explicit_percent, expected_percent):
            return None
        return {"mode": "percent_change", "value": explicit_percent}
    if expected_percent is not None:
        return {"mode": "percent_change", "value": expected_percent}
    return None


def _numbers_match(actual: Any, expected: Any) -> bool:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
    except (TypeError, ValueError):
        return False
    return abs(actual_value - expected_value) <= max(1e-6, abs(expected_value) * 1e-9)


def _horizon_is_consistent(prediction: Dict[str, Any]) -> bool:
    requested = prediction.get("forecast_horizon_minutes")
    actual = prediction.get("actual_forecast_horizon_minutes")
    window = dict(prediction.get("forecast_window") or {})
    step = float(window.get("time_step_minutes") or 0.0)
    if actual is None:
        return False
    if requested is None:
        steps = int(window.get("predict_row_count") or 0)
        return float(actual) > 0 and (not steps or abs(float(actual) - steps * step) <= max(step, 1e-6))
    return abs(float(requested) - float(actual)) <= max(step, 1e-6)


def _requested_constraints_executed(verification: Dict[str, Any]) -> bool:
    requested = set(verification.get("requested_categories") or [])
    category_status = set((verification.get("category_status") or {}).keys())
    return bool(requested and requested <= category_status and verification.get("rule_status"))


def _record_contract(record: Dict[str, Any], max_record_chars: int) -> tuple[int, bool, List[str]]:
    missing = [key for key in REQUIRED_RECORD_FIELDS if key not in record]
    size = len(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    # The master record is an audit artifact and intentionally retains the full
    # tool trajectory. Its size must not be judged against the compact SFT
    # projection limit; write_split_records enforces that limit after projection.
    return size, not missing, missing


def _check(name: str, weight: float, passed: bool, **details: Any) -> Dict[str, Any]:
    return {
        "name": name,
        "weight": weight,
        "status": "pass" if passed else "fail",
        **details,
    }
