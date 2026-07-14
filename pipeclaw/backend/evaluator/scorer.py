from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


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
}


def evaluate_native_record(
    record: Dict[str, Any],
    *,
    hard_issues: Optional[Iterable[str]] = None,
    trace_status: Optional[str] = None,
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS,
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
    issues = list(dict.fromkeys(str(item) for item in (hard_issues or record.get("quality_issues") or [])))
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
    tool_outputs = _pipeformer_outputs(record)
    tasks = _task_views(record.get("parsed_task") or {})
    predictions = [dict(item.get("prediction") or {}) for item in tool_outputs]
    verifications = [dict(item.get("verification") or {}) for item in tool_outputs]

    parsed_ok = bool(tasks) and all(_task_is_complete(task) for task in tasks)
    successful = (
        trace_status in (None, "completed")
        and bool(tool_outputs)
        and all(item.get("success") is True for item in tool_outputs)
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
    return [
        _check("parsed_task_correct", 15.0, parsed_ok, task_count=len(tasks)),
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


def _generic_checks(
    record: Dict[str, Any],
    issues: List[str],
    trace_status: Optional[str],
    max_record_chars: int,
) -> List[Dict[str, Any]]:
    outputs = list(record.get("tool_outputs") or [])
    failed_indices = [
        index
        for index, item in enumerate(outputs)
        if _tool_output_failed(item.get("output"))
    ]
    successful_output_count = sum(
        1
        for item in outputs
        if not _tool_output_failed(item.get("output"))
    )
    unresolved_failure = bool(failed_indices and not successful_output_count)
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


def _pipeformer_outputs(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    call_ids = {
        str(item.get("tool_call_id") or "")
        for item in record.get("tool_calls") or []
        if item.get("name") == PIPEFORMER_TOOL
    }
    return [
        dict(item.get("output") or {})
        for item in record.get("tool_outputs") or []
        if item.get("name") == PIPEFORMER_TOOL
        or str(item.get("tool_call_id") or "") in call_ids
    ]


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
    return bool(
        task.get("case_id")
        and variable
        and has_change
        and task.get("forecast_horizon_minutes")
        and task.get("constraint_verification_types")
        and not (task.get("unresolved_attention_targets") or [])
        and not (task.get("unresolved_output_state_variables") or [])
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

    # Multiple representations are accepted only when they describe the same
    # percentage change. A setpoint mixed with a percentage is ambiguous.
    if has_setpoint:
        if has_percentage_change or expected_percent is not None:
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
    if requested is None or actual is None:
        return False
    return abs(float(requested) - float(actual)) <= max(step, 1e-6)


def _requested_constraints_executed(verification: Dict[str, Any]) -> bool:
    requested = set(verification.get("requested_categories") or [])
    category_status = set((verification.get("category_status") or {}).keys())
    return bool(requested and requested <= category_status and verification.get("rule_status"))


def _record_contract(record: Dict[str, Any], max_record_chars: int) -> tuple[int, bool, List[str]]:
    missing = [key for key in REQUIRED_RECORD_FIELDS if key not in record]
    size = len(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    return size, not missing and size <= max_record_chars, missing


def _check(name: str, weight: float, passed: bool, **details: Any) -> Dict[str, Any]:
    return {
        "name": name,
        "weight": weight,
        "status": "pass" if passed else "fail",
        **details,
    }


def _tool_output_failed(output: Any) -> bool:
    return isinstance(output, dict) and (
        output.get("success") is False
        or bool(output.get("error"))
        or output.get("exit_code") not in (None, 0)
    )
