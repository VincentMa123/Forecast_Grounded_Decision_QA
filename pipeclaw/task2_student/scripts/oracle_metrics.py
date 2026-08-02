"""Oracle-based metrics for autonomous Task 2 rollouts.

The teacher trace is used as an oracle, not as an additional prompt.  Metrics are
semantic and denominator-aware so an inapplicable metric is reported as such rather
than silently lowering the aggregate score.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


def _first_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _task_views(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    parsed = source.get("parsed_task")
    candidates = source.get("candidate_forecasts")
    if not candidates and isinstance(parsed, Mapping):
        candidates = parsed.get("candidate_forecasts")
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)) and candidates:
        return [dict(item) for item in candidates if isinstance(item, Mapping)]
    if isinstance(parsed, Mapping) and parsed:
        return [dict(parsed)]
    return []


def _output_views(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    outputs = source.get("tool_outputs")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        return []
    views: list[dict[str, Any]] = []
    forecast_views: list[dict[str, Any]] = []
    for item in outputs:
        if not isinstance(item, Mapping):
            continue
        output = item.get("output", item)
        if isinstance(output, Mapping):
            view = dict(output)
            views.append(view)
            if item.get("name") == "run_pipeformer_forecast":
                forecast_views.append(view)
    return forecast_views or views


def _verification(output: Mapping[str, Any]) -> Mapping[str, Any]:
    verification = output.get("verification") or output.get("constraint_check")
    return verification if isinstance(verification, Mapping) else {}


def _normalise_scalar(value: Any) -> Any:
    if isinstance(value, float) and math.isfinite(value):
        return round(value, 6)
    return value


def _normalise_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalise_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set)):
        values = [_normalise_value(item) for item in value]
        try:
            return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
        except TypeError:
            return values
    return _normalise_scalar(value)


def _extract_required_constraints(tasks: Sequence[Mapping[str, Any]], outputs: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []

    def add(value: Any) -> None:
        value = str(value)
        if value not in values:
            values.append(value)

    for task in tasks:
        for key in ("constraint_verification_types", "required_constraints", "constraint_types"):
            item = task.get(key)
            if isinstance(item, str):
                add(item)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                for value in item:
                    add(value)
    for output in outputs:
        verification = _verification(output)
        statuses = verification.get("category_status")
        if isinstance(statuses, Mapping):
            for key in statuses:
                add(key)
    return values


def build_oracle(source: Mapping[str, Any]) -> dict[str, Any]:
    """Extract canonical targets and labels from one teacher source record."""

    tasks = _task_views(source)
    outputs = _output_views(source)
    first_output = outputs[0] if outputs else {}
    verification = _verification(first_output)
    tool_calls = source.get("tool_calls")
    tool_calls = tool_calls if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes)) else []
    forecast_tasks = [
        dict(item.get("arguments") or {})
        for item in tool_calls
        if isinstance(item, Mapping)
        and item.get("name") == "run_pipeformer_forecast"
        and isinstance(item.get("arguments"), Mapping)
    ]
    teacher_tool_names = [
        str(item.get("name"))
        for item in tool_calls
        if isinstance(item, Mapping) and item.get("name")
    ]
    evidence = source.get("evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    decision_summary = source.get("decision_summary")
    if not isinstance(decision_summary, Mapping):
        decision_summary = {}
    canonical_tasks = forecast_tasks or tasks
    task = dict(canonical_tasks[0]) if canonical_tasks else {}
    # Keep only canonical task fields in the compact oracle while retaining any
    # source-specific fields under ``task`` for future metrics.
    return {
        "task": _normalise_value(task),
        "tasks": [_normalise_value(item) for item in canonical_tasks],
        "required_constraints": _extract_required_constraints(tasks, outputs),
        "risk_level": verification.get("risk_level", first_output.get("risk_level")),
        "manual_intervention_label": verification.get(
            "human_intervention_label", first_output.get("human_intervention_label")
        ),
        "dispatch_recommendation": verification.get(
            "dispatch_recommendation", first_output.get("dispatch_recommendation")
        ),
        "verified_evidence": _normalise_value(dict(evidence)),
        "teacher_tool_names": teacher_tool_names,
        "has_tool_target": bool(teacher_tool_names),
        "decision_summary": _normalise_value(dict(decision_summary)),
    }


def _metric(applicable: bool, record_pass: bool = False, **details: Any) -> dict[str, Any]:
    return {"applicable": bool(applicable), "record_pass": bool(record_pass) if applicable else False, **details}


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, Mapping):
        result: list[float] = []
        for item in value.values():
            result.extend(_numeric_values(item))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for item in value:
            result.extend(_numeric_values(item))
        return result
    return []


_NUMERIC_CLAIM = re.compile(
    r"(?<![A-Za-z0-9_])[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9_])"
)


def _extract_numeric_claims(text: str) -> list[float]:
    claims: list[float] = []
    for match in _NUMERIC_CLAIM.finditer(text):
        # Ignore ordered-list markers such as ``1. ...``; they are structure,
        # not factual numeric claims.
        remainder = text[match.end() :]
        if remainder[:1] in {".", ")"} and (len(remainder) == 1 or remainder[1].isspace()):
            continue
        try:
            claims.append(float(match.group(0).replace("−", "-")))
        except ValueError:
            continue
    return claims


def _source_evidence_numbers(source: Mapping[str, Any], oracle: Mapping[str, Any]) -> list[float]:
    evidence = oracle.get("verified_evidence", {})
    numbers = _numeric_values(evidence)
    # Successful teacher tool outputs are verified evidence too.  Include their
    # structured numeric values so a faithful teacher answer is self-consistent
    # even when a value is reported in a constraint/forecast subsection rather
    # than copied into the compact top-level evidence object.
    for output in _output_views(source):
        numbers.extend(_numeric_values(output))
    state_before = source.get("state_before")
    if isinstance(state_before, Mapping):
        numbers.extend(_numeric_values(state_before.get("verified_evidence")))
    for task in oracle.get("tasks", []):
        numbers.extend(_numeric_values(task))
    # De-duplicate while preserving a stable representation.
    unique: list[float] = []
    for number in numbers:
        if not any(math.isclose(number, existing, rel_tol=1e-6, abs_tol=1e-6) for existing in unique):
            unique.append(number)
    return unique


def _student_calls(rollout: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calls = rollout.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        return []
    return [item for item in calls if isinstance(item, Mapping)]


def _student_outputs(rollout: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outputs = rollout.get("tool_outputs")
    if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
        return []
    result: list[Mapping[str, Any]] = []
    for item in outputs:
        if not isinstance(item, Mapping):
            continue
        output = item.get("output", item)
        if isinstance(output, Mapping):
            result.append(output)
    return result


def _first_successful_output(rollout: Mapping[str, Any]) -> Mapping[str, Any]:
    raw_outputs = rollout.get("tool_outputs")
    if isinstance(raw_outputs, Sequence) and not isinstance(raw_outputs, (str, bytes)):
        preferred: list[Mapping[str, Any]] = []
        fallback: list[Mapping[str, Any]] = []
        for item in raw_outputs:
            if not isinstance(item, Mapping):
                continue
            output = item.get("output", item)
            if not isinstance(output, Mapping) or output.get("success", True) is False:
                continue
            if item.get("name") == "run_pipeformer_forecast" or any(
                key in output for key in ("verification", "constraint_check", "prediction", "prediction_summary")
            ):
                preferred.append(output)
            else:
                fallback.append(output)
        if preferred:
            return preferred[0]
        if fallback:
            return fallback[0]
    for output in _student_outputs(rollout):
        if output.get("success", True) is not False:
            return output
    return {}


def _task_match(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> tuple[bool, list[str]]:
    # These fields determine the forecast request.  Additional generated fields
    # (timestamps, internal IDs, etc.) are intentionally ignored.
    fields = (
        "case_id",
        "current_operating_condition_number",
        "disturbance_variable",
        "disturbance_direction",
        "disturbance_magnitude_percent",
        "forecast_horizon_minutes",
        "task_type",
        "constraint_verification_types",
        "required_constraints",
    )
    required_fields = {"case_id", "disturbance_variable"}
    mismatches: list[str] = []
    for field in fields:
        if field not in expected:
            continue
        if field not in actual:
            # The parsed task contains classifier metadata that need not be a
            # function argument.  Core identity fields remain mandatory; other
            # fields are compared whenever the student emits them.
            if field in required_fields:
                mismatches.append(field)
            continue
        left = _normalise_value(expected[field])
        right = _normalise_value(actual[field])
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            equal = math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-5)
        else:
            equal = left == right
        if not equal:
            mismatches.append(field)
    return not mismatches, mismatches


def _student_task(rollout: Mapping[str, Any]) -> Mapping[str, Any]:
    calls = _student_calls(rollout)
    forecast_calls = [call for call in calls if call.get("name") == "run_pipeformer_forecast"]
    for call in forecast_calls:
        arguments = call.get("arguments")
        if isinstance(arguments, Mapping):
            return arguments
    output = _first_successful_output(rollout)
    prediction = output.get("prediction") or output.get("prediction_summary")
    return prediction if isinstance(prediction, Mapping) else {}


def _student_tasks(rollout: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tasks: list[Mapping[str, Any]] = []
    for call in _student_calls(rollout):
        if call.get("name") != "run_pipeformer_forecast":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, Mapping):
            tasks.append(arguments)
    if tasks:
        return tasks
    task = _student_task(rollout)
    return [task] if task else []


def _all_category_statuses(output: Mapping[str, Any]) -> Mapping[str, Any]:
    verification = _verification(output)
    statuses = verification.get("category_status")
    return statuses if isinstance(statuses, Mapping) else {}


def _label(output: Mapping[str, Any], key: str) -> Any:
    verification = _verification(output)
    if key == "human_intervention_label":
        return verification.get(key, output.get(key, output.get("manual_intervention_label")))
    return verification.get(key, output.get(key))


def evaluate_rollout(source: Mapping[str, Any], rollout: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Score one rollout against the semantic oracle extracted from its source."""

    oracle = build_oracle(source)
    expected_tasks = [task for task in oracle.get("tasks", []) if isinstance(task, Mapping)]
    expected_task = oracle.get("task", {})
    expected_task_list = expected_tasks or ([expected_task] if isinstance(expected_task, Mapping) and expected_task else [])
    actual_tasks = _student_tasks(rollout)
    task_applicable = bool(expected_task_list)
    matched_count = 0
    all_mismatches: list[str] = []
    unmatched_expected = list(expected_task_list)
    for actual_task in actual_tasks:
        match_index = next(
            (
                index
                for index, expected in enumerate(unmatched_expected)
                if _task_match(expected, actual_task)[0]
            ),
            None,
        )
        if match_index is not None:
            matched_count += 1
            unmatched_expected.pop(match_index)
        elif unmatched_expected:
            _, mismatches = _task_match(unmatched_expected[0], actual_task)
            all_mismatches.extend(mismatches)
    task_pass = task_applicable and matched_count == len(expected_task_list)
    task_metric = _metric(
        task_applicable,
        task_pass,
        expected_task_count=len(expected_task_list),
        matched_task_count=matched_count,
        mismatched_fields=sorted(set(all_mismatches)),
        matched_fields=[field for field in expected_task if field not in all_mismatches],
    )

    calls = _student_calls(rollout)
    teacher_names = set(oracle.get("teacher_tool_names", []))
    call_applicable = bool(teacher_names)
    valid_calls = [
        call
        for call in calls
        if call.get("schema_valid") is not False and call.get("execution_success") is not False
    ]
    required_tool_names: set[str] = set()
    if "run_pipeformer_forecast" in teacher_names:
        required_tool_names.add("run_pipeformer_forecast")
    if "set_decision_policy" in teacher_names:
        required_tool_names.add("set_decision_policy")
    if not required_tool_names:
        required_tool_names = set(teacher_names)
    emitted_valid_names = {str(call.get("name")) for call in valid_calls if call.get("name")}
    tool_pass = bool(valid_calls) and all(name in emitted_valid_names for name in required_tool_names) and len(valid_calls) == len(calls)
    tool_metric = _metric(
        call_applicable,
        tool_pass,
        expected_tool_names=sorted(teacher_names),
        required_tool_names=sorted(required_tool_names),
        emitted_tool_names=[str(call.get("name")) for call in calls if call.get("name")],
    )

    student_output = _first_successful_output(rollout)
    expected_constraints = set(oracle.get("required_constraints", []))
    actual_statuses = _all_category_statuses(student_output)
    expected_statuses: Mapping[str, Any] = {}
    source_outputs = _output_views(source)
    if source_outputs:
        expected_statuses = _all_category_statuses(source_outputs[0])
    constraint_applicable = bool(expected_constraints or expected_statuses)
    constraint_pass = bool(actual_statuses) and all(
        str(actual_statuses.get(key)) == str(value) for key, value in expected_statuses.items()
    )
    if constraint_applicable and expected_constraints:
        constraint_pass = constraint_pass and expected_constraints.issubset({str(key) for key in actual_statuses})
    constraint_metric = _metric(
        constraint_applicable,
        constraint_pass,
        expected_constraints=sorted(expected_constraints),
        actual_constraints=sorted(str(key) for key in actual_statuses),
    )

    expected_risk = oracle.get("risk_level")
    risk_applicable = expected_risk is not None
    actual_risk = _label(student_output, "risk_level")
    risk_metric = _metric(risk_applicable, risk_applicable and actual_risk == expected_risk, expected=expected_risk, actual=actual_risk)

    expected_intervention = oracle.get("manual_intervention_label")
    intervention_applicable = expected_intervention is not None
    actual_intervention = _label(student_output, "human_intervention_label")
    intervention_metric = _metric(
        intervention_applicable,
        intervention_applicable and actual_intervention == expected_intervention,
        expected=expected_intervention,
        actual=actual_intervention,
    )

    expected_dispatch = oracle.get("dispatch_recommendation")
    dispatch_applicable = expected_dispatch is not None
    actual_dispatch = _label(student_output, "dispatch_recommendation")
    dispatch_metric = _metric(
        dispatch_applicable,
        dispatch_applicable and actual_dispatch == expected_dispatch,
        expected=expected_dispatch,
        actual=actual_dispatch,
    )

    final_answer = rollout.get("final_answer", "")
    final_answer = final_answer if isinstance(final_answer, str) else str(final_answer)
    claimed_numbers = _extract_numeric_claims(final_answer)
    evidence_numbers = _source_evidence_numbers(source, oracle)
    unsupported = [
        value for value in claimed_numbers
        if not any(math.isclose(value, evidence, rel_tol=1e-5, abs_tol=1e-5) for evidence in evidence_numbers)
    ]
    evidence_applicable = bool(evidence_numbers) and bool(final_answer.strip())
    evidence_metric = _metric(
        evidence_applicable,
        evidence_applicable and not unsupported,
        claimed_numeric_values=claimed_numbers,
        unsupported_numeric_values=unsupported,
    )
    hallucination_metric = _metric(
        evidence_applicable,
        evidence_applicable and not unsupported,
        unsupported_numeric_values=unsupported,
    )

    answer_applicable = bool(str(source.get("final_answer") or "").strip()) or bool(final_answer.strip())
    answer_metric = _metric(answer_applicable, answer_applicable and bool(final_answer.strip()), answer_length=len(final_answer))

    json_errors = rollout.get("json_errors")
    json_errors = json_errors if isinstance(json_errors, Sequence) and not isinstance(json_errors, (str, bytes)) else []
    json_applicable = bool(teacher_names)
    json_metric = _metric(
        json_applicable,
        json_applicable and bool(calls) and not json_errors and all(call.get("schema_valid") is not False for call in calls),
        error_count=len(json_errors),
    )

    return {
        "task_parsing": task_metric,
        "tool_call": tool_metric,
        "constraint_judgment": constraint_metric,
        "risk": risk_metric,
        "manual_intervention": intervention_metric,
        "dispatch": dispatch_metric,
        "evidence_consistency": evidence_metric,
        "hallucination": hallucination_metric,
        "answer_completeness": answer_metric,
        "json_validity": json_metric,
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-record metric objects with explicit applicability counts."""

    metric_names: set[str] = set()
    for result in results:
        metrics = result.get("metrics", result)
        if isinstance(metrics, Mapping):
            metric_names.update(str(key) for key in metrics)
    summary: dict[str, Any] = {"record_count": len(results), "metrics": {}}
    for name in sorted(metric_names):
        numerator = denominator = 0
        for result in results:
            metrics = result.get("metrics", result)
            metric = metrics.get(name) if isinstance(metrics, Mapping) else None
            if not isinstance(metric, Mapping) or not metric.get("applicable", False):
                continue
            denominator += 1
            if metric.get("record_pass", False):
                numerator += 1
        summary["metrics"][name] = {
            "numerator": numerator,
            "denominator": denominator,
            "rate": (numerator / denominator) if denominator else None,
            "pass_rate": (numerator / denominator) if denominator else None,
            "failure_rate": ((denominator - numerator) / denominator) if denominator else None,
            "status": "ok" if denominator else "not_applicable",
        }
    return summary
