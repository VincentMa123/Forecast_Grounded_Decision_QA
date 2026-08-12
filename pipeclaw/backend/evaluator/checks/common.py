"""Shared helpers for canonical evaluator checks."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..models import EvaluationContext, MetricResult
from ..profiles import get_profile_policy
from .assumptions import inferred_task_fields, prediction_view


PIPEFORMER_TOOL = "run_pipeformer_forecast"
CANONICAL_METRIC_NAMES = (
    "task_parsing",
    "assumption_consistency",
    "tool_call",
    "checkpoint_inference",
    "disturbance_application",
    "forecast_horizon",
    "constraint_execution",
    "constraint_judgment",
    "verification_completeness",
    "registry_ordering",
    "risk",
    "manual_intervention",
    "dispatch",
    "evidence_consistency",
    "answer_completeness",
    "json_validity",
    "artifact_evidence",
    "record_contract",
)


def sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def metric(
    context: EvaluationContext,
    name: str,
    *,
    applicable: bool,
    passed: bool = False,
    details: Mapping[str, Any] | None = None,
    teacher_variant: str = "pipeformer",
) -> MetricResult:
    policy = get_profile_policy(
        context.profile,
        teacher_variant=teacher_variant,
    ).metric(name)
    return MetricResult(
        name=name,
        applicable=bool(applicable),
        passed=bool(passed) if applicable else False,
        weight=policy.weight,
        critical=policy.critical,
        included_in_score=policy.included_in_score,
        details=dict(details or {}),
    )


def verification_view(output: Mapping[str, Any]) -> Mapping[str, Any]:
    value = output.get("verification") or output.get("constraint_check")
    return value if isinstance(value, Mapping) else {}


def normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [normalize(item) for item in value]
        try:
            return sorted(
                values,
                key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False),
            )
        except TypeError:
            return values
    if isinstance(value, float) and math.isfinite(value):
        return round(value, 6)
    return value


def numbers_match(actual: Any, expected: Any) -> bool:
    actual_value = _finite_number(actual)
    expected_value = _finite_number(expected)
    if actual_value is None or expected_value is None:
        return False
    return math.isclose(
        actual_value,
        expected_value,
        rel_tol=1e-6,
        abs_tol=1e-6,
    )


def _finite_number(value: Any) -> float | None:
    """Coerce an evaluator value only when it is a finite non-boolean number."""

    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def task_field_comparison(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> tuple[bool, list[str], list[str]]:
    """Compare oracle and student task fields.

    Returns ``(matched, mismatched_fields, matched_fields)`` where
    ``matched_fields`` lists compared fields whose values are equal.  Assumed
    fields are excluded from comparison entirely, so they appear in neither
    list.
    """

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
    assumed_fields = inferred_task_fields(expected)
    mismatches: list[str] = []
    matched_fields: list[str] = []
    for field in fields:
        if field not in expected or field in assumed_fields:
            continue
        if field not in actual:
            if field in required_fields:
                mismatches.append(field)
            continue
        left = normalize(expected[field])
        right = normalize(actual[field])
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            equal = math.isclose(
                float(left),
                float(right),
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
        else:
            equal = left == right
        if not equal:
            mismatches.append(field)
        else:
            matched_fields.append(field)
    return not mismatches, mismatches, matched_fields


def disturbance_was_applied(
    output: Mapping[str, Any],
    task: Mapping[str, Any],
) -> bool:
    prediction = prediction_view(output)
    variable = str(
        task.get("disturbance_variable")
        or prediction.get("disturbance_variable")
        or ""
    )
    if not variable or variable != str(prediction.get("disturbance_variable") or ""):
        return False
    boundary = mapping(task.get("boundary_conditions"))
    setpoints = mapping(boundary.get("setpoints"))
    percentages = mapping(boundary.get("percentage_changes"))
    assumed = inferred_task_fields(task)
    magnitude = task.get("disturbance_magnitude_percent")
    if "disturbance_magnitude_percent" in assumed or magnitude is None:
        magnitude = prediction.get("disturbance_magnitude_percent")
    direction = str(
        (
            prediction.get("disturbance_direction")
            if "disturbance_direction" in assumed
            else task.get("disturbance_direction")
        )
        or prediction.get("disturbance_direction")
        or ""
    ).casefold()
    # Binary status variables (``*:ST``) carry their target as a scalar
    # ``disturbance_setpoint`` argument; the runtime folds it into
    # ``boundary_conditions.setpoints`` itself and forces magnitude/direction to
    # ``None``, so the request never shows a nested setpoint.  Reading only the
    # nested form left this check structurally unable to pass a binary
    # disturbance, so it graded the argument spelling rather than the action.
    setpoint_argument = task.get("disturbance_setpoint")
    if isinstance(setpoint_argument, bool):
        # The runtime rejects booleans outright; accepting them here would let
        # ``true`` masquerade as the setpoint ``1``.
        return False
    expected_mode = ""
    expected_value: Any = None
    if variable in setpoints or setpoint_argument is not None:
        if variable in percentages:
            return False
        expected_mode = "setpoint"
        if setpoint_argument is None:
            expected_value = setpoints[variable]
        else:
            expected_value = setpoint_argument
            if variable in setpoints and not numbers_match(
                setpoints[variable],
                setpoint_argument,
            ):
                # The runtime raises on this conflict, so a trace exhibiting it
                # did not apply the disturbance the caller asked for.
                return False
    elif variable in percentages:
        expected_mode = "percent_change"
        expected_value = percentages[variable]
        if magnitude is not None:
            magnitude_value = _finite_number(magnitude)
            if magnitude_value is None:
                return False
            signed = abs(magnitude_value) * (1.0 if direction == "up" else -1.0)
            if direction not in {"up", "down"} or not numbers_match(
                expected_value,
                signed,
            ):
                return False
    elif magnitude is not None and direction in {"up", "down"}:
        expected_mode = "percent_change"
        magnitude_value = _finite_number(magnitude)
        if magnitude_value is None:
            return False
        expected_value = abs(magnitude_value) * (1.0 if direction == "up" else -1.0)
    else:
        return False
    applied = sequence(
        mapping(output.get("task_resolution")).get("applied_boundary_conditions")
    )
    return any(
        isinstance(item, Mapping)
        and str(item.get("variable") or "") == variable
        and str(item.get("mode") or "") == expected_mode
        and numbers_match(item.get("value", item.get("requested_value")), expected_value)
        for item in applied
    )


def checkpoint_inference_used(output: Mapping[str, Any]) -> bool:
    """Require both the checkpoint forecast mode and its compact provenance."""

    return (
        prediction_view(output).get("forecast_mode") == "checkpoint_inference"
        and bool(mapping(output.get("provenance")).get("checkpoint_id"))
    )


def horizon_is_consistent(output: Mapping[str, Any]) -> bool:
    prediction = prediction_view(output)
    requested = prediction.get("forecast_horizon_minutes")
    actual = prediction.get("actual_forecast_horizon_minutes")
    window = mapping(prediction.get("forecast_window"))
    try:
        step = float(window.get("time_step_minutes") or 0.0)
        actual_value = float(actual)
    except (TypeError, ValueError):
        return False
    if requested is None:
        try:
            steps = int(window.get("predict_row_count") or 0)
        except (TypeError, ValueError):
            return False
        return actual_value > 0 and (
            not steps
            or abs(actual_value - steps * step) <= max(step, 1e-6)
        )
    try:
        requested_value = float(requested)
    except (TypeError, ValueError):
        return False
    return abs(requested_value - actual_value) <= max(step, 1e-6)


def requested_constraints_executed(output: Mapping[str, Any]) -> bool:
    verification = verification_view(output)
    requested = {str(item) for item in sequence(verification.get("requested_categories"))}
    category_status = {
        str(key): str(value)
        for key, value in mapping(verification.get("category_status")).items()
    }
    if not requested or not requested <= set(category_status):
        return False
    # ``rule_status`` used to be required here, but compaction drops rule-level
    # detail while keeping ``category_status``, so requiring it measured how
    # verbose the record schema is rather than whether the constraints ran.  A
    # category carrying a real evaluated status is the execution evidence; a
    # requested category left ``not_evaluated`` is a genuine miss.
    return all(category_status[name] != "not_evaluated" for name in requested)


def verification_is_complete(output: Mapping[str, Any]) -> bool:
    verification = verification_view(output)
    return bool(
        verification
        and verification.get("verification_complete") is True
        and not sequence(verification.get("not_evaluated_rules"))
    )
