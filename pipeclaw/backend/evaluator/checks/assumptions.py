from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

# Tolerance for "did the executed forecast actually apply the value we asked
# for" comparisons.  Shared by this module and ``common.numbers_match``; it is
# deliberately tighter than the 1e-5 used for task-field and CSV-row matching,
# which encode different semantics.
APPLIED_VALUE_REL_TOL = 1e-6
APPLIED_VALUE_ABS_TOL = 1e-6


_ASSUMED_FIELD_ALIASES = {
    "direction": "disturbance_direction",
    "disturbance_direction": "disturbance_direction",
    "magnitude": "disturbance_magnitude_percent",
    "magnitude_percent": "disturbance_magnitude_percent",
    "disturbance_magnitude": "disturbance_magnitude_percent",
    "disturbance_magnitude_percent": "disturbance_magnitude_percent",
}


def inferred_task_fields(task: Mapping[str, Any]) -> frozenset[str]:
    """Return canonical task fields marked as provisional assumptions."""

    assumption = task.get("disturbance_assumption")
    if isinstance(assumption, str):
        return (
            frozenset(
                {
                    "disturbance_direction",
                    "disturbance_magnitude_percent",
                }
            )
            if assumption.strip()
            else frozenset()
        )
    if not isinstance(assumption, Mapping):
        return frozenset()
    fields = assumption.get("assumed_fields")
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        return frozenset()
    return frozenset(
        canonical
        for field in fields
        if (canonical := _ASSUMED_FIELD_ALIASES.get(str(field).strip().casefold()))
    )


def prediction_view(forecast_output: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return either supported compact PipeFormer prediction shape."""

    prediction = forecast_output.get("prediction") or forecast_output.get(
        "prediction_summary"
    )
    return prediction if isinstance(prediction, Mapping) else forecast_output


def expected_applied_disturbance(
    actual_task: Mapping[str, Any],
    prediction: Mapping[str, Any],
    variable: str,
    *,
    assumed_fields: frozenset[str] | None = None,
) -> dict[str, Any] | None:
    """Return the signed percent change the runtime should have applied.

    Explicit task values win.  Provisionally assumed values fall back to the
    student's executed prediction, because an underspecified request lets the
    student choose its own direction and magnitude — that choice does not have
    to equal the teacher's sampled one.  ``assumed_fields`` comes from the
    teacher record when the caller already resolved it; otherwise it is read
    off ``actual_task``.  Returns ``None`` when the inputs cannot determine a
    valid signed magnitude.
    """

    assumed = (
        inferred_task_fields(actual_task)
        if assumed_fields is None
        else frozenset(assumed_fields)
    )
    direction = str(
        (
            prediction.get("disturbance_direction")
            if "disturbance_direction" in assumed
            else actual_task.get("disturbance_direction")
        )
        or prediction.get("disturbance_direction")
        or ""
    ).casefold()
    magnitude = (
        None
        if "disturbance_magnitude_percent" in assumed
        else actual_task.get("disturbance_magnitude_percent")
    )
    if magnitude is None:
        magnitude = prediction.get("disturbance_magnitude_percent")
    try:
        expected = abs(float(magnitude)) * (1.0 if direction == "up" else -1.0)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(expected) or not variable or direction not in {"up", "down"}:
        return None
    return {"mode": "percent_change", "value": expected}


def _applied_disturbance_matches(
    actual_task: Mapping[str, Any],
    prediction: Mapping[str, Any],
    forecast_output: Mapping[str, Any],
    assumed_fields: frozenset[str] | None = None,
) -> bool:
    resolution = forecast_output.get("task_resolution")
    if not isinstance(resolution, Mapping):
        return False
    applied = resolution.get("applied_boundary_conditions")
    if not isinstance(applied, Sequence) or isinstance(applied, (str, bytes)):
        return False

    variable = str(
        actual_task.get("disturbance_variable")
        or prediction.get("disturbance_variable")
        or ""
    )
    expected_disturbance = expected_applied_disturbance(
        actual_task,
        prediction,
        variable,
        assumed_fields=assumed_fields,
    )
    if expected_disturbance is None:
        return False
    expected = float(expected_disturbance["value"])

    for item in applied:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("variable") or "") != variable:
            continue
        if str(item.get("mode") or "").casefold() not in {
            "percent_change",
            "percentage_change",
        }:
            continue
        value = item.get("value", item.get("requested_value"))
        try:
            applied_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(applied_value) and math.isclose(
            applied_value,
            expected,
            rel_tol=APPLIED_VALUE_REL_TOL,
            abs_tol=APPLIED_VALUE_ABS_TOL,
        ):
            return True
    return False


def assumption_consistency(
    expected_tasks: Sequence[Mapping[str, Any]],
    actual_task: Mapping[str, Any],
    forecast_output: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Validate assumed values against the student's executed prediction."""

    assumed_fields = frozenset(
        field for task in expected_tasks for field in inferred_task_fields(task)
    )
    if not assumed_fields:
        return True, []

    prediction = prediction_view(forecast_output)
    mismatches: list[str] = []
    for field in sorted(assumed_fields):
        actual_value = actual_task.get(field)
        predicted_value = prediction.get(field)
        if actual_value is None:
            mismatches.append(field)
            continue
        if field == "disturbance_direction":
            if str(actual_value).casefold() not in {"up", "down"}:
                mismatches.append(field)
            elif (
                predicted_value is not None
                and str(predicted_value).casefold() != str(actual_value).casefold()
            ):
                mismatches.append(field)
            continue
        try:
            numeric_value = float(actual_value)
        except (TypeError, ValueError):
            mismatches.append(field)
            continue
        if not math.isfinite(numeric_value):
            mismatches.append(field)
            continue
        if predicted_value is not None:
            try:
                predicted_numeric = float(predicted_value)
            except (TypeError, ValueError):
                mismatches.append(field)
            else:
                if not math.isfinite(predicted_numeric) or not math.isclose(
                    numeric_value,
                    predicted_numeric,
                    rel_tol=APPLIED_VALUE_REL_TOL,
                    abs_tol=APPLIED_VALUE_ABS_TOL,
                ):
                    mismatches.append(field)
    if not mismatches and not _applied_disturbance_matches(
        actual_task,
        prediction,
        forecast_output,
        assumed_fields,
    ):
        mismatches.append("applied_boundary_conditions")
    return not mismatches, mismatches
