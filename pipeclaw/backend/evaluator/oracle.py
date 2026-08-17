from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .checks.common import mapping, normalize, sequence, verification_view
from .models import EvaluationInputError


def _task_views(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    parsed = source.get("parsed_task")
    candidates = source.get("candidate_forecasts")
    if not candidates and isinstance(parsed, Mapping):
        candidates = parsed.get("candidate_forecasts")
    if sequence(candidates):
        return [dict(item) for item in sequence(candidates) if isinstance(item, Mapping)]
    return [dict(parsed)] if isinstance(parsed, Mapping) and parsed else []


def _merge_task_fields(
    parsed_tasks: Sequence[Mapping[str, Any]],
    forecast_tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fill partial forecast arguments from parsed task defaults.

    Teacher traces may preserve only the fields emitted in a forecast tool call.
    The parsed task remains the authoritative source for omitted fields, while
    every explicit call value (including ``None``) wins over that default.
    """

    if not forecast_tasks:
        return [dict(task) for task in parsed_tasks]

    merged_tasks: list[dict[str, Any]] = []
    for index, call in enumerate(forecast_tasks):
        defaults = (
            parsed_tasks[index]
            if index < len(parsed_tasks)
            else parsed_tasks[0]
            if parsed_tasks
            else {}
        )
        merged = dict(defaults)
        merged.update(call)

        default_boundary = mapping(defaults.get("boundary_conditions"))
        call_boundary = call.get("boundary_conditions")
        if isinstance(call_boundary, Mapping):
            boundary = dict(default_boundary)
            for key, value in call_boundary.items():
                if key in {"percentage_changes", "setpoints"} and isinstance(
                    value, Mapping
                ):
                    boundary[key] = {
                        **dict(mapping(default_boundary.get(key))),
                        **dict(value),
                    }
                else:
                    boundary[key] = value
            merged["boundary_conditions"] = boundary

        # Binary disturbances commonly store the target only in the parsed
        # boundary setpoints.  Expose the canonical scalar when the call omitted
        # it, without overriding an explicit call value.
        if "disturbance_setpoint" not in call and "disturbance_setpoint" not in merged:
            variable = merged.get("disturbance_variable")
            setpoints = mapping(mapping(merged.get("boundary_conditions")).get("setpoints"))
            if variable in setpoints:
                merged["disturbance_setpoint"] = setpoints[variable]

        merged_tasks.append(merged)
    return merged_tasks


def _output_views(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    forecasts: list[dict[str, Any]] = []
    for item in sequence(source.get("tool_outputs")):
        if not isinstance(item, Mapping):
            continue
        output = item.get("output", item)
        if not isinstance(output, Mapping):
            continue
        view = dict(output)
        views.append(view)
        if item.get("name") == "run_pipeformer_forecast":
            forecasts.append(view)
    return forecasts or views


def _required_constraints(
    tasks: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> list[str]:
    values: list[str] = []

    def add(value: Any) -> None:
        text = str(value)
        if text not in values:
            values.append(text)

    for task in tasks:
        for key in (
            "constraint_verification_types",
            "required_constraints",
            "constraint_types",
        ):
            value = task.get(key)
            if isinstance(value, str):
                add(value)
            else:
                for item in sequence(value):
                    add(item)
    for output in outputs:
        statuses = verification_view(output).get("category_status")
        if isinstance(statuses, Mapping):
            for key in statuses:
                add(key)
    return values


def _inherited_assumption(source: Mapping[str, Any]) -> Mapping[str, Any] | None:
    state_before = mapping(source.get("state_before"))
    scope = mapping(state_before.get("scope"))
    disturbance = mapping(scope.get("disturbance"))
    if not disturbance.get("variable"):
        return None
    source_name = str(disturbance.get("source") or "").casefold()
    recent_text = json.dumps(source.get("recent_turns"), ensure_ascii=False, default=str).casefold()
    markers = (
        "llm provisional",
        "llm_assumption",
        "assumption source",
        "临时假设",
        "暂按",
    )
    if source_name not in {"llm_assumption", "provisional", "assumed"} and not any(
        marker in recent_text for marker in markers
    ):
        return None
    return {
        "source": "llm_assumption",
        "assumed_fields": ["direction", "magnitude_percent"],
        "statement": f"Inherited provisional disturbance for {disturbance['variable']}.",
    }


def _with_inherited_assumptions(
    source: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    copied = [dict(task) for task in tasks]
    marker = _inherited_assumption(source)
    if marker is None:
        return copied
    disturbance = mapping(
        mapping(mapping(source.get("state_before")).get("scope")).get("disturbance")
    )
    inherited_variable = disturbance.get("variable")
    for task in copied:
        if task.get("disturbance_assumption"):
            continue
        if inherited_variable and task.get("disturbance_variable") not in (
            None,
            inherited_variable,
        ):
            continue
        task["disturbance_assumption"] = dict(marker)
    return copied


def build_teacher_oracle(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Extract canonical targets and labels from a held-out teacher record."""

    if not isinstance(reference, Mapping):
        raise EvaluationInputError("Teacher reference must be a mapping.")
    source = dict(reference)
    tasks = _task_views(source)
    outputs = _output_views(source)
    first_output = outputs[0] if outputs else {}
    verification = verification_view(first_output)
    tool_calls = [
        item
        for item in sequence(source.get("tool_calls"))
        if isinstance(item, Mapping)
    ]
    forecast_tasks = [
        dict(item["arguments"])
        for item in tool_calls
        if item.get("name") == "run_pipeformer_forecast"
        and isinstance(item.get("arguments"), Mapping)
    ]
    canonical_tasks = _with_inherited_assumptions(
        source,
        _merge_task_fields(tasks, forecast_tasks),
    )
    task = canonical_tasks[0] if canonical_tasks else {}
    evidence = mapping(source.get("evidence"))
    decision_summary = mapping(source.get("decision_summary"))
    teacher_tool_names = [
        str(item.get("name"))
        for item in tool_calls
        if item.get("name")
    ]

    return {
        "task": normalize(task),
        "tasks": [normalize(item) for item in canonical_tasks],
        "required_constraints": _required_constraints(canonical_tasks, outputs),
        "risk_level": verification.get(
            "risk_level",
            first_output.get("risk_level", source.get("risk_level")),
        ),
        "manual_intervention_label": verification.get(
            "human_intervention_label",
            first_output.get(
                "human_intervention_label",
                first_output.get(
                    "manual_intervention_label",
                    source.get("manual_intervention_label"),
                ),
            ),
        ),
        "dispatch_recommendation": verification.get(
            "dispatch_recommendation",
            first_output.get(
                "dispatch_recommendation",
                source.get("dispatch_recommendation"),
            ),
        ),
        "verified_evidence": normalize(dict(evidence)),
        "teacher_tool_names": teacher_tool_names,
        "has_tool_target": bool(teacher_tool_names),
        "decision_summary": normalize(dict(decision_summary)),
    }
