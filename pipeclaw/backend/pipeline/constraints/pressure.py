from __future__ import annotations

from typing import Any, Dict, List

from ..rule_library import load_constraint_specs
from .common import run_specs


PRESSURE_SPECS = load_constraint_specs("pressure")


def run_pressure_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = run_specs(PRESSURE_SPECS, summaries, parsed_task)
    operating_window = checks[0]
    pressure_variables = operating_window["variables"]
    minimum_items = []
    maximum_items = []
    violation_nodes = []
    warning_nodes = []
    violation_duration_steps: Dict[str, int] = {}
    recovery_time_steps: Dict[str, int] = {}
    simultaneous_warning_node_count = 0

    end_user_variables = [name for name in pressure_variables if name.startswith("N_")]
    end_user_series = [summaries.get(name, {}).get("predicted_values", []) for name in end_user_variables]
    for step_index in range(max((len(values) for values in end_user_series), default=0)):
        near_lower_bound = sum(
            1
            for values in end_user_series
            if step_index < len(values) and PRESSURE_SPECS[0].fail_low <= values[step_index] < PRESSURE_SPECS[0].warning_low
        )
        simultaneous_warning_node_count = max(simultaneous_warning_node_count, near_lower_bound)

    for variable in pressure_variables:
        summary = summaries.get(variable, {})
        values = summary.get("predicted_values", [])
        labels = summary.get("prediction_labels", [])
        if not values:
            continue
        minimum_index = values.index(min(values))
        maximum_index = values.index(max(values))
        minimum_items.append((values[minimum_index], variable, minimum_index, labels[minimum_index] if minimum_index < len(labels) else None))
        maximum_items.append((values[maximum_index], variable, maximum_index, labels[maximum_index] if maximum_index < len(labels) else None))

        violation_indices = [
            index
            for index, value in enumerate(values)
            if value < PRESSURE_SPECS[0].fail_low or value > PRESSURE_SPECS[0].fail_high
        ]
        warning_indices = [
            index
            for index, value in enumerate(values)
            if index not in violation_indices
            and (value < PRESSURE_SPECS[0].warning_low or value > PRESSURE_SPECS[0].warning_high)
        ]
        if violation_indices:
            violation_nodes.append(variable)
            violation_duration_steps[variable] = len(violation_indices)
            last_violation = violation_indices[-1]
            recovered = next(
                (
                    index
                    for index in range(last_violation + 1, len(values))
                    if PRESSURE_SPECS[0].warning_low <= values[index] <= PRESSURE_SPECS[0].warning_high
                ),
                None,
            )
            recovery_time_steps[variable] = (recovered - last_violation) if recovered is not None else len(values) - last_violation
        elif warning_indices:
            warning_nodes.append(variable)

    minimum_pressure = min(minimum_items, default=None)
    maximum_pressure = max(maximum_items, default=None)
    time_step_minutes = float(parsed_task.get("forecast_time_step_minutes") or 1.0)
    operating_window.update(
        {
            "minimum_pressure": _extreme_record(minimum_pressure),
            "maximum_pressure": _extreme_record(maximum_pressure),
            "pressure_violation_nodes": violation_nodes,
            "pressure_warning_nodes": warning_nodes,
            "pressure_violation_duration_steps": violation_duration_steps,
            "pressure_violation_duration_minutes": {
                variable: round(steps * time_step_minutes, 6)
                for variable, steps in violation_duration_steps.items()
            },
            "pressure_recovery_time_steps": recovery_time_steps,
            "pressure_recovery_time_minutes": {
                variable: round(steps * time_step_minutes, 6)
                for variable, steps in recovery_time_steps.items()
            },
            "simultaneous_end_user_warning_node_count": simultaneous_warning_node_count,
        }
    )
    return checks


def _extreme_record(item: Any) -> Dict[str, Any] | None:
    if item is None:
        return None
    value, variable, step_index, timestamp = item
    return {
        "variable": variable,
        "value": value,
        "step_index": step_index,
        "timestamp": timestamp,
    }
