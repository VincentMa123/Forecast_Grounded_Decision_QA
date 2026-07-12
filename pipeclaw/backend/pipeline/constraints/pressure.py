from __future__ import annotations

from typing import Any, Dict, List

from ..rule_library import load_constraint_specs
from .common import (
    longest_episode_minutes,
    range_limits_for_variable,
    registry_index,
    run_specs,
    threshold_episodes,
    total_episode_minutes,
)


PRESSURE_SPECS = load_constraint_specs("pressure")
PRESSURE_WINDOW_SPEC = next(spec for spec in PRESSURE_SPECS if spec.name == "node_pressure_operating_window")


def run_pressure_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = run_specs(PRESSURE_SPECS, summaries, parsed_task)
    operating_window = checks[0]
    pressure_variables = operating_window["variables"]
    minimum_items = []
    maximum_items = []
    violation_nodes = []
    warning_nodes = []
    violation_episodes: Dict[str, List[Dict[str, Any]]] = {}
    warning_episodes: Dict[str, List[Dict[str, Any]]] = {}
    violation_duration_steps: Dict[str, int] = {}
    recovery_time_steps: Dict[str, int] = {}
    pressure_margins: Dict[str, Dict[str, float]] = {}
    simultaneous_warning_node_count = 0
    time_step_minutes = float(parsed_task.get("forecast_time_step_minutes") or 1.0)

    registry = registry_index(parsed_task)
    pressure_limits = {
        name: range_limits_for_variable(PRESSURE_WINDOW_SPEC, name, parsed_task)
        for name in pressure_variables
    }
    end_user_variables = [
        name
        for name in pressure_variables
        if registry.get(name, {}).get("equipment_type") == "node" or name.startswith("N_")
    ]
    end_user_series = {
        name: summaries.get(name, {}).get("predicted_values", [])
        for name in end_user_variables
    }
    for step_index in range(max((len(values) for values in end_user_series.values()), default=0)):
        near_lower_bound = sum(
            1
            for name, values in end_user_series.items()
            if step_index < len(values)
            and pressure_limits[name][2] <= values[step_index] < pressure_limits[name][0]
        )
        simultaneous_warning_node_count = max(simultaneous_warning_node_count, near_lower_bound)

    for variable in pressure_variables:
        warning_low, warning_high, fail_low, fail_high, _ = pressure_limits[variable]
        summary = summaries.get(variable, {})
        values = summary.get("predicted_values", [])
        labels = summary.get("prediction_labels", [])
        if not values:
            continue
        minimum_index = values.index(min(values))
        maximum_index = values.index(max(values))
        minimum_items.append((values[minimum_index], variable, minimum_index, labels[minimum_index] if minimum_index < len(labels) else None))
        maximum_items.append((values[maximum_index], variable, maximum_index, labels[maximum_index] if maximum_index < len(labels) else None))

        minimum = float(values[minimum_index])
        maximum = float(values[maximum_index])
        pressure_margins[variable] = {
            "warning_lower_margin": round(minimum - float(warning_low), 6),
            "warning_upper_margin": round(float(warning_high) - maximum, 6),
            "fail_lower_margin": round(minimum - float(fail_low), 6),
            "fail_upper_margin": round(float(fail_high) - maximum, 6),
        }

        violation_indices = [
            index
            for index, value in enumerate(values)
            if value < fail_low or value > fail_high
        ]
        warning_indices = [
            index
            for index, value in enumerate(values)
            if index not in violation_indices
            and (value < warning_low or value > warning_high)
        ]
        variable_violation_episodes = threshold_episodes(
            values,
            lambda value: value < fail_low or value > fail_high,
            labels,
            time_step_minutes,
        )
        variable_warning_episodes = threshold_episodes(
            values,
            lambda value: (
                fail_low <= value < warning_low
                or warning_high < value <= fail_high
            ),
            labels,
            time_step_minutes,
        )
        violation_episodes[variable] = variable_violation_episodes
        warning_episodes[variable] = variable_warning_episodes
        violation_duration_steps[variable] = len(violation_indices)
        if violation_indices:
            violation_nodes.append(variable)
            last_violation = violation_indices[-1]
            recovered = next(
                (
                    index
                    for index in range(last_violation + 1, len(values))
                    if warning_low <= values[index] <= warning_high
                ),
                None,
            )
            recovery_time_steps[variable] = (recovered - last_violation) if recovered is not None else len(values) - last_violation
        elif warning_indices:
            warning_nodes.append(variable)

    minimum_pressure = min(minimum_items, default=None)
    maximum_pressure = max(maximum_items, default=None)
    violation_duration_minutes = {
        variable: total_episode_minutes(episodes)
        for variable, episodes in violation_episodes.items()
    }
    maximum_violation_duration_minutes = max(
        (longest_episode_minutes(episodes) for episodes in violation_episodes.values()),
        default=0.0,
    )
    operating_window.update(
        {
            "minimum_pressure": _extreme_record(minimum_pressure),
            "maximum_pressure": _extreme_record(maximum_pressure),
            "pressure_violation_nodes": violation_nodes,
            "pressure_warning_nodes": warning_nodes,
            "pressure_margins": pressure_margins,
            "pressure_violation_episodes": violation_episodes,
            "pressure_warning_episodes": warning_episodes,
            "pressure_violation_duration_steps": violation_duration_steps,
            "pressure_violation_duration_minutes": violation_duration_minutes,
            "maximum_continuous_pressure_violation_minutes": round(maximum_violation_duration_minutes, 6),
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
