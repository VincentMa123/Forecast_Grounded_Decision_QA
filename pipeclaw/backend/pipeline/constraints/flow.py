from __future__ import annotations

from typing import Any, Dict, List

from ..rule_library import load_constraint_specs, load_rule_definition
from .common import (
    CATEGORY_DETAILS,
    contiguous_episodes,
    longest_episode_minutes,
    run_specs,
    status_from_threshold,
    threshold_limits_for_variable,
    threshold_episodes,
    total_episode_minutes,
    variables_for_selector,
    variables_for_spec,
)


FLOW_SPECS = load_constraint_specs("flow")
BALANCE_RULE = load_rule_definition("flow", "supply_demand_balance")
FLOW_RAMP_SPEC = next(spec for spec in FLOW_SPECS if spec.name == "flow_ramp_check")
FLOW_CAPACITY_SPEC = next(spec for spec in FLOW_SPECS if spec.name == "flow_capacity_check")
BOUNDARY_FLOW_SPEC = next(spec for spec in FLOW_SPECS if spec.name == "boundary_flow_change_rate")


def run_flow_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = run_specs(FLOW_SPECS, summaries, parsed_task)
    segment_variables = sorted(
        set(variables_for_spec(FLOW_RAMP_SPEC, summaries, parsed_task))
        | set(variables_for_spec(FLOW_CAPACITY_SPEC, summaries, parsed_task))
    )
    boundary_variables = variables_for_spec(BOUNDARY_FLOW_SPEC, summaries, parsed_task)
    abnormal_segments = _abnormal_flow_segments(summaries, parsed_task)
    flow_change_magnitude = {
        variable: summaries.get(variable, {}).get("max_abs_step_change")
        for variable in segment_variables
    }
    boundary_flow_change_rate = {
        variable: summaries.get(variable, {}).get("max_abs_step_change")
        for variable in boundary_variables
    }
    time_step_minutes = float(parsed_task.get("forecast_time_step_minutes") or 1.0)
    capacity_episodes = _capacity_excursion_episodes(summaries, time_step_minutes, parsed_task)
    ramp_events = _flow_ramp_events(summaries, parsed_task)
    balance_check = _supply_demand_balance_check(summaries, time_step_minutes, parsed_task)
    for check in checks:
        check["flow_change_magnitude"] = flow_change_magnitude
        check["boundary_flow_change_rate"] = boundary_flow_change_rate
        check["abnormal_flow_segments"] = abnormal_segments
        check["supply_demand_balance_status"] = balance_check["status"]
        if check["name"] == FLOW_RAMP_SPEC.name:
            check["flow_ramp_events"] = ramp_events
        elif check["name"] == FLOW_CAPACITY_SPEC.name:
            check["flow_capacity_excursion_episodes"] = capacity_episodes
    checks.append(balance_check)
    return checks


def _abnormal_flow_segments(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[str]:
    abnormal = set()
    for spec in (FLOW_RAMP_SPEC, FLOW_CAPACITY_SPEC):
        variables = variables_for_spec(spec, summaries, parsed_task)
        for variable in variables:
            value = summaries.get(variable, {}).get(spec.metric)
            if value is None:
                continue
            if status_from_threshold(float(value), spec.warning_threshold, spec.fail_threshold) != "pass":
                abnormal.add(variable)
    return sorted(abnormal)


def _capacity_excursion_episodes(
    summaries: Dict[str, Dict[str, Any]],
    time_step_minutes: float,
    parsed_task: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    result = {}
    variables = variables_for_spec(FLOW_CAPACITY_SPEC, summaries, parsed_task)
    for variable in variables:
        warning_threshold, fail_threshold, _ = threshold_limits_for_variable(
            FLOW_CAPACITY_SPEC, variable, parsed_task
        )
        values = summaries.get(variable, {}).get("predicted_values", [])
        labels = summaries.get(variable, {}).get("prediction_labels", [])
        warning = threshold_episodes(
            values,
            lambda value: abs(value) >= float(warning_threshold)
            and abs(value) < float(fail_threshold),
            labels,
            time_step_minutes,
        )
        failure = threshold_episodes(
            values,
            lambda value: abs(value) >= float(fail_threshold),
            labels,
            time_step_minutes,
        )
        result[variable] = {
            "warning_episodes": warning,
            "failure_episodes": failure,
            "total_out_of_limit_minutes": round(
                total_episode_minutes(warning) + total_episode_minutes(failure),
                6,
            ),
            "maximum_continuous_out_of_limit_minutes": max(
                longest_episode_minutes(warning),
                longest_episode_minutes(failure),
            ),
        }
    return result


def _flow_ramp_events(
    summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    result = {}
    variables = variables_for_spec(FLOW_RAMP_SPEC, summaries, parsed_task)
    for variable in variables:
        values = summaries.get(variable, {}).get("predicted_values", [])
        labels = summaries.get(variable, {}).get("prediction_labels", [])
        events = []
        for index in range(1, len(values)):
            change = float(values[index]) - float(values[index - 1])
            status = status_from_threshold(
                change,
                FLOW_RAMP_SPEC.warning_threshold,
                FLOW_RAMP_SPEC.fail_threshold,
            )
            if status != "pass":
                events.append(
                    {
                        "step_index": index,
                        "timestamp": labels[index] if index < len(labels) else None,
                        "change": round(change, 6),
                        "status": status,
                    }
                )
        result[variable] = events[:12]
    return result


def _supply_demand_balance_check(
    summaries: Dict[str, Dict[str, Any]],
    time_step_minutes: float,
    parsed_task: Dict[str, Any],
) -> Dict[str, Any]:
    supply_selector = BALANCE_RULE["supply_selector"]
    demand_selector = BALANCE_RULE["demand_selector"]
    supply_variables = variables_for_selector(summaries, supply_selector, parsed_task)
    demand_variables = variables_for_selector(summaries, demand_selector, parsed_task)
    variables = supply_variables + demand_variables
    usable_supply_variables = [
        name for name in supply_variables if summaries.get(name, {}).get("predicted_values")
    ]
    usable_demand_variables = [
        name for name in demand_variables if summaries.get(name, {}).get("predicted_values")
    ]
    series_lengths = [
        len(summaries[name]["predicted_values"])
        for name in usable_supply_variables + usable_demand_variables
    ]
    step_count = min(series_lengths, default=0) if usable_supply_variables and usable_demand_variables else 0
    gaps = []
    for index in range(step_count):
        supply = sum(
            summaries[name]["predicted_values"][index] for name in usable_supply_variables
        ) / len(usable_supply_variables)
        demand = sum(
            summaries[name]["predicted_values"][index] for name in usable_demand_variables
        ) / len(usable_demand_variables)
        gaps.append(supply - demand)

    usable_variables = usable_supply_variables + usable_demand_variables
    labels = summaries.get(usable_variables[0], {}).get("prediction_labels", []) if usable_variables else []
    widening_change_indices = [
        index
        for index in range(1, len(gaps))
        if abs(gaps[index]) > abs(gaps[index - 1])
    ]
    widening_episodes = contiguous_episodes(widening_change_indices, labels, time_step_minutes)

    max_gap = max((abs(value) for value in gaps), default=0.0)
    widening = bool(gaps) and abs(gaps[-1]) > abs(gaps[0])
    limits = BALANCE_RULE["limits"]
    warning_threshold = limits.get("warning_threshold")
    fail_threshold = limits.get("fail_threshold")
    threshold_status = status_from_threshold(max_gap, warning_threshold, fail_threshold)
    status = threshold_status if widening else "pass"
    if not gaps:
        status = "not_evaluated"
    flag = BALANCE_RULE["flags"].get(status)
    evaluated = {
        "metric": "max_abs_supply_demand_gap",
        "value": round(max_gap, 6),
        "initial_gap": round(gaps[0], 6) if gaps else None,
        "final_gap": round(gaps[-1], 6) if gaps else None,
        "gap_is_widening": widening,
        "widening_episodes": widening_episodes,
        "maximum_continuous_widening_minutes": longest_episode_minutes(widening_episodes),
        "warning_threshold": warning_threshold,
        "fail_threshold": fail_threshold,
        "status": status,
    }
    return {
        "name": BALANCE_RULE["rule_id"],
        "category": "flow",
        "status": status,
        "evaluation_status": "evaluated" if gaps else "not_evaluated",
        "flag": flag,
        "priority": int(BALANCE_RULE["priority"]),
        "variables": variables,
        "description": BALANCE_RULE["description"],
        "main_content": CATEGORY_DETAILS["flow"],
        "message": (
            "Supply-demand balance could not be evaluated because flow series were unavailable."
            if status == "not_evaluated"
            else "Supply-demand balance remains within the proxy tolerance."
            if status == "pass"
            else "The supply-demand gap requires review."
        ),
        "evaluated_values": [evaluated],
        "offending_values": [evaluated] if status in {"warning", "fail"} else [],
    }
