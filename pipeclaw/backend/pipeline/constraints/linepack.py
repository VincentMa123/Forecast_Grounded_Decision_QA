from __future__ import annotations

from typing import Any, Dict, List

from .rule_library import load_constraint_specs, load_rule_definition
from .common import (
    CATEGORY_DETAILS,
    contiguous_episodes,
    longest_episode_minutes,
    max_status,
    registry_index,
    run_specs,
    variables_for_selector,
)


LINEPACK_SPECS = load_constraint_specs("linepack")
LINEPACK_RECOVERY_RULE = load_rule_definition(
    "linepack", "linepack_decline_and_recovery"
)
LINEPACK_RESERVE_RULE = load_rule_definition(
    "linepack", "linepack_peak_shaving_reserve"
)


def run_linepack_checks(
    summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]
) -> List[Dict[str, Any]]:
    checks = run_specs(LINEPACK_SPECS, summaries, parsed_task)
    linepack_variables = checks[0]["variables"] if checks else []
    minimum_recovery_ratio = float(LINEPACK_RECOVERY_RULE["minimum_recovery_ratio"])
    minimum_items = []
    change_rates = {}
    recovery = {}
    decline_episodes = {}
    time_step_minutes = float(parsed_task.get("forecast_time_step_minutes") or 1.0)
    for variable in linepack_variables:
        summary = summaries.get(variable, {})
        values = summary.get("predicted_values", [])
        labels = summary.get("prediction_labels", [])
        if not values:
            continue
        minimum_index = values.index(min(values))
        minimum_items.append(
            (
                values[minimum_index],
                variable,
                minimum_index,
                labels[minimum_index] if minimum_index < len(labels) else None,
            )
        )
        decline = float(summary.get("max_decline_from_start") or 0.0)
        recovered = float(summary.get("recovery_from_minimum") or 0.0)
        recovery_target = (
            float(values[minimum_index]) + decline * minimum_recovery_ratio
        )
        recovered_index = (
            minimum_index
            if decline == 0
            else next(
                (
                    index
                    for index in range(minimum_index + 1, len(values))
                    if float(values[index]) >= recovery_target
                ),
                None,
            )
        )
        recovery_steps = (
            recovered_index - minimum_index
            if recovered_index is not None
            else max(0, len(values) - 1 - minimum_index)
        )
        decreasing_indices = [
            index
            for index in range(1, len(values))
            if float(values[index]) < float(values[index - 1])
        ]
        variable_decline_episodes = contiguous_episodes(
            decreasing_indices, labels, time_step_minutes
        )
        decline_episodes[variable] = variable_decline_episodes
        recovery[variable] = {
            "decline_from_start": round(decline, 6),
            "recovery_from_minimum": round(recovered, 6),
            "recovery_ratio": round(recovered / decline, 6) if decline > 0 else 1.0,
            "recovery_sufficient": decline == 0
            or recovered / decline >= minimum_recovery_ratio,
            "recovery_target": round(recovery_target, 6),
            "recovery_time_steps": recovery_steps,
            "recovery_time_minutes": round(recovery_steps * time_step_minutes, 6),
            "recovered_within_horizon": decline == 0 or recovered_index is not None,
            "maximum_continuous_decline_minutes": longest_episode_minutes(
                variable_decline_episodes
            ),
        }
        change_rates[variable] = summary.get("max_abs_step_change")

    minimum_linepack = min(minimum_items, default=None)
    minimum_record = None
    if minimum_linepack is not None:
        value, variable, step_index, timestamp = minimum_linepack
        minimum_record = {
            "variable": variable,
            "value": value,
            "step_index": step_index,
            "timestamp": timestamp,
        }

    insufficient_recovery = [
        {
            "variable": variable,
            "metric": "recovery_ratio",
            "value": item["recovery_ratio"],
            "status": "warning",
            "warning_threshold": minimum_recovery_ratio,
        }
        for variable, item in recovery.items()
        if not item["recovery_sufficient"]
    ]
    recovery_check = next(
        (
            check
            for check in checks
            if check["name"] == LINEPACK_RECOVERY_RULE["rule_id"]
        ),
        None,
    )
    if recovery_check is not None and insufficient_recovery:
        recovery_check["status"] = max_status([recovery_check["status"], "warning"])
        recovery_check["flag"] = LINEPACK_RECOVERY_RULE["flags"][
            recovery_check["status"]
        ]
        recovery_check["message"] = (
            f"{len(insufficient_recovery)} linepack variable(s) did not recover the configured minimum ratio."
        )
        recovery_check["offending_values"].extend(insufficient_recovery)

    reserve_check = _peak_shaving_reserve_check(summaries, parsed_task)
    checks.append(reserve_check)

    overall_linepack_status = max_status(check["status"] for check in checks)
    overall_linepack_flag = {
        "pass": "linepack_normal",
        "warning": "linepack_warning",
        "fail": "linepack_violation",
    }.get(overall_linepack_status)

    if recovery_check is not None:
        recovery_check.update(
            {
                "minimum_linepack": minimum_record,
                "linepack_recovery": recovery,
                "linepack_warning_status": overall_linepack_flag,
            }
        )
    change_rate_check = next(
        (check for check in checks if check["name"] == "linepack_change_rate"),
        None,
    )
    if change_rate_check is not None:
        change_rate_check.update(
            {
                "linepack_change_rate": change_rates,
                "linepack_decline_episodes": decline_episodes,
            }
        )
    return checks


def _peak_shaving_reserve_check(
    summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]
) -> Dict[str, Any]:
    selector = LINEPACK_RESERVE_RULE.get("selector") or {}
    variables = variables_for_selector(summaries, selector, parsed_task)
    limits = LINEPACK_RESERVE_RULE["limits"]
    warning_reserve = float(limits["warning_reserve"])
    fail_reserve = float(limits["fail_reserve"])
    registry = registry_index(parsed_task)
    evaluated = []
    capacity = {}
    statuses = []
    for variable in variables:
        metadata = registry.get(variable, {})
        lower_bound = float(metadata.get("lower_limit", limits["safe_lower_bound"]))
        limit_source = (
            "variable_registry"
            if metadata.get("lower_limit") is not None
            else "rule_library"
        )
        values = summaries.get(variable, {}).get("predicted_values", [])
        if not values:
            continue
        minimum = min(float(value) for value in values)
        reserve = minimum - lower_bound
        status = (
            "fail"
            if reserve <= fail_reserve
            else "warning"
            if reserve <= warning_reserve
            else "pass"
        )
        statuses.append(status)
        item = {
            "variable": variable,
            "metric": "minimum_linepack_reserve",
            "minimum_prediction": round(minimum, 6),
            "safe_lower_bound": lower_bound,
            "reserve": round(reserve, 6),
            "warning_reserve": warning_reserve,
            "fail_reserve": fail_reserve,
            "limit_source": limit_source,
            "status": status,
        }
        evaluated.append(item)
        capacity[variable] = item

    status = max_status(statuses)
    flags = LINEPACK_RESERVE_RULE["flags"]
    return {
        "name": LINEPACK_RESERVE_RULE["rule_id"],
        "category": "linepack",
        "status": status,
        "evaluation_status": "evaluated" if statuses else "not_evaluated",
        "flag": flags.get(status),
        "priority": int(LINEPACK_RESERVE_RULE["priority"]),
        "variables": variables,
        "description": LINEPACK_RESERVE_RULE["description"],
        "main_content": CATEGORY_DETAILS["linepack"],
        "message": (
            "Linepack peak-shaving reserve could not be evaluated."
            if status == "not_evaluated"
            else "Linepack keeps the configured short-term peak-shaving reserve."
            if status == "pass"
            else "Linepack short-term peak-shaving reserve requires review."
        ),
        "evaluated_values": evaluated,
        "offending_values": [item for item in evaluated if item["status"] != "pass"],
        "peak_shaving_capacity": capacity,
    }
