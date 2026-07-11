from __future__ import annotations

from typing import Any, Dict, List

from ..rule_library import load_constraint_specs, load_rule_definition
from .common import CATEGORY_DETAILS, max_status, run_specs, status_from_threshold, variables_matching


ABNORMALITY_SPECS = load_constraint_specs("abnormality_warning")
POTENTIAL_LEAK_RULE = load_rule_definition("abnormality_warning", "potential_leak_signal")


def run_abnormality_warning_checks(
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks = run_specs(ABNORMALITY_SPECS, summaries, parsed_task)
    leak_check = _potential_leak_check(summaries, checks)
    checks.append(leak_check)
    return checks


def _potential_leak_check(
    summaries: Dict[str, Dict[str, Any]],
    checks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_name = {check["name"]: check for check in checks}
    pressure_check = by_name.get("abnormal_pressure_drop", {})
    flow_check = by_name.get("sudden_flow_change", {})
    supply_selector = POTENTIAL_LEAK_RULE["supply_selector"]
    demand_selector = POTENTIAL_LEAK_RULE["demand_selector"]
    supply_variables = variables_matching(
        summaries,
        tuple(supply_selector.get("prefixes") or ()),
        tuple(supply_selector.get("suffixes") or ()),
    )
    demand_variables = variables_matching(
        summaries,
        tuple(demand_selector.get("prefixes") or ()),
        tuple(demand_selector.get("suffixes") or ()),
    )
    variables = list(dict.fromkeys(pressure_check.get("variables", []) + supply_variables + demand_variables))
    usable_supply_variables = [
        variable
        for variable in supply_variables
        if summaries.get(variable, {}).get("predicted_values")
    ]
    usable_demand_variables = [
        variable
        for variable in demand_variables
        if summaries.get(variable, {}).get("predicted_values")
    ]
    lengths = [
        len(summaries.get(variable, {}).get("predicted_values", []))
        for variable in usable_supply_variables + usable_demand_variables
    ]
    step_count = (
        min(lengths, default=0)
        if usable_supply_variables and usable_demand_variables
        else 0
    )
    gaps = []
    for index in range(step_count):
        supply = sum(float(summaries[name]["predicted_values"][index]) for name in usable_supply_variables)
        demand = sum(float(summaries[name]["predicted_values"][index]) for name in usable_demand_variables)
        gaps.append(supply - demand)

    max_gap = max((abs(value) for value in gaps), default=0.0)
    gap_widening = bool(gaps) and abs(gaps[-1]) > abs(gaps[0])
    limits = POTENTIAL_LEAK_RULE["limits"]
    gap_status = status_from_threshold(
        max_gap,
        limits.get("warning_gap_threshold"),
        limits.get("fail_gap_threshold"),
    )
    if not gap_widening:
        gap_status = "pass"

    pressure_status = pressure_check.get("status", "not_evaluated")
    flow_status = flow_check.get("status", "not_evaluated")
    evaluated = pressure_status != "not_evaluated" and bool(gaps)
    corroborating_status = max_status([flow_status, gap_status])
    if not evaluated:
        status = "not_evaluated"
    elif pressure_status in {"warning", "fail"} and corroborating_status in {"warning", "fail"}:
        status = max_status([pressure_status, corroborating_status])
    else:
        status = "pass"

    evidence = {
        "metric": "cross_signal_potential_leak",
        "pressure_drop_status": pressure_status,
        "sudden_flow_change_status": flow_status,
        "max_abs_supply_demand_gap": round(max_gap, 6),
        "gap_is_widening": gap_widening,
        "gap_status": gap_status,
        "status": status,
    }
    flags = POTENTIAL_LEAK_RULE["flags"]
    return {
        "name": POTENTIAL_LEAK_RULE["rule_id"],
        "category": "abnormality_warning",
        "status": status,
        "evaluation_status": "evaluated" if evaluated else "not_evaluated",
        "flag": flags.get(status),
        "priority": int(POTENTIAL_LEAK_RULE["priority"]),
        "variables": variables,
        "description": POTENTIAL_LEAK_RULE["description"],
        "main_content": CATEGORY_DETAILS["abnormality_warning"],
        "message": (
            "Potential-leak signals could not be evaluated because required pressure or flow series were unavailable."
            if status == "not_evaluated"
            else "No corroborated potential-leak signal was detected."
            if status == "pass"
            else "A pressure-drop signal is corroborated by abnormal flow behavior; engineering review is required."
        ),
        "evaluated_values": [evidence] if evaluated else [],
        "offending_values": [evidence] if status in {"warning", "fail"} else [],
        "signal_evidence": evidence,
    }
