from __future__ import annotations

from typing import Any, Dict, List

from ..rule_library import load_constraint_specs, load_rule_definition
from .common import CATEGORY_DETAILS, run_specs, status_from_threshold, variables_matching


FLOW_SPECS = load_constraint_specs("flow")
BALANCE_RULE = load_rule_definition("flow", "supply_demand_balance")


def run_flow_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = run_specs(FLOW_SPECS, summaries, parsed_task)
    flow_variables = sorted({variable for check in checks for variable in check["variables"]})
    abnormal_segments = _abnormal_flow_segments(summaries)
    flow_change_magnitude = {
        variable: summaries.get(variable, {}).get("max_abs_step_change")
        for variable in flow_variables
    }
    balance_check = _supply_demand_balance_check(summaries)
    for check in checks:
        check["flow_change_magnitude"] = flow_change_magnitude
        check["abnormal_flow_segments"] = abnormal_segments
        check["supply_demand_balance_status"] = balance_check["status"]
    checks.append(balance_check)
    return checks


def _abnormal_flow_segments(summaries: Dict[str, Dict[str, Any]]) -> List[str]:
    abnormal = set()
    for spec in FLOW_SPECS:
        variables = variables_matching(summaries, spec.prefixes, spec.suffixes)
        for variable in variables:
            value = summaries.get(variable, {}).get(spec.metric)
            if value is None:
                continue
            if status_from_threshold(float(value), spec.warning_threshold, spec.fail_threshold) != "pass":
                abnormal.add(variable)
    return sorted(abnormal)


def _supply_demand_balance_check(summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    supply_selector = BALANCE_RULE["supply_selector"]
    demand_selector = BALANCE_RULE["demand_selector"]
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
    variables = supply_variables + demand_variables
    series_lengths = [len(summaries[name].get("predicted_values", [])) for name in variables]
    step_count = min(series_lengths, default=0)
    gaps = []
    for index in range(step_count):
        supply = sum(summaries[name]["predicted_values"][index] for name in supply_variables)
        demand = sum(summaries[name]["predicted_values"][index] for name in demand_variables)
        gaps.append(supply - demand)

    max_gap = max((abs(value) for value in gaps), default=0.0)
    widening = bool(gaps) and abs(gaps[-1]) > abs(gaps[0])
    limits = BALANCE_RULE["limits"]
    warning_threshold = limits.get("warning_threshold")
    fail_threshold = limits.get("fail_threshold")
    threshold_status = status_from_threshold(max_gap, warning_threshold, fail_threshold)
    status = threshold_status if widening else "pass"
    flag = BALANCE_RULE["flags"][status]
    evaluated = {
        "metric": "max_abs_supply_demand_gap",
        "value": round(max_gap, 6),
        "initial_gap": round(gaps[0], 6) if gaps else None,
        "final_gap": round(gaps[-1], 6) if gaps else None,
        "gap_is_widening": widening,
        "warning_threshold": warning_threshold,
        "fail_threshold": fail_threshold,
        "status": status,
    }
    return {
        "name": BALANCE_RULE["rule_id"],
        "category": "flow",
        "status": status,
        "flag": flag,
        "priority": int(BALANCE_RULE["priority"]),
        "variables": variables,
        "description": BALANCE_RULE["description"],
        "main_content": CATEGORY_DETAILS["flow"],
        "message": "Supply-demand balance remains within the proxy tolerance." if status == "pass" else "The supply-demand gap requires review.",
        "evaluated_values": [evaluated],
        "offending_values": [] if status == "pass" else [evaluated],
    }
