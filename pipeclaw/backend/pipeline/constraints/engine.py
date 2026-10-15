from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .common import (
    CATEGORY_DETAILS,
    CATEGORY_ORDER,
    DISPATCH_PRIORITY_ORDER,
    DISPATCH_RULES,
    PIPELINE_CONSTRAINTS,
    STATUS_RANK,
    category_status,
    max_status,
    run_specs,
    select_requested_categories,
    status_from_threshold,
    supply_demand_gaps,
    variables_for_selector,
)
from .flow import run_flow_checks
from .linepack import run_linepack_checks
from .pressure import run_pressure_checks
from .rule_library import (
    load_constraint_specs,
    load_rule_definition,
    load_rule_document,
)
from ..forecast.result import without_none_values


# ---------------------------------------------------------------------------
# Spec-driven categories without a dedicated module
# ---------------------------------------------------------------------------

COMPRESSOR_SPECS = load_constraint_specs("compressor")
DISPATCH_PRIORITY_SPECS = load_constraint_specs("dispatch_priority")
EQUIPMENT_SPECS = load_constraint_specs("equipment_regulation")
COMPRESSOR_ENVELOPE_RULES = {
    "compressor_load_limit",
    "compressor_ratio_boundary",
    "compressor_rotational_speed_limit",
}


def run_compressor_checks(
    summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]
) -> List[Dict[str, Any]]:
    checks = run_specs(COMPRESSOR_SPECS, summaries, parsed_task)
    envelope_checks = [
        check for check in checks if check["name"] in COMPRESSOR_ENVELOPE_RULES
    ]
    if any(check["status"] == "fail" for check in envelope_checks):
        envelope_status = "outside_operating_envelope"
    elif any(check["status"] == "warning" for check in envelope_checks):
        envelope_status = "approaching_operating_envelope"
    elif any(check["status"] == "not_evaluated" for check in envelope_checks):
        envelope_status = "incomplete_operating_envelope"
    else:
        envelope_status = "inside_operating_envelope"
    for check in checks:
        check["operating_envelope_status"] = envelope_status
    return checks


ABNORMALITY_SPECS = load_constraint_specs("abnormality_warning")
POTENTIAL_LEAK_RULE = load_rule_definition(
    "abnormality_warning", "potential_leak_signal"
)


def run_abnormality_warning_checks(
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks = run_specs(ABNORMALITY_SPECS, summaries, parsed_task)
    leak_check = _potential_leak_check(summaries, checks, parsed_task)
    checks.append(leak_check)
    return checks


def _potential_leak_check(
    summaries: Dict[str, Dict[str, Any]],
    checks: List[Dict[str, Any]],
    parsed_task: Dict[str, Any],
) -> Dict[str, Any]:
    by_name = {check["name"]: check for check in checks}
    pressure_check = by_name.get("abnormal_pressure_drop", {})
    flow_check = by_name.get("sudden_flow_change", {})
    supply_selector = POTENTIAL_LEAK_RULE["supply_selector"]
    demand_selector = POTENTIAL_LEAK_RULE["demand_selector"]
    supply_variables = variables_for_selector(summaries, supply_selector, parsed_task)
    demand_variables = variables_for_selector(summaries, demand_selector, parsed_task)
    variables = list(
        dict.fromkeys(
            pressure_check.get("variables", []) + supply_variables + demand_variables
        )
    )
    _, gaps = supply_demand_gaps(summaries, supply_variables, demand_variables)

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
    elif pressure_status in {"warning", "fail"} and corroborating_status in {
        "warning",
        "fail",
    }:
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


CategoryRunner = Callable[
    [Dict[str, Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]
]

CATEGORY_RUNNERS: Dict[str, CategoryRunner] = {
    "pressure": run_pressure_checks,
    "flow": run_flow_checks,
    "linepack": run_linepack_checks,
    "compressor": run_compressor_checks,
    "equipment_regulation": partial(run_specs, EQUIPMENT_SPECS),
    "abnormality_warning": run_abnormality_warning_checks,
    "dispatch_priority": partial(run_specs, DISPATCH_PRIORITY_SPECS),
}


# ---------------------------------------------------------------------------
# Always-run policy checks: human intervention and dispatch priority
# ---------------------------------------------------------------------------

INTERVENTION_RULES = load_rule_document("human_intervention")
SAFETY_CATEGORIES = set(INTERVENTION_RULES["safety_categories"])


def intervention_label_from_checks(
    overall: str, checks: Sequence[Dict[str, Any]]
) -> str:
    if overall == INTERVENTION_RULES["no_intervention_status"]:
        return "no_intervention"
    if any(
        check["status"] == INTERVENTION_RULES["immediate_intervention_status"]
        and check["category"] in SAFETY_CATEGORIES
        for check in checks
    ):
        return "immediate_intervention_required"
    if (
        overall in INTERVENTION_RULES["operator_attention_statuses"]
        and overall == "fail"
    ):
        return "operator_attention_required"
    if any(
        check["status"] in INTERVENTION_RULES["operator_attention_statuses"]
        and check["category"] in SAFETY_CATEGORIES
        for check in checks
    ):
        return "operator_attention_required"
    return str(INTERVENTION_RULES["default_nonpass_label"])


def run_human_intervention_checks(label: str) -> List[Dict[str, Any]]:
    status = (
        "pass"
        if label == "no_intervention"
        else ("fail" if label == "immediate_intervention_required" else "warning")
    )
    return [
        {
            "name": "human_intervention_rules",
            "category": "human_intervention",
            "status": status,
            "flag": label,
            "priority": 80,
            "variables": [],
            "description": INTERVENTION_RULES["reasons"][label],
            "main_content": CATEGORY_DETAILS["human_intervention"],
            "message": f"Human-intervention label: {label}.",
            "evaluated_values": [],
            "offending_values": [],
        }
    ]


def run_dispatch_priority_policy_checks() -> List[Dict[str, Any]]:
    policy = DISPATCH_RULES["policy_rule"]
    return [
        {
            "name": policy["rule_id"],
            "category": "dispatch_priority",
            "status": "pass",
            "flag": policy["flag"],
            "priority": int(policy["priority"]),
            "variables": [],
            "description": "Apply dispatch priority order when safety and economy conflict.",
            "main_content": CATEGORY_DETAILS["dispatch_priority"],
            "message": policy["message"],
            "evaluated_values": [],
            "offending_values": [],
        }
    ]


# ---------------------------------------------------------------------------
# Engineering verification entry point and evidence projection
# ---------------------------------------------------------------------------

# Rule ids referenced by the evidence projection.  Each name must match a
# ``rule_id`` in the library JSON files; the drift test in
# test_engineering_constraints.py keeps this set and the JSON specs in sync.
PRIMARY_EVIDENCE_RULES = (
    "node_pressure_operating_window",
    "flow_ramp_check",
    "flow_capacity_check",
    "supply_demand_balance",
    "linepack_decline_and_recovery",
    "linepack_change_rate",
    "linepack_peak_shaving_reserve",
    "compressor_load_limit",
    "compressor_ratio_boundary",
    "compressor_rotational_speed_limit",
    "compressor_power_change",
)
ABNORMALITY_EVIDENCE_RULES = (
    "abnormal_pressure_drop",
    "sudden_flow_change",
    "potential_leak_signal",
    "equipment_anomaly",
)
EQUIPMENT_EVIDENCE_RULES = {
    "valve_opening_range": "valve_opening_status",
    "pressure_regulator_range": "pressure_regulator_status",
    "boundary_control_adjustment_magnitude": "boundary_adjustment_status",
}


def run_engineering_constraint_checks(
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the repository's fixed engineering category checks directly."""
    parsed_task = parsed_task or {}
    if not (parsed_task.get("_variable_registry") or []):
        raise ValueError(
            "Engineering constraint checks require variable registry metadata."
        )
    selected_categories = select_requested_categories(
        parsed_task.get("constraint_verification_types")
    )
    selected = set(selected_categories)

    checks = [
        check
        for category in CATEGORY_ORDER
        if category in selected and (runner := CATEGORY_RUNNERS.get(category))
        for check in runner(summaries, parsed_task)
    ]

    overall = max_status(check["status"] for check in checks)
    not_evaluated_rules = [
        check["name"] for check in checks if check.get("status") == "not_evaluated"
    ]
    verification_complete = not not_evaluated_rules
    risk_escalations = _risk_escalations(checks)
    risk_level = {"pass": "low", "warning": "medium", "fail": "high"}.get(
        overall, "unknown"
    )
    if not verification_complete and overall in {"pass", "not_evaluated"}:
        risk_level = "unknown"
    if risk_escalations:
        risk_level = "high"
    label = intervention_label_from_checks(overall, checks)
    if not verification_complete and label == "no_intervention":
        label = "monitoring_only"
    if risk_escalations and label in {"no_intervention", "monitoring_only"}:
        label = "operator_attention_required"
    checks.extend(run_human_intervention_checks(label))

    if "dispatch_priority" in selected:
        checks.extend(run_dispatch_priority_policy_checks())

    non_pass = [
        check
        for check in checks
        if check["status"] in {"warning", "fail"}
        and check["category"] != "human_intervention"
    ]
    non_pass.sort(
        key=lambda check: (STATUS_RANK.get(check["status"], 0), -check["priority"]),
        reverse=True,
    )
    failures = [check for check in non_pass if check["status"] == "fail"]
    warnings = [check for check in non_pass if check["status"] == "warning"]
    selected_warnings = warnings[: max(0, 5 - len(failures))]
    priority_findings = failures + selected_warnings
    dispatch_recommendation = _dispatch_recommendation(checks)
    category_statuses = category_status(checks)
    safety_checks = [
        check
        for check in checks
        if check.get("category") in {"pressure", "flow", "linepack", "abnormality_warning"}
    ]
    energy_checks = [check for check in checks if check.get("name") == "energy_consumption_cost"]
    comparison_checks = safety_checks + energy_checks
    comparison_complete = bool(safety_checks and energy_checks) and all(
        check.get("status") != "not_evaluated" for check in comparison_checks
    )
    safety_status = max_status(
        check.get("status", "not_evaluated") for check in safety_checks
    )
    energy_status = max_status(
        check.get("status", "not_evaluated") for check in energy_checks
    )

    return {
        "requested_categories": selected_categories,
        "category_status": category_statuses,
        "safety_energy_comparison": {
            "comparison_complete": comparison_complete,
            "safety_status": safety_status,
            "energy_status": energy_status,
            "consistent": safety_status == energy_status
            if comparison_complete
            else None,
        },
        "dispatch_priority_order": DISPATCH_PRIORITY_ORDER,
        "overall_status": overall,
        "verification_complete": verification_complete,
        "not_evaluated_rules": not_evaluated_rules,
        "risk_level": risk_level,
        "risk_escalations": risk_escalations,
        "rule_flags": list(
            dict.fromkeys(check.get("flag") for check in checks if check.get("flag"))
        ),
        "triggered_flags": list(
            dict.fromkeys(check.get("flag") for check in non_pass if check.get("flag"))
        ),
        "human_intervention_label": label,
        "dispatch_recommendation": dispatch_recommendation,
        "checks": checks,
        "executed_rule_ids": [check["name"] for check in checks],
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "omitted_warning_count": len(warnings) - len(selected_warnings),
        "failed_rule_ids": [check["name"] for check in failures],
        "warning_rule_ids": [check["name"] for check in warnings],
        "priority_findings": priority_findings,
        "engineering_evidence": build_engineering_evidence(checks),
    }


def build_engineering_evidence(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build one compact category-level evidence object from rule checks."""
    by_name = {check.get("name"): check for check in checks}
    (
        pressure, flow, capacity, balance,
        linepack, linepack_rate, reserve,
        compressor, compressor_ratio, compressor_speed, compressor_power,
    ) = (by_name.get(name, {}) for name in PRIMARY_EVIDENCE_RULES)

    pressure_nodes = list(
        dict.fromkeys(
            list(pressure.get("pressure_violation_nodes") or [])
            + list(pressure.get("pressure_warning_nodes") or [])
        )
    )
    pressure_margins = dict(pressure.get("pressure_margins") or {})
    insufficient_recovery = {
        variable: item
        for variable, item in dict(linepack.get("linepack_recovery") or {}).items()
        if not item.get("recovery_sufficient", True)
    }
    ramp_events = {
        variable: events
        for variable, events in dict(flow.get("flow_ramp_events") or {}).items()
        if events
    }
    capacity_episodes = {
        variable: item
        for variable, item in dict(
            capacity.get("flow_capacity_excursion_episodes") or {}
        ).items()
        if item.get("total_out_of_limit_minutes", 0) > 0
    }
    abnormality_checks = [
        by_name[name]
        for name in ABNORMALITY_EVIDENCE_RULES
        if by_name.get(name, {}).get("status") in {"warning", "fail"}
    ]

    return {
        "pressure": without_none_values(
            {
                "minimum_pressure": pressure.get("minimum_pressure"),
                "maximum_pressure": pressure.get("maximum_pressure"),
                "pressure_violation_nodes": pressure.get(
                    "pressure_violation_nodes", []
                ),
                "pressure_warning_nodes": pressure.get("pressure_warning_nodes", []),
                "at_risk_pressure_margins": {
                    variable: pressure_margins[variable]
                    for variable in pressure_nodes[:5]
                    if variable in pressure_margins
                },
                "minimum_lower_bound_margin": _minimum_pressure_margin(
                    pressure_margins, "fail_lower_margin"
                ),
                "minimum_upper_bound_margin": _minimum_pressure_margin(
                    pressure_margins, "fail_upper_margin"
                ),
                "minimum_operating_window_margin": _minimum_operating_window_margin(
                    pressure_margins
                ),
                "violation_node_count": len(
                    pressure.get("pressure_violation_nodes") or []
                ),
                "warning_node_count": len(pressure.get("pressure_warning_nodes") or []),
                "maximum_continuous_pressure_violation_minutes": pressure.get(
                    "maximum_continuous_pressure_violation_minutes"
                ),
                "simultaneous_end_user_warning_node_count": pressure.get(
                    "simultaneous_end_user_warning_node_count"
                ),
            }
        ),
        "flow": without_none_values(
            {
                "maximum_segment_flow_change": _maximum_mapping_value(
                    flow.get("flow_change_magnitude", {})
                ),
                "maximum_boundary_flow_change_rate": _maximum_mapping_value(
                    flow.get("boundary_flow_change_rate", {})
                ),
                "abnormal_flow_segments": flow.get("abnormal_flow_segments", []),
                "flow_ramp_events": ramp_events,
                "flow_capacity_excursion_episodes": capacity_episodes,
                "flow_capacity_excursion_count": len(capacity_episodes),
                "supply_demand_balance_status": balance.get("status"),
                "supply_demand_balance": (balance.get("evaluated_values") or [None])[0],
            }
        ),
        "linepack": without_none_values(
            {
                "minimum_linepack": linepack.get("minimum_linepack"),
                "maximum_linepack_change_rate": _maximum_mapping_value(
                    linepack_rate.get("linepack_change_rate", {})
                ),
                "maximum_continuous_decline_minutes": max(
                    (
                        float(item.get("maximum_continuous_decline_minutes") or 0.0)
                        for item in dict(
                            linepack.get("linepack_recovery") or {}
                        ).values()
                    ),
                    default=0.0,
                ),
                "maximum_decline_from_start": _maximum_recovery_value(
                    linepack.get("linepack_recovery"), "decline_from_start"
                ),
                "insufficient_recovery": insufficient_recovery,
                "insufficient_recovery_count": len(insufficient_recovery),
                "minimum_peak_shaving_reserve": _minimum_evaluated_value(
                    reserve, "reserve"
                ),
                "linepack_warning_status": linepack.get("linepack_warning_status"),
            }
        ),
        "compressor": without_none_values(
            {
                "operating_envelope_status": compressor.get(
                    "operating_envelope_status"
                ),
                "maximum_load": _maximum_evaluated_value(compressor),
                "maximum_compression_ratio": _maximum_evaluated_value(compressor_ratio),
                "maximum_rotational_speed": _maximum_evaluated_value(compressor_speed),
                "maximum_power_change": _maximum_evaluated_value(compressor_power),
            }
        ),
        "equipment_regulation": {
            key: by_name.get(name, {}).get("status", "not_evaluated")
            for name, key in EQUIPMENT_EVIDENCE_RULES.items()
        },
        "abnormality_warning": {
            "triggered_rule_count": len(abnormality_checks),
            "failure_rule_count": sum(
                check.get("status") == "fail" for check in abnormality_checks
            ),
            "warning_rule_count": sum(
                check.get("status") == "warning" for check in abnormality_checks
            ),
        },
    }


def _minimum_pressure_margin(
    margins: Dict[str, Dict[str, Any]], metric: str
) -> Optional[Dict[str, Any]]:
    return _extreme(
        ((variable, values.get(metric)) for variable, values in margins.items()),
        minimum=True,
    )


def _minimum_operating_window_margin(
    margins: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    return _extreme(
        (
            (
                variable,
                min(float(values["fail_lower_margin"]), float(values["fail_upper_margin"])),
            )
            for variable, values in margins.items()
            if values.get("fail_lower_margin") is not None
            and values.get("fail_upper_margin") is not None
        ),
        minimum=True,
    )


def _maximum_recovery_value(recovery: Any, metric: str) -> Optional[Dict[str, Any]]:
    return _extreme(
        ((variable, values.get(metric)) for variable, values in dict(recovery or {}).items())
    )


def _maximum_mapping_value(values: Any) -> Optional[Dict[str, Any]]:
    return _extreme(dict(values or {}).items(), absolute=True)


def _maximum_evaluated_value(check: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _extreme(
        (
            (item, item.get("value") if item.get("value") is not None else item.get("max_prediction"))
            for item in check.get("evaluated_values") or []
        ),
        absolute=True,
    )


def _minimum_evaluated_value(
    check: Dict[str, Any], metric: str
) -> Optional[Dict[str, Any]]:
    return _extreme(
        ((item, item.get(metric)) for item in check.get("evaluated_values") or []),
        minimum=True,
    )


def _extreme(
    candidates: Iterable[tuple[Any, Any]],
    *,
    minimum: bool = False,
    absolute: bool = False,
) -> Optional[Dict[str, Any]]:
    values = [(item, float(value)) for item, value in candidates if value is not None]
    if not values:
        return None
    select = min if minimum else max
    item, value = select(values, key=lambda pair: abs(pair[1]) if absolute else pair[1])
    variable = item.get("variable") if isinstance(item, dict) else str(item)
    return {"variable": variable, "value": round(value, 6)}


def _risk_escalations(checks: List[Dict[str, Any]]) -> List[str]:
    escalations = []
    pressure_check = next(
        (
            check
            for check in checks
            if check["name"] == "node_pressure_operating_window"
        ),
        None,
    )
    risk_config = PIPELINE_CONSTRAINTS["risk_escalation"]
    if pressure_check and pressure_check.get(
        "simultaneous_end_user_warning_node_count", 0
    ) >= int(risk_config["simultaneous_end_user_warning_nodes"]):
        escalations.append("multiple_end_user_nodes_near_pressure_lower_bound")
    if pressure_check and max(
        pressure_check.get("pressure_recovery_time_minutes", {}).values(), default=0.0
    ) > float(risk_config["pressure_recovery_warning_minutes"]):
        escalations.append("pressure_recovery_time_exceeds_30_minutes")
    pressure_at_risk = any(
        check.get("flag") in {"pressure_warning", "pressure_violation"}
        for check in checks
    )
    linepack_decline_check = next(
        (
            check
            for check in checks
            if check.get("name") == "linepack_decline_and_recovery"
        ),
        None,
    )
    linepack_declining = bool(
        linepack_decline_check
        and linepack_decline_check.get("status") in {"warning", "fail"}
    )
    if pressure_at_risk and linepack_declining:
        escalations.append("linepack_decline_with_pressure_near_lower_bound")
    return escalations


def _dispatch_recommendation(checks: List[Dict[str, Any]]) -> str:
    flags = {check.get("flag") for check in checks}
    for priority in DISPATCH_RULES["priority_order"]:
        if flags & set(priority["trigger_flags"]):
            return str(priority["recommendation"])
    return str(DISPATCH_RULES["default_recommendation"])


__all__ = ["build_engineering_evidence", "run_engineering_constraint_checks"]
