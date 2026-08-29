from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .common import (
    CATEGORY_ORDER,
    DISPATCH_PRIORITY_ORDER,
    DISPATCH_RULES,
    PIPELINE_CONSTRAINTS,
    STATUS_RANK,
    category_status,
    max_status,
    select_requested_categories,
)
from .compressor import run_compressor_checks
from .abnormality_warning import run_abnormality_warning_checks
from .dispatch_priority import (
    run_dispatch_priority_checks,
    run_dispatch_priority_policy_checks,
)
from .equipment_regulation import run_equipment_regulation_checks
from .flow import run_flow_checks
from .human_intervention import (
    intervention_label_from_checks,
    run_human_intervention_checks,
)
from .linepack import run_linepack_checks
from .pressure import run_pressure_checks
from ..forecast.result import without_none_values


CategoryRunner = Callable[
    [Dict[str, Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]
]

CATEGORY_RUNNERS: Dict[str, CategoryRunner] = {
    "pressure": run_pressure_checks,
    "flow": run_flow_checks,
    "linepack": run_linepack_checks,
    "compressor": run_compressor_checks,
    "equipment_regulation": run_equipment_regulation_checks,
    "abnormality_warning": run_abnormality_warning_checks,
    "dispatch_priority": run_dispatch_priority_checks,
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

    checks: List[Dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        runner = CATEGORY_RUNNERS.get(category)
        if runner is not None and category in selected:
            checks.extend(runner(summaries, parsed_task))

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
        if check.get("category")
        in {"pressure", "flow", "linepack", "abnormality_warning"}
    ]
    energy_checks = [
        check for check in checks if check.get("name") == "energy_consumption_cost"
    ]
    comparison_complete = (
        bool(safety_checks)
        and bool(energy_checks)
        and all(
            check.get("status") != "not_evaluated"
            for check in safety_checks + energy_checks
        )
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
    pressure = by_name.get("node_pressure_operating_window", {})
    flow = by_name.get("flow_ramp_check", {})
    capacity = by_name.get("flow_capacity_check", {})
    balance = by_name.get("supply_demand_balance", {})
    linepack = by_name.get("linepack_decline_and_recovery", {})
    linepack_rate = by_name.get("linepack_change_rate", {})
    reserve = by_name.get("linepack_peak_shaving_reserve", {})
    compressor = by_name.get("compressor_load_limit", {})
    compressor_ratio = by_name.get("compressor_ratio_boundary", {})
    compressor_speed = by_name.get("compressor_rotational_speed_limit", {})
    compressor_power = by_name.get("compressor_power_change", {})

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
        for name in (
            "abnormal_pressure_drop",
            "sudden_flow_change",
            "potential_leak_signal",
            "equipment_anomaly",
        )
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
            "valve_opening_status": by_name.get("valve_opening_range", {}).get(
                "status", "not_evaluated"
            ),
            "pressure_regulator_status": by_name.get(
                "pressure_regulator_range", {}
            ).get("status", "not_evaluated"),
            "boundary_adjustment_status": by_name.get(
                "boundary_control_adjustment_magnitude", {}
            ).get("status", "not_evaluated"),
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
    candidates = [
        (str(variable), float(values[metric]))
        for variable, values in margins.items()
        if values.get(metric) is not None
    ]
    if not candidates:
        return None
    variable, value = min(candidates, key=lambda item: item[1])
    return {"variable": variable, "value": round(value, 6)}


def _minimum_operating_window_margin(
    margins: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    candidates = [
        (
            str(variable),
            min(float(values["fail_lower_margin"]), float(values["fail_upper_margin"])),
        )
        for variable, values in margins.items()
        if values.get("fail_lower_margin") is not None
        and values.get("fail_upper_margin") is not None
    ]
    if not candidates:
        return None
    variable, value = min(candidates, key=lambda item: item[1])
    return {"variable": variable, "value": round(value, 6)}


def _maximum_recovery_value(recovery: Any, metric: str) -> Optional[Dict[str, Any]]:
    candidates = [
        (str(variable), float(values[metric]))
        for variable, values in dict(recovery or {}).items()
        if values.get(metric) is not None
    ]
    if not candidates:
        return None
    variable, value = max(candidates, key=lambda item: item[1])
    return {"variable": variable, "value": round(value, 6)}


def _maximum_mapping_value(values: Any) -> Optional[Dict[str, Any]]:
    candidates = [
        (str(variable), float(value))
        for variable, value in dict(values or {}).items()
        if value is not None
    ]
    if not candidates:
        return None
    variable, value = max(candidates, key=lambda item: abs(item[1]))
    return {"variable": variable, "value": round(value, 6)}


def _maximum_evaluated_value(check: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = []
    for item in check.get("evaluated_values") or []:
        value = item.get("value")
        if value is None:
            value = item.get("max_prediction")
        if value is not None:
            candidates.append((dict(item), float(value)))
    if not candidates:
        return None
    item, value = max(candidates, key=lambda candidate: abs(candidate[1]))
    return {"variable": item.get("variable"), "value": round(value, 6)}


def _minimum_evaluated_value(
    check: Dict[str, Any], metric: str
) -> Optional[Dict[str, Any]]:
    candidates = [
        (dict(item), float(item[metric]))
        for item in check.get("evaluated_values") or []
        if item.get(metric) is not None
    ]
    if not candidates:
        return None
    item, value = min(candidates, key=lambda candidate: candidate[1])
    return {"variable": item.get("variable"), "value": round(value, 6)}


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
