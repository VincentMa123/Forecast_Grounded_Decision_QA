from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .constraints.common import (
    CATEGORY_ORDER,
    DISPATCH_PRIORITY_ORDER,
    STATUS_RANK,
    category_status,
    max_status,
    select_requested_categories,
    variables_matching,
)
from .constraints.compressor import run_compressor_checks
from .constraints.abnormality_warning import run_abnormality_warning_checks
from .constraints.dispatch_priority import run_dispatch_priority_checks, run_dispatch_priority_policy_checks
from .constraints.equipment_regulation import run_equipment_regulation_checks
from .constraints.flow import run_flow_checks
from .constraints.human_intervention import intervention_label_from_checks, run_human_intervention_checks
from .constraints.linepack import run_linepack_checks
from .constraints.pressure import run_pressure_checks
from .rule_library import load_pipeline_constraints, load_rule_document


CategoryRunner = Callable[[Dict[str, Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]]

CATEGORY_RUNNERS: Dict[str, CategoryRunner] = {
    "pressure": run_pressure_checks,
    "flow": run_flow_checks,
    "linepack": run_linepack_checks,
    "compressor": run_compressor_checks,
    "equipment_regulation": run_equipment_regulation_checks,
    "abnormality_warning": run_abnormality_warning_checks,
    "dispatch_priority": run_dispatch_priority_checks,
}
PIPELINE_CONSTRAINTS = load_pipeline_constraints()
DISPATCH_RULES = load_rule_document("dispatch_priority")


def run_engineering_constraint_checks(
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    parsed_task = parsed_task or {}
    selected_categories = select_requested_categories(parsed_task.get("constraint_verification_types") or parsed_task.get("requested_checks"))
    selected = set(selected_categories)

    checks: List[Dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        runner = CATEGORY_RUNNERS.get(category)
        if runner is not None and category in selected:
            checks.extend(runner(summaries, parsed_task))

    overall = max_status(check["status"] for check in checks)
    not_evaluated_rules = [
        check["name"]
        for check in checks
        if check.get("status") == "not_evaluated"
    ]
    verification_complete = not not_evaluated_rules
    risk_escalations = _risk_escalations(checks)
    risk_level = {"pass": "low", "warning": "medium", "fail": "high"}.get(overall, "unknown")
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
        if check["status"] in {"warning", "fail"} and check["category"] != "human_intervention"
    ]
    non_pass.sort(key=lambda check: (STATUS_RANK.get(check["status"], 0), -check["priority"]), reverse=True)
    dispatch_recommendation = _dispatch_recommendation(checks)

    return {
        "requested_categories": selected_categories,
        "category_status": category_status(checks),
        "dispatch_priority_order": DISPATCH_PRIORITY_ORDER,
        "overall_status": overall,
        "verification_complete": verification_complete,
        "not_evaluated_rules": not_evaluated_rules,
        "risk_level": risk_level,
        "risk_escalations": risk_escalations,
        "rule_flags": list(dict.fromkeys(check.get("flag") for check in checks if check.get("flag"))),
        "human_intervention_label": label,
        "dispatch_recommendation": dispatch_recommendation,
        "checks": checks,
        "executed_rule_ids": [check["name"] for check in checks],
        "priority_findings": non_pass[:5],
    }


def _risk_escalations(checks: List[Dict[str, Any]]) -> List[str]:
    escalations = []
    pressure_check = next((check for check in checks if check["name"] == "node_pressure_operating_window"), None)
    risk_config = PIPELINE_CONSTRAINTS["risk_escalation"]
    if pressure_check and pressure_check.get("simultaneous_end_user_warning_node_count", 0) >= int(risk_config["simultaneous_end_user_warning_nodes"]):
        escalations.append("multiple_end_user_nodes_near_pressure_lower_bound")
    if pressure_check and max(pressure_check.get("pressure_recovery_time_minutes", {}).values(), default=0.0) > float(risk_config["pressure_recovery_warning_minutes"]):
        escalations.append("pressure_recovery_time_exceeds_30_minutes")
    pressure_at_risk = any(
        check.get("flag") in {"pressure_warning", "pressure_violation"}
        for check in checks
    )
    linepack_decline_check = next(
        (check for check in checks if check.get("name") == "linepack_decline_and_recovery"),
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


__all__ = ["run_engineering_constraint_checks", "variables_matching"]
