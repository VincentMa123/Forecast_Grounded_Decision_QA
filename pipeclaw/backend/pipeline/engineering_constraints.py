from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .constraints.abnormality_warning import run_abnormality_warning_checks
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
from .constraints.dispatch_priority import run_dispatch_priority_checks, run_dispatch_priority_policy_checks
from .constraints.equipment_regulation import run_equipment_regulation_checks
from .constraints.flow import run_flow_checks
from .constraints.human_intervention import intervention_label_from_checks, run_human_intervention_checks
from .constraints.linepack import run_linepack_checks
from .constraints.pressure import run_pressure_checks


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
    label = intervention_label_from_checks(overall, checks)
    checks.extend(run_human_intervention_checks(label))

    if "dispatch_priority" in selected:
        checks.extend(run_dispatch_priority_policy_checks())

    non_pass = [check for check in checks if check["status"] != "pass" and check["category"] != "human_intervention"]
    non_pass.sort(key=lambda check: (STATUS_RANK.get(check["status"], 0), -check["priority"]), reverse=True)

    return {
        "method": "engineering_constraint_library_v1",
        "value_space": "normalized PipeFormer forecast variables; replace proxy thresholds with real station/segment limits when available",
        "rule_source": "Forecast-Grounded Decision QA Task 1 engineering constraint table",
        "requested_categories": selected_categories,
        "category_status": category_status(checks),
        "dispatch_priority_order": DISPATCH_PRIORITY_ORDER,
        "overall_status": overall,
        "human_intervention_label": label,
        "checks": checks,
        "priority_findings": non_pass[:5],
        "notes": (
            "This implements the engineering constraint categories in the teacher-trace pipeline. "
            "Current limits are normalized mock-data proxy thresholds until real engineering limits are supplied."
        ),
    }


__all__ = ["run_engineering_constraint_checks", "variables_matching"]
