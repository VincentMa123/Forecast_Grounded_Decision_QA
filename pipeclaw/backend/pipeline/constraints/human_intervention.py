from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .common import CATEGORY_DETAILS, SAFETY_CATEGORIES


def intervention_label_from_checks(overall: str, checks: Sequence[Dict[str, Any]]) -> str:
    if overall == "pass":
        return "no_intervention"
    if any(check["status"] == "fail" and check["category"] in SAFETY_CATEGORIES for check in checks):
        return "immediate_intervention_required"
    if overall == "fail":
        return "operator_attention_required"
    if any(check["status"] == "warning" and check["category"] in SAFETY_CATEGORIES for check in checks):
        return "operator_attention_required"
    return "monitoring_only"


def run_human_intervention_checks(label: str) -> List[Dict[str, Any]]:
    status = "pass" if label == "no_intervention" else ("fail" if label == "immediate_intervention_required" else "warning")
    return [
        {
            "name": "human_intervention_rules",
            "category": "human_intervention",
            "status": status,
            "priority": 80,
            "variables": [],
            "description": "Convert engineering check severity into the intervention labels.",
            "main_content": CATEGORY_DETAILS["human_intervention"],
            "message": f"Human-intervention label: {label}.",
            "offending_values": [],
        }
    ]