from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .rule_library import load_rule_document
from .common import CATEGORY_DETAILS


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
