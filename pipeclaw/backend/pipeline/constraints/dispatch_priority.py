from __future__ import annotations

from typing import Any, Dict, List

from .rule_library import load_constraint_specs, load_rule_document
from .common import CATEGORY_DETAILS, run_specs


DISPATCH_RULES = load_rule_document("dispatch_priority")
DISPATCH_PRIORITY_SPECS = load_constraint_specs("dispatch_priority")


def run_dispatch_priority_checks(
    summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]
) -> List[Dict[str, Any]]:
    return run_specs(DISPATCH_PRIORITY_SPECS, summaries, parsed_task)


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
