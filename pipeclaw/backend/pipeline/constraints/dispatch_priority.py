from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import ConstraintSpec
from .common import CATEGORY_DETAILS, run_specs


DISPATCH_PRIORITY_SPECS: Tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        name="energy_consumption_cost_proxy",
        category="dispatch_priority",
        description="Energy and cost are audited after safety, supply assurance, and equipment protection.",
        priority=70,
        metric="mean_abs_delta_vs_observed",
        prefixes=("TE_", "C_"),
        suffixes=("_v000", "_v001"),
        warning_threshold=0.6,
        fail_threshold=1.5,
    ),
)


def run_dispatch_priority_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_specs(DISPATCH_PRIORITY_SPECS, summaries, parsed_task)


def run_dispatch_priority_policy_checks() -> List[Dict[str, Any]]:
    return [
        {
            "name": "dispatch_priority_rules",
            "category": "dispatch_priority",
            "status": "pass",
            "priority": 90,
            "variables": [],
            "description": "Apply dispatch priority order when safety and economy conflict.",
            "main_content": CATEGORY_DETAILS["dispatch_priority"],
            "message": "Dispatch priorities are safety, supply assurance, equipment protection, then energy/cost.",
            "offending_values": [],
        }
    ]