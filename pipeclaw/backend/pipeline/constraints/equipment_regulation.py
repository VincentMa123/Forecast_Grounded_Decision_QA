from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import ConstraintSpec
from .common import run_specs


EQUIPMENT_REGULATION_SPECS: Tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        name="boundary_control_adjustment_magnitude",
        category="equipment_regulation",
        description="Boundary-control adjustment should stay within the allowable dispatch magnitude.",
        priority=50,
        metric="boundary_change_percent",
        warning_threshold=10.0,
        fail_threshold=20.0,
    ),
)


def run_equipment_regulation_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_specs(EQUIPMENT_REGULATION_SPECS, summaries, parsed_task)