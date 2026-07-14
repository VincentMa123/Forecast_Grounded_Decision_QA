from __future__ import annotations

from typing import Any, Dict, List

from ..rule_library import load_constraint_specs
from .common import run_specs


EQUIPMENT_SPECS = load_constraint_specs("equipment_regulation")


def run_equipment_regulation_checks(
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return run_specs(EQUIPMENT_SPECS, summaries, parsed_task)
