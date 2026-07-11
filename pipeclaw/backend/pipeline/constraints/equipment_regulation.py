from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..rule_library import load_constraint_specs
from .common import run_specs


EQUIPMENT_SPECS = load_constraint_specs("equipment_regulation")


def run_equipment_regulation_checks(
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks = run_specs(EQUIPMENT_SPECS, summaries, parsed_task)
    state = {
        "valve_opening": _state_values(summaries, ("B_",), ("_v000", "_opening")),
        "pressure_regulator": _state_values(summaries, ("R_",), ("_v000", "_range")),
        "boundary_controls": _state_values(summaries, ("T_",), ()),
    }
    for check in checks:
        check["equipment_regulation_state"] = state
    return checks


def _state_values(
    summaries: Dict[str, Dict[str, Any]],
    prefixes: Tuple[str, ...],
    suffixes: Tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    return {
        variable: {
            "mean_prediction": summary.get("mean_prediction"),
            "minimum_prediction": summary.get("minimum_prediction"),
            "maximum_prediction": summary.get("maximum_prediction"),
        }
        for variable, summary in summaries.items()
        if variable.startswith(prefixes) and (not suffixes or variable.endswith(suffixes))
    }
