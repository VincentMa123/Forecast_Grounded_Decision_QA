from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..rule_library import load_constraint_specs
from .common import registry_index, run_specs


EQUIPMENT_SPECS = load_constraint_specs("equipment_regulation")


def run_equipment_regulation_checks(
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    checks = run_specs(EQUIPMENT_SPECS, summaries, parsed_task)
    state = {
        "valve_opening": _state_values(summaries, parsed_task, ("valve_opening",), ("B_",), ("_v000", "_opening")),
        "pressure_regulator": _state_values(summaries, parsed_task, ("regulator_range",), ("R_",), ("_v000", "_range")),
        "boundary_controls": _boundary_control_values(summaries, parsed_task),
    }
    for check in checks:
        check["equipment_regulation_state"] = state
    return checks


def _state_values(
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Dict[str, Any],
    physical_quantities: Tuple[str, ...],
    prefixes: Tuple[str, ...],
    suffixes: Tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    registry = registry_index(parsed_task)
    return {
        variable: {
            "mean_prediction": summary.get("mean_prediction"),
            "minimum_prediction": summary.get("minimum_prediction"),
            "maximum_prediction": summary.get("maximum_prediction"),
        }
        for variable, summary in summaries.items()
        if (
            registry.get(variable, {}).get("physical_quantity") in physical_quantities
            if variable in registry
            else variable.startswith(prefixes) and (not suffixes or variable.endswith(suffixes))
        )
    }


def _boundary_control_values(
    summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    registry = registry_index(parsed_task)
    return {
        variable: {
            "mean_prediction": summary.get("mean_prediction"),
            "minimum_prediction": summary.get("minimum_prediction"),
            "maximum_prediction": summary.get("maximum_prediction"),
        }
        for variable, summary in summaries.items()
        if (
            registry.get(variable, {}).get("role") == "input"
            and bool(registry.get(variable, {}).get("controllable"))
        )
        or (not registry and ":" in variable)
    }
