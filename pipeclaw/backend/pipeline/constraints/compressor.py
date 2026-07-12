from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..rule_library import load_constraint_specs
from .common import registry_index, run_specs


COMPRESSOR_SPECS = load_constraint_specs("compressor")


def run_compressor_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = run_specs(COMPRESSOR_SPECS, summaries, parsed_task)
    state = {
        "load": _state_values(summaries, parsed_task, ("compressor_load",), ("C_",), ("_v000",)),
        "compression_ratio": _state_values(summaries, parsed_task, ("compression_ratio",), ("C_",), ("_v001",)),
        "rotational_speed": _state_values(summaries, parsed_task, ("rotational_speed",), ("C_",), ("_v002", "_speed")),
        "power": _state_values(summaries, parsed_task, ("power",), ("TE_",), ("_v000",)),
    }
    envelope_rule_names = {
        "compressor_load_limit",
        "compressor_ratio_boundary",
        "compressor_rotational_speed_limit",
    }
    envelope_checks = [check for check in checks if check["name"] in envelope_rule_names]
    if any(check["status"] == "fail" for check in envelope_checks):
        envelope_status = "outside_operating_envelope"
    elif any(check["status"] == "warning" for check in envelope_checks):
        envelope_status = "approaching_operating_envelope"
    elif any(check["status"] == "not_evaluated" for check in envelope_checks):
        envelope_status = "incomplete_operating_envelope"
    else:
        envelope_status = "inside_operating_envelope"
    regulation_margin = {}
    for check in checks:
        check["compressor_state"] = state
        check["operating_envelope_status"] = envelope_status
        for item in check["evaluated_values"]:
            item["regulation_margin_to_fail"] = item.get("fail_margin")
            variable = item.get("variable")
            if variable:
                regulation_margin[variable] = item.get("fail_margin")
        check["regulation_margin_to_fail"] = regulation_margin
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
            "maximum_prediction": summary.get("maximum_prediction"),
            "max_abs_prediction": summary.get("max_abs_prediction"),
        }
        for variable, summary in summaries.items()
        if (
            registry.get(variable, {}).get("physical_quantity") in physical_quantities
            if variable in registry
            else variable.startswith(prefixes) and variable.endswith(suffixes)
        )
    }
