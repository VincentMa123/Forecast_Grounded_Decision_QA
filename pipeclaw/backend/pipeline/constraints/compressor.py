from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..rule_library import load_constraint_specs
from .common import run_specs


COMPRESSOR_SPECS = load_constraint_specs("compressor")


def run_compressor_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = run_specs(COMPRESSOR_SPECS, summaries, parsed_task)
    state = {
        "load": _state_values(summaries, ("C_",), ("_v000",)),
        "compression_ratio": _state_values(summaries, ("C_",), ("_v001",)),
        "power": _state_values(summaries, ("TE_",), ("_v000",)),
    }
    envelope_checks = checks[:2]
    if any(check["status"] == "fail" for check in envelope_checks):
        envelope_status = "outside_operating_envelope"
    elif any(check["status"] == "warning" for check in envelope_checks):
        envelope_status = "approaching_operating_envelope"
    else:
        envelope_status = "inside_operating_envelope"
    for check in checks:
        check["compressor_state"] = state
        check["operating_envelope_status"] = envelope_status
        for item in check["evaluated_values"]:
            item["regulation_margin_to_fail"] = item.get("fail_margin")
    return checks


def _state_values(
    summaries: Dict[str, Dict[str, Any]],
    prefixes: Tuple[str, ...],
    suffixes: Tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    return {
        variable: {
            "mean_prediction": summary.get("mean_prediction"),
            "maximum_prediction": summary.get("maximum_prediction"),
            "max_abs_prediction": summary.get("max_abs_prediction"),
        }
        for variable, summary in summaries.items()
        if variable.startswith(prefixes) and variable.endswith(suffixes)
    }
