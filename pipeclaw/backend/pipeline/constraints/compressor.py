from __future__ import annotations

from typing import Any, Dict, List

from .rule_library import load_constraint_specs
from .common import run_specs


COMPRESSOR_SPECS = load_constraint_specs("compressor")


def run_compressor_checks(
    summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]
) -> List[Dict[str, Any]]:
    checks = run_specs(COMPRESSOR_SPECS, summaries, parsed_task)
    envelope_rule_names = {
        "compressor_load_limit",
        "compressor_ratio_boundary",
        "compressor_rotational_speed_limit",
    }
    envelope_checks = [
        check for check in checks if check["name"] in envelope_rule_names
    ]
    if any(check["status"] == "fail" for check in envelope_checks):
        envelope_status = "outside_operating_envelope"
    elif any(check["status"] == "warning" for check in envelope_checks):
        envelope_status = "approaching_operating_envelope"
    elif any(check["status"] == "not_evaluated" for check in envelope_checks):
        envelope_status = "incomplete_operating_envelope"
    else:
        envelope_status = "inside_operating_envelope"
    for check in checks:
        check["operating_envelope_status"] = envelope_status
    return checks
