from __future__ import annotations

from typing import Any, Dict, List

from ..rule_library import load_constraint_specs, load_rule_definition
from .common import max_status, run_specs


LINEPACK_SPECS = load_constraint_specs("linepack")
LINEPACK_RECOVERY_RULE = load_rule_definition("linepack", "linepack_decline_and_recovery")


def run_linepack_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks = run_specs(LINEPACK_SPECS, summaries, parsed_task)
    linepack_variables = checks[0]["variables"] if checks else []
    minimum_recovery_ratio = float(LINEPACK_RECOVERY_RULE["minimum_recovery_ratio"])
    minimum_items = []
    change_rates = {}
    recovery = {}
    for variable in linepack_variables:
        summary = summaries.get(variable, {})
        values = summary.get("predicted_values", [])
        labels = summary.get("prediction_labels", [])
        if not values:
            continue
        minimum_index = values.index(min(values))
        minimum_items.append(
            (values[minimum_index], variable, minimum_index, labels[minimum_index] if minimum_index < len(labels) else None)
        )
        decline = float(summary.get("max_decline_from_start") or 0.0)
        recovered = float(summary.get("recovery_from_minimum") or 0.0)
        recovery[variable] = {
            "decline_from_start": round(decline, 6),
            "recovery_from_minimum": round(recovered, 6),
            "recovery_ratio": round(recovered / decline, 6) if decline > 0 else 1.0,
            "recovery_sufficient": decline == 0 or recovered / decline >= minimum_recovery_ratio,
        }
        change_rates[variable] = summary.get("max_abs_step_change")

    minimum_linepack = min(minimum_items, default=None)
    minimum_record = None
    if minimum_linepack is not None:
        value, variable, step_index, timestamp = minimum_linepack
        minimum_record = {
            "variable": variable,
            "value": value,
            "step_index": step_index,
            "timestamp": timestamp,
        }

    insufficient_recovery = [
        {
            "variable": variable,
            "metric": "recovery_ratio",
            "value": item["recovery_ratio"],
            "status": "warning",
            "warning_threshold": minimum_recovery_ratio,
        }
        for variable, item in recovery.items()
        if not item["recovery_sufficient"]
    ]
    recovery_check = next(
        (check for check in checks if check["name"] == LINEPACK_RECOVERY_RULE["rule_id"]),
        None,
    )
    if recovery_check is not None and insufficient_recovery:
        recovery_check["status"] = max_status([recovery_check["status"], "warning"])
        recovery_check["flag"] = LINEPACK_RECOVERY_RULE["flags"][recovery_check["status"]]
        recovery_check["message"] = (
            f"{len(insufficient_recovery)} linepack variable(s) did not recover the configured minimum ratio."
        )
        recovery_check["offending_values"].extend(insufficient_recovery)

    for check in checks:
        check["minimum_linepack"] = minimum_record
        check["linepack_change_rate"] = change_rates
        check["linepack_recovery"] = recovery
        check["linepack_warning_status"] = check["flag"]
    return checks
