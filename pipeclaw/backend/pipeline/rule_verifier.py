from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def variables_matching(names: Iterable[str], prefixes: Tuple[str, ...], suffixes: Tuple[str, ...] = ()) -> List[str]:
    result = []
    for name in names:
        if prefixes and not name.startswith(prefixes):
            continue
        if suffixes and not name.endswith(suffixes):
            continue
        result.append(name)
    return result


def check_range(
    name: str,
    variables: List[str],
    summaries: Dict[str, Dict[str, Any]],
    low: float,
    high: float,
) -> Dict[str, Any]:
    offending = []
    for variable in variables:
        for value in summaries.get(variable, {}).get("predicted_values", []):
            if value < low or value > high:
                offending.append({"variable": variable, "value": value, "allowed_range": [low, high]})
    if offending:
        status = "warning" if len(offending) <= 3 else "fail"
        message = f"{len(offending)} predicted value(s) outside proxy range."
    else:
        status = "pass"
        message = "All selected proxy variables are inside the configured range."
    return {
        "name": name,
        "status": status,
        "variables": variables,
        "message": message,
        "offending_values": offending,
    }


def run_constraint_checks(summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    names = list(summaries)
    checks = [
        check_range("pressure_proxy", variables_matching(names, ("N_", "P_"), ("_v000",)), summaries, -3.0, 3.0),
        check_range("flow_proxy", variables_matching(names, ("B_", "P_"), ("_v001",)), summaries, -3.0, 3.0),
        check_range("linepack_proxy", variables_matching(names, ("P_", "N_"), ("_v000",)), summaries, -3.0, 3.0),
        check_range("compressor_load_proxy", variables_matching(names, ("C_",), ("_v000", "_v001")), summaries, -1.2, 1.2),
        check_range("energy_proxy", variables_matching(names, ("C_", "R_"), ("_v000", "_v001")), summaries, -1.5, 1.5),
    ]

    if any(check["status"] == "fail" for check in checks):
        overall = "fail"
    elif any(check["status"] == "warning" for check in checks):
        overall = "warning"
    else:
        overall = "pass"
    return {
        "method": "mock_proxy_rules",
        "value_space": "normalized sample_prediction values from PipeFormer mock output",
        "overall_status": overall,
        "checks": checks,
    }