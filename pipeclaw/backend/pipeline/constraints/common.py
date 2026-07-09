from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schemas import ConstraintSpec


STATUS_RANK = {"pass": 0, "warning": 1, "fail": 2}
CATEGORY_ORDER = [
    "pressure",
    "flow",
    "linepack",
    "compressor",
    "equipment_regulation",
    "abnormality_warning",
    "human_intervention",
    "dispatch_priority",
]
CATEGORY_DETAILS = {
    "pressure": "Node pressure lower/upper bounds, key-node pressure margin, and out-of-limit duration.",
    "flow": "Segment flow change, maximum transmission capacity, boundary flow change rate, and supply-demand balance.",
    "linepack": "Linepack decline magnitude, linepack recovery capability, short-term peak-shaving capacity, and warning threshold.",
    "compressor": "Compressor load, compression ratio, rotational speed, power, operating envelope, and regulation margin.",
    "equipment_regulation": "Valve opening, pressure-regulating equipment range, and allowable adjustment magnitude of boundary control variables.",
    "abnormality_warning": "Rules for reviewing abnormal pressure drops, sudden flow changes, and potential leaks or equipment anomalies.",
    "human_intervention": "no_intervention, monitoring_only, operator_attention_required, immediate_intervention_required.",
    "dispatch_priority": "Safety first, then supply assurance, then equipment protection, and finally energy consumption and cost.",
}
DISPATCH_PRIORITY_ORDER = [
    "safety",
    "supply_assurance",
    "equipment_protection",
    "energy_consumption_and_cost",
]
SAFETY_CATEGORIES = {"pressure", "flow", "abnormality_warning"}
ALWAYS_RUN_CATEGORIES = {"equipment_regulation", "abnormality_warning", "human_intervention", "dispatch_priority"}
MAX_OFFENDING_VALUES = 12


def select_requested_categories(values: Optional[Iterable[str]]) -> List[str]:
    requested = []
    seen = set()
    for raw in values or []:
        category = str(raw).strip()
        if not category or category not in CATEGORY_ORDER or category in seen:
            continue
        seen.add(category)
        requested.append(category)

    categories = set(requested or CATEGORY_ORDER)
    categories.update(ALWAYS_RUN_CATEGORIES)
    return [category for category in CATEGORY_ORDER if category in categories]


def variables_matching(names: Iterable[str], prefixes: Tuple[str, ...], suffixes: Tuple[str, ...] = ()) -> List[str]:
    result = []
    for name in names:
        if prefixes and not name.startswith(prefixes):
            continue
        if suffixes and not name.endswith(suffixes):
            continue
        result.append(name)
    return result


def max_status(statuses: Iterable[str]) -> str:
    return max(statuses, key=lambda status: STATUS_RANK.get(status, 0), default="pass")


def status_from_threshold(value: float, warning: Optional[float], fail: Optional[float]) -> str:
    magnitude = abs(value)
    if fail is not None and magnitude >= fail:
        return "fail"
    if warning is not None and magnitude >= warning:
        return "warning"
    return "pass"


def category_status(checks: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    result = {category: "pass" for category in CATEGORY_ORDER}
    for check in checks:
        category = check["category"]
        result[category] = max_status([result.get(category, "pass"), check["status"]])
    return result


def base_check(spec: ConstraintSpec, variables: Sequence[str]) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "category": spec.category,
        "status": "pass",
        "priority": spec.priority,
        "variables": list(variables),
        "description": spec.description,
        "main_content": CATEGORY_DETAILS[spec.category],
        "message": "No violation detected.",
        "offending_values": [],
    }


def evaluate_range(spec: ConstraintSpec, summaries: Dict[str, Dict[str, Any]], variables: Sequence[str]) -> Dict[str, Any]:
    check = base_check(spec, variables)
    if not variables:
        check["message"] = "No matching variables were available for this rule."
        return check

    statuses = []
    for variable in variables:
        for index, value in enumerate(summaries.get(variable, {}).get("predicted_values", [])):
            status = "pass"
            if _outside_range(value, spec.fail_low, spec.fail_high):
                status = "fail"
            elif _outside_range(value, spec.warning_low, spec.warning_high):
                status = "warning"
            statuses.append(status)
            if status != "pass" and len(check["offending_values"]) < MAX_OFFENDING_VALUES:
                check["offending_values"].append(
                    {
                        "variable": variable,
                        "step_index": index,
                        "value": value,
                        "status": status,
                        "warning_range": [spec.warning_low, spec.warning_high],
                        "fail_range": [spec.fail_low, spec.fail_high],
                    }
                )

    check["status"] = max_status(statuses)
    if check["status"] == "pass":
        check["message"] = "All selected variables are inside the configured operating window."
    else:
        check["message"] = f"{len(check['offending_values'])} value(s) crossed the configured operating window."
    return check


def evaluate_summary_metric(spec: ConstraintSpec, summaries: Dict[str, Dict[str, Any]], variables: Sequence[str]) -> Dict[str, Any]:
    check = base_check(spec, variables)
    if not variables:
        check["message"] = "No matching variables were available for this rule."
        return check

    statuses = []
    for variable in variables:
        value = summaries.get(variable, {}).get(spec.metric)
        if value is None:
            continue
        status = status_from_threshold(float(value), spec.warning_threshold, spec.fail_threshold)
        statuses.append(status)
        if status != "pass" and len(check["offending_values"]) < MAX_OFFENDING_VALUES:
            check["offending_values"].append(
                {
                    "variable": variable,
                    "metric": spec.metric,
                    "value": value,
                    "status": status,
                    "warning_threshold": spec.warning_threshold,
                    "fail_threshold": spec.fail_threshold,
                }
            )

    check["status"] = max_status(statuses)
    if check["status"] == "pass":
        check["message"] = f"All selected variables pass {spec.metric}."
    else:
        check["message"] = f"{len(check['offending_values'])} variable(s) crossed {spec.metric} threshold."
    return check


def evaluate_boundary_change(spec: ConstraintSpec, parsed_task: Dict[str, Any]) -> Dict[str, Any]:
    changed_variable = parsed_task.get("disturbance_variable") or parsed_task.get("changed_variable")
    variables = [changed_variable] if changed_variable else []
    check = base_check(spec, variables)
    change_percent = parsed_task.get("disturbance_magnitude_percent")
    if change_percent is None:
        change_percent = parsed_task.get("change_percent")
    if change_percent is None:
        check["message"] = "No boundary-control adjustment magnitude was parsed."
        return check

    magnitude = abs(float(change_percent))
    status = status_from_threshold(magnitude, spec.warning_threshold, spec.fail_threshold)
    check["status"] = status
    if status == "pass":
        check["message"] = "Boundary-control adjustment is within the allowable magnitude."
    else:
        check["message"] = "Boundary-control adjustment magnitude requires review."
        check["offending_values"].append(
            {
                "variable": changed_variable,
                "metric": "abs_change_percent",
                "value": magnitude,
                "status": status,
                "warning_threshold": spec.warning_threshold,
                "fail_threshold": spec.fail_threshold,
            }
        )
    return check


def evaluate_spec(
    spec: ConstraintSpec,
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Dict[str, Any],
) -> Dict[str, Any]:
    if spec.metric == "boundary_change_percent":
        return evaluate_boundary_change(spec, parsed_task)

    names = list(summaries)
    variables = variables_matching(names, spec.prefixes, spec.suffixes)
    if spec.metric == "predicted_range":
        return evaluate_range(spec, summaries, variables)
    return evaluate_summary_metric(spec, summaries, variables)


def run_specs(
    specs: Sequence[ConstraintSpec],
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [evaluate_spec(spec, summaries, parsed_task) for spec in specs]


def _outside_range(value: float, low: Optional[float], high: Optional[float]) -> bool:
    return (low is not None and value < low) or (high is not None and value > high)
