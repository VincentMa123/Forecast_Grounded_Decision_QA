from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..rule_library import load_pipeline_constraints, load_rule_document
from ..schemas import ConstraintSpec


STATUS_RANK = {"pass": 0, "warning": 1, "fail": 2}
PIPELINE_CONSTRAINTS = load_pipeline_constraints()
DISPATCH_RULES = load_rule_document("dispatch_priority")
CATEGORY_ORDER = list(PIPELINE_CONSTRAINTS["category_order"])
CATEGORY_DETAILS = dict(PIPELINE_CONSTRAINTS["category_details"])
DISPATCH_PRIORITY_ORDER = [item["id"] for item in DISPATCH_RULES["priority_order"]]
ALWAYS_RUN_CATEGORIES = set(PIPELINE_CONSTRAINTS["always_run_categories"])
MAX_OFFENDING_VALUES = 12
MAX_EVALUATED_VALUES = 50


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
        "flag": _flag_for_status(spec, "pass"),
        "priority": spec.priority,
        "variables": list(variables),
        "description": spec.description,
        "main_content": CATEGORY_DETAILS[spec.category],
        "message": "No violation detected.",
        "evaluated_values": [],
        "offending_values": [],
    }


def evaluate_range(spec: ConstraintSpec, summaries: Dict[str, Dict[str, Any]], variables: Sequence[str]) -> Dict[str, Any]:
    check = base_check(spec, variables)
    if not variables:
        check["message"] = "No matching variables were available for this rule."
        return check

    statuses = []
    for variable in variables:
        predicted_values = summaries.get(variable, {}).get("predicted_values", [])
        variable_statuses = []
        for index, value in enumerate(predicted_values):
            status = "pass"
            if _outside_range(value, spec.fail_low, spec.fail_high):
                status = "fail"
            elif _outside_range(value, spec.warning_low, spec.warning_high):
                status = "warning"
            statuses.append(status)
            variable_statuses.append(status)
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

        if predicted_values and len(check["evaluated_values"]) < MAX_EVALUATED_VALUES:
            minimum = min(predicted_values)
            maximum = max(predicted_values)
            check["evaluated_values"].append(
                {
                    "variable": variable,
                    "metric": spec.metric,
                    "min_prediction": minimum,
                    "min_step_index": predicted_values.index(minimum),
                    "max_prediction": maximum,
                    "max_step_index": predicted_values.index(maximum),
                    "status": max_status(variable_statuses),
                    "warning_range": [spec.warning_low, spec.warning_high],
                    "fail_range": [spec.fail_low, spec.fail_high],
                    "warning_margin": _range_margin(minimum, maximum, spec.warning_low, spec.warning_high),
                    "fail_margin": _range_margin(minimum, maximum, spec.fail_low, spec.fail_high),
                }
            )

    check["status"] = max_status(statuses)
    check["flag"] = _flag_for_status(spec, check["status"])
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
        if len(check["evaluated_values"]) < MAX_EVALUATED_VALUES:
            evaluated = {
                "variable": variable,
                "metric": spec.metric,
                "value": value,
                "status": status,
                "warning_threshold": spec.warning_threshold,
                "fail_threshold": spec.fail_threshold,
                "warning_margin": _threshold_margin(float(value), spec.warning_threshold),
                "fail_margin": _threshold_margin(float(value), spec.fail_threshold),
            }
            if spec.metric == "max_abs_prediction":
                predicted_values = summaries.get(variable, {}).get("predicted_values", [])
                if predicted_values:
                    peak_step_index = max(range(len(predicted_values)), key=lambda index: abs(predicted_values[index]))
                    evaluated["peak_value"] = predicted_values[peak_step_index]
                    evaluated["peak_step_index"] = peak_step_index
            check["evaluated_values"].append(evaluated)
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
    check["flag"] = _flag_for_status(spec, check["status"])
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
    check["flag"] = _flag_for_status(spec, status)
    check["evaluated_values"].append(
        {
            "variable": changed_variable,
            "metric": "abs_change_percent",
            "value": magnitude,
            "status": status,
            "warning_threshold": spec.warning_threshold,
            "fail_threshold": spec.fail_threshold,
            "warning_margin": _threshold_margin(magnitude, spec.warning_threshold),
            "fail_margin": _threshold_margin(magnitude, spec.fail_threshold),
        }
    )
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


def _flag_for_status(spec: ConstraintSpec, status: str) -> str:
    configured = {
        "pass": spec.pass_flag,
        "warning": spec.warning_flag,
        "fail": spec.fail_flag,
    }.get(status)
    return configured or f"{spec.category}_{status}"


def _threshold_margin(value: float, threshold: Optional[float]) -> Optional[float]:
    if threshold is None:
        return None
    return round(float(threshold) - abs(float(value)), 6)


def _range_margin(
    minimum: float,
    maximum: float,
    low: Optional[float],
    high: Optional[float],
) -> Optional[float]:
    margins = []
    if low is not None:
        margins.append(float(minimum) - float(low))
    if high is not None:
        margins.append(float(high) - float(maximum))
    return round(min(margins), 6) if margins else None
