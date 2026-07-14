from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..rule_library import load_pipeline_constraints, load_rule_document
from ..schemas import ConstraintSpec


STATUS_RANK = {"not_evaluated": -1, "pass": 0, "warning": 1, "fail": 2}
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


def registry_index(parsed_task: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    entries = (parsed_task or {}).get("_variable_registry") or []
    return {
        str(item.get("variable")): dict(item)
        for item in entries
        if isinstance(item, dict) and item.get("variable")
    }


def variables_matching(
    names: Iterable[str],
    *,
    registry: Dict[str, Dict[str, Any]],
    physical_quantities: Tuple[str, ...] = (),
    equipment_types: Tuple[str, ...] = (),
    roles: Tuple[str, ...] = (),
) -> List[str]:
    if not registry:
        raise ValueError("Variable registry metadata is required for constraint selection.")
    result = []
    for name in names:
        metadata = registry.get(name, {})
        if physical_quantities and metadata.get("physical_quantity") not in physical_quantities:
            continue
        if equipment_types and metadata.get("equipment_type") not in equipment_types:
            continue
        if roles and metadata.get("role") not in roles:
            continue
        result.append(name)
    return result


def variables_for_spec(
    spec: ConstraintSpec,
    names: Iterable[str],
    parsed_task: Optional[Dict[str, Any]],
) -> List[str]:
    return variables_matching(
        names,
        registry=registry_index(parsed_task),
        physical_quantities=spec.physical_quantities,
        equipment_types=spec.equipment_types,
        roles=spec.roles,
    )


def variables_for_selector(
    names: Iterable[str],
    selector: Dict[str, Any],
    parsed_task: Optional[Dict[str, Any]],
) -> List[str]:
    return variables_matching(
        names,
        registry=registry_index(parsed_task),
        physical_quantities=tuple(selector.get("physical_quantities") or ()),
        equipment_types=tuple(selector.get("equipment_types") or ()),
        roles=tuple(selector.get("roles") or ()),
    )


def range_limits_for_variable(
    spec: ConstraintSpec,
    variable: str,
    parsed_task: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str]:
    metadata = registry_index(parsed_task).get(variable, {})
    if spec.use_registry_limits and metadata:
        warning_low = metadata.get("warning_lower_limit", spec.warning_low)
        warning_high = metadata.get("warning_upper_limit", spec.warning_high)
        fail_low = metadata.get("lower_limit", spec.fail_low)
        fail_high = metadata.get("upper_limit", spec.fail_high)
        if any(value is not None for value in (warning_low, warning_high, fail_low, fail_high)):
            return warning_low, warning_high, fail_low, fail_high, "variable_registry"
    return spec.warning_low, spec.warning_high, spec.fail_low, spec.fail_high, "rule_library"


def threshold_limits_for_variable(
    spec: ConstraintSpec,
    variable: str,
    parsed_task: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], Optional[float], str]:
    metadata = registry_index(parsed_task).get(variable, {})
    if spec.use_registry_limits and metadata:
        warning = _absolute_limit(metadata.get("warning_lower_limit"), metadata.get("warning_upper_limit"))
        fail = _absolute_limit(metadata.get("lower_limit"), metadata.get("upper_limit"))
        if warning is not None or fail is not None:
            return warning if warning is not None else spec.warning_threshold, fail if fail is not None else spec.fail_threshold, "variable_registry"
    return spec.warning_threshold, spec.fail_threshold, "rule_library"


def max_status(statuses: Iterable[str]) -> str:
    return max(statuses, key=lambda status: STATUS_RANK.get(status, -1), default="not_evaluated")


def status_from_threshold(value: float, warning: Optional[float], fail: Optional[float]) -> str:
    magnitude = abs(value)
    if fail is not None and magnitude >= fail:
        return "fail"
    if warning is not None and magnitude >= warning:
        return "warning"
    return "pass"


def category_status(checks: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    result = {category: "not_evaluated" for category in CATEGORY_ORDER}
    for check in checks:
        category = check["category"]
        result[category] = max_status([result.get(category, "not_evaluated"), check["status"]])
    return result


def base_check(spec: ConstraintSpec, variables: Sequence[str]) -> Dict[str, Any]:
    return {
        "name": spec.name,
        "category": spec.category,
        "status": "not_evaluated",
        "evaluation_status": "not_evaluated",
        "flag": None,
        "priority": spec.priority,
        "variables": list(variables),
        "description": spec.description,
        "main_content": CATEGORY_DETAILS[spec.category],
        "message": "The rule was not evaluated because required input values were unavailable.",
        "evaluated_values": [],
        "offending_values": [],
    }


def evaluate_range(
    spec: ConstraintSpec,
    summaries: Dict[str, Dict[str, Any]],
    variables: Sequence[str],
    parsed_task: Dict[str, Any],
) -> Dict[str, Any]:
    check = base_check(spec, variables)
    if not variables:
        check["message"] = "No matching variables were available for this rule."
        return check

    statuses = []
    for variable in variables:
        warning_low, warning_high, fail_low, fail_high, limit_source = range_limits_for_variable(
            spec, variable, parsed_task
        )
        predicted_values = summaries.get(variable, {}).get("predicted_values", [])
        variable_statuses = []
        for index, value in enumerate(predicted_values):
            status = "pass"
            if _outside_range(value, fail_low, fail_high):
                status = "fail"
            elif _outside_range(value, warning_low, warning_high):
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
                        "warning_range": [warning_low, warning_high],
                        "fail_range": [fail_low, fail_high],
                        "limit_source": limit_source,
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
                    "warning_range": [warning_low, warning_high],
                    "fail_range": [fail_low, fail_high],
                    "warning_margin": _range_margin(minimum, maximum, warning_low, warning_high),
                    "fail_margin": _range_margin(minimum, maximum, fail_low, fail_high),
                    "limit_source": limit_source,
                }
            )

    check["status"] = max_status(statuses)
    check["evaluation_status"] = "evaluated" if statuses else "not_evaluated"
    check["flag"] = _flag_for_status(spec, check["status"])
    if check["status"] == "not_evaluated":
        check["message"] = "Matching variables did not contain forecast values for this rule."
    elif check["status"] == "pass":
        check["message"] = "All selected variables are inside the configured operating window."
    else:
        check["message"] = f"{len(check['offending_values'])} value(s) crossed the configured operating window."
    return check


def evaluate_summary_metric(
    spec: ConstraintSpec,
    summaries: Dict[str, Dict[str, Any]],
    variables: Sequence[str],
    parsed_task: Dict[str, Any],
) -> Dict[str, Any]:
    check = base_check(spec, variables)
    if not variables:
        check["message"] = "No matching variables were available for this rule."
        return check

    statuses = []
    for variable in variables:
        warning_threshold, fail_threshold, limit_source = threshold_limits_for_variable(
            spec, variable, parsed_task
        )
        value = summaries.get(variable, {}).get(spec.metric)
        if value is None:
            continue
        status = status_from_threshold(float(value), warning_threshold, fail_threshold)
        statuses.append(status)
        if len(check["evaluated_values"]) < MAX_EVALUATED_VALUES:
            evaluated = {
                "variable": variable,
                "metric": spec.metric,
                "value": value,
                "status": status,
                "warning_threshold": warning_threshold,
                "fail_threshold": fail_threshold,
                "warning_margin": _threshold_margin(float(value), warning_threshold),
                "fail_margin": _threshold_margin(float(value), fail_threshold),
                "limit_source": limit_source,
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
                    "warning_threshold": warning_threshold,
                    "fail_threshold": fail_threshold,
                    "limit_source": limit_source,
                }
            )

    check["status"] = max_status(statuses)
    check["evaluation_status"] = "evaluated" if statuses else "not_evaluated"
    check["flag"] = _flag_for_status(spec, check["status"])
    if check["status"] == "not_evaluated":
        check["message"] = f"Matching variables did not provide the {spec.metric} metric."
    elif check["status"] == "pass":
        check["message"] = f"All selected variables pass {spec.metric}."
    else:
        check["message"] = f"{len(check['offending_values'])} variable(s) crossed {spec.metric} threshold."
    return check


def evaluate_boundary_change(spec: ConstraintSpec, parsed_task: Dict[str, Any]) -> Dict[str, Any]:
    disturbance_variable = parsed_task.get("disturbance_variable")
    variables = [disturbance_variable] if disturbance_variable else []
    check = base_check(spec, variables)
    disturbance_percent = parsed_task.get("disturbance_magnitude_percent")
    if disturbance_percent is None:
        check["message"] = "No boundary-control adjustment magnitude was parsed."
        return check

    magnitude = abs(float(disturbance_percent))
    status = status_from_threshold(magnitude, spec.warning_threshold, spec.fail_threshold)
    check["status"] = status
    check["evaluation_status"] = "evaluated"
    check["flag"] = _flag_for_status(spec, status)
    check["evaluated_values"].append(
        {
            "variable": disturbance_variable,
            "metric": "abs_disturbance_percent",
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
                "variable": disturbance_variable,
                "metric": "abs_disturbance_percent",
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
    if spec.metric == "boundary_disturbance_percent":
        return evaluate_boundary_change(spec, parsed_task)

    names = list(summaries)
    variables = variables_for_spec(spec, names, parsed_task)
    if spec.metric == "predicted_range":
        return evaluate_range(spec, summaries, variables, parsed_task)
    return evaluate_summary_metric(spec, summaries, variables, parsed_task)


def run_specs(
    specs: Sequence[ConstraintSpec],
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [evaluate_spec(spec, summaries, parsed_task) for spec in specs]


def contiguous_episodes(
    matching_indices: Iterable[int],
    labels: Sequence[Any],
    time_step_minutes: float,
) -> List[Dict[str, Any]]:
    indices = sorted(set(int(index) for index in matching_indices))
    if not indices:
        return []

    groups: List[List[int]] = [[indices[0]]]
    for index in indices[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])

    episodes = []
    for group in groups:
        start_index = group[0]
        end_index = group[-1]
        episodes.append(
            {
                "start_step_index": start_index,
                "end_step_index": end_index,
                "start_timestamp": labels[start_index] if start_index < len(labels) else None,
                "end_timestamp": labels[end_index] if end_index < len(labels) else None,
                "duration_steps": len(group),
                "duration_minutes": round(len(group) * time_step_minutes, 6),
            }
        )
    return episodes


def threshold_episodes(
    values: Sequence[float],
    predicate: Callable[[float], bool],
    labels: Sequence[Any],
    time_step_minutes: float,
) -> List[Dict[str, Any]]:
    return contiguous_episodes(
        (index for index, value in enumerate(values) if predicate(float(value))),
        labels,
        time_step_minutes,
    )


def longest_episode_minutes(episodes: Sequence[Dict[str, Any]]) -> float:
    return max((float(item.get("duration_minutes") or 0.0) for item in episodes), default=0.0)


def total_episode_minutes(episodes: Sequence[Dict[str, Any]]) -> float:
    return round(sum(float(item.get("duration_minutes") or 0.0) for item in episodes), 6)


def _outside_range(value: float, low: Optional[float], high: Optional[float]) -> bool:
    return (low is not None and value < low) or (high is not None and value > high)


def _absolute_limit(low: Any, high: Any) -> Optional[float]:
    limits = [abs(float(value)) for value in (low, high) if value is not None]
    return max(limits) if limits else None


def _flag_for_status(spec: ConstraintSpec, status: str) -> Optional[str]:
    if status == "not_evaluated":
        return None
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
