from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional


REGISTRY_SEARCH_TOOL = "search_pipeformer_registry"
FORECAST_TOOL = "run_pipeformer_forecast"


def _normalized(value: Any) -> str:
    return str(value or "").strip().casefold()


def _successful_search(call: Dict[str, Any]) -> bool:
    if call.get("name") != REGISTRY_SEARCH_TOOL:
        return False
    output = call.get("output")
    return (
        isinstance(output, dict)
        and output.get("success") is True
        and not output.get("error")
        and output.get("exit_code") in (None, 0)
    )


def _returned_entries(call: Dict[str, Any], variable: str) -> List[Dict[str, Any]]:
    target = _normalized(variable)
    output = dict(call.get("output") or {})
    return [
        dict(item)
        for item in output.get("variables") or []
        if isinstance(item, dict) and _normalized(item.get("variable")) == target
    ]


def _search_terms(arguments: Dict[str, Any]) -> set[str]:
    terms = set()
    query = _normalized(arguments.get("query"))
    if query:
        terms.add(query)
    for key in (
        "equipment_ids",
        "equipment_types",
        "physical_quantities",
        "attention_targets",
    ):
        terms.update(
            _normalized(value)
            for value in arguments.get(key) or []
            if _normalized(value)
        )
    return terms


def _normalization_provenance(
    forecast_arguments: Dict[str, Any],
    variable: str,
    search_arguments: Dict[str, Any],
) -> bool:
    target = _normalized(variable)
    search_terms = _search_terms(search_arguments)
    if not search_terms:
        return False
    for item in forecast_arguments.get("vocabulary_normalizations") or []:
        if not isinstance(item, dict):
            continue
        canonical = {
            _normalized(value) for value in item.get("canonical_variables") or []
        }
        requested_term = _normalized(item.get("requested_term"))
        if (
            item.get("normalization_source") == "registry_search"
            and target in canonical
            and requested_term in search_terms
        ):
            return True
    return False


def _disturbance_search_authorizes(
    call: Dict[str, Any],
    forecast_arguments: Dict[str, Any],
    disturbance_variable: str,
) -> bool:
    if not _successful_search(call) or not _returned_entries(
        call, disturbance_variable
    ):
        return False
    arguments = dict(call.get("arguments") or {})
    if _normalized(arguments.get("query")) == _normalized(disturbance_variable):
        return True
    return _normalization_provenance(
        forecast_arguments,
        disturbance_variable,
        arguments,
    )


def _candidate_search_authorizes(
    call: Dict[str, Any],
    candidate_variable: str,
) -> bool:
    if not _successful_search(call):
        return False
    arguments = dict(call.get("arguments") or {})
    if (
        _normalized(arguments.get("role")) != "input"
        or _normalized(arguments.get("controllable")) != "true"
    ):
        return False
    return any(
        _normalized(item.get("role")) == "input" and item.get("controllable") is True
        for item in _returned_entries(call, candidate_variable)
    )


def _as_mapping(value: Any) -> Dict[str, Any]:
    """Trust boundary: accept mappings and single-keyed-dict lists; {} otherwise."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list) and all(
        isinstance(item, Mapping) and len(item) == 1 for item in value
    ):
        return {key: item[key] for item in value for key in item}
    return {}


def candidate_action_variables(forecast_arguments: Dict[str, Any]) -> List[str]:
    """Return stable, unique candidate boundary-action variable IDs."""
    boundary = _as_mapping(forecast_arguments.get("boundary_conditions"))
    variables = {
        str(variable)
        for key in ("percentage_changes", "setpoints")
        for variable in _as_mapping(boundary.get(key))
        if str(variable).strip()
    }
    disturbance_variable = str(
        forecast_arguments.get("disturbance_variable") or ""
    ).strip()
    if disturbance_variable.endswith(":ST") and disturbance_variable in _as_mapping(
        boundary.get("setpoints")
    ):
        variables.discard(disturbance_variable)
    return sorted(variables, key=str.casefold)


def authorize_forecast_registry(
    forecast_arguments: Dict[str, Any],
    completed_tool_calls: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify that prior successful registry results authorize forecast variables."""
    calls = [dict(item) for item in completed_tool_calls]
    disturbance_variable = str(
        forecast_arguments.get("disturbance_variable") or ""
    ).strip()
    candidate_variables = candidate_action_variables(forecast_arguments)
    candidate_id = str(forecast_arguments.get("candidate_id") or "").strip()
    candidate_role = str(
        forecast_arguments.get("candidate_role") or "candidate"
    ).casefold()
    disturbance_search_call_ids: List[str] = []
    candidate_search_call_ids: Dict[str, List[str]] = {}
    issues: List[Dict[str, str]] = []

    if candidate_id and candidate_role != "baseline" and not candidate_variables:
        issues.append(
            {
                "code": "candidate_action_missing",
                "message": (
                    f"Dispatch candidate {candidate_id} has no boundary-control "
                    "setpoint or percentage change."
                ),
                "retry_instruction": (
                    "For a dispatch candidate, add at least one different registered "
                    "controllable input under boundary_conditions.setpoints or "
                    "boundary_conditions.percentage_changes. For a disturbance-only "
                    "reference forecast, use candidate_role=baseline. For a single "
                    "prediction, omit candidate_id."
                ),
            }
        )

    if not disturbance_variable:
        issues.append(
            {
                "code": "missing_disturbance_variable",
                "message": "A canonical disturbance_variable is required.",
                "retry_instruction": (
                    "Search the registry for the exact disturbance variable or a meaningful "
                    "equipment/physical-quantity/attention target, then retry with the returned "
                    "canonical disturbance_variable and normalization provenance."
                ),
            }
        )
    else:
        disturbance_search_call_ids = [
            str(call.get("tool_call_id") or "")
            for call in calls
            if _disturbance_search_authorizes(
                call,
                forecast_arguments,
                disturbance_variable,
            )
        ]
        if not disturbance_search_call_ids:
            issues.append(
                {
                    "code": "disturbance_registry_evidence_missing",
                    "message": (
                        f"No preceding successful relevant registry search returned "
                        f"disturbance variable {disturbance_variable}."
                    ),
                    "retry_instruction": (
                        f"Search exact ID {disturbance_variable!r}, or search by meaningful "
                        "equipment/physical quantity/attention target and provide a "
                        "registry_search vocabulary normalization. Zero-match and broad "
                        "role-only searches do not authorize a disturbance mapping."
                    ),
                }
            )

    for variable in candidate_variables:
        matching_call_ids = [
            str(call.get("tool_call_id") or "")
            for call in calls
            if _candidate_search_authorizes(call, variable)
        ]
        if matching_call_ids:
            candidate_search_call_ids[variable] = matching_call_ids
        else:
            issues.append(
                {
                    "code": "candidate_registry_evidence_missing",
                    "message": (
                        f"No preceding role=input, controllable=true registry result returned "
                        f"candidate action variable {variable}."
                    ),
                    "retry_instruction": (
                        "Run search_pipeformer_registry with role=input and controllable=true, "
                        f"select {variable} only if it appears in the returned variables with "
                        "role=input and controllable=true, then retry."
                    ),
                }
            )

    if disturbance_variable and _normalized(disturbance_variable) in {
        _normalized(value) for value in candidate_variables
    }:
        issues.append(
            {
                "code": "same_variable_disturbance_action",
                "message": (
                    f"Candidate actions cannot modify background disturbance variable "
                    f"{disturbance_variable}."
                ),
                "retry_instruction": (
                    "Keep the disturbance variable only in the disturbance fields and choose "
                    "a different registered controllable input for the candidate action."
                ),
            }
        )

    return {
        "authorized": not issues,
        "disturbance_variable": disturbance_variable or None,
        "candidate_action_variables": candidate_variables,
        "disturbance_search_call_ids": [
            value for value in disturbance_search_call_ids if value
        ],
        "candidate_search_call_ids": {
            variable: [value for value in call_ids if value]
            for variable, call_ids in candidate_search_call_ids.items()
        },
        "issues": issues,
    }


def forecast_registry_failure_result(
    forecast_arguments: Dict[str, Any],
    completed_tool_calls: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return a structured retry result when registry preconditions are not met."""
    authorization = authorize_forecast_registry(
        forecast_arguments,
        completed_tool_calls,
    )
    if authorization["authorized"]:
        return None
    instructions = [
        issue["retry_instruction"]
        for issue in authorization["issues"]
        if issue.get("retry_instruction")
    ]
    return {
        "success": False,
        "record_in_teacher_trace": False,
        "error_code": "forecast_registry_precondition_failed",
        "error": "PipeFormer forecast was not executed because registry preconditions failed.",
        "retry_instructions": instructions,
        "registry_authorization": authorization,
    }
