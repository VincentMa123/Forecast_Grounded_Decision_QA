"""Canonical compact projection of PipeFormer tool output."""

from __future__ import annotations

from typing import Any, Dict, List


COMPACT_COMPARABLE_METRIC_KEYS = (
    "energy_consumption",
    "energy_consumption_delta",
    "energy_unit",
    "energy_variable_count",
    "baseline_reference",
)


def _without_none_values(value: Dict[str, Any]) -> Dict[str, Any]:
    """Keep stable list/dict fields while omitting values that are truly absent."""
    return {key: item for key, item in value.items() if item is not None}


def _compact_parsed_task(output: Dict[str, Any]) -> Dict[str, Any]:
    parsed = dict(output.get("parsed_task") or {})
    boundary = dict(parsed.get("boundary_conditions") or {})
    compact_boundary = {
        key: boundary.get(key)
        for key in ("keep_other_boundary_controls", "setpoints", "percentage_changes")
        if key in boundary
    }
    keys = (
        "case_id",
        "current_operating_condition_number",
        "disturbance_variable",
        "disturbance_direction",
        "disturbance_magnitude_percent",
        "disturbance_assumption",
        "disturbance_source",
        "forecast_horizon_minutes",
        "attention_targets",
        "output_state_variables",
        "constraint_verification_types",
        "task_type",
        "forecast_time_step_minutes",
        "unresolved_attention_targets",
        "unresolved_output_state_variables",
        "variable_normalizations",
        "vocabulary_normalizations",
        "invalid_normalized_variables",
    )
    compact = {key: parsed.get(key) for key in keys}
    compact["resolved_attention_variable_count"] = len(
        parsed.get("resolved_attention_variables") or []
    )
    compact["resolved_output_variable_count"] = len(
        parsed.get("resolved_output_variables") or []
    )
    compact["boundary_conditions"] = compact_boundary
    return _without_none_values(compact)


def _relevant_forecast_variables(
    output: Dict[str, Any],
    limit: int = 8,
) -> List[str]:
    relevant: List[str] = []
    summaries = dict(
        (output.get("prediction_summary") or {}).get("output_forecast_summary") or {}
    )
    available = set(summaries)

    def add(value: Any) -> None:
        variable = str(value or "").strip()
        if variable in available and variable not in relevant and len(relevant) < limit:
            relevant.append(variable)

    evidence = dict(output.get("evidence") or {})
    for key in ("top_watch_variables", "key_observation_variables"):
        for item in evidence.get(key) or []:
            add(item.get("variable"))
    verification = dict(output.get("constraint_check") or {})
    for finding in verification.get("priority_findings") or []:
        for item in list(finding.get("offending_values") or []) + list(
            finding.get("evaluated_values") or []
        ):
            add(item.get("variable"))
    add((output.get("parsed_task") or {}).get("disturbance_variable"))
    for variable in summaries:
        add(variable)
    return relevant


def _compact_prediction_summary(output: Dict[str, Any]) -> Dict[str, Any]:
    prediction = dict(output.get("prediction_summary") or {})
    metadata = dict(output.get("forecast_metadata") or {})
    keys = (
        "forecast_mode",
        "case_id",
        "current_operating_condition_number",
        "forecast_horizon_minutes",
        "disturbance_variable",
        "disturbance_direction",
        "disturbance_magnitude_percent",
        "disturbance_assumption",
        "disturbance_source",
        "counterfactual_comparison",
        "output_forecast_summary",
    )
    compact = {
        key: prediction.get(key)
        for key in keys
        if key != "output_forecast_summary"
    }
    summaries = dict(prediction.get("output_forecast_summary") or {})
    relevant_variables = _relevant_forecast_variables(output)
    compact["output_forecast_summary"] = {
        variable: summaries[variable]
        for variable in relevant_variables
        if variable in summaries
    }
    compact["total_output_variable_count"] = len(summaries)
    compact.update(
        {
            "forecast_window": metadata.get("forecast_window"),
            "actual_forecast_steps": metadata.get("actual_forecast_steps"),
            "actual_forecast_horizon_minutes": metadata.get(
                "actual_forecast_horizon_minutes"
            ),
        }
    )
    return _without_none_values(compact)


def _compact_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
    evaluated = [
        item
        for item in finding.get("evaluated_values", [])
        if item.get("status") in {"warning", "fail"}
    ]
    values = evaluated or list(finding.get("offending_values", []))
    compact = {
        key: finding.get(key)
        for key in (
            "name",
            "category",
            "status",
            "evaluation_status",
            "flag",
            "priority",
            "message",
        )
    }
    compact["evaluated_variable_count"] = len(finding.get("variables") or [])
    compact["affected_variables"] = list(
        dict.fromkeys(
            str(item.get("variable")) for item in values if item.get("variable")
        )
    )[:3]
    compact["evaluated_values"] = values[:3]
    if finding.get("operating_envelope_status"):
        compact["operating_envelope_status"] = finding["operating_envelope_status"]
    return _without_none_values(compact)


def _compact_constraint_check(output: Dict[str, Any]) -> Dict[str, Any]:
    verification = dict(output.get("constraint_check") or {})
    checks = list(verification.get("checks") or [])
    findings = [
        _compact_finding(item) for item in verification.get("priority_findings", [])
    ]
    rule_status = {
        str(check.get("name")): check.get("status")
        for check in checks
        if check.get("name")
    }
    compact = {
        "requested_categories": verification.get("requested_categories"),
        "category_status": verification.get("category_status"),
        "safety_energy_comparison": verification.get("safety_energy_comparison"),
        "rule_status": rule_status,
        "overall_status": verification.get("overall_status"),
        "verification_complete": verification.get("verification_complete"),
        "not_evaluated_rules": verification.get("not_evaluated_rules"),
        "risk_level": verification.get("risk_level"),
        "risk_escalations": verification.get("risk_escalations"),
        "failure_count": verification.get("failure_count", 0),
        "warning_count": verification.get("warning_count", 0),
        "omitted_warning_count": verification.get("omitted_warning_count", 0),
        "failed_rule_ids": verification.get("failed_rule_ids", []),
        "warning_rule_ids": verification.get("warning_rule_ids", []),
        "triggered_flags": verification.get("triggered_flags", []),
        "human_intervention_label": verification.get("human_intervention_label"),
        "dispatch_recommendation": verification.get("dispatch_recommendation"),
        "priority_findings": findings,
        "engineering_evidence": verification.get("engineering_evidence", {}),
    }
    comparable_metrics = dict(verification.get("comparable_metrics") or {})
    if comparable_metrics:
        compact["comparable_metrics"] = {
            key: comparable_metrics[key]
            for key in COMPACT_COMPARABLE_METRIC_KEYS
            if key in comparable_metrics
        }
        if "energy_evaluation_status" in comparable_metrics:
            compact["comparable_metrics"]["energy_evaluation_status"] = (
                comparable_metrics["energy_evaluation_status"]
            )
    return _without_none_values(compact)


def project_pipeformer_output(output: Dict[str, Any]) -> Dict[str, Any]:
    parsed_task = _compact_parsed_task(output)
    metadata = dict(output.get("forecast_metadata") or {})
    task_resolution = {
        "resolved_attention_variable_count": parsed_task.get(
            "resolved_attention_variable_count", 0
        ),
        "resolved_output_variable_count": parsed_task.get(
            "resolved_output_variable_count", 0
        ),
        "unresolved_attention_targets": parsed_task.get(
            "unresolved_attention_targets", []
        ),
        "unresolved_output_state_variables": parsed_task.get(
            "unresolved_output_state_variables", []
        ),
        "applied_boundary_conditions": metadata.get("applied_boundary_conditions", []),
        "variable_normalizations": parsed_task.get("variable_normalizations", []),
        "vocabulary_normalizations": parsed_task.get(
            "vocabulary_normalizations", []
        ),
        "invalid_normalized_variables": parsed_task.get(
            "invalid_normalized_variables", []
        ),
    }
    provenance = {
        "checkpoint_id": metadata.get("checkpoint_id"),
        "data_case_id": metadata.get("data_case_id"),
        "device": metadata.get("device"),
        "model_input_projection_type": metadata.get("model_input_projection_type"),
        "data_provenance": metadata.get("data_provenance"),
    }
    return {
        "parsed_task": parsed_task,
        "prediction_summary": _compact_prediction_summary(output),
        "constraint_check": _compact_constraint_check(output),
        "evidence": dict(output.get("evidence") or {}),
        "risk_level": output.get("risk_level"),
        "manual_intervention_label": output.get("manual_intervention_label"),
        "dispatch_recommendation": output.get("dispatch_recommendation"),
        "task_resolution": _without_none_values(task_resolution),
        "provenance": _without_none_values(provenance),
    }


def compact_pipeformer_output(projection: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": True,
        "task_resolution": projection["task_resolution"],
        "prediction": projection["prediction_summary"],
        "verification": projection["constraint_check"],
        "evidence": projection["evidence"],
        "risk_level": projection.get("risk_level"),
        "manual_intervention_label": projection.get("manual_intervention_label"),
        "dispatch_recommendation": projection.get("dispatch_recommendation"),
        "provenance": projection["provenance"],
    }
