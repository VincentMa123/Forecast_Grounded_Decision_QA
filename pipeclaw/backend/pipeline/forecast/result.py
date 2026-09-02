from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


COMPACT_COMPARABLE_METRIC_KEYS = (
    "energy_consumption",
    "energy_consumption_delta",
    "energy_unit",
    "energy_variable_count",
    "baseline_reference",
)

COMPACT_OUTPUT_SUMMARY_KEYS = (
    "mean_prediction",
    "minimum_prediction",
    "minimum_step_index",
    "maximum_prediction",
    "maximum_step_index",
    "max_abs_prediction",
    "peak_value",
    "peak_step_index",
    "prediction_change",
    "max_abs_step_change",
    "max_abs_step_change_index",
    "max_step_decline",
    "max_step_decline_index",
    "max_decline_from_start",
    "recovery_from_minimum",
)


def without_none_values(value: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _resolved_variable_count(
    parsed: Mapping[str, Any],
    *,
    count_key: str,
    variable_key: str,
) -> int:
    """Keep canonical counts when their source lists were intentionally omitted."""
    explicit = parsed.get(count_key)
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
        return explicit
    return len(parsed.get(variable_key) or [])


def compact_parsed_task(output: Mapping[str, Any]) -> Dict[str, Any]:
    """Create the one bounded parsed-task representation used by public results."""
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
    compact["resolved_attention_variable_count"] = _resolved_variable_count(
        parsed,
        count_key="resolved_attention_variable_count",
        variable_key="resolved_attention_variables",
    )
    compact["resolved_output_variable_count"] = _resolved_variable_count(
        parsed,
        count_key="resolved_output_variable_count",
        variable_key="resolved_output_variables",
    )
    compact["boundary_conditions"] = compact_boundary
    return without_none_values(compact)


def _relevant_forecast_variables(
    output: Mapping[str, Any],
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


def compact_forecast_window(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    stored = metadata.get("forecast_window")
    if isinstance(stored, Mapping):
        return dict(stored)

    real_rows = metadata.get("real_rows")
    predict_rows = metadata.get("predict_rows")
    labels = metadata.get("forecast_time_labels")
    if not isinstance(labels, list):
        labels = []
    if not labels:
        source_rows = predict_rows if isinstance(predict_rows, list) else real_rows
        labels = [
            str(label).removesuffix("_real").removesuffix("_predict")
            for label in source_rows or []
        ]
    window = {
        "start_time": labels[0] if labels else None,
        "end_time": labels[-1] if labels else None,
        "time_step_minutes": metadata.get("time_step_minutes"),
        "real_row_count": len(real_rows) if isinstance(real_rows, list) else 0,
        "predict_row_count": len(predict_rows) if isinstance(predict_rows, list) else 0,
    }
    return without_none_values(window)


def source_name(value: Any) -> Optional[str]:
    return Path(str(value)).name if value else None


def _compact_output_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: summary[key]
        for key in COMPACT_OUTPUT_SUMMARY_KEYS
        if summary.get(key) is not None
    }


def compact_output_summaries(
    summaries: Mapping[str, Any], variables: Iterable[str]
) -> Dict[str, Any]:
    return {
        variable: _compact_output_summary(dict(summaries[variable] or {}))
        for variable in variables
        if variable in summaries
    }


def _compact_prediction(output: Mapping[str, Any]) -> Dict[str, Any]:
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
    )
    compact = {key: prediction.get(key) for key in keys}
    summaries = dict(prediction.get("output_forecast_summary") or {})
    compact["output_forecast_summary"] = compact_output_summaries(
        summaries, _relevant_forecast_variables(output)
    )
    compact["total_output_variable_count"] = len(summaries)
    forecast_window = compact_forecast_window(metadata)
    compact.update(
        {
            "forecast_window": forecast_window or None,
            "actual_forecast_steps": metadata.get("actual_forecast_steps"),
            "actual_forecast_horizon_minutes": metadata.get(
                "actual_forecast_horizon_minutes"
            ),
        }
    )
    return without_none_values(compact)


def _compact_finding(finding: Mapping[str, Any]) -> Dict[str, Any]:
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
    return without_none_values(compact)


def _compact_verification(output: Mapping[str, Any]) -> Dict[str, Any]:
    verification = dict(output.get("constraint_check") or {})
    checks = list(verification.get("checks") or [])
    compact = {
        "requested_categories": verification.get("requested_categories"),
        "category_status": verification.get("category_status"),
        "safety_energy_comparison": verification.get("safety_energy_comparison"),
        "rule_status": {
            str(check.get("name")): check.get("status")
            for check in checks
            if check.get("name")
        },
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
        "priority_findings": [
            _compact_finding(item) for item in verification.get("priority_findings", [])
        ],
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
    return without_none_values(compact)


def _compact_execution(output: Mapping[str, Any]) -> Dict[str, Any]:
    parsed_task = compact_parsed_task(output)
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
        "vocabulary_normalizations": parsed_task.get("vocabulary_normalizations", []),
        "invalid_normalized_variables": parsed_task.get(
            "invalid_normalized_variables", []
        ),
    }
    provenance = {
        "checkpoint_id": metadata.get("checkpoint_id")
        or source_name(metadata.get("checkpoint_dir")),
        "data_case_id": metadata.get("data_case_id")
        or source_name(metadata.get("data_case_dir")),
        "device": metadata.get("device"),
        "model_input_projection_type": metadata.get("model_input_projection_type"),
        "data_provenance": metadata.get("data_provenance"),
    }
    return {
        "success": True,
        "parsed_task": parsed_task,
        "task_resolution": without_none_values(task_resolution),
        "prediction": _compact_prediction(output),
        "verification": _compact_verification(output),
        "evidence": dict(output.get("evidence") or {}),
        "risk_level": output.get("risk_level"),
        "manual_intervention_label": output.get("manual_intervention_label"),
        "dispatch_recommendation": output.get("dispatch_recommendation"),
        "provenance": without_none_values(provenance),
    }


class ForecastResult(BaseModel):
    """The released compact forecast-result mapping with a typed boundary."""

    success: Literal[True]
    parsed_task: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    task_resolution: Dict[str, Any]
    prediction: Dict[str, Any]
    verification: Dict[str, Any]
    evidence: Dict[str, Any]
    risk_level: Optional[str] = None
    manual_intervention_label: Optional[str] = None
    dispatch_recommendation: Optional[str] = None
    provenance: Dict[str, Any]

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_execution(cls, legacy: Mapping[str, Any]) -> "ForecastResult":
        """Validate a successful detailed execution as the compact contract."""
        if not isinstance(legacy, Mapping):
            raise TypeError("legacy execution must be a mapping")
        if legacy.get("success") is not True:
            raise ValueError("ForecastResult requires a successful execution payload")
        return cls.model_validate(_compact_execution(legacy))

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "ForecastResult":
        """Accept either a detailed execution or a public result in a trace envelope."""
        if not isinstance(value, Mapping):
            raise TypeError("forecast payload must be a mapping")
        public_fields = {"task_resolution", "prediction", "verification", "provenance"}
        if public_fields <= value.keys():
            return cls.model_validate(
                {
                    key: item
                    for key, item in value.items()
                    if key not in {"candidate_id", "candidate_role"}
                }
            )
        return cls.from_execution(value)


__all__ = [
    "COMPACT_COMPARABLE_METRIC_KEYS",
    "ForecastResult",
    "compact_forecast_window",
    "compact_parsed_task",
]
