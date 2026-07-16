from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .condition_parser import (
    CATEGORY_ATTENTION_TARGETS,
    CATEGORY_OUTPUT_STATE_VARIABLES,
    DEFAULT_CONSTRAINT_VERIFICATION_TYPES,
    PIPEFORMER_TASK_SCHEMA_VERSION,
    parse_condition,
)
from .engineering_constraints import EngineeringConstraintEngine
from .evidence_extractor import summarize_variables, top_variables
from .pipeformer_inference import (
    PipeFormerInferenceConfig,
    PipeFormerInferenceEngine,
    find_default_checkpoint_dir,
)


REGISTRY_GROUP_RULES = {
    "nodes": {"equipment_types": {"node"}, "roles": {"output"}},
    "segments": {"equipment_types": {"pipeline_segment", "ball_valve"}, "roles": {"output"}},
    "linepack": {"physical_quantities": {"linepack"}, "roles": {"output"}},
    "compressors": {"equipment_types": {"compressor", "compressor_power"}, "roles": {"output"}},
    "pressure": {"physical_quantities": {"pressure"}, "roles": {"output"}},
    "flow": {"physical_quantities": {"flow"}, "roles": {"output"}},
    "compressor_load": {"physical_quantities": {"compressor_load"}, "roles": {"output"}},
    "compressor": {"equipment_types": {"compressor", "compressor_power"}, "roles": {"output"}},
    "compression_ratio": {"physical_quantities": {"compression_ratio"}, "roles": {"output"}},
    "compressor_speed": {"physical_quantities": {"rotational_speed"}, "roles": {"output"}},
    "compressor_power": {"physical_quantities": {"power"}, "roles": {"output"}},
    "power": {"physical_quantities": {"power"}, "roles": {"output"}},
    "energy": {"physical_quantities": {"power"}, "roles": {"output"}},
    "energy_consumption": {"physical_quantities": {"power"}, "roles": {"output"}},
    "energy_cost": {"physical_quantities": {"power"}, "roles": {"output"}},
    "operating_cost": {"physical_quantities": {"power"}, "roles": {"output"}},
    "valves": {"equipment_types": {"ball_valve"}},
    "pressure_regulators": {"equipment_types": {"pressure_regulator"}},
    "boundary_controls": {"roles": {"input"}, "controllable": True},
    "valve_opening": {"physical_quantities": {"valve_opening"}, "roles": {"output"}},
    "regulator_range": {"physical_quantities": {"regulator_range"}, "roles": {"output"}},
    "boundary_control_adjustment": {"roles": {"input"}, "controllable": True},
    "dispatch_priority_audit": {"roles": {"output"}},
}
COMPACT_OUTPUT_KEYS = (
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
logger = logging.getLogger(__name__)


def _repo_root_from_backend_root(backend_root: Path) -> Path:
    return Path(backend_root).resolve().parents[1]


def _optional_path(path_value: Optional[str]) -> Optional[Path]:
    return Path(path_value).expanduser().resolve() if path_value else None


def _compact_source_name(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    return Path(str(path_value)).name


def _strip_row_suffix(label: str) -> str:
    for suffix in ("_real", "_predict"):
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def _row_label(row: Any) -> str:
    return str(getattr(row, "label", row))


def _forecast_window_summary(forecast_context: Dict[str, Any]) -> Dict[str, Any]:
    real_rows = forecast_context.get("real_rows") or []
    predict_rows = forecast_context.get("predict_rows") or []
    labels = list(forecast_context.get("forecast_time_labels") or [])
    if not labels:
        labels = [_strip_row_suffix(_row_label(row)) for row in predict_rows or real_rows]

    window = {
        "start_time": labels[0] if labels else None,
        "end_time": labels[-1] if labels else None,
        "time_step_minutes": forecast_context.get("time_step_minutes"),
        "real_row_count": len(real_rows),
        "predict_row_count": len(predict_rows),
    }
    return {key: value for key, value in window.items() if value is not None}


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _env_optional_path(name: str, default: Optional[Path] = None) -> Optional[Path]:
    value = os.getenv(name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve() if default else None


def _first_path(*candidates: Optional[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate is not None:
            return candidate.resolve()
    return None


def _clean_checks(requested_categories: Optional[List[str]]) -> List[str]:
    allowed = set(DEFAULT_CONSTRAINT_VERIFICATION_TYPES)
    checks = []
    for item in requested_categories or []:
        check = str(item).strip()
        if check and check in allowed and check not in checks:
            checks.append(check)
    return checks or DEFAULT_CONSTRAINT_VERIFICATION_TYPES.copy()


def _unique_targets(checks: List[str], mapping: Dict[str, List[str]]) -> List[str]:
    result = []
    for check in checks:
        for value in mapping.get(check, []):
            if value not in result:
                result.append(value)
    return result


def _resolve_requested_variables(
    requested: List[str],
    variable_names: List[str],
    registry_entries: List[Dict[str, Any]],
) -> tuple[List[str], List[str]]:
    if not registry_entries:
        raise ValueError("Variable registry metadata is required to resolve PipeFormer targets.")
    registry = {
        str(item.get("variable")): item
        for item in registry_entries
        if isinstance(item, dict) and item.get("variable")
    }
    resolved = []
    unresolved = []
    for raw in requested:
        target = str(raw).strip()
        matches: List[str] = []
        if target in variable_names:
            matches = [target]
        elif target in REGISTRY_GROUP_RULES:
            rule = REGISTRY_GROUP_RULES[target]
            matches = [
                name
                for name in variable_names
                if _registry_group_match(registry.get(name, {}), rule)
            ]
        else:
            matches = [
                name
                for name in variable_names
                if name.startswith(f"{target}_") or name.startswith(f"{target}:")
            ]
        if not matches:
            unresolved.append(target)
        for name in matches:
            if name not in resolved:
                resolved.append(name)
    return resolved, unresolved


def _registry_group_match(metadata: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    if not metadata:
        return False
    for key in ("physical_quantities", "equipment_types", "roles"):
        allowed = rule.get(key)
        metadata_key = {
            "physical_quantities": "physical_quantity",
            "equipment_types": "equipment_type",
            "roles": "role",
        }[key]
        if allowed and metadata.get(metadata_key) not in allowed:
            return False
    if "controllable" in rule and bool(metadata.get("controllable")) != bool(rule["controllable"]):
        return False
    return True


def _compact_output_summaries(
    summaries: Dict[str, Dict[str, Any]],
    variables: List[str],
) -> Dict[str, Dict[str, Any]]:
    return {
        variable: {
            key: summaries[variable].get(key)
            for key in COMPACT_OUTPUT_KEYS
            if summaries[variable].get(key) is not None
        }
        for variable in variables
        if variable in summaries
    }


def _counterfactual_comparison(
    baseline_rows: List[Any],
    disturbed_rows: List[Any],
    output_variables: List[str],
) -> Dict[str, Any]:
    comparisons = []
    for variable in output_variables:
        baseline = [float(row.values[variable]) for row in baseline_rows if variable in row.values]
        disturbed = [float(row.values[variable]) for row in disturbed_rows if variable in row.values]
        compared_steps = min(len(baseline), len(disturbed))
        if not compared_steps:
            continue
        deltas = [disturbed[index] - baseline[index] for index in range(compared_steps)]
        mean_delta = sum(deltas) / len(deltas)
        peak_index = max(range(len(deltas)), key=lambda index: abs(deltas[index]))
        comparisons.append(
            {
                "variable": variable,
                "mean_delta": round(mean_delta, 6),
                "final_delta": round(deltas[-1], 6),
                "max_abs_delta": round(abs(deltas[peak_index]), 6),
                "max_abs_delta_step_index": peak_index,
                "direction": "increase" if mean_delta > 0 else "decrease" if mean_delta < 0 else "unchanged",
            }
        )
    comparisons.sort(key=lambda item: item["max_abs_delta"], reverse=True)
    impacted = [item for item in comparisons if item["max_abs_delta"] > 1e-9]
    return {
        "mode": "baseline_vs_disturbed",
        "compared_step_count": min(len(baseline_rows), len(disturbed_rows)),
        "compared_output_variable_count": len(comparisons),
        "nonzero_impacted_variable_count": len(impacted),
        "top_impacted_variables": impacted[:5],
    }


def _condition_number_from_case_id(case_id: Optional[str]) -> Optional[int]:
    if not case_id:
        return None
    digits = "".join(ch for ch in case_id if ch.isdigit())
    return int(digits) if digits else None


def _normalize_pipeformer_task(parsed: Dict[str, Any]) -> Dict[str, Any]:
    checks = _clean_checks(parsed.get("constraint_verification_types"))
    parsed["constraint_verification_types"] = checks

    parsed.setdefault("case_id", None)
    parsed.setdefault("current_operating_condition_number", _condition_number_from_case_id(parsed.get("case_id")))
    parsed.setdefault("forecast_horizon_minutes", None)
    parsed.setdefault("disturbance_direction", "unknown")
    parsed.setdefault("disturbance_magnitude_percent", None)
    parsed.setdefault("attention_targets", _unique_targets(checks, CATEGORY_ATTENTION_TARGETS))
    parsed.setdefault("output_state_variables", _unique_targets(checks, CATEGORY_OUTPUT_STATE_VARIABLES))

    boundary_conditions = dict(parsed.get("boundary_conditions") or {})
    boundary_conditions.setdefault("keep_other_boundary_controls", True)
    boundary_conditions.setdefault("disturbance_variable", parsed.get("disturbance_variable"))
    boundary_conditions.setdefault("disturbance_direction", parsed.get("disturbance_direction"))
    boundary_conditions.setdefault("disturbance_magnitude_percent", parsed.get("disturbance_magnitude_percent"))
    parsed["boundary_conditions"] = boundary_conditions

    parsed.setdefault("task_type", "prediction_and_verification")
    parsed.setdefault("parse_schema_version", PIPEFORMER_TASK_SCHEMA_VERSION)
    return parsed


def build_pipeformer_task(
    *,
    question: str,
    case_id: Optional[str] = None,
    forecast_horizon_minutes: Optional[int] = None,
    current_operating_condition_number: Optional[int] = None,
    boundary_conditions: Optional[Dict[str, Any]] = None,
    disturbance_variable: Optional[str] = None,
    disturbance_direction: Optional[str] = None,
    disturbance_magnitude_percent: Optional[float] = None,
    attention_targets: Optional[List[str]] = None,
    output_state_variables: Optional[List[str]] = None,
    constraint_verification_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    parse_error: Optional[str] = None
    if question:
        try:
            parsed = parse_condition(question)
        except Exception as exc:
            parse_error = str(exc)

    if case_id is not None:
        parsed["case_id"] = case_id
    if current_operating_condition_number is not None:
        parsed["current_operating_condition_number"] = int(current_operating_condition_number)
    if boundary_conditions is not None:
        parsed["boundary_conditions"] = dict(boundary_conditions)
    if disturbance_variable is not None:
        parsed["disturbance_variable"] = disturbance_variable
    if disturbance_direction is not None:
        parsed["disturbance_direction"] = disturbance_direction
    if disturbance_magnitude_percent is not None:
        parsed["disturbance_magnitude_percent"] = float(disturbance_magnitude_percent)
    if forecast_horizon_minutes is not None:
        parsed["forecast_horizon_minutes"] = int(forecast_horizon_minutes)
    if attention_targets is not None:
        parsed["attention_targets"] = list(attention_targets)
    if output_state_variables is not None:
        parsed["output_state_variables"] = list(output_state_variables)
    if constraint_verification_types is not None:
        parsed["constraint_verification_types"] = _clean_checks(constraint_verification_types)

    parsed = _normalize_pipeformer_task(parsed)
    if not parsed.get("disturbance_variable"):
        raise ValueError(f"PipeFormer forecast requires disturbance_variable. Parse error: {parse_error or 'not parsed'}")
    if parsed.get("disturbance_magnitude_percent") is not None and parsed.get("disturbance_direction") not in {"up", "down"}:
        raise ValueError("PipeFormer forecast requires disturbance_direction to be 'up' or 'down' when disturbance_magnitude_percent is set.")
    return parsed


class PipeFormerForecastService:
    """Coordinate task parsing, checkpoint inference, constraints, and evidence."""

    def __init__(self, backend_root: Path) -> None:
        self.backend_root = Path(backend_root).resolve()

    def analyze(self, **request: Any) -> Dict[str, Any]:
        return run_pipeformer_forecast_analysis(
            backend_root=self.backend_root,
            **request,
        )


def run_pipeformer_forecast_analysis(
    *,
    question: str,
    backend_root: Path,
    case_id: Optional[str] = None,
    forecast_horizon_minutes: Optional[int] = None,
    current_operating_condition_number: Optional[int] = None,
    boundary_conditions: Optional[Dict[str, Any]] = None,
    disturbance_variable: Optional[str] = None,
    disturbance_direction: Optional[str] = None,
    disturbance_magnitude_percent: Optional[float] = None,
    attention_targets: Optional[List[str]] = None,
    output_state_variables: Optional[List[str]] = None,
    constraint_verification_types: Optional[List[str]] = None,
    include_baseline_comparison: bool = False,
    pipeformer_root: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    static_dir: Optional[str] = None,
    mapping_csv: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info("PipeFormer forecast tool started")
    repo_root = _repo_root_from_backend_root(backend_root)
    resolved_pipeformer_root = _optional_path(pipeformer_root) or _env_path("PIPEFORMER_ROOT", repo_root / "pipeFormer")
    resolved_checkpoint_dir = _first_path(
        _optional_path(checkpoint_dir),
        _env_optional_path("PIPEFORMER_CHECKPOINT_DIR"),
    )
    if resolved_checkpoint_dir is None:
        resolved_checkpoint_dir = find_default_checkpoint_dir(repo_root).resolve()
    resolved_data_dir = _optional_path(data_dir) or _env_optional_path("PIPEFORMER_DATA_DIR")
    resolved_static_dir = _optional_path(static_dir) or _env_optional_path("PIPEFORMER_STATIC_DIR")
    mapping_override = _optional_path(mapping_csv) or _env_optional_path("PIPEFORMER_MAPPING_CSV")
    resolved_device = device or os.getenv("PIPEFORMER_DEVICE", "cpu")
    logger.info(
        "PipeFormer path overrides resolved: root=%s checkpoint=%s static=%s mapping=%s device=%s",
        resolved_pipeformer_root,
        resolved_checkpoint_dir,
        resolved_static_dir,
        mapping_override,
        resolved_device,
    )
    inference_engine = PipeFormerInferenceEngine(
        PipeFormerInferenceConfig(
            checkpoint_dir=resolved_checkpoint_dir,
            pipeformer_root=resolved_pipeformer_root,
            data_dir=resolved_data_dir,
            static_dir=resolved_static_dir,
            mapping_path=mapping_override,
            device=resolved_device,
        )
    )
    constraint_engine = EngineeringConstraintEngine()

    parsed_task = build_pipeformer_task(
        question=question,
        case_id=case_id,
        forecast_horizon_minutes=forecast_horizon_minutes,
        current_operating_condition_number=current_operating_condition_number,
        boundary_conditions=boundary_conditions,
        disturbance_variable=disturbance_variable,
        disturbance_direction=disturbance_direction,
        disturbance_magnitude_percent=disturbance_magnitude_percent,
        attention_targets=attention_targets,
        output_state_variables=output_state_variables,
        constraint_verification_types=constraint_verification_types,
    )
    logger.info("PipeFormer parsed task: %s", parsed_task)
    forecast_context = inference_engine.forecast(parsed_task)
    logger.info("PipeFormer forecast context ready: mode=%s", forecast_context.get("mode"))
    parsed_task["forecast_time_step_minutes"] = forecast_context.get("time_step_minutes")
    registry_entries = list(forecast_context.get("variable_registry") or [])
    parsed_task["_variable_registry"] = registry_entries
    variable_summaries = summarize_variables(forecast_context["real_rows"], forecast_context["predict_rows"])
    logger.info("PipeFormer variable summaries built: variables=%d", len(variable_summaries))
    variable_names = list(variable_summaries)
    resolved_attention, unresolved_attention = _resolve_requested_variables(
        parsed_task.get("attention_targets") or [],
        variable_names,
        registry_entries,
    )
    resolved_outputs, unresolved_outputs = _resolve_requested_variables(
        parsed_task.get("output_state_variables") or [],
        variable_names,
        registry_entries,
    )
    parsed_task["resolved_attention_variables"] = resolved_attention
    parsed_task["resolved_output_variables"] = resolved_outputs
    parsed_task["unresolved_attention_targets"] = unresolved_attention
    parsed_task["unresolved_output_state_variables"] = unresolved_outputs
    counterfactual_comparison = None
    if include_baseline_comparison:
        baseline_task = copy.deepcopy(parsed_task)
        baseline_task["disturbance_magnitude_percent"] = None
        baseline_task["disturbance_direction"] = "unknown"
        baseline_boundary = dict(baseline_task.get("boundary_conditions") or {})
        for key in ("setpoints", "percentage_changes"):
            values = dict(baseline_boundary.get(key) or {})
            values.pop(str(parsed_task.get("disturbance_variable") or ""), None)
            if values:
                baseline_boundary[key] = values
            else:
                baseline_boundary.pop(key, None)
        for key in ("disturbance_variable", "disturbance_direction", "disturbance_magnitude_percent"):
            baseline_boundary.pop(key, None)
        baseline_boundary["keep_other_boundary_controls"] = True
        baseline_task["boundary_conditions"] = baseline_boundary
        baseline_context = inference_engine.forecast(baseline_task)
        counterfactual_comparison = _counterfactual_comparison(
            baseline_context["predict_rows"],
            forecast_context["predict_rows"],
            resolved_outputs,
        )
        counterfactual_comparison["disturbance_variable"] = parsed_task.get("disturbance_variable")
        counterfactual_comparison["applied_disturbance"] = next(
            (
                item
                for item in forecast_context.get("applied_boundary_conditions") or []
                if item.get("variable") == parsed_task.get("disturbance_variable")
            ),
            None,
        )
    verification = constraint_engine.evaluate(variable_summaries, parsed_task=parsed_task)
    logger.info("PipeFormer constraint checks finished: overall_status=%s", verification.get("overall_status"))
    priority_evidence_variables = []
    for finding in verification.get("priority_findings", []):
        for value in list(finding.get("offending_values") or []) + list(finding.get("evaluated_values") or []):
            variable = value.get("variable")
            if variable and variable not in priority_evidence_variables:
                priority_evidence_variables.append(variable)
    disturbance_variable = parsed_task.get("disturbance_variable")
    output_variable_set = set(resolved_outputs)
    evidence_candidate_names = output_variable_set | set(priority_evidence_variables)
    if disturbance_variable:
        evidence_candidate_names.add(disturbance_variable)
    evidence_summaries = {
        variable: summary
        for variable, summary in variable_summaries.items()
        if variable in evidence_candidate_names
    }
    evidence_variables = top_variables(
        evidence_summaries,
        limit=3,
        preferred_variables=resolved_outputs,
        priority_variables=priority_evidence_variables,
    )
    observation_variables = top_variables(
        {
            variable: summary
            for variable, summary in variable_summaries.items()
            if variable in output_variable_set
        },
        limit=2,
        preferred_variables=resolved_attention,
        priority_variables=[
            variable
            for variable in priority_evidence_variables
            if variable in output_variable_set
        ],
    )
    logger.info(
        "PipeFormer evidence assembled: manual_intervention=%s top_variables=%s",
        verification.get("human_intervention_label"),
        [item.get("variable") for item in evidence_variables],
    )

    prediction_summary = {
        "forecast_mode": forecast_context["mode"],
        "case_id": parsed_task.get("case_id"),
        "current_operating_condition_number": parsed_task.get("current_operating_condition_number"),
        "forecast_horizon_minutes": parsed_task.get("forecast_horizon_minutes"),
        "disturbance_variable": parsed_task["disturbance_variable"],
        "disturbance_direction": parsed_task["disturbance_direction"],
        "disturbance_magnitude_percent": parsed_task["disturbance_magnitude_percent"],
        "attention_targets": parsed_task["attention_targets"],
        "output_state_variables": parsed_task["output_state_variables"],
        "constraint_verification_types": parsed_task["constraint_verification_types"],
        "resolved_attention_variables": resolved_attention,
        "resolved_output_variables": resolved_outputs,
        "output_forecast_summary": _compact_output_summaries(variable_summaries, resolved_outputs),
        "top_watch_variables": evidence_variables,
    }
    if counterfactual_comparison is not None:
        prediction_summary["counterfactual_comparison"] = counterfactual_comparison
    forecast_metadata = {
        "mode": forecast_context["mode"],
        "disturbance_variable_mapping": forecast_context["disturbance_variable_mapping"],
        "forecast_window": _forecast_window_summary(forecast_context),
        "baseline_comparison_included": counterfactual_comparison is not None,
    }
    for key in (
        "sequence_length",
        "time_step_offset",
        "requested_forecast_horizon_minutes",
        "requested_forecast_steps",
        "time_step_minutes",
        "actual_forecast_steps",
        "actual_forecast_horizon_minutes",
        "actual_forecast_horizon_source",
        "disturbance_timing_mode",
        "adjusted_input_step_count",
        "device",
        "model_input_projection_type",
        "data_provenance",
        "operating_condition_number_used",
        "applied_boundary_conditions",
    ):
        if key in forecast_context:
            forecast_metadata[key] = forecast_context[key]

    compact_source_keys = {
        "checkpoint_dir": "checkpoint_id",
        "data_case_dir": "data_case_id",
    }
    for source_key, metadata_key in compact_source_keys.items():
        if source_key in forecast_context:
            forecast_metadata[metadata_key] = _compact_source_name(forecast_context[source_key])

    parsed_task.pop("_variable_registry", None)
    result = {
        "success": True,
        "parsed_task": parsed_task,
        "prediction_summary": prediction_summary,
        "constraint_check": verification,
        "evidence": {
            "top_watch_variables": evidence_variables,
            "key_observation_variables": observation_variables,
        },
        "risk_level": verification["risk_level"],
        "manual_intervention_label": verification["human_intervention_label"],
        "dispatch_recommendation": verification.get("dispatch_recommendation"),
        "quality_flag": (
            "needs_review"
            if unresolved_attention
            or unresolved_outputs
            or not verification.get("verification_complete", True)
            else "pass"
        ),
        "forecast_metadata": forecast_metadata,
    }
    if counterfactual_comparison is not None:
        result["counterfactual_comparison"] = counterfactual_comparison
    return result
