from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .condition_parser import parse_condition
from .evidence_extractor import summarize_variables, top_variables
from .pipeformer_inference import (
    find_default_checkpoint_dir,
    find_default_forecast_csv,
    load_pipeformer_forecast_context,
)
from .rule_verifier import run_constraint_checks
from .teacher_answer import build_teacher_answer, final_answer_text, risk_level_from_status


DEFAULT_REQUESTED_CHECKS = ["pressure", "flow", "linepack", "compressor", "equipment_regulation", "abnormality_warning", "dispatch_priority"]
CHECK_TO_ATTENTION_TARGETS = {
    "pressure": ["nodes"],
    "flow": ["segments"],
    "linepack": ["linepack"],
    "compressor": ["compressors"],
    "equipment_regulation": ["valves", "pressure_regulators", "boundary_controls"],
    "abnormality_warning": ["abnormal_pressure_drops", "sudden_flow_changes", "leak_or_equipment_anomaly_signals"],
    "dispatch_priority": ["dispatch_priority_audit"],
}
CHECK_TO_OUTPUT_STATE_VARIABLES = {
    "pressure": ["pressure"],
    "flow": ["flow"],
    "linepack": ["linepack"],
    "compressor": ["compressor_load", "compression_ratio", "compressor_power"],
    "equipment_regulation": ["valve_opening", "regulator_range", "boundary_control_adjustment"],
    "abnormality_warning": ["pressure_drop", "flow_ramp", "leak_or_equipment_anomaly_score"],
    "dispatch_priority": ["energy_consumption", "operating_cost"],
}
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


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clean_checks(requested_checks: Optional[List[str]]) -> List[str]:
    allowed = set(DEFAULT_REQUESTED_CHECKS)
    checks = []
    for item in requested_checks or []:
        check = str(item).strip()
        if check and check in allowed and check not in checks:
            checks.append(check)
    return checks or DEFAULT_REQUESTED_CHECKS.copy()


def _unique_targets(checks: List[str], mapping: Dict[str, List[str]]) -> List[str]:
    result = []
    for check in checks:
        for value in mapping.get(check, []):
            if value not in result:
                result.append(value)
    return result


def _condition_number_from_case_id(case_id: Optional[str]) -> Optional[int]:
    if not case_id:
        return None
    digits = "".join(ch for ch in case_id if ch.isdigit())
    return int(digits) if digits else None


def _sync_pipeformer_task_aliases(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not parsed.get("disturbance_variable") and parsed.get("changed_variable"):
        parsed["disturbance_variable"] = parsed["changed_variable"]
    if not parsed.get("changed_variable") and parsed.get("disturbance_variable"):
        parsed["changed_variable"] = parsed["disturbance_variable"]

    if not parsed.get("disturbance_direction") and parsed.get("change_direction"):
        parsed["disturbance_direction"] = parsed["change_direction"]
    if not parsed.get("change_direction") and parsed.get("disturbance_direction"):
        parsed["change_direction"] = parsed["disturbance_direction"]

    if parsed.get("disturbance_magnitude_percent") is None and parsed.get("change_percent") is not None:
        parsed["disturbance_magnitude_percent"] = float(parsed["change_percent"])
    if parsed.get("change_percent") is None and parsed.get("disturbance_magnitude_percent") is not None:
        parsed["change_percent"] = float(parsed["disturbance_magnitude_percent"])

    checks = parsed.get("constraint_verification_types") or parsed.get("requested_checks")
    checks = _clean_checks(checks)
    parsed["constraint_verification_types"] = checks
    parsed["requested_checks"] = checks

    parsed.setdefault("case_id", None)
    parsed.setdefault("current_operating_condition_number", _condition_number_from_case_id(parsed.get("case_id")))
    parsed.setdefault("forecast_horizon_minutes", None)
    parsed.setdefault("disturbance_direction", "unknown")
    parsed.setdefault("change_direction", parsed["disturbance_direction"])
    parsed.setdefault("disturbance_magnitude_percent", None)
    parsed.setdefault("change_percent", parsed["disturbance_magnitude_percent"])
    parsed.setdefault("attention_targets", _unique_targets(checks, CHECK_TO_ATTENTION_TARGETS))
    parsed.setdefault("output_state_variables", _unique_targets(checks, CHECK_TO_OUTPUT_STATE_VARIABLES))

    boundary_conditions = dict(parsed.get("boundary_conditions") or {})
    keep_other = parsed.get("keep_other_boundary_controls")
    if keep_other is None:
        keep_other = bool(boundary_conditions.get("keep_other_boundary_controls", False))
    parsed["keep_other_boundary_controls"] = bool(keep_other)
    boundary_conditions.setdefault("keep_other_boundary_controls", parsed["keep_other_boundary_controls"])
    boundary_conditions.setdefault("disturbance_variable", parsed.get("disturbance_variable"))
    boundary_conditions.setdefault("disturbance_direction", parsed.get("disturbance_direction"))
    boundary_conditions.setdefault("disturbance_magnitude_percent", parsed.get("disturbance_magnitude_percent"))
    parsed["boundary_conditions"] = boundary_conditions

    parsed.setdefault("task_type", "prediction_and_verification")
    parsed.setdefault("parse_schema_version", "pipeformer_task_v2_pdf_terms")
    return parsed


def build_pipeformer_task(
    *,
    question: str,
    case_id: Optional[str] = None,
    changed_variable: Optional[str] = None,
    change_direction: Optional[str] = None,
    change_percent: Optional[float] = None,
    forecast_horizon_minutes: Optional[int] = None,
    keep_other_boundary_controls: Optional[bool] = None,
    requested_checks: Optional[List[str]] = None,
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
    elif changed_variable is not None:
        parsed["disturbance_variable"] = changed_variable
    if disturbance_direction is not None:
        parsed["disturbance_direction"] = disturbance_direction
    elif change_direction is not None:
        parsed["disturbance_direction"] = change_direction
    if disturbance_magnitude_percent is not None:
        parsed["disturbance_magnitude_percent"] = float(disturbance_magnitude_percent)
    elif change_percent is not None:
        parsed["disturbance_magnitude_percent"] = float(change_percent)
    if forecast_horizon_minutes is not None:
        parsed["forecast_horizon_minutes"] = int(forecast_horizon_minutes)
    if keep_other_boundary_controls is not None:
        parsed["keep_other_boundary_controls"] = bool(keep_other_boundary_controls)
    if attention_targets is not None:
        parsed["attention_targets"] = list(attention_targets)
    if output_state_variables is not None:
        parsed["output_state_variables"] = list(output_state_variables)
    if constraint_verification_types is not None:
        parsed["constraint_verification_types"] = _clean_checks(constraint_verification_types)
    elif requested_checks is not None:
        parsed["constraint_verification_types"] = _clean_checks(requested_checks)

    parsed = _sync_pipeformer_task_aliases(parsed)
    if not parsed.get("disturbance_variable"):
        raise ValueError(f"PipeFormer forecast requires disturbance_variable. Parse error: {parse_error or 'not parsed'}")
    if parsed.get("disturbance_magnitude_percent") is not None and parsed.get("disturbance_direction") not in {"up", "down"}:
        raise ValueError("PipeFormer forecast requires disturbance_direction to be 'up' or 'down' when disturbance_magnitude_percent is set.")
    return parsed


def run_pipeformer_forecast_analysis(
    *,
    question: str,
    backend_root: Path,
    case_id: Optional[str] = None,
    changed_variable: Optional[str] = None,
    change_direction: Optional[str] = None,
    change_percent: Optional[float] = None,
    forecast_horizon_minutes: Optional[int] = None,
    keep_other_boundary_controls: Optional[bool] = None,
    requested_checks: Optional[List[str]] = None,
    current_operating_condition_number: Optional[int] = None,
    boundary_conditions: Optional[Dict[str, Any]] = None,
    disturbance_variable: Optional[str] = None,
    disturbance_direction: Optional[str] = None,
    disturbance_magnitude_percent: Optional[float] = None,
    attention_targets: Optional[List[str]] = None,
    output_state_variables: Optional[List[str]] = None,
    constraint_verification_types: Optional[List[str]] = None,
    pipeformer_root: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    static_dir: Optional[str] = None,
    mapping_csv: Optional[str] = None,
    forecast_csv: Optional[str] = None,
    device: Optional[str] = None,
    use_sample_csv: Optional[bool] = None,
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
    resolved_mapping_csv = _optional_path(mapping_csv) or _env_path(
        "PIPEFORMER_MAPPING_CSV",
        resolved_pipeformer_root / "data" / "mock_tiny" / "static" / "mock_tiny" / "index_variable_mapping.csv",
    )
    resolved_use_sample_csv = _env_bool("PIPEFORMER_USE_SAMPLE_CSV", False) if use_sample_csv is None else bool(use_sample_csv)
    resolved_forecast_csv = _first_path(
        _optional_path(forecast_csv),
        _env_optional_path("PIPEFORMER_FORECAST_CSV"),
    )
    if resolved_forecast_csv is None:
        resolved_forecast_csv = (
            find_default_forecast_csv(repo_root).resolve()
            if resolved_use_sample_csv
            else resolved_pipeformer_root / "outputs" / "mock_tiny_decoder" / "sample_predictions" / "eval_16.csv"
        )
    resolved_data_dir = _optional_path(data_dir) or _env_optional_path("PIPEFORMER_DATA_DIR")
    resolved_static_dir = _optional_path(static_dir) or _env_optional_path("PIPEFORMER_STATIC_DIR")
    resolved_device = device or os.getenv("PIPEFORMER_DEVICE", "cpu")
    logger.info(
        "PipeFormer paths resolved: root=%s checkpoint=%s mapping=%s device=%s use_sample_csv=%s",
        resolved_pipeformer_root,
        resolved_checkpoint_dir,
        resolved_mapping_csv,
        resolved_device,
        resolved_use_sample_csv,
    )

    parsed_task = build_pipeformer_task(
        question=question,
        case_id=case_id,
        changed_variable=changed_variable,
        change_direction=change_direction,
        change_percent=change_percent,
        forecast_horizon_minutes=forecast_horizon_minutes,
        keep_other_boundary_controls=keep_other_boundary_controls,
        requested_checks=requested_checks,
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
    forecast_context = load_pipeformer_forecast_context(
        parsed_task=parsed_task,
        forecast_csv=resolved_forecast_csv,
        mapping_path=resolved_mapping_csv,
        checkpoint_dir=resolved_checkpoint_dir,
        pipeformer_root=resolved_pipeformer_root,
        data_dir=resolved_data_dir,
        static_dir=resolved_static_dir,
        device=resolved_device,
        use_sample_csv=resolved_use_sample_csv,
    )
    logger.info("PipeFormer forecast context ready: mode=%s", forecast_context.get("mode"))
    variable_summaries = summarize_variables(forecast_context["real_rows"], forecast_context["predict_rows"])
    logger.info("PipeFormer variable summaries built: variables=%d", len(variable_summaries))
    verification = run_constraint_checks(variable_summaries, parsed_task=parsed_task)
    logger.info("PipeFormer constraint checks finished: overall_status=%s", verification.get("overall_status"))
    evidence_variables = top_variables(variable_summaries, limit=3)
    answer = build_teacher_answer(parsed_task, verification, evidence_variables)
    logger.info(
        "PipeFormer teacher answer assembled: manual_intervention=%s top_variables=%s",
        answer.get("requires_manual_intervention"),
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
        "top_watch_variables": evidence_variables,
    }
    forecast_metadata = {
        "mode": forecast_context["mode"],
        "changed_variable_mapping": forecast_context["changed_variable_mapping"],
        "forecast_window": _forecast_window_summary(forecast_context),
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
        "device",
        "model_input_projection_type",
    ):
        if key in forecast_context:
            forecast_metadata[key] = forecast_context[key]

    compact_source_keys = {
        "checkpoint_dir": "checkpoint_id",
        "data_case_dir": "data_case_id",
        "forecast_csv": "forecast_csv_name",
    }
    for source_key, metadata_key in compact_source_keys.items():
        if source_key in forecast_context:
            forecast_metadata[metadata_key] = _compact_source_name(forecast_context[source_key])

    return {
        "success": True,
        "parsed_task": parsed_task,
        "prediction_summary": prediction_summary,
        "constraint_check": verification,
        "evidence": {
            "top_watch_variables": evidence_variables,
            "key_observation_variables": answer["key_observation_variables"],
        },
        "risk_level": risk_level_from_status(verification["overall_status"]),
        "manual_intervention_label": answer.get("manual_intervention_label", "no_intervention"),
        "dispatch_recommendation": "N/A - prediction-only scenario; no dispatch action was requested.",
        "final_answer": final_answer_text(answer),
        "quality_flag": "pass",
        "forecast_metadata": forecast_metadata,
    }
