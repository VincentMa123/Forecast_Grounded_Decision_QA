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
from .teacher_answer import build_teacher_answer
from .trace_formatter import final_answer_text, risk_level_from_status, row_labels


DEFAULT_REQUESTED_CHECKS = ["pressure", "flow", "linepack", "compressor_load", "energy"]
logger = logging.getLogger(__name__)


def _repo_root_from_backend_root(backend_root: Path) -> Path:
    return Path(backend_root).resolve().parents[1]


def _optional_path(path_value: Optional[str]) -> Optional[Path]:
    return Path(path_value).expanduser().resolve() if path_value else None


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
    checks = [str(item).strip() for item in (requested_checks or []) if str(item).strip()]
    checks = [item for item in checks if item in allowed]
    return checks or DEFAULT_REQUESTED_CHECKS.copy()


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
    if changed_variable is not None:
        parsed["changed_variable"] = changed_variable
    if change_direction is not None:
        parsed["change_direction"] = change_direction
    if change_percent is not None:
        parsed["change_percent"] = float(change_percent)
    if forecast_horizon_minutes is not None:
        parsed["forecast_horizon_minutes"] = int(forecast_horizon_minutes)
    if keep_other_boundary_controls is not None:
        parsed["keep_other_boundary_controls"] = bool(keep_other_boundary_controls)
    if requested_checks is not None:
        parsed["requested_checks"] = _clean_checks(requested_checks)

    if "changed_variable" not in parsed or not parsed["changed_variable"]:
        raise ValueError(f"PipeFormer forecast requires changed_variable. Parse error: {parse_error or 'not parsed'}")
    if parsed.get("change_percent") is not None and parsed.get("change_direction") not in {"up", "down"}:
        raise ValueError("PipeFormer forecast requires change_direction to be 'up' or 'down' when change_percent is set.")

    parsed.setdefault("case_id", None)
    parsed.setdefault("change_direction", "unknown")
    parsed.setdefault("change_percent", None)
    parsed.setdefault("forecast_horizon_minutes", None)
    parsed.setdefault("keep_other_boundary_controls", False)
    parsed["requested_checks"] = _clean_checks(parsed.get("requested_checks"))
    parsed.setdefault("task_type", "prediction_and_verification")
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
    verification = run_constraint_checks(variable_summaries)
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
        "forecast_horizon_minutes": parsed_task["forecast_horizon_minutes"],
        "changed_variable": parsed_task["changed_variable"],
        "change_direction": parsed_task["change_direction"],
        "change_percent": parsed_task["change_percent"],
        "top_watch_variables": evidence_variables,
    }
    forecast_metadata = {
        "mode": forecast_context["mode"],
        "changed_variable_mapping": forecast_context["changed_variable_mapping"],
        "real_rows": row_labels(forecast_context["real_rows"]),
        "predict_rows": row_labels(forecast_context["predict_rows"]),
    }
    for key in (
        "checkpoint_dir",
        "weights_path",
        "model_config_path",
        "training_config_path",
        "data_dir",
        "static_dir",
        "data_case_dir",
        "sequence_length",
        "time_step_offset",
        "device",
        "model_input_projection_type",
        "forecast_csv",
        "mapping_csv",
    ):
        if key in forecast_context:
            forecast_metadata[key] = forecast_context[key]

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
        "manual_intervention_label": "required" if answer["requires_manual_intervention"] else "not_required",
        "dispatch_recommendation": "N/A - prediction-only scenario; no dispatch action was requested.",
        "final_answer": final_answer_text(answer),
        "quality_flag": "pass",
        "forecast_metadata": forecast_metadata,
    }
