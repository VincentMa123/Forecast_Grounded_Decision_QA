from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def risk_level_from_status(status: str) -> str:
    return {
        "pass": "low",
        "warning": "medium",
        "fail": "high",
    }.get(status, "unknown")


def final_answer_text(answer: Dict[str, Any]) -> str:
    indicators = ", ".join(item["variable"] for item in answer.get("top_3_watch_indicators", []))
    key_variables = ", ".join(item["variable"] for item in answer.get("key_observation_variables", []))
    intervention = "yes" if answer.get("requires_manual_intervention") else "no"
    parts = [
        answer.get("most_likely_operating_consequence", ""),
        f"Top watch indicators: {indicators}." if indicators else "",
        f"Manual intervention required: {intervention}.",
        f"Priority audit constraint: {answer.get('priority_audit_constraint')}." if answer.get("priority_audit_constraint") else "",
        f"Key evidence variables: {key_variables}." if key_variables else "",
        answer.get("scope_note", ""),
    ]
    return " ".join(part for part in parts if part)


def row_labels(rows: List[Any]) -> List[str]:
    return [row.label for row in rows]


def forecast_tool_call(parsed_task: Dict[str, Any], mapping_path: Path, forecast_context: Dict[str, Any]) -> Dict[str, Any]:
    if forecast_context["mode"] == "checkpoint_inference":
        return {
            "call_id": "tool_002",
            "tool_name": "run_pipeformer_checkpoint_inference",
            "arguments": {
                "checkpoint_dir": forecast_context["checkpoint_dir"],
                "data_case_dir": forecast_context["data_case_dir"],
                "mapping_csv": mapping_path.as_posix(),
                "changed_variable": parsed_task["changed_variable"],
                "change_percent": parsed_task["change_percent"],
                "device": forecast_context["device"],
            },
        }
    return {
        "call_id": "tool_002",
        "tool_name": "load_pipeformer_forecast_csv",
        "arguments": {
            "forecast_csv": forecast_context.get("forecast_csv"),
            "mapping_csv": mapping_path.as_posix(),
            "changed_variable": parsed_task["changed_variable"],
        },
    }


def forecast_tool_output(forecast_context: Dict[str, Any], real_rows: List[str], predict_rows: List[str]) -> Dict[str, Any]:
    output = {
        "mode": forecast_context["mode"],
        "changed_variable_mapping": forecast_context["changed_variable_mapping"],
        "real_rows": real_rows,
        "predict_rows": predict_rows,
    }
    for key in (
        "checkpoint_dir",
        "weights_path",
        "model_config_path",
        "training_config_path",
        "data_case_dir",
        "sequence_length",
        "time_step_offset",
        "model_input_projection_type",
        "forecast_csv",
    ):
        if key in forecast_context:
            output[key] = forecast_context[key]
    return {"call_id": "tool_002", "output": output}


def build_teacher_trace_record(
    scenario: Dict[str, Any],
    scenario_path: Path,
    question: str,
    parsed_task: Dict[str, Any],
    mapping_path: Path,
    forecast_context: Dict[str, Any],
    verification: Dict[str, Any],
    evidence_variables: List[Dict[str, Any]],
    answer: Dict[str, Any],
) -> Dict[str, Any]:
    scenario_id = scenario.get("scenario_id")
    scenario_type = scenario.get("scenario_type")
    real_rows = row_labels(forecast_context["real_rows"])
    predict_rows = row_labels(forecast_context["predict_rows"])

    tool_calls = [
        {
            "call_id": "tool_001",
            "tool_name": "parse_pipeformer_condition",
            "arguments": {
                "scenario_file": scenario_path.as_posix(),
                "user_input": question,
            },
        },
        forecast_tool_call(parsed_task, mapping_path, forecast_context),
        {
            "call_id": "tool_003",
            "tool_name": "check_engineering_constraints",
            "arguments": {"requested_checks": parsed_task["requested_checks"]},
        },
    ]
    tool_outputs = [
        {"call_id": "tool_001", "output": parsed_task},
        forecast_tool_output(forecast_context, real_rows, predict_rows),
        {"call_id": "tool_003", "output": verification},
    ]

    return {
        "sample_id": f"sample_{scenario_id}",
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "user_input": question,
        "parsed_task": parsed_task,
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
        "prediction_summary": {
            "forecast_mode": forecast_context["mode"],
            "forecast_horizon_minutes": parsed_task["forecast_horizon_minutes"],
            "changed_variable": parsed_task["changed_variable"],
            "change_direction": parsed_task["change_direction"],
            "change_percent": parsed_task["change_percent"],
            "top_watch_variables": evidence_variables,
        },
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
    }