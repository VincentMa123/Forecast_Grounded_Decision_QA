from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.io_utils import write_json, write_jsonl
from pipeline.scenario_loader import load_scenarios
from pipeline.scenario_preflight import (
    require_supported_scenario_sources,
    validate_scenario_sources,
)


logger = logging.getLogger("teacher_trace")


def backend_root() -> Path:
    return Path(__file__).resolve().parent


def default_scenario_files() -> List[Path]:
    root = backend_root() / "pipeclaw_data"
    return [
        root / "pipeclaw_dataset_v2.json",
        root / "Pipeline_Full_Life_Cycle_Test_Dataset-v4.json",
        root / "Pipeline_Full_Life_Cycle_Test_Dataset-v7.json",
    ]


def load_scenario_sources(paths: List[Path]) -> List[Dict[str, Any]]:
    sources = []
    seen_names = set()
    for path in paths:
        resolved = Path(path).resolve()
        source_name = resolved.stem
        if source_name in seen_names:
            raise ValueError(f"Duplicate dataset source name {source_name!r}; rename one source file to keep record ids unique.")
        seen_names.add(source_name)
        scenarios = load_scenarios(resolved)
        for scenario in scenarios:
            scenario["dataset_source"] = source_name
            scenario["source_file"] = resolved.name
        sources.append(
            {
                "dataset_source": source_name,
                "source_file": resolved.name,
                "path": resolved,
                "scenarios": scenarios,
            }
        )
    return sources


def flatten_source_scenarios(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [scenario for source in sources for scenario in source.get("scenarios") or []]


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def short_text(value: Any, limit: int = 700) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def load_backend_env() -> None:
    env_path = backend_root() / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            os.environ.setdefault(key, value.strip().strip("'\""))
    else:
        load_dotenv(env_path, override=False)


def build_parser() -> argparse.ArgumentParser:
    root = backend_root()
    parser = argparse.ArgumentParser(description="Run PipeClaw and export teacher_trace records.")
    parser.add_argument(
        "--scenario-file",
        type=Path,
        action="append",
        default=None,
        help="Scenario source JSON; repeat to combine sources. Defaults to PipeClaw v2, lifecycle v4, and lifecycle v7.",
    )
    parser.add_argument("--dataset-source", default=None, help="Disambiguate --scenario-id when it exists in multiple sources.")
    parser.add_argument("--scenario-id", default=None, help="Generate one scenario. Omit to generate all scenarios.")
    parser.add_argument("--agent-id", default="teacher_trace")
    parser.add_argument("--session-id", default=None, help="Only valid with --scenario-id.")
    parser.add_argument("--device", default=os.getenv("PIPEFORMER_DEVICE", "cpu"))
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=root / "generated_teacher_traces" / "teacher_trace.jsonl",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "generated_teacher_traces" / "teacher_trace.json",
    )
    parser.add_argument(
        "--session-output-jsonl",
        type=Path,
        default=root / "generated_teacher_traces" / "teacher_trace_sessions.jsonl",
        help="One complete multi-turn evaluation record per source session.",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=root / "generated_teacher_traces" / "splits",
        help="Directory for scenario-isolated train/valid/test turn JSONL files.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=root.parents[1] / "pipeFormer" / "data" / "mock_lifecycle" / "static" / "mock_lifecycle" / "index_variable_mapping.csv",
        help="Active PipeFormer mapping used for scenario-variable preflight.",
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=root / "generated_teacher_traces" / "scenario_preflight.json",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--split-seed", default="pipeclaw-lifecycle-v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level",
        default=os.getenv("TEACHER_TRACE_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Terminal logging verbosity for teacher trace generation.",
    )
    return parser


def parse_tool_output(tool_call: Dict[str, Any]) -> Any:
    if "result" in tool_call:
        return tool_call["result"]
    raw = tool_call.get("result_summary")
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _strip_row_suffix(label: str) -> str:
    for suffix in ("_real", "_predict"):
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def _compact_source_name(path_value: Any) -> Optional[str]:
    if not path_value:
        return None
    return Path(str(path_value)).name


def _forecast_window_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(metadata.get("forecast_window"), dict):
        return dict(metadata["forecast_window"])

    real_rows = metadata.get("real_rows") if isinstance(metadata.get("real_rows"), list) else []
    predict_rows = metadata.get("predict_rows") if isinstance(metadata.get("predict_rows"), list) else []
    labels = metadata.get("forecast_time_labels") if isinstance(metadata.get("forecast_time_labels"), list) else []
    if not labels:
        labels = [_strip_row_suffix(str(label)) for label in (predict_rows or real_rows)]

    window = {
        "start_time": labels[0] if labels else None,
        "end_time": labels[-1] if labels else None,
        "time_step_minutes": metadata.get("time_step_minutes"),
        "real_row_count": len(real_rows),
        "predict_row_count": len(predict_rows),
    }
    return {key: value for key, value in window.items() if value is not None}


def sanitize_forecast_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    bulky_keys = {"real_rows", "predict_rows", "time_labels", "forecast_time_labels"}
    path_keys = {
        "checkpoint_dir",
        "weights_path",
        "model_config_path",
        "training_config_path",
        "data_dir",
        "static_dir",
        "data_case_dir",
        "mapping_csv",
    }
    cleaned = {
        key: sanitize_tool_output(value)
        for key, value in metadata.items()
        if key not in bulky_keys and key not in path_keys
    }
    cleaned["forecast_window"] = _forecast_window_from_metadata(metadata)

    compact_source_keys = {
        "checkpoint_dir": "checkpoint_id",
        "data_case_dir": "data_case_id",
    }
    for source_key, metadata_key in compact_source_keys.items():
        compact_name = _compact_source_name(metadata.get(source_key))
        if compact_name:
            cleaned.setdefault(metadata_key, compact_name)
    return cleaned


def sanitize_tool_output(value: Any) -> Any:
    if isinstance(value, dict):
        if isinstance(value.get("forecast_metadata"), dict):
            value = dict(value)
            value["forecast_metadata"] = sanitize_forecast_metadata(value["forecast_metadata"])
        return {key: sanitize_tool_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_tool_output(item) for item in value]
    return value


def tool_call_id(tool_call: Dict[str, Any], index: int) -> str:
    return str(tool_call.get("tool_call_id") or f"tool_{index:03d}")


def _without_empty_values(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item is not None and item != {} and item != []
    }


def compact_parsed_task(output: Dict[str, Any]) -> Dict[str, Any]:
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
        "forecast_horizon_minutes",
        "attention_targets",
        "output_state_variables",
        "constraint_verification_types",
        "task_type",
        "forecast_time_step_minutes",
        "unresolved_attention_targets",
        "unresolved_output_state_variables",
    )
    compact = {key: parsed.get(key) for key in keys}
    compact["resolved_attention_variable_count"] = len(parsed.get("resolved_attention_variables") or [])
    compact["resolved_output_variable_count"] = len(parsed.get("resolved_output_variables") or [])
    compact["boundary_conditions"] = compact_boundary
    return _without_empty_values(compact)


def compact_prediction_summary(output: Dict[str, Any]) -> Dict[str, Any]:
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
        "output_forecast_summary",
    )
    compact = {key: prediction.get(key) for key in keys if key != "output_forecast_summary"}
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
            "actual_forecast_horizon_minutes": metadata.get("actual_forecast_horizon_minutes"),
        }
    )
    return _without_empty_values(compact)


def _relevant_forecast_variables(output: Dict[str, Any], limit: int = 8) -> List[str]:
    relevant: List[str] = []

    def add(value: Any) -> None:
        variable = str(value or "").strip()
        if variable and variable not in relevant and len(relevant) < limit:
            relevant.append(variable)

    evidence = dict(output.get("evidence") or {})
    for key in ("top_watch_variables", "key_observation_variables"):
        for item in evidence.get(key) or []:
            add(item.get("variable"))
    verification = dict(output.get("constraint_check") or {})
    for finding in verification.get("priority_findings") or []:
        for item in list(finding.get("offending_values") or []) + list(finding.get("evaluated_values") or []):
            add(item.get("variable"))
    add((output.get("parsed_task") or {}).get("disturbance_variable"))
    return relevant


def _compact_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
    evaluated = [
        item
        for item in finding.get("evaluated_values", [])
        if item.get("status") in {"warning", "fail"}
    ]
    values = evaluated or list(finding.get("offending_values", []))
    compact = {
        key: finding.get(key)
        for key in ("name", "category", "status", "evaluation_status", "flag", "priority", "message")
    }
    compact["evaluated_variable_count"] = len(finding.get("variables") or [])
    compact["affected_variables"] = list(
        dict.fromkeys(str(item.get("variable")) for item in values if item.get("variable"))
    )[:3]
    compact["evaluated_values"] = values[:3]
    if finding.get("operating_envelope_status"):
        compact["operating_envelope_status"] = finding["operating_envelope_status"]
    return _without_empty_values(compact)


def _engineering_evidence(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_name = {check.get("name"): check for check in checks}
    pressure = by_name.get("node_pressure_operating_window", {})
    flow = by_name.get("flow_ramp_check", {})
    balance = by_name.get("supply_demand_balance", {})
    linepack = by_name.get("linepack_decline_and_recovery", {})
    compressor = by_name.get("compressor_load_limit", {})
    pressure_nodes = list(
        dict.fromkeys(
            list(pressure.get("pressure_violation_nodes") or [])
            + list(pressure.get("pressure_warning_nodes") or [])
        )
    )
    pressure_margins = dict(pressure.get("pressure_margins") or {})
    linepack_recovery = {
        variable: item
        for variable, item in dict(linepack.get("linepack_recovery") or {}).items()
        if not item.get("recovery_sufficient", True)
    }
    ramp_events = {
        variable: events
        for variable, events in dict(flow.get("flow_ramp_events") or {}).items()
        if events
    }
    capacity_episodes = {
        variable: item
        for variable, item in dict(
            by_name.get("flow_capacity_check", {}).get("flow_capacity_excursion_episodes") or {}
        ).items()
        if item.get("total_out_of_limit_minutes", 0) > 0
    }

    abnormality_checks = [
        by_name[name]
        for name in ("abnormal_pressure_drop", "sudden_flow_change", "potential_leak_signal", "equipment_anomaly")
        if by_name.get(name, {}).get("status") in {"warning", "fail"}
    ]
    abnormality = {}
    if abnormality_checks:
        abnormality = {
            "triggered_rule_count": len(abnormality_checks),
            "failure_rule_count": sum(check.get("status") == "fail" for check in abnormality_checks),
            "warning_rule_count": sum(check.get("status") == "warning" for check in abnormality_checks),
        }

    evidence = {
        "pressure": _without_empty_values(
            {
                "minimum_pressure": pressure.get("minimum_pressure"),
                "maximum_pressure": pressure.get("maximum_pressure"),
                "pressure_violation_nodes": pressure.get("pressure_violation_nodes"),
                "pressure_warning_nodes": pressure.get("pressure_warning_nodes"),
                "at_risk_pressure_margins": {
                    variable: pressure_margins[variable]
                    for variable in pressure_nodes[:5]
                    if variable in pressure_margins
                },
                "maximum_continuous_pressure_violation_minutes": pressure.get(
                    "maximum_continuous_pressure_violation_minutes"
                ),
                "simultaneous_end_user_warning_node_count": pressure.get(
                    "simultaneous_end_user_warning_node_count"
                ),
            }
        ),
        "flow": _without_empty_values(
            {
                "abnormal_flow_segments": flow.get("abnormal_flow_segments"),
                "flow_ramp_events": ramp_events,
                "flow_capacity_excursion_episodes": capacity_episodes,
                "supply_demand_balance_status": flow.get("supply_demand_balance_status"),
                "supply_demand_balance": (balance.get("evaluated_values") or [None])[0],
            }
        ),
        "linepack": _without_empty_values(
            {
                "minimum_linepack": linepack.get("minimum_linepack"),
                "insufficient_recovery": linepack_recovery,
                "linepack_warning_status": linepack.get("linepack_warning_status"),
            }
        ),
        "compressor": _without_empty_values(
            {
                "operating_envelope_status": compressor.get("operating_envelope_status"),
            }
        ),
        "equipment_regulation": _without_empty_values(
            {
                "valve_opening_status": by_name.get("valve_opening_range", {}).get("status"),
                "pressure_regulator_status": by_name.get("pressure_regulator_range", {}).get("status"),
                "boundary_adjustment_status": by_name.get("boundary_control_adjustment_magnitude", {}).get("status"),
            }
        ),
        "abnormality_warning": abnormality,
    }
    return _without_empty_values(evidence)


def compact_constraint_check(output: Dict[str, Any]) -> Dict[str, Any]:
    verification = dict(output.get("constraint_check") or {})
    checks = list(verification.get("checks") or [])
    findings = [_compact_finding(item) for item in verification.get("priority_findings", [])]
    non_pass_checks = [
        check
        for check in checks
        if check.get("status") in {"warning", "fail"} and check.get("category") != "human_intervention"
    ]
    failures = [check for check in non_pass_checks if check.get("status") == "fail"]
    warnings = [check for check in non_pass_checks if check.get("status") == "warning"]
    detailed_warning_count = sum(item.get("status") == "warning" for item in findings)
    rule_status = {
        str(check.get("name")): check.get("status")
        for check in checks
        if check.get("name")
    }
    compact = {
        "requested_categories": verification.get("requested_categories"),
        "category_status": verification.get("category_status"),
        "rule_status": rule_status,
        "overall_status": verification.get("overall_status"),
        "verification_complete": verification.get("verification_complete"),
        "not_evaluated_rules": verification.get("not_evaluated_rules"),
        "risk_level": verification.get("risk_level"),
        "risk_escalations": verification.get("risk_escalations"),
        "failure_count": verification.get("failure_count", len(failures)),
        "warning_count": verification.get("warning_count", len(warnings)),
        "omitted_warning_count": verification.get(
            "omitted_warning_count",
            max(0, len(warnings) - detailed_warning_count),
        ),
        "failed_rule_ids": verification.get("failed_rule_ids") or [check.get("name") for check in failures],
        "warning_rule_ids": verification.get("warning_rule_ids") or [check.get("name") for check in warnings],
        "triggered_flags": list(
            dict.fromkeys(check.get("flag") for check in non_pass_checks if check.get("flag"))
        ),
        "human_intervention_label": verification.get("human_intervention_label"),
        "dispatch_recommendation": verification.get("dispatch_recommendation"),
        "priority_findings": findings,
        "engineering_evidence": _engineering_evidence(checks),
    }
    return _without_empty_values(compact)


def compact_pipeformer_output(output: Dict[str, Any]) -> Dict[str, Any]:
    parsed_task = compact_parsed_task(output)
    metadata = dict(output.get("forecast_metadata") or {})
    task_resolution = {
        "resolved_attention_variable_count": parsed_task.get("resolved_attention_variable_count", 0),
        "resolved_output_variable_count": parsed_task.get("resolved_output_variable_count", 0),
        "unresolved_attention_targets": parsed_task.get("unresolved_attention_targets", []),
        "unresolved_output_state_variables": parsed_task.get("unresolved_output_state_variables", []),
        "applied_boundary_conditions": metadata.get("applied_boundary_conditions", []),
    }
    provenance = {
        "checkpoint_id": metadata.get("checkpoint_id"),
        "data_case_id": metadata.get("data_case_id"),
        "device": metadata.get("device"),
        "model_input_projection_type": metadata.get("model_input_projection_type"),
        "data_provenance": metadata.get("data_provenance"),
    }
    return {
        "success": True,
        "task_resolution": _without_empty_values(task_resolution),
        "prediction": compact_prediction_summary(output),
        "verification": compact_constraint_check(output),
        "evidence": output.get("evidence"),
        "provenance": _without_empty_values(provenance),
    }


def exported_tool_calls(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for index, item in enumerate(trace.get("tool_calls", []), start=1):
        call_id = tool_call_id(item, index)
        tool_name = str(item.get("tool_name") or "")
        calls.append(
            {
                "tool_call_id": call_id,
                "name": tool_name,
                "arguments": item.get("args", {}),
            }
        )
    return calls


def exported_tool_outputs(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for index, item in enumerate(trace.get("tool_calls", []), start=1):
        call_id = tool_call_id(item, index)
        tool_name = str(item.get("tool_name") or "")
        output = sanitize_tool_output(parse_tool_output(item))
        if tool_name == "run_pipeformer_forecast" and isinstance(output, dict) and output.get("success"):
            output = compact_pipeformer_output(output)
        outputs.append(
            {
                "tool_call_id": call_id,
                "name": tool_name,
                "output": output,
            }
        )
    return outputs


def final_answer(trace: Dict[str, Any]) -> str:
    for message in reversed(trace.get("messages", [])):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def successful_pipeformer_outputs(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    outputs = []
    for item in trace.get("tool_calls", []):
        if item.get("tool_name") != "run_pipeformer_forecast":
            continue
        output = parse_tool_output(item)
        if isinstance(output, dict) and output.get("success"):
            outputs.append(output)
    return outputs


UNSUPPORTED_HISTORY_CLAIM = re.compile(
    r"\b(?:reproduc(?:ed|ible|ibility|tion)?|previous runs?|prior runs?|stable across runs?|times stable)\b"
    r"|\u590d\u73b0|\u6b64\u524d.*(?:\u7ed3\u679c|\u8fd0\u884c)|\u524d(?:\u51e0|[\u4e00-\u5341\d]+)\u6b21.*\u4e00\u81f4|\u7a33\u5b9a.*(?:\u590d\u73b0|\u8fd0\u884c)",
    re.IGNORECASE,
)
NO_DISPATCH_REQUEST = re.compile(
    r"\u4e0d\u8981.{0,12}\u8c03\u5ea6(?:\u52a8\u4f5c|\u5efa\u8bae)"
    r"|(?:do\s+not|don't)\s+(?:give|provide|include).{0,30}dispatch",
    re.IGNORECASE,
)
DISPATCH_ADVICE = re.compile(
    r"\s*(?:\u8c03\u5ea6\u5efa\u8bae|dispatch\s+recommendation)\s*[:\uff1a][^\n]*",
    re.IGNORECASE,
)
SAFETY_ENERGY_INCONSISTENCY_CLAIM = re.compile(
    r"\u5b89\u5168\u4fa7\u4e0e\u80fd\u8017(?:/\u8bbe\u5907)?\u4fa7\u7ed3\u8bba\u4e0d\u4e00\u81f4",
    re.IGNORECASE,
)
UNSUPPORTED_UNIQUENESS_CLAIM = re.compile(
    r"(?:\s*[,，]\s*)?(?:\u552f\u4e00(?:\u8d8a\u9650|\u544a\u8b66|\u5f02\u5e38)?\u53d8\u91cf|the\s+only\s+(?:violating|warning|abnormal)\s+variable)",
    re.IGNORECASE,
)


def llm_answer_quality_issues(answer: str, *, check_history_claims: bool) -> List[str]:
    issues = []
    if not answer.strip():
        issues.append("missing_llm_final_answer")
    if check_history_claims and UNSUPPORTED_HISTORY_CLAIM.search(answer):
        issues.append("unsupported_execution_history_or_repeatability_claim")
    return issues


def remove_unsupported_history_claims(answer: str) -> str:
    """Remove unsupported repeatability claims before exporting an SFT target."""
    kept_lines = [line for line in answer.splitlines() if not UNSUPPORTED_HISTORY_CLAIM.search(line)]
    cleaned = "\n".join(kept_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _safety_and_energy_checks_pass(pipeformer: Optional[Dict[str, Any]]) -> bool:
    checks = ((pipeformer or {}).get("constraint_check") or {}).get("checks") or []
    safety_checks = [
        item
        for item in checks
        if item.get("category") in {"pressure", "flow", "linepack", "abnormality_warning"}
    ]
    energy_checks = [item for item in checks if item.get("name") == "energy_consumption_cost"]
    return (
        bool(safety_checks)
        and all(item.get("status") == "pass" for item in safety_checks)
        and bool(energy_checks)
        and all(item.get("status") == "pass" for item in energy_checks)
    )


def enforce_requested_answer_scope(
    answer: str,
    question: str,
    pipeformer: Optional[Dict[str, Any]],
) -> str:
    if NO_DISPATCH_REQUEST.search(question):
        answer = DISPATCH_ADVICE.sub("", answer)
    if _safety_and_energy_checks_pass(pipeformer):
        answer = SAFETY_ENERGY_INCONSISTENCY_CLAIM.sub(
            "\u5b89\u5168\u4fa7\u4e0e\u80fd\u8017\u6210\u672c\u5747\u901a\u8fc7\uff1b\u538b\u7f29\u673a/\u8bbe\u5907\u4fa7\u53e6\u6709\u544a\u8b66",
            answer,
        )
    return answer.strip()


def enforce_grounded_variable_descriptions(answer: str, pipeformer: Optional[Dict[str, Any]]) -> str:
    if not pipeformer:
        return answer

    constraint_check = dict(pipeformer.get("constraint_check") or {})
    findings = list(constraint_check.get("priority_findings") or [])
    finding_variables = {
        str(value.get("variable"))
        for finding in findings
        for value in list(finding.get("evaluated_values") or []) + list(finding.get("offending_values") or [])
        if value.get("variable")
    }
    distinct_nonpass_variables = set(finding_variables)
    if len(findings) != 1 or len(distinct_nonpass_variables) != 1:
        answer = UNSUPPORTED_UNIQUENESS_CLAIM.sub("", answer)

    evidence = dict(pipeformer.get("evidence") or {})
    evidence_items: Dict[str, Dict[str, Any]] = {}
    for key in ("top_watch_variables", "key_observation_variables"):
        for item in evidence.get(key) or []:
            variable = item.get("variable")
            if variable:
                evidence_items[str(variable)] = dict(item)

    for variable, item in evidence_items.items():
        if variable in finding_variables:
            continue
        facts = []
        for key in ("mean_prediction", "mean_abs_delta_vs_observed"):
            value = item.get(key)
            if value is not None:
                facts.append(f"{key}={float(value):.3f}")
        escaped = re.escape(variable)
        pattern = re.compile(rf"(?P<token>`?{escaped}`?)\s*[（(][^（）()\n]*[）)]")
        replacement_suffix = f"（{', '.join(facts)}）" if facts else ""
        answer = pattern.sub(lambda match: f"{match.group('token')}{replacement_suffix}", answer)
    return answer.strip()


def build_teacher_record(
    scenario: Dict[str, Any],
    question: str,
    trace: Dict[str, Any],
    *,
    source_session_id: str = "session_001",
    turn_id: int = 1,
    conversation_context: Optional[List[Dict[str, Any]]] = None,
    split: str = "train",
) -> Dict[str, Any]:
    dataset_source = str(scenario.get("dataset_source") or "unknown_source")
    scenario_id = str(scenario.get("scenario_id") or "unknown_scenario")
    record_id = f"{dataset_source}:{source_session_id}::turn_{turn_id:03d}"
    pipeformer_outputs = successful_pipeformer_outputs(trace)
    pipeformer = pipeformer_outputs[0] if len(pipeformer_outputs) == 1 else None
    raw_answer = final_answer(trace)
    has_pipeformer = bool(pipeformer_outputs)
    raw_answer_issues = llm_answer_quality_issues(raw_answer, check_history_claims=has_pipeformer)
    answer = remove_unsupported_history_claims(raw_answer) if has_pipeformer else raw_answer
    answer = enforce_requested_answer_scope(answer, question, pipeformer)
    answer = enforce_grounded_variable_descriptions(answer, pipeformer)
    quality_issues = llm_answer_quality_issues(answer, check_history_claims=has_pipeformer)
    quality_flag = "pass" if trace.get("status") == "completed" else "needs_review"
    parsed_task: Dict[str, Any] = {}
    prediction_summary: Dict[str, Any] = {}
    constraint_check: Dict[str, Any] = {}
    evidence: Dict[str, Any] = {}
    risk_level = "low"
    manual_intervention_label = "no_intervention"
    dispatch_recommendation = ""
    if pipeformer:
        parsed_task = compact_parsed_task(pipeformer)
        prediction_summary = compact_prediction_summary(pipeformer)
        constraint_check = compact_constraint_check(pipeformer)
        evidence = dict(pipeformer.get("evidence") or {})
        risk_level = pipeformer.get("risk_level")
        manual_intervention_label = pipeformer.get("manual_intervention_label")
        dispatch_recommendation = pipeformer.get("dispatch_recommendation")
        quality_flag = pipeformer.get("quality_flag", quality_flag)
    elif len(pipeformer_outputs) > 1:
        parsed_task = {
            "candidate_forecasts": [compact_parsed_task(output) for output in pipeformer_outputs]
        }
        prediction_summary = {
            "candidate_forecasts": [compact_prediction_summary(output) for output in pipeformer_outputs]
        }
        constraint_check = {
            "candidate_forecasts": [compact_constraint_check(output) for output in pipeformer_outputs]
        }
        evidence = {
            "candidate_forecasts": [dict(output.get("evidence") or {}) for output in pipeformer_outputs]
        }
        risk_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        intervention_rank = {
            "no_intervention": 0,
            "monitoring_only": 1,
            "operator_attention_required": 2,
            "immediate_intervention_required": 3,
        }
        risk_level = max(
            (str(output.get("risk_level") or "low") for output in pipeformer_outputs),
            key=lambda value: risk_rank.get(value, -1),
        )
        manual_intervention_label = max(
            (str(output.get("manual_intervention_label") or "no_intervention") for output in pipeformer_outputs),
            key=lambda value: intervention_rank.get(value, -1),
        )
        dispatch_recommendation = next(
            (str(output.get("dispatch_recommendation")) for output in pipeformer_outputs if output.get("dispatch_recommendation")),
            "",
        )
        if any(output.get("quality_flag") != "pass" for output in pipeformer_outputs):
            quality_flag = "needs_review"

    if quality_issues:
        quality_flag = "needs_review"
    if raw_answer_issues and not quality_issues:
        logger.warning(
            "Removed unsupported claims from exported final_answer: %s",
            ", ".join(raw_answer_issues),
        )

    return {
        "sample_id": record_id,
        "dataset_source": dataset_source,
        "source_scenario_id": scenario_id,
        "scenario_id": scenario_id,
        "split_group_id": scenario_id,
        "session_id": source_session_id,
        "turn_id": turn_id,
        "scenario_type": scenario.get("scenario_type"),
        "split": split,
        "conversation_context": sanitize_tool_output(conversation_context or []),
        "user_input": question,
        "parsed_task": sanitize_tool_output(parsed_task),
        "tool_calls": exported_tool_calls(trace),
        "tool_outputs": exported_tool_outputs(trace),
        "prediction_summary": sanitize_tool_output(prediction_summary),
        "constraint_check": sanitize_tool_output(constraint_check),
        "evidence": sanitize_tool_output(evidence),
        "risk_level": risk_level,
        "manual_intervention_label": manual_intervention_label,
        "dispatch_recommendation": dispatch_recommendation,
        "final_answer": answer,
        "quality_flag": quality_flag,
    }


def _turn_trace(full_trace: Dict[str, Any], message_start: int, tool_start: int) -> Dict[str, Any]:
    return {
        "session_id": full_trace.get("session_id"),
        "agent_id": full_trace.get("agent_id"),
        "status": full_trace.get("status"),
        "messages": list(full_trace.get("messages") or [])[message_start:],
        "tool_calls": list(full_trace.get("tool_calls") or [])[tool_start:],
    }


def _history_turn(record: Dict[str, Any]) -> Dict[str, Any]:
    return _without_empty_values(
        {
            "turn_id": record["turn_id"],
            "user_input": record["user_input"],
            "tool_calls": record.get("tool_calls"),
            "tool_outputs": record.get("tool_outputs"),
            "assistant_output": record.get("final_answer"),
        }
    )


def scenario_split_map(scenarios: List[Dict[str, Any]], seed: str) -> Dict[str, str]:
    """Create deterministic 80/10/10 splits, stratified by scenario type."""
    groups: Dict[str, List[str]] = {}
    scenario_types: Dict[str, str] = {}
    for scenario in scenarios:
        scenario_id = str(scenario.get("scenario_id") or "")
        scenario_type = str(scenario.get("scenario_type") or "unknown")
        prior_type = scenario_types.get(scenario_id)
        if prior_type is not None and prior_type != scenario_type:
            raise ValueError(
                f"Canonical scenario {scenario_id!r} has conflicting types: {prior_type!r} and {scenario_type!r}."
            )
        scenario_types[scenario_id] = scenario_type
    for scenario_id, scenario_type in scenario_types.items():
        groups.setdefault(scenario_type, []).append(scenario_id)
    result: Dict[str, str] = {}
    for scenario_type, scenario_ids in groups.items():
        ordered = sorted(
            scenario_ids,
            key=lambda value: hashlib.sha256(f"{seed}:{scenario_type}:{value}".encode("utf-8")).hexdigest(),
        )
        count = len(ordered)
        valid_count = max(1, round(count * 0.1)) if count >= 3 else 0
        test_count = max(1, round(count * 0.1)) if count >= 3 else 0
        train_end = count - valid_count - test_count
        for index, scenario_id in enumerate(ordered):
            result[scenario_id] = "train" if index < train_end else "valid" if index < train_end + valid_count else "test"
    return result


def run_backend_session(
    scenario: Dict[str, Any],
    source_session: Dict[str, Any],
    args: argparse.Namespace,
    scenario_index: int,
    session_index: int,
    run_stamp: str,
    split: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from agent.orchestrator import AgentOrchestrator
    from agent.schemas import AgentChatRequest

    scenario_id = str(scenario.get("scenario_id") or f"scenario_{scenario_index:06d}")
    dataset_source = str(scenario.get("dataset_source") or "unknown_source")
    source_session_id = str(source_session.get("session_id") or f"{scenario_id}_session_{session_index:03d}")
    runtime_session_id = args.session_id or f"teacher_{scenario_index:04d}_{session_index:03d}_{run_stamp}"
    os.environ["PIPEFORMER_DEVICE"] = str(args.device)

    logger.info("Session started: scenario=%s source_session=%s runtime_session=%s", scenario_id, source_session_id, runtime_session_id)

    orchestrator = AgentOrchestrator(
        data_loader=None,
        agent_id=args.agent_id,
        session_id=runtime_session_id,
        enable_skills=False,
        workspace_root_base=backend_root() / ".openclaw",
    )
    records: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    message_count = 0
    tool_count = 0
    for fallback_turn_id, turn in enumerate(source_session.get("dialogue") or [], start=1):
        question = str(turn.get("user_input") or "").strip()
        if not question:
            continue
        turn_id = int(turn.get("turn_id") or fallback_turn_id)
        logger.info("Turn started: scenario=%s session=%s turn=%d question=%s", scenario_id, source_session_id, turn_id, short_text(question, 300))
        try:
            result = orchestrator.run_agent(
                AgentChatRequest(agent_id=args.agent_id, session_id=runtime_session_id, message=question)
            )
            trace_path = Path(result.trace_summary.trace_path)
            full_trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace = _turn_trace(full_trace, message_count, tool_count)
            message_count = len(full_trace.get("messages") or [])
            tool_count = len(full_trace.get("tool_calls") or [])
            record = build_teacher_record(
                scenario,
                question,
                trace,
                source_session_id=source_session_id,
                turn_id=turn_id,
                conversation_context=history,
                split=split,
            )
            records.append(record)
            history.append(_history_turn(record))
            logger.info("Turn finished: scenario=%s session=%s turn=%d quality=%s", scenario_id, source_session_id, turn_id, record["quality_flag"])
            if trace.get("status") != "completed":
                errors.append(
                    {
                        "turn_id": turn_id,
                        "user_input": question,
                        "error": f"Agent run ended with status {trace.get('status') or 'unknown'}.",
                    }
                )
                break
        except Exception as exc:
            logger.exception("Turn failed: scenario=%s session=%s turn=%d", scenario_id, source_session_id, turn_id)
            errors.append({"turn_id": turn_id, "user_input": question, "error": str(exc)})
            break

    session_record = {
        "session_record_id": f"{dataset_source}:{source_session_id}",
        "dataset_source": dataset_source,
        "source_scenario_id": scenario_id,
        "scenario_id": scenario_id,
        "split_group_id": scenario_id,
        "session_id": source_session_id,
        "scenario_type": scenario.get("scenario_type"),
        "offset_hours": source_session.get("offset_hours"),
        "split": split,
        "turns": [
            {
                "turn_id": record["turn_id"],
                "user_input": record["user_input"],
                "expected_answer": record["final_answer"],
                "tool_calls": record["tool_calls"],
                "tool_outputs": record["tool_outputs"],
                "quality_flag": record["quality_flag"],
            }
            for record in records
        ],
        "complete": not errors and len(records) == len(source_session.get("dialogue") or []),
        "errors": errors,
    }
    return records, session_record


def write_split_records(output_dir: Path, records: List[Dict[str, Any]], force: bool) -> None:
    for split in ("train", "valid", "test"):
        write_jsonl(output_dir / f"teacher_trace_{split}.jsonl", [item for item in records if item.get("split") == split], force=force)


def main() -> int:
    load_backend_env()
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    scenario_files = list(args.scenario_file or default_scenario_files())
    all_sources = load_scenario_sources(scenario_files)
    selected_sources = [
        source
        for source in all_sources
        if args.dataset_source is None or source["dataset_source"] == args.dataset_source
    ]
    if args.dataset_source and not selected_sources:
        available = ", ".join(source["dataset_source"] for source in all_sources)
        raise ValueError(f"Unknown --dataset-source {args.dataset_source!r}. Available sources: {available}")
    if args.scenario_id:
        matches = [
            (source, scenario)
            for source in selected_sources
            for scenario in source.get("scenarios") or []
            if str(scenario.get("scenario_id")) == args.scenario_id
        ]
        if not matches:
            raise KeyError(f"Scenario {args.scenario_id!r} not found in the selected sources.")
        if len(matches) > 1:
            candidates = ", ".join(source["dataset_source"] for source, _ in matches)
            raise ValueError(
                f"Scenario {args.scenario_id!r} exists in multiple sources ({candidates}); pass --dataset-source."
            )
        source, scenario = matches[0]
        selected_sources = [{**source, "scenarios": [scenario]}]
    scenarios = flatten_source_scenarios(selected_sources)
    all_scenarios = flatten_source_scenarios(all_sources)
    source_session_count = sum(len(scenario.get("sessions") or []) for scenario in scenarios)
    if args.session_id and (len(scenarios) != 1 or source_session_count != 1):
        raise ValueError("--session-id requires one selected scenario containing exactly one source session.")

    preflight = validate_scenario_sources(selected_sources, args.mapping_csv)
    write_json(args.preflight_output, preflight, force=args.force)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    require_supported_scenario_sources(selected_sources, args.mapping_csv)

    logger.info(
        "Teacher trace generation started: scenarios=%d device=%s scenario_files=%s",
        len(scenarios),
        args.device,
        [str(path) for path in scenario_files],
    )
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    records: List[Dict[str, Any]] = []
    session_records: List[Dict[str, Any]] = []
    split_map = scenario_split_map(all_scenarios, args.split_seed)
    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = str(scenario.get("scenario_id") or f"scenario_{index:06d}")
        split = split_map[scenario_id]
        for session_index, source_session in enumerate(scenario.get("sessions") or [], start=1):
            turn_records, session_record = run_backend_session(
                scenario, source_session, args, index, session_index, run_stamp, split
            )
            records.extend(turn_records)
            session_records.append(session_record)

    logger.info("Writing JSONL output: %s", args.output_jsonl)
    write_jsonl(args.output_jsonl, records, force=args.force)
    logger.info("Writing pretty JSON output: %s", args.output_json)
    write_json(args.output_json, records[0] if len(records) == 1 else records, force=args.force)
    logger.info("Writing session evaluation JSONL: %s", args.session_output_jsonl)
    write_jsonl(args.session_output_jsonl, session_records, force=args.force)
    logger.info("Writing scenario-isolated split files: %s", args.split_output_dir)
    write_split_records(args.split_output_dir, records, force=args.force)
    failed_sessions = sum(not bool(item.get("complete")) for item in session_records)
    run_status = "ok" if failed_sessions == 0 else "completed_with_errors"
    logger.info(
        "Teacher trace generation complete: records=%d failed_sessions=%d",
        len(records),
        failed_sessions,
    )
    print(
        json.dumps(
            {
                "status": run_status,
                "records": len(records),
                "sessions": len(session_records),
                "failed_sessions": failed_sessions,
                "output_jsonl": args.output_jsonl.as_posix(),
                "output_json": args.output_json.as_posix(),
                "session_output_jsonl": args.session_output_jsonl.as_posix(),
                "split_output_dir": args.split_output_dir.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if failed_sessions == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
