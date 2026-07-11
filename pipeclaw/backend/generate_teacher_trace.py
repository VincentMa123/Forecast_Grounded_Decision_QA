from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.io_utils import write_json, write_jsonl
from pipeline.scenario_loader import find_scenario, load_scenarios


logger = logging.getLogger("teacher_trace")


def backend_root() -> Path:
    return Path(__file__).resolve().parent


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
        default=root / "pipeclaw_data" / "mock_pipeformer_tiny_scenarios.json",
    )
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level",
        default=os.getenv("TEACHER_TRACE_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Terminal logging verbosity for teacher trace generation.",
    )
    return parser


def first_user_input(scenario: Dict[str, Any]) -> str:
    for session in scenario.get("sessions", []):
        for turn in session.get("dialogue", []):
            text = str(turn.get("user_input") or "").strip()
            if text:
                return text
    raise ValueError(f"Scenario {scenario.get('scenario_id')} has no non-empty user_input.")


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
        "resolved_attention_variables",
        "resolved_output_variables",
        "unresolved_attention_targets",
        "unresolved_output_state_variables",
    )
    compact = {key: parsed.get(key) for key in keys}
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
    compact = {key: prediction.get(key) for key in keys}
    compact.update(
        {
            "forecast_window": metadata.get("forecast_window"),
            "actual_forecast_steps": metadata.get("actual_forecast_steps"),
            "actual_forecast_horizon_minutes": metadata.get("actual_forecast_horizon_minutes"),
        }
    )
    return _without_empty_values(compact)


def _compact_finding(finding: Dict[str, Any]) -> Dict[str, Any]:
    evaluated = [
        item
        for item in finding.get("evaluated_values", [])
        if item.get("status") in {"warning", "fail"}
    ]
    values = evaluated or list(finding.get("offending_values", []))
    compact = {
        key: finding.get(key)
        for key in ("name", "category", "status", "evaluation_status", "flag", "priority", "variables", "message")
    }
    compact["evaluated_values"] = values[:12]
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
    equipment = by_name.get("valve_opening_range", {})
    abnormal_pressure = by_name.get("abnormal_pressure_drop", {})
    sudden_flow = by_name.get("sudden_flow_change", {})
    potential_leak = by_name.get("potential_leak_signal", {})
    equipment_anomaly = by_name.get("equipment_anomaly", {})

    evidence = {
        "pressure": _without_empty_values(
            {
                key: pressure.get(key)
                for key in (
                    "minimum_pressure",
                    "maximum_pressure",
                    "pressure_violation_nodes",
                    "pressure_warning_nodes",
                    "pressure_margins",
                    "pressure_violation_episodes",
                    "pressure_warning_episodes",
                    "pressure_violation_duration_minutes",
                    "maximum_continuous_pressure_violation_minutes",
                    "pressure_recovery_time_minutes",
                    "simultaneous_end_user_warning_node_count",
                )
            }
        ),
        "flow": _without_empty_values(
            {
                "flow_change_magnitude": flow.get("flow_change_magnitude"),
                "boundary_flow_change_rate": flow.get("boundary_flow_change_rate"),
                "flow_ramp_events": flow.get("flow_ramp_events"),
                "abnormal_flow_segments": flow.get("abnormal_flow_segments"),
                "flow_capacity_excursion_episodes": by_name.get("flow_capacity_check", {}).get("flow_capacity_excursion_episodes"),
                "supply_demand_balance_status": flow.get("supply_demand_balance_status"),
                "supply_demand_balance": (balance.get("evaluated_values") or [None])[0],
            }
        ),
        "linepack": _without_empty_values(
            {
                key: linepack.get(key)
                for key in (
                    "minimum_linepack",
                    "linepack_change_rate",
                    "linepack_decline_episodes",
                    "linepack_recovery",
                    "short_term_peak_shaving_capacity",
                    "linepack_warning_status",
                )
            }
        ),
        "compressor": _without_empty_values(
            {
                "compressor_state": compressor.get("compressor_state"),
                "operating_envelope_status": compressor.get("operating_envelope_status"),
                "regulation_margin_to_fail": compressor.get("regulation_margin_to_fail"),
            }
        ),
        "equipment_regulation": _without_empty_values(
            {
                "equipment_regulation_state": equipment.get("equipment_regulation_state"),
                "valve_opening_status": by_name.get("valve_opening_range", {}).get("status"),
                "pressure_regulator_status": by_name.get("pressure_regulator_range", {}).get("status"),
                "boundary_adjustment_status": by_name.get("boundary_control_adjustment_magnitude", {}).get("status"),
            }
        ),
        "abnormality_warning": _without_empty_values(
            {
                "abnormal_pressure_drop": (abnormal_pressure.get("evaluated_values") or [None])[0],
                "sudden_flow_change": (sudden_flow.get("evaluated_values") or [None])[0],
                "potential_leak_signal": potential_leak.get("signal_evidence"),
                "equipment_anomaly": (equipment_anomaly.get("evaluated_values") or [None])[0],
            }
        ),
    }
    return _without_empty_values(evidence)


def compact_constraint_check(output: Dict[str, Any]) -> Dict[str, Any]:
    verification = dict(output.get("constraint_check") or {})
    checks = list(verification.get("checks") or [])
    findings = [_compact_finding(item) for item in verification.get("priority_findings", [])]
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
        "triggered_flags": [item.get("flag") for item in findings if item.get("flag")],
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
        "resolved_attention_variables": parsed_task.get("resolved_attention_variables", []),
        "resolved_output_variables": parsed_task.get("resolved_output_variables", []),
        "unresolved_attention_targets": parsed_task.get("unresolved_attention_targets", []),
        "unresolved_output_state_variables": parsed_task.get("unresolved_output_state_variables", []),
        "applied_boundary_conditions": metadata.get("applied_boundary_conditions", []),
    }
    provenance = {
        "checkpoint_id": metadata.get("checkpoint_id"),
        "data_case_id": metadata.get("data_case_id"),
        "device": metadata.get("device"),
        "model_input_projection_type": metadata.get("model_input_projection_type"),
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


def successful_pipeformer_output(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for item in trace.get("tool_calls", []):
        if item.get("tool_name") != "run_pipeformer_forecast":
            continue
        output = parse_tool_output(item)
        if isinstance(output, dict) and output.get("success"):
            return output
    return None


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


def build_teacher_record(scenario: Dict[str, Any], question: str, trace: Dict[str, Any]) -> Dict[str, Any]:
    pipeformer = successful_pipeformer_output(trace)
    raw_answer = final_answer(trace)
    raw_answer_issues = llm_answer_quality_issues(raw_answer, check_history_claims=pipeformer is not None)
    answer = remove_unsupported_history_claims(raw_answer) if pipeformer else raw_answer
    answer = enforce_requested_answer_scope(answer, question, pipeformer)
    answer = enforce_grounded_variable_descriptions(answer, pipeformer)
    quality_issues = llm_answer_quality_issues(answer, check_history_claims=pipeformer is not None)
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

    if quality_issues:
        quality_flag = "needs_review"
    if raw_answer_issues and not quality_issues:
        logger.warning(
            "Removed unsupported claims from exported final_answer: %s",
            ", ".join(raw_answer_issues),
        )

    return {
        "sample_id": f"sample_{scenario.get('scenario_id')}",
        "scenario_id": scenario.get("scenario_id"),
        "scenario_type": scenario.get("scenario_type"),
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


def run_backend_trace(scenario: Dict[str, Any], args: argparse.Namespace, index: int, run_stamp: str) -> tuple[str, Dict[str, Any]]:
    from agent.orchestrator import AgentOrchestrator
    from agent.schemas import AgentChatRequest

    question = first_user_input(scenario)
    scenario_id = str(scenario.get("scenario_id") or f"scenario_{index:06d}")
    session_id = args.session_id or f"teacher_{index:06d}_{scenario_id}_{run_stamp}"
    os.environ["PIPEFORMER_DEVICE"] = str(args.device)

    logger.info("Scenario %s started (%d): session_id=%s", scenario_id, index, session_id)
    logger.info("User question: %s", short_text(question, limit=500))

    orchestrator = AgentOrchestrator(
        data_loader=None,
        agent_id=args.agent_id,
        session_id=session_id,
        enable_skills=False,
        workspace_root_base=backend_root() / ".openclaw",
    )
    result = orchestrator.run_agent(AgentChatRequest(agent_id=args.agent_id, session_id=session_id, message=question))
    trace_path = Path(result.trace_summary.trace_path)
    logger.info("Scenario %s finished: status=%s trace=%s", scenario_id, result.trace_summary.status, trace_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    logger.info("Loaded trace: messages=%d tool_calls=%d", len(trace.get("messages", [])), len(trace.get("tool_calls", [])))
    trace["_trace_path"] = trace_path.as_posix()
    return question, trace


def selected_scenarios(args: argparse.Namespace) -> List[Dict[str, Any]]:
    scenarios = load_scenarios(args.scenario_file)
    if args.scenario_id:
        return [find_scenario(scenarios, args.scenario_id)]
    return scenarios


def main() -> int:
    load_backend_env()
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    scenarios = selected_scenarios(args)
    if args.session_id and len(scenarios) != 1:
        raise ValueError("--session-id can only be used with --scenario-id.")

    logger.info(
        "Teacher trace generation started: scenarios=%d device=%s scenario_file=%s",
        len(scenarios),
        args.device,
        args.scenario_file,
    )
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    records = []
    for index, scenario in enumerate(scenarios, start=1):
        question, trace = run_backend_trace(scenario, args, index, run_stamp)
        logger.info("Building teacher record for scenario=%s", scenario.get("scenario_id"))
        records.append(build_teacher_record(scenario, question, trace))

    logger.info("Writing JSONL output: %s", args.output_jsonl)
    write_jsonl(args.output_jsonl, records, force=args.force)
    logger.info("Writing pretty JSON output: %s", args.output_json)
    write_json(args.output_json, records[0] if len(records) == 1 else records, force=args.force)
    logger.info("Teacher trace generation complete: records=%d", len(records))
    print(
        json.dumps(
            {
                "status": "ok",
                "records": len(records),
                "output_jsonl": args.output_jsonl.as_posix(),
                "output_json": args.output_json.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
