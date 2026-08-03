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

from evaluator.csv_evidence import build_csv_evidence
from evaluator.decision_trace_state import (
    DecisionTraceState,
    bounded_recent_turns,
    serialize_verified_decision_state,
)
from evaluator.grounding_contract import (
    GroundingContractBuilder,
    comparison_answer_issues,
    finalize_applied_disturbance_disclosure,
    grounded_fallback_answer,
)
from evaluator.teacher_quality import (
    answer_quality_issues,
    evaluate_teacher_quality,
    grounded_numeric_claim_values,
    llm_answer_quality_issues,
    numeric_claim_values,
    numeric_claims_are_grounded,
    numeric_grounding_evidence,
    safety_and_energy_checks_pass as _safety_and_energy_checks_pass,
    tool_output_failed,
)
from evaluator.tool_evidence import attach_tool_arguments, classify_tool_evidence, requested_artifacts
from evaluator.topology_evidence import topology_summary_from_tool_outputs
from evaluator.scorer import NativeTraceEvaluator
from pipeline.forecast_registry_contract import authorize_forecast_registry
from pipeline.io_utils import write_json, write_jsonl
from pipeline.scenario_preflight import validate_scenario_sources
from pipeline.teacher_trace_store import (
    TeacherTraceStore,
    load_existing_records,
    merge_records,
    merge_session_records,
    scenario_sample_ids,
    scenario_session_record_ids,
    validate_combined_splits,
)


logger = logging.getLogger("teacher_trace")

SFT_MAX_RECORD_CHARS = 35_000
SFT_MAX_TOOL_TEXT_CHARS = 4_000
SFT_MAX_GENERIC_TOOL_PAIRS = 6
SFT_MAX_GENERIC_OUTPUT_CHARS = 2_500
SFT_MAX_PIPEFORMER_VARIABLES = 3
SFT_REGISTRY_ID_FIELDS = (
    "variable",
    "role",
    "controllable",
)
COMPACT_COMPARABLE_METRIC_KEYS = (
    "energy_consumption",
    "energy_consumption_delta",
    "energy_unit",
    "energy_variable_count",
    "baseline_reference",
)
SFT_VARIABLE_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?(?![A-Za-z0-9_:])"
)
SFT_FILE_REFERENCE = re.compile(r"(?i)\b[\w.-]+\.(?:csv|jsonl?|xlsx?|parquet)\b")
SFT_OMITTED_TOOL_KEYS = {
    "abs_path",
    "cmd",
    "cwd",
    "duration_s",
    "output_dir",
    "run_dir",
    "session_id",
    "timestamp",
    "workspace",
}
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
        scenarios = json.loads(resolved.read_text(encoding="utf-8-sig"))
        if not isinstance(scenarios, list):
            raise TypeError(f"Scenario file must contain a JSON list: {resolved}")
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


def combined_preflight_sources(
    all_sources: List[Dict[str, Any]],
    selected_sources: List[Dict[str, Any]],
    existing_records: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[str]]:
    required_pairs = {
        (
            str(record.get("dataset_source") or ""),
            str(record.get("scenario_id") or ""),
        )
        for record in existing_records
        if record.get("dataset_source") and record.get("scenario_id")
    }
    required_pairs.update(
        (
            str(source.get("dataset_source") or ""),
            str(scenario.get("scenario_id") or ""),
        )
        for source in selected_sources
        for scenario in source.get("scenarios") or []
    )
    available_pairs = {
        (str(source.get("dataset_source") or ""), str(scenario.get("scenario_id") or ""))
        for source in all_sources
        for scenario in source.get("scenarios") or []
    }
    selected = []
    for source in all_sources:
        source_name = str(source.get("dataset_source") or "")
        scenarios = [
            scenario
            for scenario in source.get("scenarios") or []
            if (source_name, str(scenario.get("scenario_id") or "")) in required_pairs
        ]
        if scenarios:
            selected.append({**source, "scenarios": scenarios})
    missing = sorted(f"{source}:{scenario}" for source, scenario in required_pairs - available_pairs)
    return selected, missing


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


def scenario_evidence_text(scenario: Dict[str, Any], question: str) -> str:
    description = str(scenario.get("scenario_description") or "").strip()
    return "\n".join(value for value in (description, question) if value)


def agent_turn_message(
    scenario: Dict[str, Any],
    question: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    del history  # Prior verified state is injected by the runtime state manager.
    description = str(scenario.get("scenario_description") or "").strip()
    parts = []
    if description:
        parts.extend([
            "Scenario context (scope and task intent only; verify factual claims with tools):",
            description,
        ])
    parts.extend(["Current user request:", question])
    return "\n\n".join(parts)


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing outputs. Without this flag, new unique records are appended.",
    )
    parser.add_argument(
        "--replace-selected-scenario",
        action="store_true",
        help="Replace only the scenario selected by both --dataset-source and --scenario-id.",
    )
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


def compact_sft_tool_output(value: Any) -> Any:
    """Compatibility wrapper around the configured trace projector."""
    return DEFAULT_PROJECTOR.compact_sft_output(value)


def tool_call_id(tool_call: Dict[str, Any], index: int) -> str:
    return str(tool_call.get("tool_call_id") or f"tool_{index:03d}")


def _without_none_values(value: Dict[str, Any]) -> Dict[str, Any]:
    """Keep stable list/dict fields while omitting values that are truly absent."""
    return {key: item for key, item in value.items() if item is not None}


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
    compact["resolved_attention_variable_count"] = len(parsed.get("resolved_attention_variables") or [])
    compact["resolved_output_variable_count"] = len(parsed.get("resolved_output_variables") or [])
    compact["boundary_conditions"] = compact_boundary
    return _without_none_values(compact)


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
        "disturbance_assumption",
        "disturbance_source",
        "counterfactual_comparison",
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
    return _without_none_values(compact)


def _relevant_forecast_variables(output: Dict[str, Any], limit: int = 8) -> List[str]:
    relevant: List[str] = []
    summaries = dict((output.get("prediction_summary") or {}).get("output_forecast_summary") or {})
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
        for item in list(finding.get("offending_values") or []) + list(finding.get("evaluated_values") or []):
            add(item.get("variable"))
    add((output.get("parsed_task") or {}).get("disturbance_variable"))
    for variable in summaries:
        add(variable)
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
    return _without_none_values(compact)


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
            compact["comparable_metrics"]["energy_evaluation_status"] = comparable_metrics[
                "energy_evaluation_status"
            ]
    return _without_none_values(compact)


def project_pipeformer_output(output: Dict[str, Any]) -> Dict[str, Any]:
    parsed_task = compact_parsed_task(output)
    metadata = dict(output.get("forecast_metadata") or {})
    task_resolution = {
        "resolved_attention_variable_count": parsed_task.get("resolved_attention_variable_count", 0),
        "resolved_output_variable_count": parsed_task.get("resolved_output_variable_count", 0),
        "unresolved_attention_targets": parsed_task.get("unresolved_attention_targets", []),
        "unresolved_output_state_variables": parsed_task.get("unresolved_output_state_variables", []),
        "applied_boundary_conditions": metadata.get("applied_boundary_conditions", []),
        "variable_normalizations": parsed_task.get("variable_normalizations", []),
        "vocabulary_normalizations": parsed_task.get("vocabulary_normalizations", []),
        "invalid_normalized_variables": parsed_task.get("invalid_normalized_variables", []),
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
        "prediction_summary": compact_prediction_summary(output),
        "constraint_check": compact_constraint_check(output),
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


def export_trace_tools(
    trace: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    calls: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    pipeformer_results: List[Dict[str, Any]] = []
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
        output = sanitize_tool_output(parse_tool_output(item))
        if tool_name == "run_pipeformer_forecast" and isinstance(output, dict) and output.get("success"):
            candidate_id = (item.get("args") or {}).get("candidate_id") or output.get("candidate_id")
            candidate_role = (item.get("args") or {}).get("candidate_role") or output.get("candidate_role")
            projection = project_pipeformer_output(output)
            pipeformer_results.append({"tool_call_id": call_id, "output": output, "projection": projection})
            output = compact_pipeformer_output(projection)
            if candidate_id:
                output["candidate_id"] = candidate_id
            if candidate_role:
                output["candidate_role"] = candidate_role
        outputs.append(
            {
                "tool_call_id": call_id,
                "name": tool_name,
                "output": output,
            }
        )
    return calls, outputs, pipeformer_results


def final_answer(trace: Dict[str, Any]) -> str:
    for message in reversed(trace.get("messages", [])):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


class TeacherTraceProjector:
    """Convert raw agent/tool traces into compact, stable training fields."""

    def __init__(
        self,
        *,
        max_tool_text_chars: int = SFT_MAX_TOOL_TEXT_CHARS,
        omitted_tool_keys: Optional[set[str]] = None,
    ) -> None:
        self.max_tool_text_chars = max_tool_text_chars
        self.omitted_tool_keys = frozenset(omitted_tool_keys or SFT_OMITTED_TOOL_KEYS)

    @staticmethod
    def sanitize(value: Any) -> Any:
        return sanitize_tool_output(value)

    def compact_sft_output(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self.compact_sft_output(item)
                for key, item in value.items()
                if key not in self.omitted_tool_keys
            }
        if isinstance(value, list):
            return [self.compact_sft_output(item) for item in value]
        if isinstance(value, str) and len(value) > self.max_tool_text_chars:
            return value[: self.max_tool_text_chars] + "... [truncated for SFT]"
        return value

    def compact_sft_trajectory(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_outputs: List[Dict[str, Any]],
        answer: str,
        *,
        max_pipeformer_variables: Optional[int] = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Keep the smallest evidence-bearing tool trajectory for SFT."""
        outputs_by_id = {
            str(item.get("tool_call_id") or ""): item for item in tool_outputs
        }
        pairs = [
            (index, call, outputs_by_id.get(str(call.get("tool_call_id") or "")))
            for index, call in enumerate(tool_calls)
        ]
        successful = [
            pair
            for pair in pairs
            if pair[2] is not None and not tool_output_failed(pair[2])
        ]
        successful_pipeformer = [
            pair for pair in successful if pair[1].get("name") == "run_pipeformer_forecast"
        ]
        successful_decision_policies = [
            pair for pair in successful if pair[1].get("name") == "set_decision_policy"
        ]
        if successful_pipeformer:
            referenced_variables = self._reference_tokens(answer)
            action_forecasts = [
                pair
                for pair in successful_pipeformer
                if self._forecast_action_variables(pair[1]) & referenced_variables
            ]
            # A current comparison needs every cited action forecast, but an
            # actionless baseline duplicate is already represented by the
            # candidate forecasts' applied-disturbance evidence.
            if action_forecasts:
                successful_pipeformer = action_forecasts
            multiple_pipeformer = len(successful_pipeformer) > 1
            required_registry_call_ids = self._registry_calls_required_for_forecasts(
                successful,
                successful_pipeformer,
            )
            selected_registry_searches = [
                pair
                for pair in successful
                if (
                    pair[1].get("name") == "search_pipeformer_registry"
                    and str(pair[1].get("tool_call_id") or "")
                    in required_registry_call_ids
                )
            ]
            selected = sorted(
                [
                    *selected_registry_searches,
                    *successful_decision_policies[-1:],
                    *successful_pipeformer,
                ],
                key=lambda pair: pair[0],
            )
        elif successful:
            multiple_pipeformer = False
            selected = self._select_generic_evidence_pairs(successful, answer)
        else:
            multiple_pipeformer = False
            selected = pairs[-1:]

        compact_calls = []
        compact_outputs = []
        for _, call, output in selected:
            compact_calls.append(
                {
                    "tool_call_id": call.get("tool_call_id"),
                    "name": call.get("name"),
                    "arguments": self._compact_call_arguments(
                        call.get("arguments") or {},
                        pipeformer=call.get("name") == "run_pipeformer_forecast",
                    ),
                }
            )
            if output is None:
                continue
            raw_output = output.get("output")
            if (
                call.get("name") == "search_pipeformer_registry"
                and isinstance(raw_output, dict)
            ):
                raw_output = self._compact_registry_sft_output(raw_output)
            elif call.get("name") == "run_pipeformer_forecast" and isinstance(raw_output, dict):
                raw_output = self._compact_pipeformer_sft_output(
                    raw_output,
                    answer,
                    include_auxiliary_variables=not multiple_pipeformer,
                    max_variables=(
                        SFT_MAX_PIPEFORMER_VARIABLES
                        if max_pipeformer_variables is None
                        else max_pipeformer_variables
                    ),
                )
            else:
                raw_output = self._compact_generic_sft_output(
                    raw_output,
                    answer,
                    tool_name=str(call.get("name") or ""),
                    arguments=dict(call.get("arguments") or {}),
                )
            compact_outputs.append(
                {
                    "tool_call_id": output.get("tool_call_id"),
                    "name": output.get("name"),
                    "output": raw_output,
                }
            )
        return compact_calls, compact_outputs

    @staticmethod
    def _registry_calls_required_for_forecasts(
        successful: List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
        forecasts: List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
    ) -> set[str]:
        """Select only searches that actually authorize a retained forecast."""
        required: set[str] = set()
        for forecast_index, forecast_call, _ in forecasts:
            preceding = []
            for index, call, output_record in successful:
                if index >= forecast_index:
                    continue
                preceding.append(
                    {
                        "tool_call_id": call.get("tool_call_id"),
                        "name": call.get("name"),
                        "arguments": dict(call.get("arguments") or {}),
                        "output": (output_record or {}).get("output"),
                    }
                )
            authorization = authorize_forecast_registry(
                dict(forecast_call.get("arguments") or {}),
                preceding,
            )
            required.update(
                str(value)
                for value in authorization.get("disturbance_search_call_ids") or []
                if str(value)
            )
            for call_ids in (authorization.get("candidate_search_call_ids") or {}).values():
                required.update(str(value) for value in call_ids if str(value))
        return required

    @staticmethod
    def _compact_registry_sft_output(
        value: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keep registry authorization evidence while bounding broad result pages.

        Search pages can contain 50 variables with nested effect targets.  The
        forecast contract only needs the canonical ID (and role/controllability)
        for every returned entry.  The full semantic registry entry remains in
        the audit transcript and verified state; this bounded projection keeps
        registry-before-forecast teaching evidence without oversized SFT records.
        """
        compact: Dict[str, Any] = {
            key: value[key]
            for key in (
                "success",
                "matched_variable_count",
                "matched_total_count",
                "offset",
                "next_offset",
                "error",
                "exit_code",
            )
            if key in value and value[key] not in (None, "")
        }
        variables = []
        for raw in value.get("variables") or []:
            if not isinstance(raw, dict) or not raw.get("variable"):
                continue
            item = {
                key: raw[key]
                for key in SFT_REGISTRY_ID_FIELDS
                if key in raw
            }
            variables.append(item)
        compact["variables"] = variables
        return compact

    @staticmethod
    def _forecast_action_variables(call: Dict[str, Any]) -> set[str]:
        boundary = dict(dict(call.get("arguments") or {}).get(
            "boundary_conditions"
        ) or {})
        return {
            str(variable)
            for key in ("percentage_changes", "setpoints")
            for variable in dict(boundary.get(key) or {})
            if str(variable)
        }

    def _select_generic_evidence_pairs(
        self,
        pairs: List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
        answer: str,
    ) -> List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]]:
        references = self._reference_tokens(answer)
        remaining = set(references)
        available = list(pairs)
        selected = []
        while available and len(selected) < SFT_MAX_GENERIC_TOOL_PAIRS:
            ranked = []
            for pair in available:
                blob = json.dumps((pair[2] or {}).get("output") or {}, ensure_ascii=False).casefold()
                normalized = blob.replace(",", "")
                covered = {
                    value for value in remaining
                    if value.casefold().replace(",", "") in normalized
                }
                ranked.append((len(covered), self._evidence_score(pair[2], answer), -pair[0], pair, covered))
            _, _, _, best, covered = max(ranked, key=lambda item: item[:3])
            selected.append(best)
            remaining.difference_update(covered)
            available.remove(best)
            if not remaining and selected:
                break
        return sorted(selected, key=lambda pair: pair[0])

    @staticmethod
    def _reference_tokens(answer: str) -> set[str]:
        values = set(SFT_VARIABLE_REFERENCE.findall(answer))
        values.update(SFT_FILE_REFERENCE.findall(answer))
        values.update(str(value) for value in numeric_claim_values(answer))
        return {value for value in values if value}

    def _compact_generic_sft_output(
        self,
        value: Any,
        answer: str,
        *,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """Retain a small excerpt containing every answer-grounding token possible."""
        if not isinstance(value, (dict, list, str)):
            return value
        if isinstance(value, dict):
            status = {
                key: value[key]
                for key in ("success", "exit_code", "error", "stderr")
                if key in value and value[key] not in (None, "")
            }
        else:
            status = {}
        if isinstance(value, dict):
            assessment = classify_tool_evidence(
                {"name": tool_name, "arguments": arguments, "output": value}
            )
            evidence_kind = {
                "file_content_read": "file_content",
                "command_content_or_computation": "command_content",
            }.get(assessment.reason)
            if evidence_kind:
                status["evidence_kind"] = evidence_kind
                source_payload = {
                    "path": value.get("path"),
                    "abs_path": value.get("abs_path"),
                    "cmd": value.get("cmd"),
                    "arguments": arguments,
                }
                source_artifacts = list(
                    requested_artifacts(json.dumps(source_payload, ensure_ascii=False))
                )
                if source_artifacts:
                    status["source_artifacts"] = source_artifacts
        compact_value = self.compact_sft_output(value)
        text_value = (
            compact_value
            if isinstance(compact_value, str)
            else json.dumps(compact_value, ensure_ascii=False, indent=2)
        )
        lines = text_value.splitlines() or [text_value]
        references = self._reference_tokens(answer)
        selected_indices = {0, len(lines) - 1}
        normalized_lines = [line.casefold().replace(",", "") for line in lines]
        for reference in references:
            token = reference.casefold().replace(",", "")
            for index, line in enumerate(normalized_lines):
                if token in line:
                    selected_indices.update({max(0, index - 1), index, min(len(lines) - 1, index + 1)})
                    break
        excerpt_lines = [lines[index] for index in sorted(selected_indices)]
        excerpt = "\n".join(excerpt_lines)
        if len(selected_indices) <= 2 and len(excerpt) < min(800, len(text_value)):
            excerpt = text_value[: min(len(text_value), SFT_MAX_GENERIC_OUTPUT_CHARS)]
        if len(excerpt) > SFT_MAX_GENERIC_OUTPUT_CHARS:
            excerpt = excerpt[:SFT_MAX_GENERIC_OUTPUT_CHARS].rsplit("\n", 1)[0]
        return {**status, "evidence_excerpt": excerpt}

    def compact_record_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        compact = {
            key: value
            for key, value in evidence.items()
            if key not in {
                "top_watch_variables",
                "key_observation_variables",
                "verified_numeric_claims",
                "candidate_forecasts",
            }
        }
        for key in ("top_watch_variables", "key_observation_variables"):
            if not evidence.get(key):
                continue
            compact[key] = [
                {
                    item_key: item[item_key]
                    for item_key in (
                        "variable",
                        "role",
                        "metric",
                        "value",
                        "status",
                        "mean_prediction",
                        "mean_abs_delta_vs_observed",
                    )
                    if item_key in item
                }
                for item in list(evidence.get(key) or [])[:3]
            ]
        return self.compact_sft_output(compact)

    def compact_sft_decision_summary(self, value: Any) -> Dict[str, Any]:
        """Keep decision labels and ordering while leaving metric detail to tools."""
        summary = dict(value or {})
        raw_policy = summary.get("ranking_policy")
        if isinstance(raw_policy, dict):
            compact_policy = {
                "source": raw_policy.get("source"),
                "hard_constraints": list(
                    raw_policy.get("hard_constraints") or []
                ),
                "objectives": [
                    {
                        key: objective[key]
                        for key in ("metric", "direction", "tolerance")
                        if key in objective
                    }
                    for objective in raw_policy.get("objectives") or []
                    if isinstance(objective, dict)
                ],
            }
        elif isinstance(raw_policy, str) and raw_policy.strip():
            compact_policy = {
                "source": "legacy_named_policy",
                "policy_id": raw_policy.strip(),
            }
        else:
            compact_policy = {}
        compact_summary = {
            key: summary[key]
            for key in (
                "status",
                "selected_candidate_id",
                "ranked_candidate_ids",
                "ranked_candidate_groups",
                "eliminated_candidates",
                "missing_metrics",
            )
            if key in summary
        }
        if compact_policy:
            compact_summary["ranking_policy"] = compact_policy
        return compact_summary

    def _compact_call_arguments(self, value: Any, *, pipeformer: bool) -> Any:
        # Keep the complete argument object.  In particular, ``question`` is a
        # required field of ``run_pipeformer_forecast`` even though it repeats
        # the record's user_input.  Dropping it here creates schema-invalid SFT
        # tool calls.  ``pipeformer`` remains part of this helper's interface
        # for compatibility with existing projection call sites.
        del pipeformer
        if isinstance(value, dict):
            return {key: self._compact_call_arguments(item, pipeformer=False) for key, item in value.items()}
        if isinstance(value, list):
            return [self._compact_call_arguments(item, pipeformer=False) for item in value]
        if isinstance(value, str) and len(value) > 2_000:
            return value[:2_000] + "... [truncated for SFT]"
        return value

    @staticmethod
    def _evidence_score(output: Optional[Dict[str, Any]], answer: str) -> int:
        blob = json.dumps((output or {}).get("output") or {}, ensure_ascii=False).casefold()
        references = set(SFT_VARIABLE_REFERENCE.findall(answer))
        references.update(SFT_FILE_REFERENCE.findall(answer))
        references.update(str(value) for value in numeric_claim_values(answer))
        score = sum(3 for value in references if value.casefold() in blob)
        if '"stdout"' in blob or '"content"' in blob:
            score += 1
        return score

    def _compact_pipeformer_sft_output(
        self,
        output: Dict[str, Any],
        answer: str,
        *,
        include_auxiliary_variables: bool,
        max_variables: int,
    ) -> Dict[str, Any]:
        prediction = dict(output.get("prediction") or {})
        verification = dict(output.get("verification") or {})
        evidence = dict(output.get("evidence") or {})
        referenced = list(dict.fromkeys(SFT_VARIABLE_REFERENCE.findall(answer)))
        if include_auxiliary_variables:
            for key in ("top_watch_variables", "key_observation_variables"):
                referenced.extend(
                    str(item.get("variable"))
                    for item in evidence.get(key) or []
                    if item.get("variable")
                )
            for finding in verification.get("priority_findings") or []:
                referenced.extend(str(value) for value in finding.get("affected_variables") or [])
        referenced = list(dict.fromkeys(referenced))[:max_variables]
        summary = dict(prediction.get("output_forecast_summary") or {})
        metric_keys = {
            "mean_prediction",
            "minimum_prediction",
            "maximum_prediction",
            "max_abs_prediction",
            "prediction_change",
            "max_abs_step_change",
            "max_step_decline",
            "max_decline_from_start",
            "recovery_from_minimum",
        }
        compact_summary = {
            variable: {
                key: value
                for key, value in dict(summary.get(variable) or {}).items()
                if key in metric_keys
            }
            for variable in referenced
            if variable in summary
        }
        prediction_keys = (
            "forecast_mode",
            "case_id",
            "current_operating_condition_number",
            "forecast_horizon_minutes",
            "actual_forecast_horizon_minutes",
            "actual_forecast_steps",
            "disturbance_variable",
            "disturbance_direction",
            "disturbance_magnitude_percent",
            "disturbance_assumption",
            "disturbance_source",
            "forecast_window",
            "counterfactual_comparison",
            "total_output_variable_count",
        )
        compact_prediction = {
            key: prediction[key] for key in prediction_keys if key in prediction
        }
        if not include_auxiliary_variables and "counterfactual_comparison" in compact_prediction:
            comparison = dict(compact_prediction["counterfactual_comparison"] or {})
            compact_prediction["counterfactual_comparison"] = {
                key: comparison[key]
                for key in (
                    "mode",
                    "compared_step_count",
                    "compared_output_variable_count",
                    "nonzero_impacted_variable_count",
                    "baseline_reference",
                    "disturbance_variable",
                    "applied_disturbance",
                )
                if key in comparison
            }
        compact_prediction["output_forecast_summary"] = compact_summary
        verification_keys = (
            "requested_categories",
            "category_status",
            "safety_energy_comparison",
            "rule_status",
            "overall_status",
            "verification_complete",
            "not_evaluated_rules",
            "risk_level",
            "risk_escalations",
            "failure_count",
            "warning_count",
            "failed_rule_ids",
            "warning_rule_ids",
            "triggered_flags",
            "human_intervention_label",
            "dispatch_recommendation",
            "priority_findings",
        )
        compact_verification = {
            key: verification[key]
            for key in verification_keys
            if key in verification and key not in {"rule_status", "risk_escalations", "priority_findings"}
        }
        if "comparable_metrics" in verification:
            metrics = dict(verification.get("comparable_metrics") or {})
            compact_verification["comparable_metrics"] = {
                key: metrics[key]
                for key in COMPACT_COMPARABLE_METRIC_KEYS
                if key in metrics
            }
            if "energy_evaluation_status" in metrics:
                compact_verification["comparable_metrics"]["evaluation_status"] = metrics[
                    "energy_evaluation_status"
                ]
        compact_verification["priority_findings"] = [
            {
                key: finding[key]
                for key in (
                    "name",
                    "category",
                    "status",
                    "evaluation_status",
                    "flag",
                    "priority",
                    "affected_variables",
                )
                if key in finding
            }
            for finding in (verification.get("priority_findings") or [])[:5]
        ]
        compact_evidence = {
            key: [
                {
                    item_key: item[item_key]
                    for item_key in (
                        "variable",
                        "role",
                        "metric",
                        "value",
                        "status",
                        "mean_prediction",
                        "mean_abs_delta_vs_observed",
                    )
                    if item_key in item
                }
                for item in list(evidence.get(key) or [])[:3]
            ]
            for key in ("top_watch_variables", "key_observation_variables")
            if include_auxiliary_variables and evidence.get(key)
        }
        if evidence.get("boundary_application_evidence"):
            compact_evidence["boundary_application_evidence"] = [
                {
                    key: item[key]
                    for key in (
                        "variable",
                        "mode",
                        "requested_value",
                        "input_values_applied",
                        "verified",
                    )
                    if key in item
                }
                for item in evidence.get("boundary_application_evidence") or []
            ]
        task_resolution = dict(output.get("task_resolution") or {})
        compact_resolution = {
            key: task_resolution[key]
            for key in (
                "resolved_attention_variable_count",
                "resolved_output_variable_count",
                "unresolved_attention_targets",
                "unresolved_output_state_variables",
                "applied_boundary_conditions",
            )
            if key in task_resolution
        }
        provenance = dict(output.get("provenance") or {})
        compact_provenance = {
            key: provenance[key]
            for key in ("checkpoint_id", "forecast_mode", "device")
            if key in provenance
        }
        return {
            "success": output.get("success") is True,
            "task_resolution": compact_resolution,
            "prediction": compact_prediction,
            "verification": compact_verification,
            "evidence": compact_evidence,
            "provenance": compact_provenance,
        }

    @staticmethod
    def export_tools(
        trace: Dict[str, Any],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        return export_trace_tools(trace)


class TeacherTraceRecordBuilder:
    """Build one evaluated SFT record from one assistant-turn trace."""

    def __init__(
        self,
        projector: Optional[TeacherTraceProjector] = None,
        evaluator: Optional[NativeTraceEvaluator] = None,
    ) -> None:
        self.projector = projector or TeacherTraceProjector()
        self.evaluator = evaluator or NativeTraceEvaluator()

    def build(
        self,
        scenario: Dict[str, Any],
        question: str,
        trace: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return _build_teacher_record(
            scenario,
            question,
            trace,
            projector=self.projector,
            native_evaluator=self.evaluator,
            **kwargs,
        )


DEFAULT_PROJECTOR = TeacherTraceProjector()
DEFAULT_RECORD_BUILDER = TeacherTraceRecordBuilder(projector=DEFAULT_PROJECTOR)


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
    """Compatibility entry point used by tests and existing integrations."""
    return DEFAULT_RECORD_BUILDER.build(
        scenario,
        question,
        trace,
        source_session_id=source_session_id,
        turn_id=turn_id,
        conversation_context=conversation_context,
        split=split,
    )










def _ensure_applied_disturbance_disclosure(
    answer: str,
    question: str,
    grounding_contract: Dict[str, Any],
) -> str:
    """Apply the exact machine-generated disclosure block to any forecast answer."""
    del question  # Kept in the public helper signature for existing callers.
    return finalize_applied_disturbance_disclosure(answer, grounding_contract)


def _build_teacher_record(
    scenario: Dict[str, Any],
    question: str,
    trace: Dict[str, Any],
    *,
    projector: TeacherTraceProjector,
    native_evaluator: NativeTraceEvaluator,
    source_session_id: str = "session_001",
    turn_id: int = 1,
    conversation_context: Optional[List[Dict[str, Any]]] = None,
    split: str = "train",
) -> Dict[str, Any]:
    dataset_source = str(scenario.get("dataset_source") or "unknown_source")
    scenario_id = str(scenario.get("scenario_id") or "unknown_scenario")
    record_id = f"{dataset_source}:{source_session_id}::turn_{turn_id:03d}"
    tool_calls, tool_outputs, pipeformer_results = projector.export_tools(trace)
    pipeformer_call_count = sum(
        item.get("tool_name") == "run_pipeformer_forecast"
        for item in trace.get("tool_calls", [])
    )
    grounded_tool_outputs = attach_tool_arguments(tool_outputs, tool_calls)
    history_state = DecisionTraceState.from_history(conversation_context or [])
    state_before = serialize_verified_decision_state(
        history_state,
        max_chars=int(os.getenv("VERIFIED_STATE_MAX_CHARS", "16000")),
    )
    recent_turns = bounded_recent_turns(
        conversation_context or [],
        max_turns=2,
        max_chars=int(os.getenv("RECENT_TURNS_MAX_CHARS", "4000")),
    )
    grounding_contract = GroundingContractBuilder().build(
        question,
        grounded_tool_outputs,
        prior_candidate_results=history_state.candidate_results,
        prior_decision_policy=history_state.decision_policy,
        prior_decision_policy_source_question=(
            history_state.decision_policy_source_question
        ),
        prior_applied_disturbances=history_state.applied_disturbances,
        require_decision_policy=True,
    )
    candidate_ids = {
        str(item.get("tool_call_id") or ""): str(item.get("candidate_id") or "")
        for item in grounding_contract.get("candidate_results") or []
    }
    if candidate_ids:
        pipeformer_results = [
            item
            for item in pipeformer_results
            if str(item.get("tool_call_id") or "") in candidate_ids
        ]
    pipeformer_outputs = [item["output"] for item in pipeformer_results]
    projections = [item["projection"] for item in pipeformer_results]
    pipeformer = pipeformer_outputs[0] if len(pipeformer_outputs) == 1 else None
    projection = projections[0] if len(projections) == 1 else None
    answer = final_answer(trace).strip()
    original_answer = answer
    answer = _ensure_applied_disturbance_disclosure(
        answer,
        question,
        grounding_contract,
    )
    disclosure_repair_applied = answer != original_answer
    parsed_task: Dict[str, Any] = {}
    prediction_summary: Dict[str, Any] = {}
    constraint_check: Dict[str, Any] = {}
    evidence: Dict[str, Any] = {}
    risk_level = None
    manual_intervention_label = None
    dispatch_recommendation = None
    if pipeformer and projection:
        parsed_task = projection["parsed_task"]
        prediction_summary = projection["prediction_summary"]
        constraint_check = projection["constraint_check"]
        evidence = projection["evidence"]
        risk_level = pipeformer.get("risk_level")
        manual_intervention_label = pipeformer.get("manual_intervention_label")
        dispatch_recommendation = pipeformer.get("dispatch_recommendation")
    elif len(pipeformer_outputs) > 1:
        def candidate_projection(item: Dict[str, Any], field: str) -> Dict[str, Any]:
            return {
                "candidate_id": candidate_ids[str(item["tool_call_id"])],
                "tool_call_id": str(item["tool_call_id"]),
                **dict(item["projection"].get(field) or {}),
            }

        parsed_task = {
            "candidate_forecasts": [
                candidate_projection(item, "parsed_task")
                for item in pipeformer_results
            ]
        }
        prediction_summary = {
            "candidate_forecasts": [
                candidate_projection(item, "prediction_summary")
                for item in pipeformer_results
            ]
        }
        constraint_check = {
            "candidate_forecasts": [
                candidate_projection(item, "constraint_check")
                for item in pipeformer_results
            ]
        }
        evidence = {
            "candidate_forecasts": [
                candidate_projection(item, "evidence")
                for item in pipeformer_results
            ]
        }
        risk_level = grounding_contract.get("worst_case_risk_level")
        manual_intervention_label = grounding_contract.get(
            "worst_case_intervention_label"
        )
        dispatch_recommendation = str(
            (grounding_contract.get("decision_summary") or {}).get(
                "selected_dispatch_recommendation"
            )
            or ""
        )
    quality_question = scenario_evidence_text(scenario, question)
    csv_evidence = build_csv_evidence(
        tool_calls,
        tool_outputs,
        answer,
        scope_text=quality_question,
    )
    if csv_evidence:
        evidence["csv_evidence"] = csv_evidence
    topology_summary = topology_summary_from_tool_outputs(grounded_tool_outputs)
    if topology_summary:
        evidence["topology_summary"] = topology_summary
    decision_summary = dict(grounding_contract.get("decision_summary") or {})
    fallback_applied = False
    answer_quality_flag, quality_issues = evaluate_teacher_quality(
        answer=answer,
        question=question,
        pipeformer=pipeformer,
        trace_status=trace.get("status"),
        pipeformer_call_count=pipeformer_call_count,
        pipeformer_outputs=pipeformer_outputs,
        conversation_context=conversation_context,
        tool_outputs=grounded_tool_outputs,
        record_evidence=evidence,
        grounding_contract=grounding_contract,
    )
    if (
        grounding_contract.get("answer_mode") == "dispatch_comparison"
        and quality_issues
    ):
        fallback_answer = grounded_fallback_answer(question, grounding_contract)
        fallback_flag, fallback_issues = evaluate_teacher_quality(
            answer=fallback_answer,
            question=question,
            pipeformer=pipeformer,
            trace_status=trace.get("status"),
            pipeformer_call_count=pipeformer_call_count,
            pipeformer_outputs=pipeformer_outputs,
            conversation_context=conversation_context,
            tool_outputs=grounded_tool_outputs,
            record_evidence=evidence,
            grounding_contract=grounding_contract,
        )
        if len(fallback_issues) < len(quality_issues):
            answer = fallback_answer
            answer_quality_flag = fallback_flag
            quality_issues = fallback_issues
            fallback_applied = True
    verified_numeric_claims = [
        value
        for value in dict.fromkeys(numeric_claim_values(answer))
        if numeric_claims_are_grounded(
            str(value),
            "",
            {"pipeformer_outputs": pipeformer_outputs},
        )
    ]
    if verified_numeric_claims:
        evidence["verified_numeric_claims"] = verified_numeric_claims
    if quality_issues:
        logger.warning("Teacher answer requires review: %s", ", ".join(quality_issues))

    record = {
        "sample_id": record_id,
        "dataset_source": dataset_source,
        "source_scenario_id": scenario_id,
        "scenario_id": scenario_id,
        "split_group_id": scenario_id,
        "session_id": source_session_id,
        "turn_id": turn_id,
        "scenario_type": scenario.get("scenario_type"),
        "split": split,
        "answer_mode": grounding_contract.get("answer_mode"),
        "grounding_contract": projector.sanitize(grounding_contract),
        "decision_summary": projector.sanitize(decision_summary),
        "conversation_context": projector.sanitize(conversation_context or []),
        "state_before": projector.sanitize(state_before),
        "recent_turns": projector.sanitize(recent_turns),
        "user_input": question,
        "parsed_task": projector.sanitize(parsed_task),
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
        "prediction_summary": projector.sanitize(prediction_summary),
        "constraint_check": projector.sanitize(constraint_check),
        "evidence": projector.sanitize(evidence),
        "risk_level": risk_level,
        "manual_intervention_label": manual_intervention_label,
        "dispatch_recommendation": dispatch_recommendation,
        "final_answer": answer,
        "trace_status": trace.get("status"),
        "quality_flag": answer_quality_flag,
        "quality_issues": quality_issues,
    }
    if grounding_contract.get("decision_policy"):
        record["decision_policy"] = projector.sanitize(
            grounding_contract["decision_policy"]
        )
    if disclosure_repair_applied:
        record["repair_provenance"] = {
            "method": "deterministic_disturbance_disclosure",
            "external_llm_calls": 0,
            "reason": (
                "Canonical applied-disturbance wording was prepended from stored "
                "execution evidence without changing the model's substantive answer."
            ),
        }
    elif fallback_applied:
        record["repair_provenance"] = {
            "method": "deterministic_grounding_contract",
            "external_llm_calls": 0,
            "reason": "Multi-candidate answer rebuilt from stored tool evidence.",
        }
    native_quality = native_evaluator.evaluate(
        record,
        hard_issues=quality_issues,
        trace_status=trace.get("status"),
    )
    record["quality_flag"] = native_quality["quality_flag"]
    record["quality_score"] = native_quality["quality_score"]
    record["quality_profile"] = native_quality["profile"]
    record["quality_failed_checks"] = native_quality["failed_checks"]
    return record


def _turn_trace(full_trace: Dict[str, Any], message_start: int, tool_start: int) -> Dict[str, Any]:
    return {
        "session_id": full_trace.get("session_id"),
        "agent_id": full_trace.get("agent_id"),
        "status": full_trace.get("status"),
        "messages": list(full_trace.get("messages") or [])[message_start:],
        "tool_calls": list(full_trace.get("tool_calls") or [])[tool_start:],
    }


def _pipeformer_history_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    prediction = dict(record.get("prediction_summary") or {})
    verification = dict(record.get("constraint_check") or {})
    evidence = dict(record.get("evidence") or {})
    stored_contract = dict(record.get("grounding_contract") or {})
    if (
        stored_contract.get("answer_mode") == "dispatch_comparison"
        and stored_contract.get("candidate_results")
    ):
        forecast_arguments = {
            str(item.get("tool_call_id") or ""): dict(
                item.get("arguments") or {}
            )
            for item in record.get("tool_calls") or []
            if item.get("name") == "run_pipeformer_forecast"
        }
        applied_disturbances = list(
            stored_contract.get("applied_disturbances") or []
        )
        scoped_candidates = []
        for source_candidate in stored_contract.get("candidate_results") or []:
            candidate = dict(source_candidate)
            arguments = forecast_arguments.get(
                str(candidate.get("tool_call_id") or ""),
                {},
            )
            if arguments:
                candidate["case_id"] = arguments.get("case_id")
                candidate["forecast_horizon_minutes"] = arguments.get(
                    "forecast_horizon_minutes"
                )
                disturbance_variable = str(
                    arguments.get("disturbance_variable") or ""
                )
                matching = next(
                    (
                        dict(item)
                        for item in applied_disturbances
                        if str(dict(item).get("variable") or "")
                        == disturbance_variable
                    ),
                    {},
                )
                if matching:
                    candidate["disturbance"] = matching
            scoped_candidates.append(_without_none_values(candidate))
        return _without_none_values({
            "candidate_results": scoped_candidates,
            "decision_summary": stored_contract.get("decision_summary") or {},
            "comparison_leaders": stored_contract.get("comparison_leaders") or {},
            "applied_disturbances": applied_disturbances,
            "worst_case_risk_level": stored_contract.get(
                "worst_case_risk_level"
            ),
            "worst_case_intervention_label": stored_contract.get(
                "worst_case_intervention_label"
            ),
        })

    def project(
        candidate_prediction: Dict[str, Any],
        candidate_verification: Dict[str, Any],
        candidate_evidence: Dict[str, Any],
        *,
        include_record_decision: bool,
    ) -> Dict[str, Any]:
        summary = {
            key: candidate_prediction.get(key)
            for key in ("candidate_id", "tool_call_id", "forecast_mode", "case_id", "forecast_horizon_minutes")
            if candidate_prediction.get(key) is not None
        }
        disturbance = _without_none_values({
            "variable": candidate_prediction.get("disturbance_variable"),
            "direction": candidate_prediction.get("disturbance_direction"),
            "magnitude_percent": candidate_prediction.get("disturbance_magnitude_percent"),
        })
        applied_disturbances = (
            list(stored_contract.get("applied_disturbances") or [])
            if include_record_decision
            else []
        )
        if applied_disturbances:
            matching_disturbance = next(
                (
                    item
                    for item in applied_disturbances
                    if item.get("variable") == disturbance.get("variable")
                ),
                applied_disturbances[0],
            )
            if matching_disturbance.get("mode") == "setpoint":
                disturbance["setpoint"] = matching_disturbance.get(
                    "requested_value"
                )
        if disturbance:
            summary["disturbance"] = disturbance
        if applied_disturbances:
            summary["applied_disturbances"] = applied_disturbances
        counterfactual = dict(candidate_prediction.get("counterfactual_comparison") or {})
        if counterfactual:
            if counterfactual.get("top_impacted_variables"):
                counterfactual["top_impacted_variables"] = list(counterfactual["top_impacted_variables"])[:5]
            summary["counterfactual_comparison"] = counterfactual
        summary.update(_without_none_values({
            "category_status": candidate_verification.get("category_status"),
            "verification_complete": candidate_verification.get("verification_complete"),
            "risk_level": (
                candidate_verification.get("risk_level")
                or (record.get("risk_level") if include_record_decision else None)
            ),
            "human_intervention_label": (
                candidate_verification.get("human_intervention_label")
                or (record.get("manual_intervention_label") if include_record_decision else None)
            ),
        }))
        comparable_metrics = dict(candidate_verification.get("comparable_metrics") or {})
        if comparable_metrics:
            summary["comparable_metrics"] = {
                key: comparable_metrics[key]
                for key in (*COMPACT_COMPARABLE_METRIC_KEYS, "energy_evaluation_status")
                if key in comparable_metrics
            }
        findings = []
        for finding in list(candidate_verification.get("priority_findings") or [])[:5]:
            compact = {
                key: finding.get(key)
                for key in ("name", "category", "status", "flag", "priority", "affected_variables")
                if finding.get(key) is not None
            }
            compact["evaluated_values"] = [
                {
                    key: item.get(key)
                    for key in (
                        "variable",
                        "metric",
                        "value",
                        "status",
                        "warning_threshold",
                        "fail_threshold",
                        "warning_margin",
                        "fail_margin",
                    )
                    if item.get(key) is not None
                }
                for item in list(finding.get("evaluated_values") or [])[:3]
            ]
            findings.append(_without_none_values(compact))
        if findings:
            summary["priority_findings"] = findings
        watches = []
        for item in list(candidate_evidence.get("top_watch_variables") or [])[:3]:
            watches.append({
                key: item.get(key)
                for key in (
                    "variable",
                    "role",
                    "metric",
                    "value",
                    "status",
                    "mean_prediction",
                    "mean_abs_delta_vs_observed",
                )
                if item.get(key) is not None
            })
        if watches:
            summary["top_watch_variables"] = watches
        return summary

    candidate_predictions = list(prediction.get("candidate_forecasts") or [])
    if not candidate_predictions:
        return project(prediction, verification, evidence, include_record_decision=True)

    contract = GroundingContractBuilder().build(
        str(record.get("user_input") or ""),
        attach_tool_arguments(
            record.get("tool_outputs") or [],
            record.get("tool_calls") or [],
        ),
        decision_policy=dict(record.get("decision_policy") or {}) or None,
    )
    if contract.get("answer_mode") == "dispatch_comparison":
        return _without_none_values({
            "candidate_results": contract.get("candidate_results") or [],
            "decision_summary": contract.get("decision_summary") or {},
            "comparison_leaders": contract.get("comparison_leaders") or {},
            "worst_case_risk_level": contract.get("worst_case_risk_level"),
            "worst_case_intervention_label": contract.get(
                "worst_case_intervention_label"
            ),
        })

    candidate_checks = list(verification.get("candidate_forecasts") or [])
    candidate_evidence = list(evidence.get("candidate_forecasts") or [])

    def matching(items: List[Dict[str, Any]], candidate: Dict[str, Any]) -> Dict[str, Any]:
        for item in items:
            if any(
                candidate.get(key) is not None and candidate.get(key) == item.get(key)
                for key in ("tool_call_id", "candidate_id")
            ):
                return item
        return {}

    return _without_none_values({
        "candidate_forecasts": [
            project(
                candidate,
                matching(candidate_checks, candidate),
                matching(candidate_evidence, candidate),
                include_record_decision=False,
            )
            for candidate in candidate_predictions
        ],
        "worst_case_risk_level": record.get("risk_level"),
        "worst_case_intervention_label": record.get("manual_intervention_label"),
    })


def _history_turn(record: Dict[str, Any]) -> Dict[str, Any]:
    requested = requested_artifacts(str(record.get("user_input") or ""))
    outputs = attach_tool_arguments(
        record.get("tool_outputs") or [], record.get("tool_calls") or []
    )
    assessments = [
        classify_tool_evidence(item, requested=requested)
        for item in outputs
    ]
    evidence_artifacts = sorted({
        artifact
        for assessment in assessments if assessment.evidence_found
        for artifact in assessment.matched_artifacts
    })
    evidence_found = any(assessment.evidence_found for assessment in assessments)
    record_evidence = dict(record.get("evidence") or {})
    verified_evidence_summary = {
        key: record_evidence[key]
        for key in ("csv_evidence", "topology_summary")
        if evidence_found and record_evidence.get(key)
    }
    registry_variables: List[Dict[str, Any]] = []
    for item in outputs:
        if str(item.get("name") or "").casefold() != "search_pipeformer_registry":
            continue
        output = dict(item.get("output") or {})
        if output.get("success") is not True or output.get("error"):
            continue
        arguments = dict(item.get("arguments") or {})
        for variable in output.get("variables") or []:
            if not isinstance(variable, dict) or not variable.get("variable"):
                continue
            registry_variables.append(
                {
                    **dict(variable),
                    "provenance": {
                        "tool_call_id": item.get("tool_call_id"),
                        "query": arguments.get("query"),
                        "role": arguments.get("role"),
                        "controllable": arguments.get("controllable"),
                    },
                    "execution_authorization": False,
                }
            )
    if registry_variables:
        verified_evidence_summary["registry_variables"] = registry_variables
    pipeformer_evidence_found = any(
        assessment.evidence_found
        and str(item.get("name") or "").casefold() == "run_pipeformer_forecast"
        for item, assessment in zip(outputs, assessments)
    )
    forecast_outputs = [
        dict(item.get("output") or {})
        for item in outputs
        if str(item.get("name") or "").casefold()
        == "run_pipeformer_forecast"
    ]
    verified_forecast_outputs = bool(forecast_outputs) and all(
        output.get("success") is True
        and dict(
            output.get("verification")
            or output.get("constraint_check")
            or {}
        ).get("verification_complete")
        is True
        and bool(
            dict(output.get("evidence") or {}).get(
                "boundary_application_evidence"
            )
        )
        and all(
            isinstance(application, dict)
            and application.get("verified") is True
            for application in dict(output.get("evidence") or {}).get(
                "boundary_application_evidence"
            )
            or []
        )
        for output in forecast_outputs
    )
    if pipeformer_evidence_found and verified_forecast_outputs:
        pipeformer_summary = _pipeformer_history_summary(record)
        if pipeformer_summary:
            verified_evidence_summary["pipeformer"] = pipeformer_summary
        turn_state = DecisionTraceState().updated_from_tool_results(
            str(record.get("session_id") or ""),
            int(record.get("turn_id") or 0),
            str(record.get("user_input") or ""),
            outputs,
        )
        snapshot = dict(
            turn_state.verified_evidence.get("single_forecast_snapshot") or {}
        )
        if snapshot:
            verified_evidence_summary["single_forecast_snapshot"] = snapshot
    policy_outputs = [
        dict(item.get("output") or {})
        for item, assessment in zip(outputs, assessments)
        if (
            assessment.evidence_found
            and str(item.get("name") or "").casefold()
            == "set_decision_policy"
            and dict(item.get("output") or {}).get("success") is True
        )
    ]
    if policy_outputs and not verified_evidence_summary.get("pipeformer"):
        policy = dict(policy_outputs[-1].get("decision_policy") or {})
        if policy.get("source") == "llm_tool":
            verified_evidence_summary["pipeformer"] = {
                "decision_policy": policy,
            }
    compact_calls = [
        {
            "tool_call_id": item.get("tool_call_id"),
            "name": item.get("name"),
            "arguments": DEFAULT_PROJECTOR._compact_call_arguments(
                item.get("arguments") or {},
                pipeformer=item.get("name") == "run_pipeformer_forecast",
            ),
        }
        for item in record.get("tool_calls") or []
        if str(item.get("tool_call_id") or "") in {
            str(output.get("tool_call_id") or "")
            for output, assessment in zip(outputs, assessments)
            if assessment.evidence_found
        }
    ]
    return _without_none_values(
        {
            "session_id": record["session_id"],
            "turn_id": record["turn_id"],
            "user_input": record["user_input"],
            "assistant_output": record.get("final_answer"),
            "quality_flag": record.get("quality_flag"),
            "verified_state_eligible": bool(verified_evidence_summary),
            "grounding_verified": (
                record.get("quality_flag") == "pass"
                and bool(verified_evidence_summary)
            ),
            "tool_evidence_verified": (
                bool(verified_evidence_summary)
            ),
            "evidence_artifacts": evidence_artifacts,
            "verified_evidence_summary": verified_evidence_summary or None,
            "parsed_task": compact_parsed_task(dict(record.get("parsed_task") or {})),
            "tool_calls": compact_calls or None,
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
    scenario_history: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from agent.orchestrator import AgentOrchestrator
    from agent.schemas import AgentChatRequest

    scenario_id = str(scenario.get("scenario_id") or f"scenario_{scenario_index:06d}")
    dataset_source = str(scenario.get("dataset_source") or "unknown_source")
    source_session_id = str(source_session.get("session_id") or f"{scenario_id}_session_{session_index:03d}")
    runtime_session_id = args.session_id or f"teacher_{scenario_index:04d}_{session_index:03d}_{run_stamp}"
    os.environ["PIPEFORMER_DEVICE"] = str(args.device)

    logger.info("Session started: scenario=%s source_session=%s runtime_session=%s", scenario_id, source_session_id, runtime_session_id)

    run_hash = hashlib.sha256(f"{dataset_source}:{scenario_id}:{run_stamp}".encode("utf-8")).hexdigest()[:10]
    run_namespace = f"r{scenario_index:04d}_{run_hash}"
    orchestrator = AgentOrchestrator(
        data_loader=None,
        agent_id=args.agent_id,
        session_id=runtime_session_id,
        enable_skills=False,
        workspace_root_base=backend_root() / ".openclaw" / "tt_runs" / run_namespace,
    )
    orchestrator.verified_state_manager.commit(
        runtime_session_id,
        DecisionTraceState.from_history(scenario_history),
    )
    records: List[Dict[str, Any]] = []
    history = scenario_history
    errors: List[Dict[str, Any]] = []
    raw_trace_paths: Dict[int, str] = {}
    message_count = 0
    tool_count = 0
    for fallback_turn_id, turn in enumerate(source_session.get("dialogue") or [], start=1):
        question = str(turn.get("user_input") or "").strip()
        if not question:
            continue
        turn_id = int(turn.get("turn_id") or fallback_turn_id)
        logger.info("Turn started: scenario=%s session=%s turn=%d question=%s", scenario_id, source_session_id, turn_id, short_text(question, 300))
        try:
            def validate_answer(answer: str, completed_calls: List[Dict[str, Any]]) -> List[str]:
                history_state = DecisionTraceState.from_history(history)
                outputs = [
                    item["output"]
                    for item in completed_calls
                    if item.get("name") == "run_pipeformer_forecast"
                    and isinstance(item.get("output"), dict)
                    and item["output"].get("success")
                ]
                issues = answer_quality_issues(
                    answer,
                    question,
                    outputs[0] if len(outputs) == 1 else None,
                    conversation_context=history,
                    tool_outputs=completed_calls,
                    record_evidence={
                        "topology_summary": topology_summary_from_tool_outputs(completed_calls)
                    },
                )
                contract = GroundingContractBuilder().build(
                    question,
                    completed_calls,
                    require_decision_policy=True,
                    prior_candidate_results=history_state.candidate_results,
                    prior_decision_policy=history_state.decision_policy,
                    prior_decision_policy_source_question=(
                        history_state.decision_policy_source_question
                    ),
                    prior_applied_disturbances=history_state.applied_disturbances,
                )
                issues.extend(comparison_answer_issues(answer, contract))
                return list(dict.fromkeys(issues))

            result = orchestrator.run_agent(
                AgentChatRequest(
                    agent_id=args.agent_id,
                    session_id=runtime_session_id,
                    message=agent_turn_message(scenario, question, history),
                ),
                answer_validator=validate_answer,
            )
            trace_path = Path(result.trace_summary.trace_path)
            raw_trace_paths[turn_id] = str(trace_path.resolve())
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
                "state_before": record.get("state_before"),
                "recent_turns": record.get("recent_turns"),
                "tool_calls": record["tool_calls"],
                "tool_outputs": record["tool_outputs"],
                "quality_flag": record["quality_flag"],
                "quality_score": record["quality_score"],
                "quality_profile": record["quality_profile"],
                "quality_failed_checks": record["quality_failed_checks"],
                "quality_issues": record["quality_issues"],
                "raw_trace_path": raw_trace_paths.get(record["turn_id"]),
            }
            for record in records
        ],
        "complete": not errors and len(records) == len(source_session.get("dialogue") or []),
        "errors": errors,
    }
    return records, session_record


def write_split_records(
    output_dir: Path,
    records: List[Dict[str, Any]],
    force: bool,
) -> int:
    eligible_records = [
        item
        for item in records
        if item.get("quality_flag") == "pass"
        and not item.get("sft_exclusion_reason")
        # A turn can be valid for audit while still being unsafe for SFT when
        # its prompt contains an unresolved earlier assistant answer.
        and all(
            turn.get("quality_flag") == "pass"
            for turn in item.get("conversation_context") or []
        )
    ]
    sft_fields = (
        "sample_id",
        "scenario_id",
        "session_id",
        "turn_id",
        "scenario_type",
        "state_before",
        "recent_turns",
        "user_input",
        "parsed_task",
        "tool_calls",
        "tool_outputs",
        "evidence",
        "decision_summary",
        "final_answer",
    )
    written_count = 0
    for split in ("train", "valid", "test"):
        split_records = []
        for item in eligible_records:
            if item.get("split") != split:
                continue
            projected = {key: item[key] for key in sft_fields if key in item}
            if "state_before" not in projected:
                projected["state_before"] = serialize_verified_decision_state(
                    DecisionTraceState.from_history(
                        item.get("conversation_context") or []
                    ),
                    max_chars=int(
                        os.getenv("VERIFIED_STATE_MAX_CHARS", "16000")
                    ),
                )
            if "recent_turns" not in projected:
                projected["recent_turns"] = bounded_recent_turns(
                    item.get("conversation_context") or [],
                    max_turns=2,
                    max_chars=int(
                        os.getenv("RECENT_TURNS_MAX_CHARS", "4000")
                    ),
                )
            full_grounding_evidence = numeric_grounding_evidence(item)
            projected["tool_calls"], projected["tool_outputs"] = DEFAULT_PROJECTOR.compact_sft_trajectory(
                list(projected.get("tool_calls") or []),
                list(projected.get("tool_outputs") or []),
                str(projected.get("final_answer") or ""),
            )
            projected_evidence = DEFAULT_PROJECTOR.compact_record_evidence(
                dict(projected.get("evidence") or {})
            )
            projected["decision_summary"] = (
                DEFAULT_PROJECTOR.compact_sft_decision_summary(
                    projected.get("decision_summary")
                )
            )
            answer_text = str(projected.get("final_answer") or "")
            supporting_values = list(dict.fromkeys(
                grounded_numeric_claim_values(
                    answer_text,
                    str(projected.get("user_input") or ""),
                    full_grounding_evidence,
                )
            ))
            if supporting_values:
                projected_evidence["supporting_numeric_values"] = supporting_values
            projected["evidence"] = projected_evidence
            compact_evidence = numeric_grounding_evidence(projected)
            if not numeric_claims_are_grounded(
                str(projected.get("final_answer") or ""),
                str(projected.get("user_input") or ""),
                compact_evidence,
            ):
                logger.warning("Skipping SFT record with evidence removed by compaction: sample=%s", item.get("sample_id"))
                continue
            size = len(json.dumps(projected, ensure_ascii=False))
            # The normal projection retains up to three referenced forecast
            # variables.  For a rare near-limit record, retry only that
            # bounded projection with fewer variables and accept it only when
            # every numeric and canonical variable claim remains grounded.
            for max_variables in range(SFT_MAX_PIPEFORMER_VARIABLES - 1, -1, -1):
                if size <= SFT_MAX_RECORD_CHARS:
                    break
                trial = dict(projected)
                trial["tool_calls"], trial["tool_outputs"] = (
                    DEFAULT_PROJECTOR.compact_sft_trajectory(
                        list(item.get("tool_calls") or []),
                        list(item.get("tool_outputs") or []),
                        answer_text,
                        max_pipeformer_variables=max_variables,
                    )
                )
                trial_evidence = numeric_grounding_evidence(trial)
                if not numeric_claims_are_grounded(
                    answer_text,
                    str(trial.get("user_input") or ""),
                    trial_evidence,
                ):
                    continue
                claimed_variables = {
                    variable.casefold()
                    for variable in SFT_VARIABLE_REFERENCE.findall(answer_text)
                }
                supported_variables = {
                    variable.casefold()
                    for variable in SFT_VARIABLE_REFERENCE.findall(
                        json.dumps(trial_evidence, ensure_ascii=False)
                    )
                }
                if not claimed_variables <= supported_variables:
                    continue
                trial_size = len(json.dumps(trial, ensure_ascii=False))
                if trial_size < size:
                    projected, size = trial, trial_size
            if size > SFT_MAX_RECORD_CHARS:
                logger.warning("Skipping oversized SFT record: sample=%s chars=%d", item.get("sample_id"), size)
                continue
            split_records.append(projected)
        write_jsonl(
            output_dir / f"teacher_trace_{split}.jsonl",
            split_records,
            force=force,
        )
        written_count += len(split_records)
    return written_count


class TeacherTraceGenerator:
    """Coordinate scenario selection, agent execution, merging, and export."""

    def __init__(self, args: argparse.Namespace, store: Optional[TeacherTraceStore] = None) -> None:
        self.args = args
        self.store = store or TeacherTraceStore.from_args(args)

    def run(self) -> int:
        return run_teacher_trace_generation(self.args, self.store)


def run_teacher_trace_generation(args: argparse.Namespace, store: TeacherTraceStore) -> int:
    configure_logging(args.log_level)
    replacement_mode = bool(getattr(args, "replace_selected_scenario", False))
    if replacement_mode:
        if args.force:
            raise ValueError("--replace-selected-scenario cannot be combined with --force.")
        if not args.dataset_source or not args.scenario_id:
            raise ValueError(
                "--replace-selected-scenario requires both --dataset-source and --scenario-id."
            )
        if args.session_id:
            raise ValueError("Scenario replacement must regenerate every session; omit --session-id.")

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

    existing_records: List[Dict[str, Any]] = []
    existing_session_records: List[Dict[str, Any]] = []
    if not args.force:
        existing_records = store.load_master()
        existing_session_records = store.load_sessions()

    if replacement_mode:
        missing_targets = []
        target = {
            "dataset_source": str(args.dataset_source),
            "scenario_id": str(args.scenario_id),
        }
        if not store.contains_scenario(existing_records, **target):
            missing_targets.append("master teacher trace")
        if not store.contains_scenario(existing_session_records, **target):
            missing_targets.append("session teacher trace")
        if missing_targets:
            raise ValueError(
                "Cannot replace the selected scenario because it is absent from "
                + " and ".join(missing_targets)
                + "; no LLM calls were made."
            )
    preflight_sources, missing_preflight_pairs = combined_preflight_sources(
        all_sources,
        selected_sources,
        existing_records,
    )
    preflight = validate_scenario_sources(preflight_sources or selected_sources, args.mapping_csv)
    selected_preflight = validate_scenario_sources(selected_sources, args.mapping_csv)
    preflight["selected_data_files"] = {
        "required_data_file_count": selected_preflight["required_data_file_count"],
        "required_data_files": selected_preflight["required_data_files"],
        "missing_data_file_count": selected_preflight["missing_data_file_count"],
        "missing_data_files": selected_preflight["missing_data_files"],
    }
    # Data availability blocks only the scenarios selected for this run. The
    # broader report still audits existing sources for collisions and coverage.
    preflight["supported"] = (
        not preflight["unsupported_variables"]
        and preflight["variable_registry"]["supported"]
        and not selected_preflight["missing_data_files"]
        and not selected_preflight.get("missing_target_mappings")
    )
    preflight["append_mode"] = not args.force and not replacement_mode
    preflight["replacement_mode"] = replacement_mode
    preflight["existing_record_count"] = len(existing_records)
    preflight["unavailable_existing_scenarios"] = missing_preflight_pairs
    write_json(args.preflight_output, preflight, force=True)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    if not preflight["supported"]:
        problems = []
        if preflight["unsupported_variables"]:
            problems.append(
                "variables absent from mapping: "
                + ", ".join(preflight["unsupported_variables"])
            )
        if selected_preflight["missing_data_files"]:
            problems.append(
                "pipeline data files not found: "
                + ", ".join(selected_preflight["missing_data_files"])
            )
        if selected_preflight.get("missing_target_mappings"):
            problems.append(
                "consumer supply points missing canonical topology mappings: "
                + ", ".join(selected_preflight["missing_target_mappings"])
            )
        problems.extend(preflight["variable_registry"]["errors"])
        raise ValueError("Scenario preflight failed; " + "; ".join(problems))

    logger.info(
        "Teacher trace generation started: scenarios=%d device=%s scenario_files=%s",
        len(scenarios),
        args.device,
        [str(path) for path in scenario_files],
    )
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    records: List[Dict[str, Any]] = []
    session_records: List[Dict[str, Any]] = []
    split_map = scenario_split_map(all_scenarios, args.split_seed)
    existing_sample_ids = {
        str(record.get("sample_id"))
        for record in existing_records
        if record.get("sample_id")
    }
    existing_session_ids = {
        str(record.get("session_record_id"))
        for record in existing_session_records
        if record.get("session_record_id")
    }
    skipped_scenarios = 0
    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = str(scenario.get("scenario_id") or f"scenario_{index:06d}")
        expected_sample_ids = store.sample_ids(scenario)
        expected_session_ids = store.session_ids(scenario)
        if (
            not replacement_mode
            and expected_sample_ids
            and set(expected_sample_ids) <= existing_sample_ids
            and set(expected_session_ids) <= existing_session_ids
        ):
            skipped_scenarios += 1
            logger.info(
                "Skipping existing scenario without LLM calls: dataset=%s scenario=%s records=%d",
                scenario.get("dataset_source"),
                scenario_id,
                len(expected_sample_ids),
            )
            continue
        split = split_map[scenario_id]
        scenario_history: List[Dict[str, Any]] = []
        for session_index, source_session in enumerate(scenario.get("sessions") or [], start=1):
            turn_records, session_record = run_backend_session(
                scenario,
                source_session,
                args,
                index,
                session_index,
                run_stamp,
                split,
                scenario_history,
            )
            records.extend(turn_records)
            session_records.append(session_record)

    failed_sessions = sum(not bool(item.get("complete")) for item in session_records)
    removed_record_count = 0
    removed_session_count = 0
    if replacement_mode:
        if failed_sessions:
            raise RuntimeError(
                "Scenario replacement was not written because at least one regenerated session failed."
            )
        selected_scenario = scenarios[0]
        expected_sample_ids = set(store.sample_ids(selected_scenario))
        expected_session_ids = set(store.session_ids(selected_scenario))
        generated_sample_ids = {str(item.get("sample_id") or "") for item in records}
        generated_session_ids = {
            str(item.get("session_record_id") or "") for item in session_records
        }
        if generated_sample_ids != expected_sample_ids:
            raise ValueError(
                "Scenario replacement is incomplete: generated sample ids do not match the source definition."
            )
        if generated_session_ids != expected_session_ids:
            raise ValueError(
                "Scenario replacement is incomplete: generated session ids do not match the source definition."
            )
        combined_records, removed_record_count = store.replace_scenario(
            existing_records,
            records,
            dataset_source=str(args.dataset_source),
            scenario_id=str(args.scenario_id),
            id_field="sample_id",
        )
        combined_session_records, removed_session_count = store.replace_scenario(
            existing_session_records,
            session_records,
            dataset_source=str(args.dataset_source),
            scenario_id=str(args.scenario_id),
            id_field="session_record_id",
        )
        duplicate_record_count = 0
        updated_session_count = 0
        appended_record_count = 0
    else:
        combined_records, duplicate_record_count = store.merge_records(
            existing_records,
            records,
            id_field="sample_id",
        )
        combined_session_records, updated_session_count = store.merge_sessions(
            existing_session_records,
            session_records,
        )
        appended_record_count = len(combined_records) - len(existing_records)
    store.validate_splits(combined_records)

    logger.info("Writing combined JSONL output: %s", args.output_jsonl)
    logger.info("Writing pretty JSON output: %s", args.output_json)
    store.write_master(combined_records)
    logger.info("Writing session evaluation JSONL: %s", args.session_output_jsonl)
    store.write_sessions(combined_session_records)
    logger.info("Writing scenario-isolated split files: %s", args.split_output_dir)
    sft_record_count = write_split_records(
        args.split_output_dir,
        combined_records,
        force=True,
    )
    total_failed_sessions = sum(not bool(item.get("complete")) for item in combined_session_records)
    quality_pass_records = sum(item.get("quality_flag") == "pass" for item in combined_records)
    if failed_sessions:
        run_status = "completed_with_errors"
    elif not replacement_mode and not appended_record_count and existing_records:
        run_status = "no_changes"
    elif sft_record_count < len(combined_records):
        run_status = "completed_with_quality_issues"
    else:
        run_status = "ok"
    logger.info(
        "Teacher trace generation complete: generated=%d appended=%d total=%d quality_pass_records=%d "
        "sft_records=%d failed_sessions=%d",
        len(records),
        appended_record_count,
        len(combined_records),
        quality_pass_records,
        sft_record_count,
        failed_sessions,
    )
    print(
        json.dumps(
            {
                "status": run_status,
                "append_mode": not args.force and not replacement_mode,
                "replacement_mode": replacement_mode,
                "replaced_dataset_source": args.dataset_source if replacement_mode else None,
                "replaced_scenario_id": args.scenario_id if replacement_mode else None,
                "removed_records": removed_record_count,
                "removed_sessions": removed_session_count,
                "records": len(combined_records),
                "generated_records": len(records),
                "appended_records": appended_record_count,
                "duplicate_records_skipped": duplicate_record_count,
                "skipped_existing_scenarios": skipped_scenarios,
                "quality_pass_records": quality_pass_records,
                "sft_records": sft_record_count,
                "sessions": len(combined_session_records),
                "failed_sessions": failed_sessions,
                "total_failed_sessions": total_failed_sessions,
                "updated_existing_sessions": updated_session_count,
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


def main() -> int:
    load_backend_env()
    args = build_parser().parse_args()
    return TeacherTraceGenerator(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
