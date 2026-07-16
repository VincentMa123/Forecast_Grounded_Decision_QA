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

from evaluator.teacher_quality import (
    answer_quality_issues,
    evaluate_teacher_quality,
    llm_answer_quality_issues,
    numeric_claim_values,
    numeric_claims_are_grounded,
    safety_and_energy_checks_pass as _safety_and_energy_checks_pass,
    tool_output_failed,
)
from evaluator.scorer import NativeTraceEvaluator
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

SFT_MAX_RECORD_CHARS = 24_000
SFT_MAX_TOOL_TEXT_CHARS = 4_000
SFT_MAX_GENERIC_TOOL_PAIRS = 6
SFT_MAX_GENERIC_OUTPUT_CHARS = 2_500
SFT_MAX_PIPEFORMER_VARIABLES = 3
SFT_VARIABLE_REFERENCE = re.compile(r"\b[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?\b")
SFT_FILE_REFERENCE = re.compile(r"(?i)\b[\w.-]+\.(?:csv|jsonl?|xlsx?|parquet)\b")
SFT_OMITTED_TOOL_KEYS = {
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
            projection = project_pipeformer_output(output)
            pipeformer_results.append({"output": output, "projection": projection})
            output = compact_pipeformer_output(projection)
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
            if pair[2] is not None and not tool_output_failed(pair[2].get("output"))
        ]
        successful_pipeformer = [
            pair for pair in successful if pair[1].get("name") == "run_pipeformer_forecast"
        ]
        multiple_pipeformer = len(successful_pipeformer) > 1
        if successful_pipeformer:
            selected = successful_pipeformer
        elif successful:
            selected = self._select_generic_evidence_pairs(successful, answer)
        else:
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
            if call.get("name") == "run_pipeformer_forecast" and isinstance(raw_output, dict):
                raw_output = self._compact_pipeformer_sft_output(
                    raw_output,
                    answer,
                    include_auxiliary_variables=not multiple_pipeformer,
                )
            else:
                raw_output = self._compact_generic_sft_output(raw_output, answer)
            compact_outputs.append(
                {
                    "tool_call_id": output.get("tool_call_id"),
                    "name": output.get("name"),
                    "output": raw_output,
                }
            )
        return compact_calls, compact_outputs

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

    def _compact_generic_sft_output(self, value: Any, answer: str) -> Any:
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
        text_value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
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

    def _compact_call_arguments(self, value: Any, *, pipeformer: bool) -> Any:
        if pipeformer and isinstance(value, dict):
            value = {key: item for key, item in value.items() if key != "question"}
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
        referenced = list(dict.fromkeys(referenced))[:SFT_MAX_PIPEFORMER_VARIABLES]
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
            "forecast_window",
            "counterfactual_comparison",
            "total_output_variable_count",
        )
        compact_prediction = {
            key: prediction[key] for key in prediction_keys if key in prediction
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
    pipeformer_outputs = [item["output"] for item in pipeformer_results]
    projections = [item["projection"] for item in pipeformer_results]
    pipeformer_call_count = sum(
        item.get("tool_name") == "run_pipeformer_forecast"
        for item in trace.get("tool_calls", [])
    )
    pipeformer = pipeformer_outputs[0] if len(pipeformer_outputs) == 1 else None
    projection = projections[0] if len(projections) == 1 else None
    answer = final_answer(trace).strip()
    answer_quality_flag, quality_issues = evaluate_teacher_quality(
        answer=answer,
        question=question,
        pipeformer=pipeformer,
        trace_status=trace.get("status"),
        pipeformer_call_count=pipeformer_call_count,
        pipeformer_outputs=pipeformer_outputs,
        conversation_context=conversation_context,
        tool_outputs=tool_outputs,
    )
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
        parsed_task = {
            "candidate_forecasts": [item["parsed_task"] for item in projections]
        }
        prediction_summary = {
            "candidate_forecasts": [item["prediction_summary"] for item in projections]
        }
        constraint_check = {
            "candidate_forecasts": [item["constraint_check"] for item in projections]
        }
        evidence = {
            "candidate_forecasts": [item["evidence"] for item in projections]
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
        "conversation_context": projector.sanitize(conversation_context or []),
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


def _history_turn(record: Dict[str, Any]) -> Dict[str, Any]:
    return _without_none_values(
        {
            "session_id": record["session_id"],
            "turn_id": record["turn_id"],
            "user_input": record["user_input"],
            "assistant_output": record.get("final_answer"),
            "quality_flag": record.get("quality_flag"),
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
                outputs = [
                    item["output"]
                    for item in completed_calls
                    if item.get("name") == "run_pipeformer_forecast"
                    and isinstance(item.get("output"), dict)
                    and item["output"].get("success")
                ]
                return answer_quality_issues(
                    answer,
                    question,
                    outputs[0] if len(outputs) == 1 else None,
                    conversation_context=history,
                    tool_outputs=completed_calls,
                )

            result = orchestrator.run_agent(
                AgentChatRequest(agent_id=args.agent_id, session_id=runtime_session_id, message=question),
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
        "conversation_context",
        "user_input",
        "parsed_task",
        "tool_calls",
        "tool_outputs",
        "evidence",
        "final_answer",
    )
    written_count = 0
    for split in ("train", "valid", "test"):
        split_records = []
        for item in eligible_records:
            if item.get("split") != split:
                continue
            projected = {key: item[key] for key in sft_fields if key in item}
            full_grounding_evidence = {
                "conversation_context": projected.get("conversation_context", []),
                "tool_outputs": projected.get("tool_outputs", []),
                "evidence": projected.get("evidence", {}),
            }
            projected["tool_calls"], projected["tool_outputs"] = DEFAULT_PROJECTOR.compact_sft_trajectory(
                list(projected.get("tool_calls") or []),
                list(projected.get("tool_outputs") or []),
                str(projected.get("final_answer") or ""),
            )
            projected_evidence = DEFAULT_PROJECTOR.compact_record_evidence(
                dict(projected.get("evidence") or {})
            )
            supporting_values = []
            answer_text = str(projected.get("final_answer") or "")
            for value in numeric_claim_values(answer_text):
                token = str(value)
                if numeric_claims_are_grounded(
                    token,
                    str(projected.get("user_input") or ""),
                    full_grounding_evidence,
                ):
                    if value not in supporting_values:
                        supporting_values.append(value)
            if supporting_values:
                projected_evidence["supporting_numeric_values"] = supporting_values
            projected["evidence"] = projected_evidence
            compact_evidence = {
                "conversation_context": projected.get("conversation_context", []),
                "tool_outputs": projected["tool_outputs"],
                "evidence": projected.get("evidence", {}),
            }
            if not numeric_claims_are_grounded(
                str(projected.get("final_answer") or ""),
                str(projected.get("user_input") or ""),
                compact_evidence,
            ):
                logger.warning("Skipping SFT record with evidence removed by compaction: sample=%s", item.get("sample_id"))
                continue
            size = len(json.dumps(projected, ensure_ascii=False))
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

    preflight_sources, missing_preflight_pairs = combined_preflight_sources(
        all_sources,
        selected_sources,
        existing_records,
    )
    preflight = validate_scenario_sources(preflight_sources or selected_sources, args.mapping_csv)
    preflight["append_mode"] = not args.force
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
            expected_sample_ids
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

    combined_records, duplicate_record_count = store.merge_records(
        existing_records,
        records,
        id_field="sample_id",
    )
    combined_session_records, updated_session_count = store.merge_sessions(
        existing_session_records,
        session_records,
    )
    store.validate_splits(combined_records)
    appended_record_count = len(combined_records) - len(existing_records)

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
    failed_sessions = sum(not bool(item.get("complete")) for item in session_records)
    total_failed_sessions = sum(not bool(item.get("complete")) for item in combined_session_records)
    quality_pass_records = sum(item.get("quality_flag") == "pass" for item in combined_records)
    if failed_sessions:
        run_status = "completed_with_errors"
    elif not appended_record_count and existing_records:
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
                "append_mode": not args.force,
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
