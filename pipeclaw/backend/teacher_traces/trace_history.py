from __future__ import annotations

import json
import ntpath
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Mapping, Tuple

from pipeclaw.backend.grounding.decision_trace_state import VerifiedDecisionState
from pipeclaw.backend.grounding.evidence.tool import (
    ToolEvidenceAssessment,
    attach_tool_arguments,
    classify_tool_evidence,
    requested_artifacts,
)
from pipeclaw.backend.pipeline.forecast.result import (
    compact_parsed_task,
    without_none_values,
)


OMITTED_CALL_ARGUMENT_KEYS = frozenset({"cwd"})
MAX_HISTORY_SUMMARY_CHARS = 1_900

# Marker appended to every value the SFT projection truncates.  The marker is
# persisted, so all compaction paths must emit byte-identical text.
SFT_TRUNCATION_MARKER = "... [truncated for SFT]"


def _host_absolute_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    raw = value.strip()
    normalized = raw.replace("\\", "/")
    return bool(
        Path(normalized).is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or ntpath.splitdrive(raw)[0]
    )


def compact_tool_call_arguments(value: Any) -> Any:
    """Keep actionable prior-call arguments without host-specific paths."""
    if isinstance(value, Mapping):
        compacted: Dict[str, Any] = {}
        for key, item in value.items():
            if key in OMITTED_CALL_ARGUMENT_KEYS:
                if item is None or _host_absolute_path(item):
                    continue
                if isinstance(item, str):
                    item = item.replace("\\", "/")
            compacted[str(key)] = compact_tool_call_arguments(item)
        return compacted
    if isinstance(value, list):
        return [compact_tool_call_arguments(item) for item in value]
    if isinstance(value, str) and len(value) > 2_000:
        return value[:2_000] + SFT_TRUNCATION_MARKER
    return value


@dataclass(frozen=True)
class ToolEvidenceSummary:
    """One classification pass shared by history and offline evaluation."""

    outputs: List[Dict[str, Any]]
    assessments: Tuple[ToolEvidenceAssessment, ...]
    evidence_artifacts: Tuple[str, ...]

    @property
    def evidence_found(self) -> bool:
        return any(assessment.evidence_found for assessment in self.assessments)


def summarize_record_tool_evidence(record: Mapping[str, Any]) -> ToolEvidenceSummary:
    """Attach arguments and classify successful evidence exactly once."""
    question = str(record.get("user_input") or "")
    outputs = attach_tool_arguments(
        record.get("tool_outputs") or [],
        record.get("tool_calls") or [],
    )
    requested = requested_artifacts(question)
    assessments = tuple(
        classify_tool_evidence(item, requested=requested) for item in outputs
    )
    artifacts = tuple(
        sorted(
            {
                artifact
                for assessment in assessments
                if assessment.evidence_found
                for artifact in assessment.matched_artifacts
            }
        )
    )
    return ToolEvidenceSummary(outputs, assessments, artifacts)


def _comparison_state(state: VerifiedDecisionState) -> Dict[str, Any]:
    """Persist the state-owned candidate facts without a second projection."""
    if not state.candidates:
        return {}
    return without_none_values(
        {
            "scope": deepcopy(state.scope),
            "candidate_results": deepcopy(state.candidates),
            "decision_policy": deepcopy(state.decision_policy),
            "applied_disturbances": deepcopy(state.applied_disturbances),
        }
    )


def _policy_memory_summary(value: Any) -> Dict[str, Any]:
    policy = dict(value or {})
    return without_none_values(
        {
            "source": policy.get("source"),
            "hard_constraints": list(policy.get("hard_constraints") or [])[:2],
            "objectives": [
                {
                    key: objective[key]
                    for key in ("metric", "direction")
                    if objective.get(key) is not None
                }
                for objective in policy.get("objectives") or []
                if isinstance(objective, Mapping)
            ][:2],
        }
    )


def _comparison_memory_summary(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep prompt memory small; the complete candidate facts remain in state."""
    candidates = [
        {
            key: deepcopy(candidate[key])
            for key in (
                "candidate_id",
                "action",
                "failure_count",
                "warning_count",
                "risk_level",
                "energy_consumption",
            )
            if candidate.get(key) is not None
        }
        for candidate in value.get("candidate_results") or []
        if isinstance(candidate, Mapping)
    ]
    return without_none_values(
        {
            "candidate_results": candidates[:3],
            "decision_policy": _policy_memory_summary(value.get("decision_policy")),
            "applied_disturbances": list(value.get("applied_disturbances") or [])[:2],
        }
    )


def _compact_csv_value(value: Any, *, depth: int = 0) -> Any:
    """Retain a small, readable CSV fact without carrying result tables."""
    if isinstance(value, str):
        return value if len(value) <= 320 else value[:320] + "... [truncated]"
    if depth >= 2:
        return deepcopy(value)
    if isinstance(value, Mapping):
        return {
            str(key): _compact_csv_value(item, depth=depth + 1)
            for key, item in list(value.items())[:6]
        }
    if isinstance(value, list):
        return [_compact_csv_value(item, depth=depth + 1) for item in value[:2]]
    return deepcopy(value)


def _compact_csv_result(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return _compact_csv_value(value)
    source = dict(value)
    result = {
        key: _compact_csv_value(source[key])
        for key in (
            "tool_call_id",
            "source_file",
            "operation",
            "variable",
            "status",
            "value",
        )
        if source.get(key) is not None
    }
    return result or _compact_csv_value(source)


def _minimal_csv_evidence(value: Any) -> Dict[str, Any]:
    source = dict(value or {})
    return without_none_values(
        {
            "source_file_count": source.get("source_file_count"),
            "source_files": list(source.get("source_files") or [])[:2],
            **{
                key: [_compact_csv_result(source[key][0])]
                for key in ("computed_results", "derived_results")
                if source.get(key)
            },
        }
    )


def _compact_csv_evidence(value: Any) -> Dict[str, Any]:
    source = dict(value or {})
    compact = {
        "source_file_count": source.get("source_file_count"),
    }
    if source.get("selection_summary") is not None:
        compact["selection_summary"] = _compact_csv_value(source["selection_summary"])
    if source.get("source_files"):
        compact["source_files"] = list(source["source_files"])[:3]
    for key, limit in (("computed_results", 1), ("derived_results", 2)):
        if source.get(key):
            compact[key] = [
                _compact_csv_result(item) for item in list(source[key])[:limit]
            ]
    rows = []
    for item in list(source.get("answer_rows") or [])[:3]:
        if not isinstance(item, Mapping):
            continue
        values = dict(item.get("values") or {})
        rows.append(
            {
                "source_file": item.get("source_file"),
                "values": {
                    key: _compact_csv_value(values[key]) for key in list(values)[:6]
                },
            }
        )
    if rows:
        compact["answer_rows"] = rows
    return without_none_values(compact)


def _bounded_verified_evidence_summary(
    value: Dict[str, Any],
) -> Dict[str, Any]:
    """Keep one history-memory entry eligible for PromptBuilder rendering."""
    if (
        len(json.dumps(value, ensure_ascii=False, default=str))
        <= MAX_HISTORY_SUMMARY_CHARS
    ):
        return value
    primary = (
        "pipeformer"
        if value.get("pipeformer")
        else "single_forecast_snapshot"
        if value.get("single_forecast_snapshot")
        else None
    )
    if primary:
        return {primary: value[primary]}
    for key in ("topology_summary", "csv_evidence", "registry_summary"):
        if value.get(key):
            primary_value = value[key]
            if key == "csv_evidence":
                primary_value = _minimal_csv_evidence(primary_value)
            return {key: primary_value}
    return {}


def _history_parsed_task(record: Mapping[str, Any]) -> Dict[str, Any]:
    parsed = dict(record.get("parsed_task") or {})
    if {
        "resolved_attention_variables",
        "resolved_output_variables",
    } & set(parsed):
        return compact_parsed_task({"parsed_task": parsed})
    return deepcopy(parsed)


def build_history_turn(
    record: Mapping[str, Any],
    *,
    evidence_summary: ToolEvidenceSummary | None = None,
) -> Dict[str, Any]:
    """Project one completed record into verified cross-turn history."""
    summary = evidence_summary or summarize_record_tool_evidence(record)
    outputs = summary.outputs
    assessments = summary.assessments
    record_evidence = dict(record.get("evidence") or {})
    verified_evidence_summary = {}
    if summary.evidence_found and record_evidence.get("csv_evidence"):
        verified_evidence_summary["csv_evidence"] = _compact_csv_evidence(
            record_evidence["csv_evidence"]
        )
    if summary.evidence_found and record_evidence.get("topology_summary"):
        verified_evidence_summary["topology_summary"] = record_evidence[
            "topology_summary"
        ]
    registry_variables: List[Dict[str, Any]] = []
    for item in outputs:
        if str(item.get("name") or "").casefold() != "search_pipeformer_registry":
            continue
        output = dict(item.get("output") or {})
        if output.get("success") is not True or output.get("error"):
            continue
        for variable in output.get("variables") or []:
            if not isinstance(variable, dict) or not variable.get("variable"):
                continue
            registry_variables.append(
                {
                    key: variable[key]
                    for key in ("variable", "role", "controllable")
                    if variable.get(key) is not None
                }
            )
            registry_variables[-1]["provenance"] = {
                "tool_call_id": item.get("tool_call_id"),
            }
    if registry_variables:
        verified_evidence_summary["registry_summary"] = {
            "returned_variable_count": len(registry_variables),
            "returned_variable_ids": [
                item["variable"] for item in registry_variables[:12]
            ],
        }

    pipeformer_evidence_found = any(
        assessment.evidence_found
        and str(item.get("name") or "").casefold() == "run_pipeformer_forecast"
        for item, assessment in zip(outputs, assessments)
    )
    forecast_outputs = [
        dict(item.get("output") or {})
        for item in outputs
        if str(item.get("name") or "").casefold() == "run_pipeformer_forecast"
    ]
    verified_forecast_outputs = bool(forecast_outputs) and all(
        output.get("success") is True
        and dict(
            output.get("verification") or output.get("constraint_check") or {}
        ).get("verification_complete")
        is True
        and bool(
            dict(output.get("evidence") or {}).get("boundary_application_evidence")
        )
        and all(
            isinstance(application, dict) and application.get("verified") is True
            for application in dict(output.get("evidence") or {}).get(
                "boundary_application_evidence"
            )
            or []
        )
        for output in forecast_outputs
    )
    failed_forecast_request = (
        next(
            (
                {"arguments": compact_tool_call_arguments(item.get("arguments") or {})}
                for item in reversed(outputs)
                if str(item.get("name") or "").casefold() == "run_pipeformer_forecast"
            ),
            None,
        )
        if forecast_outputs
        and all(output.get("success") is not True for output in forecast_outputs)
        else None
    )
    comparison_state: Dict[str, Any] = {}
    if pipeformer_evidence_found and verified_forecast_outputs:
        turn_state = VerifiedDecisionState().updated_from_tool_results(
            str(record.get("session_id") or ""),
            int(record.get("turn_id") or 0),
            str(record.get("user_input") or ""),
            outputs,
        )
        comparison_state = _comparison_state(turn_state)
        if comparison_state:
            verified_evidence_summary["pipeformer"] = _comparison_memory_summary(
                comparison_state
            )
        else:
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
            and str(item.get("name") or "").casefold() == "set_decision_policy"
            and dict(item.get("output") or {}).get("success") is True
        )
    ]
    if policy_outputs and not verified_evidence_summary.get("pipeformer"):
        policy = dict(policy_outputs[-1].get("decision_policy") or {})
        if policy.get("source") == "llm_tool":
            verified_evidence_summary["pipeformer"] = {
                "decision_policy": _policy_memory_summary(policy)
            }
    evidence_call_ids = {
        str(output.get("tool_call_id") or "")
        for output, assessment in zip(outputs, assessments)
        if assessment.evidence_found
    }
    compact_calls = [
        {
            "tool_call_id": item.get("tool_call_id"),
            "name": item.get("name"),
            "arguments": compact_tool_call_arguments(item.get("arguments") or {}),
        }
        for item in record.get("tool_calls") or []
        if str(item.get("tool_call_id") or "") in evidence_call_ids
    ]
    verified_evidence_summary = _bounded_verified_evidence_summary(
        verified_evidence_summary
    )
    return without_none_values(
        {
            "session_id": record.get("session_id"),
            "turn_id": record.get("turn_id"),
            "user_input": record.get("user_input"),
            "assistant_output": record.get("final_answer"),
            "quality_flag": record.get("quality_flag"),
            "verified_state_eligible": bool(verified_evidence_summary),
            "grounding_verified": (
                record.get("quality_flag") == "pass" and bool(verified_evidence_summary)
            ),
            "tool_evidence_verified": bool(verified_evidence_summary),
            "evidence_artifacts": list(summary.evidence_artifacts),
            "verified_evidence_summary": verified_evidence_summary or None,
            "comparison_state": comparison_state or None,
            "registry_variables": registry_variables or None,
            "failed_forecast_request": failed_forecast_request,
            "parsed_task": _history_parsed_task(record),
            "tool_calls": compact_calls or None,
        }
    )


__all__ = [
    "ToolEvidenceSummary",
    "build_history_turn",
    "compact_tool_call_arguments",
    "summarize_record_tool_evidence",
]
