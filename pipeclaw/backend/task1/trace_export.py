from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from pipeclaw.backend.evaluator.numeric_grounding import (
    grounded_numeric_claim_values,
    numeric_claims_are_grounded,
    numeric_grounding_evidence,
)
from pipeclaw.backend.evaluator.quality_references import variable_references
from pipeclaw.backend.grounding.decision_trace_state import (
    VerifiedDecisionState,
    bounded_recent_turns,
    serialize_verified_decision_state,
)
from pipeclaw.backend.pipeline.io_utils import write_jsonl
from pipeclaw.backend.task1.trace_projection import (
    DEFAULT_PROJECTOR,
    SFT_MAX_PIPEFORMER_VARIABLES,
)


SFT_MAX_RECORD_CHARS = 35_000
SFT_PRIOR_TOOL_CALL_ARGUMENT_KEYS: Dict[str, frozenset[str] | None] = {
    "read_file": frozenset({"path"}),
    "write_file": frozenset({"path"}),
    "edit_file": frozenset({"path"}),
    "run_command": frozenset({"cmd"}),
    "search_pipeformer_registry": frozenset({"query", "role", "controllable"}),
    "run_pipeformer_forecast": frozenset(
        {
            "case_id",
            "disturbance_variable",
            "disturbance_direction",
            "disturbance_magnitude_percent",
            "output_state_variables",
        }
    ),
    "analyze_pipeline_topology": None,
    "set_decision_policy": frozenset(),
}

logger = logging.getLogger("teacher_trace")


def prior_tool_call_provenance(
    conversation_context: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep identifying arguments for prior calls without their bulk payloads."""
    summarized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for turn in conversation_context or []:
        for source_call in dict(turn).get("tool_calls") or []:
            call = dict(source_call)
            name = str(call.get("name") or "")
            if not name:
                continue
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (TypeError, ValueError):
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            retained_keys = SFT_PRIOR_TOOL_CALL_ARGUMENT_KEYS.get(name, frozenset())
            projected_arguments = (
                dict(arguments)
                if retained_keys is None
                else {
                    key: arguments[key]
                    for key in sorted(arguments)
                    if key in retained_keys
                }
            )
            entry: Dict[str, Any] = {"name": name}
            if projected_arguments:
                entry["arguments"] = projected_arguments
            fingerprint = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            if fingerprint not in seen:
                seen.add(fingerprint)
                summarized.append(entry)
    return summarized


def write_split_records(
    output_dir: Path,
    records: List[Dict[str, Any]],
    force: bool,
) -> int:
    """Write only compact, quality-passing records that remain evidence-grounded."""
    eligible_records = [
        item
        for item in records
        if item.get("quality_flag") == "pass"
        and not item.get("sft_exclusion_reason")
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
            rebuilt = serialize_verified_decision_state(
                VerifiedDecisionState.from_history(item.get("conversation_context") or []),
                max_chars=int(os.getenv("VERIFIED_STATE_MAX_CHARS", "16000")),
            )
            state_before = dict(projected.get("state_before") or {})
            for key, value in rebuilt.items():
                if value and not state_before.get(key):
                    state_before[key] = value
            projected["state_before"] = state_before
            if "recent_turns" not in projected:
                projected["recent_turns"] = bounded_recent_turns(
                    item.get("conversation_context") or [],
                    max_turns=2,
                    max_chars=int(os.getenv("RECENT_TURNS_MAX_CHARS", "4000")),
                )
            prior_tool_calls = prior_tool_call_provenance(
                item.get("conversation_context") or []
            )
            state_before = projected.get("state_before")
            if prior_tool_calls and isinstance(state_before, dict):
                provenance = dict(state_before.get("provenance") or {})
                provenance["prior_tool_calls"] = prior_tool_calls
                projected["state_before"] = {
                    **state_before,
                    "provenance": provenance,
                }
            full_grounding_evidence = numeric_grounding_evidence(item)
            projected["tool_calls"], projected["tool_outputs"] = (
                DEFAULT_PROJECTOR.select_sft_trajectory(
                    list(projected.get("tool_calls") or []),
                    list(projected.get("tool_outputs") or []),
                    str(projected.get("final_answer") or ""),
                )
            )
            projected_evidence = DEFAULT_PROJECTOR.serialize_sft_record_evidence(
                dict(projected.get("evidence") or {})
            )
            projected["decision_summary"] = (
                DEFAULT_PROJECTOR.serialize_sft_decision_summary(
                    projected.get("decision_summary")
                )
            )
            answer_text = str(projected.get("final_answer") or "")
            supporting_values = list(
                dict.fromkeys(
                    grounded_numeric_claim_values(
                        answer_text,
                        str(projected.get("user_input") or ""),
                        full_grounding_evidence,
                    )
                )
            )
            if supporting_values:
                projected_evidence["supporting_numeric_values"] = supporting_values
            projected["evidence"] = projected_evidence
            compact_evidence = numeric_grounding_evidence(projected)
            if not numeric_claims_are_grounded(
                answer_text,
                str(projected.get("user_input") or ""),
                compact_evidence,
            ):
                logger.warning(
                    "Skipping SFT record with evidence removed by compaction: sample=%s",
                    item.get("sample_id"),
                )
                continue
            size = len(json.dumps(projected, ensure_ascii=False))
            for max_variables in range(SFT_MAX_PIPEFORMER_VARIABLES - 1, -1, -1):
                if size <= SFT_MAX_RECORD_CHARS:
                    break
                trial = dict(projected)
                trial["tool_calls"], trial["tool_outputs"] = (
                    DEFAULT_PROJECTOR.select_sft_trajectory(
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
                    for variable in variable_references(answer_text)
                }
                supported_variables = {
                    variable.casefold()
                    for variable in variable_references(
                        json.dumps(trial_evidence, ensure_ascii=False)
                    )
                }
                if not claimed_variables <= supported_variables:
                    continue
                trial_size = len(json.dumps(trial, ensure_ascii=False))
                if trial_size < size:
                    projected, size = trial, trial_size
            if size > SFT_MAX_RECORD_CHARS:
                logger.warning(
                    "Skipping oversized SFT record: sample=%s chars=%d",
                    item.get("sample_id"),
                    size,
                )
                continue
            split_records.append(projected)
        write_jsonl(output_dir / f"teacher_trace_{split}.jsonl", split_records, force=force)
        written_count += len(split_records)
    return written_count


__all__ = ["prior_tool_call_provenance", "write_split_records"]
