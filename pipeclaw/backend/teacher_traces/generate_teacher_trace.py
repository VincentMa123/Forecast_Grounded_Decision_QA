from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_CSV = (
    BACKEND_ROOT.parents[1]
    / "pipeFormer"
    / "data"
    / "mock_lifecycle"
    / "static"
    / "mock_lifecycle"
    / "index_variable_mapping.csv"
)
DEFAULT_SPLIT_SEED = "pipeclaw-lifecycle-v1"

from pipeclaw.backend.grounding.decision_trace_state import (
    DEFAULT_RECENT_TURNS_MAX_CHARS,
    DEFAULT_STATE_MAX_CHARS,
    VerifiedDecisionState,
    bounded_recent_turns,
    serialize_verified_decision_state,
)
from pipeclaw.backend.grounding.contract import (
    build_grounding_contract,
    comparison_answer_issues,
    finalize_applied_disturbance_disclosure,
    grounded_fallback_answer,
)
from pipeclaw.backend.evaluator.answer_quality import (
    answer_quality_issues,
    evaluate_quality_context,
)
from pipeclaw.backend.evaluator.numeric_grounding import (
    numeric_claims_are_grounded,
)
from pipeclaw.backend.evaluator.quality_references import (
    numeric_claim_values,
)
from pipeclaw.backend.grounding.evidence.csv import record_csv_evidence
from pipeclaw.backend.grounding.evidence.tool import (
    attach_tool_arguments,
)
from pipeclaw.backend.grounding.evidence.topology import (
    topology_summary_from_tool_outputs,
)
from pipeclaw.backend.evaluator.scorer import (
    NativeTraceEvaluator,
    apply_quality_aliases,
)
from pipeclaw.backend.evaluator.quality_context import build_quality_context
from pipeclaw.backend.pipeline.scenario_preflight import validate_scenario_sources
from pipeclaw.backend.teacher_traces.teacher_trace_store import TeacherTraceStore
from pipeclaw.backend.teacher_traces.trace_history import build_history_turn
from pipeclaw.backend.teacher_traces.trace_projection import (
    export_trace_tools,
    final_answer,
)
from pipeclaw.backend.teacher_traces.trace_export import write_split_records
from pipeclaw.backend.teacher_traces.trace_sources import (
    combined_preflight_sources,
    default_scenario_files,
    flatten_source_scenarios,
    load_scenario_sources,
)


logger = logging.getLogger("teacher_trace")


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
) -> str:
    description = str(scenario.get("scenario_description") or "").strip()
    parts = []
    if description:
        parts.extend(
            [
                "Scenario context (scope and task intent only; verify factual claims with tools):",
                description,
            ]
        )
    parts.extend(["Current user request:", question])
    return "\n\n".join(parts)


def load_backend_env() -> None:
    env_path = BACKEND_ROOT / ".env"
    if not env_path.exists():
        return
    from dotenv import load_dotenv

    load_dotenv(env_path, override=False)


def build_parser() -> argparse.ArgumentParser:
    root = BACKEND_ROOT
    parser = argparse.ArgumentParser(
        description="Run PipeClaw and export teacher_trace records."
    )
    parser.add_argument(
        "--scenario-file",
        type=Path,
        action="append",
        default=None,
        help="Scenario source JSON; repeat to combine sources. Defaults to PipeClaw v2, lifecycle v4, and lifecycle v7.",
    )
    parser.add_argument(
        "--dataset-source",
        default=None,
        help="Disambiguate --scenario-id when it exists in multiple sources.",
    )
    parser.add_argument(
        "--scenario-id",
        default=None,
        help="Generate one scenario. Omit to generate all scenarios.",
    )
    parser.add_argument("--agent-id", default="teacher_trace")
    parser.add_argument(
        "--session-id", default=None, help="Only valid with --scenario-id."
    )
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
    parser.add_argument("--preflight-only", action="store_true")
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


DEFAULT_NATIVE_EVALUATOR = NativeTraceEvaluator()


def _project_forecast_fields(
    question: str,
    trace: Dict[str, Any],
    conversation_context: Optional[List[Dict[str, Any]]],
    *,
    verified_state_max_chars: int,
    recent_turns_max_chars: int,
) -> Dict[str, Any]:
    """Project tool output into legacy forecast fields and trusted state."""

    tool_calls, tool_outputs, pipeformer_results = export_trace_tools(trace)
    pipeformer_call_count = sum(
        item.get("name") == "run_pipeformer_forecast" for item in tool_calls
    )
    grounded_tool_outputs = attach_tool_arguments(tool_outputs, tool_calls)
    history_state = VerifiedDecisionState.from_history(conversation_context or [])
    state_before = serialize_verified_decision_state(
        history_state,
        max_chars=verified_state_max_chars,
    )
    recent_turns = bounded_recent_turns(
        conversation_context or [],
        max_turns=2,
        max_chars=recent_turns_max_chars,
    )
    grounding_contract = build_grounding_contract(
        question,
        grounded_tool_outputs,
        prior_state=history_state,
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
    parsed_task: Dict[str, Any] = {}
    prediction_summary: Dict[str, Any] = {}
    constraint_check: Dict[str, Any] = {}
    evidence: Dict[str, Any] = {}
    risk_level = None
    manual_intervention_label = None
    dispatch_recommendation = None
    if pipeformer and projection:
        parsed_task = deepcopy(projection["parsed_task"])
        prediction_summary = deepcopy(projection["prediction_summary"])
        constraint_check = deepcopy(projection["constraint_check"])
        evidence = deepcopy(projection["evidence"])
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

        projected_fields = {
            field: {
                "candidate_forecasts": [
                    candidate_projection(item, field) for item in pipeformer_results
                ]
            }
            for field in (
                "parsed_task",
                "prediction_summary",
                "constraint_check",
                "evidence",
            )
        }
        parsed_task = projected_fields["parsed_task"]
        prediction_summary = projected_fields["prediction_summary"]
        constraint_check = projected_fields["constraint_check"]
        evidence = projected_fields["evidence"]
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
    return {
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
        "grounded_tool_outputs": grounded_tool_outputs,
        "pipeformer_call_count": pipeformer_call_count,
        "pipeformer_results": pipeformer_results,
        "pipeformer_outputs": pipeformer_outputs,
        "pipeformer": pipeformer,
        "grounding_contract": grounding_contract,
        "state_before": state_before,
        "recent_turns": recent_turns,
        "parsed_task": parsed_task,
        "prediction_summary": prediction_summary,
        "constraint_check": constraint_check,
        "evidence": evidence,
        "risk_level": risk_level,
        "manual_intervention_label": manual_intervention_label,
        "dispatch_recommendation": dispatch_recommendation,
    }


def _evaluate_and_repair_answer(
    scenario: Dict[str, Any],
    question: str,
    trace: Dict[str, Any],
    conversation_context: Optional[List[Dict[str, Any]]],
    forecast: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate the answer and apply deterministic grounding repairs."""

    answer = final_answer(trace).strip()
    grounding_contract = forecast["grounding_contract"]
    evidence = deepcopy(forecast["evidence"])
    original_answer = answer
    answer = finalize_applied_disturbance_disclosure(answer, grounding_contract)
    disclosure_repair_applied = answer != original_answer
    quality_question = scenario_evidence_text(scenario, question)
    csv_evidence = record_csv_evidence(
        {
            "tool_calls": forecast["tool_calls"],
            "tool_outputs": forecast["tool_outputs"],
            "final_answer": answer,
        },
        scope_text=quality_question,
    )
    if csv_evidence:
        evidence["csv_evidence"] = csv_evidence
    topology_summary = topology_summary_from_tool_outputs(
        forecast["grounded_tool_outputs"]
    )
    if topology_summary:
        evidence["topology_summary"] = topology_summary
    fallback_applied = False

    def score_answer(answer_text: str) -> tuple[Any, list[Any]]:
        context = build_quality_context(
            answer=answer_text,
            question=question,
            pipeformer=forecast["pipeformer"],
            conversation_context=conversation_context,
            tool_outputs=forecast["grounded_tool_outputs"],
            record_evidence=evidence,
        )
        forecasts_pass = (
            forecast["pipeformer_call_count"] == 0
            or bool(forecast["pipeformer_outputs"])
        ) and all(
            output.get("quality_flag") == "pass"
            for output in forecast["pipeformer_outputs"]
        )
        return evaluate_quality_context(
            context,
            grounding_contract,
            trace_status=trace.get("status"),
            forecasts_pass=forecasts_pass,
        )

    answer_quality_flag, quality_issues = score_answer(answer)
    if (
        grounding_contract.get("answer_mode") == "dispatch_comparison"
        and quality_issues
    ):
        fallback_answer = grounded_fallback_answer(question, grounding_contract)
        fallback_flag, fallback_issues = score_answer(fallback_answer)
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
            {"pipeformer_outputs": forecast["pipeformer_outputs"]},
        )
    ]
    if verified_numeric_claims:
        evidence["verified_numeric_claims"] = verified_numeric_claims
    return {
        "answer": answer,
        "evidence": evidence,
        "answer_quality_flag": answer_quality_flag,
        "quality_issues": quality_issues,
        "disclosure_repair_applied": disclosure_repair_applied,
        "fallback_applied": fallback_applied,
    }


def _assemble_teacher_record(
    scenario: Dict[str, Any],
    question: str,
    trace: Dict[str, Any],
    *,
    source_session_id: str,
    turn_id: int,
    conversation_context: Optional[List[Dict[str, Any]]],
    split: str,
    forecast: Dict[str, Any],
    answer: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the canonical record and persist native evaluator aliases."""

    dataset_source = str(scenario.get("dataset_source") or "unknown_source")
    scenario_id = str(scenario.get("scenario_id") or "unknown_scenario")
    record_id = f"{dataset_source}:{source_session_id}::turn_{turn_id:03d}"
    grounding_contract = forecast["grounding_contract"]
    quality_issues = answer["quality_issues"]
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
        "grounding_contract": deepcopy(grounding_contract),
        "decision_summary": deepcopy(
            dict(grounding_contract.get("decision_summary") or {})
        ),
        "conversation_context": deepcopy(conversation_context or []),
        "state_before": deepcopy(forecast["state_before"]),
        "recent_turns": deepcopy(forecast["recent_turns"]),
        "user_input": question,
        "parsed_task": deepcopy(forecast["parsed_task"]),
        "tool_calls": forecast["tool_calls"],
        "tool_outputs": forecast["tool_outputs"],
        "prediction_summary": deepcopy(forecast["prediction_summary"]),
        "constraint_check": deepcopy(forecast["constraint_check"]),
        "evidence": deepcopy(answer["evidence"]),
        "risk_level": forecast["risk_level"],
        "manual_intervention_label": forecast["manual_intervention_label"],
        "dispatch_recommendation": forecast["dispatch_recommendation"],
        "final_answer": answer["answer"],
        "trace_status": trace.get("status"),
        "quality_flag": answer["answer_quality_flag"],
        "quality_issues": quality_issues,
    }
    if grounding_contract.get("decision_policy"):
        record["decision_policy"] = deepcopy(grounding_contract["decision_policy"])
    if answer["disclosure_repair_applied"]:
        record["repair_provenance"] = {
            "method": "deterministic_disturbance_disclosure",
            "external_llm_calls": 0,
            "reason": (
                "Canonical applied-disturbance wording was prepended from stored "
                "execution evidence without changing the model's substantive answer."
            ),
        }
    elif answer["fallback_applied"]:
        record["repair_provenance"] = {
            "method": "deterministic_grounding_contract",
            "external_llm_calls": 0,
            "reason": "Multi-candidate answer rebuilt from stored tool evidence.",
        }
    native_quality = DEFAULT_NATIVE_EVALUATOR.evaluate(
        record,
        hard_issues=quality_issues,
        trace_status=trace.get("status"),
    )
    # The quality_* fields are aliases of the canonical schema-v3 report; the
    # generator never computes a second score of its own.
    apply_quality_aliases(
        record,
        native_quality,
        aliases=(
            "quality_flag",
            "quality_score",
            "quality_profile",
            "quality_failed_checks",
        ),
    )
    return record


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
    """Build one canonical teacher record from a completed turn trace."""

    forecast = _project_forecast_fields(
        question,
        trace,
        conversation_context,
        verified_state_max_chars=int(
            os.getenv("VERIFIED_STATE_MAX_CHARS", DEFAULT_STATE_MAX_CHARS)
        ),
        recent_turns_max_chars=int(
            os.getenv("RECENT_TURNS_MAX_CHARS", DEFAULT_RECENT_TURNS_MAX_CHARS)
        ),
    )
    answer = _evaluate_and_repair_answer(
        scenario,
        question,
        trace,
        conversation_context,
        forecast,
    )
    if answer["quality_issues"]:
        logger.warning(
            "Teacher answer requires review: %s",
            ", ".join(answer["quality_issues"]),
        )
    return _assemble_teacher_record(
        scenario,
        question,
        trace,
        source_session_id=source_session_id,
        turn_id=turn_id,
        conversation_context=conversation_context,
        split=split,
        forecast=forecast,
        answer=answer,
    )


def _turn_trace(
    full_trace: Dict[str, Any], message_start: int, tool_start: int
) -> Dict[str, Any]:
    return {
        "session_id": full_trace.get("session_id"),
        "agent_id": full_trace.get("agent_id"),
        "status": full_trace.get("status"),
        "messages": list(full_trace.get("messages") or [])[message_start:],
        "tool_calls": list(full_trace.get("tool_calls") or [])[tool_start:],
    }


def _build_session_record(
    scenario: Dict[str, Any],
    source_session: Dict[str, Any],
    source_session_id: str,
    records: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    raw_trace_paths: Dict[int, str],
    split: str = "train",
) -> Dict[str, Any]:
    """Finalize one source session from completed turn records."""

    dataset_source = str(scenario.get("dataset_source") or "unknown_source")
    scenario_id = str(scenario.get("scenario_id") or "unknown_scenario")
    return {
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
        "complete": not errors
        and len(records) == len(source_session.get("dialogue") or []),
        "errors": errors,
    }


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
            key=lambda value: hashlib.sha256(
                f"{seed}:{scenario_type}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        count = len(ordered)
        holdout_count = max(1, round(count * 0.1)) if count >= 3 else 0
        valid_count = holdout_count
        test_count = holdout_count
        train_end = count - valid_count - test_count
        for index, scenario_id in enumerate(ordered):
            result[scenario_id] = (
                "train"
                if index < train_end
                else "valid"
                if index < train_end + valid_count
                else "test"
            )
    return result


@dataclass(frozen=True)
class _GenerationScope:
    scenario_files: tuple[Path, ...]
    all_sources: tuple[Dict[str, Any], ...]
    selected_sources: tuple[Dict[str, Any], ...]
    scenarios: tuple[Dict[str, Any], ...]
    replacement_mode: bool


@dataclass(frozen=True)
class _GenerationExecution:
    records: List[Dict[str, Any]]
    session_records: List[Dict[str, Any]]
    skipped_scenarios: int
    failed_session_count: int


@dataclass(frozen=True)
class _MergedGeneration:
    records: List[Dict[str, Any]]
    session_records: List[Dict[str, Any]]
    removed_record_count: int
    removed_session_count: int
    duplicate_record_count: int
    updated_session_count: int
    appended_record_count: int


class TeacherTraceGenerator:
    """Coordinate scenario selection, generation, persistence, and SFT export."""

    def __init__(
        self, args: argparse.Namespace, store: Optional[TeacherTraceStore] = None
    ) -> None:
        self.args = args
        self.store = store if store is not None else TeacherTraceStore.from_args(args)

    def run(self) -> int:
        configure_logging(self.args.log_level)
        scope = self._select_scope()
        existing_records, existing_session_records = self._load_existing_records()
        self._validate_replacement_target(
            scope, existing_records, existing_session_records
        )
        preflight, selected_preflight = self._run_preflight(scope, existing_records)
        if self.args.preflight_only:
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            return 0
        self._require_supported_preflight(preflight, selected_preflight)
        execution = self._execute_sessions(
            scope,
            existing_records,
            existing_session_records,
        )
        merged = self._merge_records(
            scope,
            existing_records,
            existing_session_records,
            execution,
        )
        return self._persist_and_report(scope, existing_records, execution, merged)

    def _select_scope(self) -> _GenerationScope:
        args = self.args
        replacement_mode = bool(getattr(args, "replace_selected_scenario", False))
        if replacement_mode:
            if args.force:
                raise ValueError(
                    "--replace-selected-scenario cannot be combined with --force."
                )
            if not args.dataset_source or not args.scenario_id:
                raise ValueError(
                    "--replace-selected-scenario requires both --dataset-source and --scenario-id."
                )
            if args.session_id:
                raise ValueError(
                    "Scenario replacement must regenerate every session; omit --session-id."
                )

        scenario_files = tuple(args.scenario_file or default_scenario_files())
        all_sources = tuple(load_scenario_sources(list(scenario_files)))
        selected_sources = [
            source
            for source in all_sources
            if args.dataset_source is None
            or source["dataset_source"] == args.dataset_source
        ]
        if args.dataset_source and not selected_sources:
            available = ", ".join(source["dataset_source"] for source in all_sources)
            raise ValueError(
                f"Unknown --dataset-source {args.dataset_source!r}. Available sources: {available}"
            )
        if args.scenario_id:
            matches = [
                (source, scenario)
                for source in selected_sources
                for scenario in source.get("scenarios") or []
                if str(scenario.get("scenario_id")) == args.scenario_id
            ]
            if not matches:
                raise KeyError(
                    f"Scenario {args.scenario_id!r} not found in the selected sources."
                )
            if len(matches) > 1:
                candidates = ", ".join(
                    source["dataset_source"] for source, _ in matches
                )
                raise ValueError(
                    f"Scenario {args.scenario_id!r} exists in multiple sources ({candidates}); "
                    "pass --dataset-source."
                )
            source, scenario = matches[0]
            selected_sources = [{**source, "scenarios": [scenario]}]
        scenarios = tuple(flatten_source_scenarios(selected_sources))
        source_session_count = sum(
            len(scenario.get("sessions") or []) for scenario in scenarios
        )
        if args.session_id and (len(scenarios) != 1 or source_session_count != 1):
            raise ValueError(
                "--session-id requires one selected scenario containing exactly one source session."
            )
        return _GenerationScope(
            scenario_files=scenario_files,
            all_sources=all_sources,
            selected_sources=tuple(selected_sources),
            scenarios=scenarios,
            replacement_mode=replacement_mode,
        )

    def _load_existing_records(
        self,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if self.args.force:
            return [], []
        return self.store.load_master(), self.store.load_sessions()

    def _validate_replacement_target(
        self,
        scope: _GenerationScope,
        existing_records: List[Dict[str, Any]],
        existing_session_records: List[Dict[str, Any]],
    ) -> None:
        if not scope.replacement_mode:
            return
        target = {
            "dataset_source": str(self.args.dataset_source),
            "scenario_id": str(self.args.scenario_id),
        }
        missing_targets = []
        if not self.store.contains_scenario(existing_records, **target):
            missing_targets.append("master teacher trace")
        if not self.store.contains_scenario(existing_session_records, **target):
            missing_targets.append("session teacher trace")
        if missing_targets:
            raise ValueError(
                "Cannot replace the selected scenario because it is absent from "
                + " and ".join(missing_targets)
                + "; no LLM calls were made."
            )

    def _run_preflight(
        self,
        scope: _GenerationScope,
        existing_records: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        preflight_sources, missing_preflight_pairs = combined_preflight_sources(
            list(scope.all_sources),
            list(scope.selected_sources),
            existing_records,
        )
        preflight = validate_scenario_sources(
            preflight_sources or list(scope.selected_sources),
            DEFAULT_MAPPING_CSV,
        )
        selected_preflight = validate_scenario_sources(
            list(scope.selected_sources),
            DEFAULT_MAPPING_CSV,
        )
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
        preflight["append_mode"] = not self.args.force and not scope.replacement_mode
        preflight["replacement_mode"] = scope.replacement_mode
        preflight["existing_record_count"] = len(existing_records)
        preflight["unavailable_existing_scenarios"] = missing_preflight_pairs
        return preflight, selected_preflight

    @staticmethod
    def _require_supported_preflight(
        preflight: Dict[str, Any],
        selected_preflight: Dict[str, Any],
    ) -> None:
        if preflight["supported"]:
            return
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

    def _run_session(
        self,
        scenario: Dict[str, Any],
        source_session: Dict[str, Any],
        scenario_index: int,
        session_index: int,
        run_stamp: str,
        split: str,
        scenario_history: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run one source session using this generator's runtime settings."""

        from pipeclaw.backend.agent.orchestrator import AgentOrchestrator
        from pipeclaw.backend.agent.schemas import AgentChatRequest

        args = self.args
        scenario_id = str(
            scenario.get("scenario_id") or f"scenario_{scenario_index:06d}"
        )
        dataset_source = str(scenario.get("dataset_source") or "unknown_source")
        source_session_id = str(
            source_session.get("session_id")
            or f"{scenario_id}_session_{session_index:03d}"
        )
        runtime_session_id = (
            args.session_id
            or f"teacher_{scenario_index:04d}_{session_index:03d}_{run_stamp}"
        )
        os.environ["PIPEFORMER_DEVICE"] = str(args.device)

        logger.info(
            "Session started: scenario=%s source_session=%s runtime_session=%s",
            scenario_id,
            source_session_id,
            runtime_session_id,
        )

        run_hash = hashlib.sha256(
            f"{dataset_source}:{scenario_id}:{run_stamp}".encode("utf-8")
        ).hexdigest()[:10]
        run_namespace = f"r{scenario_index:04d}_{run_hash}"
        orchestrator = AgentOrchestrator(
            data_loader=None,
            agent_id=args.agent_id,
            session_id=runtime_session_id,
            enable_skills=False,
            workspace_root_base=BACKEND_ROOT / ".openclaw" / "tt_runs" / run_namespace,
        )
        orchestrator.verified_state_manager.commit(
            runtime_session_id,
            VerifiedDecisionState.from_history(scenario_history),
        )
        records: List[Dict[str, Any]] = []
        history = scenario_history
        errors: List[Dict[str, Any]] = []
        raw_trace_paths: Dict[int, str] = {}
        message_count = 0
        tool_count = 0
        for fallback_turn_id, turn in enumerate(
            source_session.get("dialogue") or [], start=1
        ):
            question = str(turn.get("user_input") or "").strip()
            if not question:
                continue
            turn_id = int(turn.get("turn_id") or fallback_turn_id)
            logger.info(
                "Turn started: scenario=%s session=%s turn=%d question=%s",
                scenario_id,
                source_session_id,
                turn_id,
                short_text(question, 300),
            )
            try:

                def validate_answer(
                    answer: str, completed_calls: List[Dict[str, Any]]
                ) -> List[str]:
                    history_state = VerifiedDecisionState.from_history(history)
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
                            "topology_summary": topology_summary_from_tool_outputs(
                                completed_calls
                            )
                        },
                    )
                    contract = build_grounding_contract(
                        question,
                        completed_calls,
                        require_decision_policy=True,
                        prior_state=history_state,
                    )
                    issues.extend(comparison_answer_issues(answer, contract))
                    return list(dict.fromkeys(issues))

                result = orchestrator.run_agent(
                    AgentChatRequest(
                        agent_id=args.agent_id,
                        session_id=runtime_session_id,
                        message=agent_turn_message(scenario, question),
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
                history.append(build_history_turn(record))
                logger.info(
                    "Turn finished: scenario=%s session=%s turn=%d quality=%s",
                    scenario_id,
                    source_session_id,
                    turn_id,
                    record["quality_flag"],
                )
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
                logger.exception(
                    "Turn failed: scenario=%s session=%s turn=%d",
                    scenario_id,
                    source_session_id,
                    turn_id,
                )
                errors.append(
                    {"turn_id": turn_id, "user_input": question, "error": str(exc)}
                )
                break

        session_record = _build_session_record(
            scenario,
            source_session,
            source_session_id,
            records,
            errors,
            raw_trace_paths,
            split,
        )
        return records, session_record

    def _execute_sessions(
        self,
        scope: _GenerationScope,
        existing_records: List[Dict[str, Any]],
        existing_session_records: List[Dict[str, Any]],
    ) -> _GenerationExecution:
        logger.info(
            "Teacher trace generation started: scenarios=%d device=%s scenario_files=%s",
            len(scope.scenarios),
            self.args.device,
            [str(path) for path in scope.scenario_files],
        )
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        records: List[Dict[str, Any]] = []
        session_records: List[Dict[str, Any]] = []
        split_map = scenario_split_map(
            flatten_source_scenarios(list(scope.all_sources)), DEFAULT_SPLIT_SEED
        )
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
        for index, scenario in enumerate(scope.scenarios, start=1):
            scenario_id = str(scenario.get("scenario_id") or f"scenario_{index:06d}")
            expected_sample_ids = self.store.sample_ids(scenario)
            expected_session_ids = self.store.session_ids(scenario)
            if (
                not scope.replacement_mode
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
            for session_index, source_session in enumerate(
                scenario.get("sessions") or [], start=1
            ):
                turn_records, session_record = self._run_session(
                    scenario,
                    source_session,
                    index,
                    session_index,
                    run_stamp,
                    split,
                    scenario_history,
                )
                records.extend(turn_records)
                session_records.append(session_record)
        failed_session_count = sum(
            not bool(item.get("complete")) for item in session_records
        )
        return _GenerationExecution(
            records, session_records, skipped_scenarios, failed_session_count
        )

    def _merge_records(
        self,
        scope: _GenerationScope,
        existing_records: List[Dict[str, Any]],
        existing_session_records: List[Dict[str, Any]],
        execution: _GenerationExecution,
    ) -> _MergedGeneration:
        failed_sessions = execution.failed_session_count
        if scope.replacement_mode:
            if failed_sessions:
                raise RuntimeError(
                    "Scenario replacement was not written because at least one regenerated session failed."
                )
            selected_scenario = scope.scenarios[0]
            expected_sample_ids = set(self.store.sample_ids(selected_scenario))
            expected_session_ids = set(self.store.session_ids(selected_scenario))
            generated_sample_ids = {
                str(item.get("sample_id") or "") for item in execution.records
            }
            generated_session_ids = {
                str(item.get("session_record_id") or "")
                for item in execution.session_records
            }
            if generated_sample_ids != expected_sample_ids:
                raise ValueError(
                    "Scenario replacement is incomplete: generated sample ids do not match the source definition."
                )
            if generated_session_ids != expected_session_ids:
                raise ValueError(
                    "Scenario replacement is incomplete: generated session ids do not match the source definition."
                )
            combined_records, removed_record_count = self.store.replace_scenario(
                existing_records,
                execution.records,
                dataset_source=str(self.args.dataset_source),
                scenario_id=str(self.args.scenario_id),
                id_field="sample_id",
            )
            combined_session_records, removed_session_count = (
                self.store.replace_scenario(
                    existing_session_records,
                    execution.session_records,
                    dataset_source=str(self.args.dataset_source),
                    scenario_id=str(self.args.scenario_id),
                    id_field="session_record_id",
                )
            )
            merged = _MergedGeneration(
                combined_records,
                combined_session_records,
                removed_record_count,
                removed_session_count,
                0,
                0,
                0,
            )
        else:
            combined_records, duplicate_record_count = self.store.merge_records(
                existing_records,
                execution.records,
                id_field="sample_id",
            )
            combined_session_records, updated_session_count = self.store.merge_sessions(
                existing_session_records,
                execution.session_records,
            )
            merged = _MergedGeneration(
                combined_records,
                combined_session_records,
                0,
                0,
                duplicate_record_count,
                updated_session_count,
                len(combined_records) - len(existing_records),
            )
        self.store.validate_splits(merged.records)
        return merged

    def _persist_and_report(
        self,
        scope: _GenerationScope,
        existing_records: List[Dict[str, Any]],
        execution: _GenerationExecution,
        merged: _MergedGeneration,
    ) -> int:
        logger.info("Writing combined JSONL output: %s", self.args.output_jsonl)
        logger.info("Writing pretty JSON output: %s", self.args.output_json)
        self.store.write_master(merged.records)
        logger.info(
            "Writing session evaluation JSONL: %s", self.args.session_output_jsonl
        )
        self.store.write_sessions(merged.session_records)
        logger.info(
            "Writing scenario-isolated split files: %s", self.args.split_output_dir
        )
        sft_record_count = write_split_records(
            self.args.split_output_dir,
            merged.records,
        )
        failed_sessions = execution.failed_session_count
        total_failed_sessions = sum(
            not bool(item.get("complete")) for item in merged.session_records
        )
        quality_pass_records = sum(
            item.get("quality_flag") == "pass" for item in merged.records
        )
        if failed_sessions:
            run_status = "completed_with_errors"
        elif (
            not scope.replacement_mode
            and not merged.appended_record_count
            and existing_records
        ):
            run_status = "no_changes"
        elif sft_record_count < len(merged.records):
            run_status = "completed_with_quality_issues"
        else:
            run_status = "ok"
        logger.info(
            "Teacher trace generation complete: generated=%d appended=%d total=%d quality_pass_records=%d "
            "sft_records=%d failed_sessions=%d",
            len(execution.records),
            merged.appended_record_count,
            len(merged.records),
            quality_pass_records,
            sft_record_count,
            failed_sessions,
        )
        print(
            json.dumps(
                {
                    "status": run_status,
                    "append_mode": not self.args.force and not scope.replacement_mode,
                    "replacement_mode": scope.replacement_mode,
                    "replaced_dataset_source": self.args.dataset_source
                    if scope.replacement_mode
                    else None,
                    "replaced_scenario_id": self.args.scenario_id
                    if scope.replacement_mode
                    else None,
                    "removed_records": merged.removed_record_count,
                    "removed_sessions": merged.removed_session_count,
                    "records": len(merged.records),
                    "generated_records": len(execution.records),
                    "appended_records": merged.appended_record_count,
                    "duplicate_records_skipped": merged.duplicate_record_count,
                    "skipped_existing_scenarios": execution.skipped_scenarios,
                    "quality_pass_records": quality_pass_records,
                    "sft_records": sft_record_count,
                    "sessions": len(merged.session_records),
                    "failed_sessions": failed_sessions,
                    "total_failed_sessions": total_failed_sessions,
                    "updated_existing_sessions": merged.updated_session_count,
                    "output_jsonl": self.args.output_jsonl.as_posix(),
                    "output_json": self.args.output_json.as_posix(),
                    "session_output_jsonl": self.args.session_output_jsonl.as_posix(),
                    "split_output_dir": self.args.split_output_dir.as_posix(),
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
