"""Teacher-record assembly: question parsing, projection, repair, and packaging.

Extracted verbatim from ``generate_teacher_trace`` (Round 7b, finding 2) so the
CLI/generator facade stays focused on orchestration. ``generate_teacher_trace``
re-exports these names, so existing ``from generate_teacher_trace import
build_teacher_record`` call sites keep working unchanged.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pipeclaw.backend.grounding.decision_trace_state import (
    DEFAULT_RECENT_TURNS_MAX_CHARS,
    DEFAULT_STATE_MAX_CHARS,
    VerifiedDecisionState,
    bounded_recent_turns,
    serialize_verified_decision_state,
)
from pipeclaw.backend.grounding.contract import (
    build_grounding_contract,
    finalize_applied_disturbance_disclosure,
    grounded_fallback_answer,
)
from pipeclaw.backend.evaluator.answer_quality import evaluate_quality_context
from pipeclaw.backend.evaluator.numeric_grounding import numeric_claims_are_grounded
from pipeclaw.backend.evaluator.quality_references import numeric_claim_values
from pipeclaw.backend.grounding.evidence.csv import record_csv_evidence
from pipeclaw.backend.grounding.evidence.tool import attach_tool_arguments
from pipeclaw.backend.grounding.evidence.topology import topology_summary_from_tool_outputs
from pipeclaw.backend.evaluator.scorer import NativeTraceEvaluator, apply_quality_aliases
from pipeclaw.backend.evaluator.quality_context import build_quality_context
from pipeclaw.backend.teacher_traces.trace_projection import (
    export_trace_tools,
    final_answer,
)

logger = logging.getLogger("teacher_trace")


def scenario_evidence_text(scenario: Dict[str, Any], question: str) -> str:
    description = str(scenario.get("scenario_description") or "").strip()
    return "\n".join(value for value in (description, question) if value)




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
