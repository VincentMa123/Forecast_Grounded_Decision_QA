from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class DecisionTraceState:
    """Verified session-wide inputs for forecast comparison and rendering."""

    candidate_results: List[Dict[str, Any]] = field(default_factory=list)
    decision_policy: Optional[Dict[str, Any]] = None
    decision_policy_source_question: Optional[str] = None
    applied_disturbances: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_history(
        cls,
        conversation_context: Iterable[Dict[str, Any]],
    ) -> "DecisionTraceState":
        candidates: Dict[str, Dict[str, Any]] = {}
        candidate_order: List[str] = []
        disturbances: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        disturbance_order: List[tuple[str, str, str]] = []
        decision_policy: Optional[Dict[str, Any]] = None
        decision_policy_source_question: Optional[str] = None

        for turn in conversation_context:
            if turn.get("grounding_verified") is not True:
                continue
            summary = dict(turn.get("verified_evidence_summary") or {})
            pipeformer = dict(summary.get("pipeformer") or {})
            explicit_candidate_ids = {
                str(arguments.get("candidate_id"))
                for call in turn.get("tool_calls") or []
                if call.get("name") == "run_pipeformer_forecast"
                for arguments in [dict(call.get("arguments") or {})]
                if arguments.get("candidate_id")
            }
            for candidate in pipeformer.get("candidate_results") or []:
                item = dict(candidate or {})
                candidate_id = str(item.get("candidate_id") or "")
                if not candidate_id or candidate_id not in explicit_candidate_ids:
                    continue
                key = candidate_id.casefold()
                if key not in candidates:
                    candidate_order.append(key)
                candidates[key] = deepcopy(item)

            decision = dict(pipeformer.get("decision_summary") or {})
            missing = {
                str(value)
                for value in decision.get("missing_metrics") or []
            }
            policy = dict(decision.get("ranking_policy") or {})
            source_error = any(
                "source_not_in_user_request" in value
                for value in missing
                if value.startswith("decision_policy_")
            )
            if (
                policy.get("source") == "llm_tool"
                and not source_error
            ):
                decision_policy = deepcopy(policy)
                decision_policy_source_question = str(
                    turn.get("user_input") or ""
                ) or None

            for disturbance in pipeformer.get("applied_disturbances") or []:
                item = dict(disturbance or {})
                variable = str(item.get("variable") or "")
                mode = str(item.get("mode") or "")
                if not variable or not mode:
                    continue
                key = (
                    variable.casefold(),
                    mode.casefold(),
                    repr(item.get("requested_value")),
                )
                if key not in disturbances:
                    disturbance_order.append(key)
                disturbances[key] = deepcopy(item)

        return cls(
            candidate_results=[candidates[key] for key in candidate_order],
            decision_policy=decision_policy,
            decision_policy_source_question=decision_policy_source_question,
            applied_disturbances=[
                disturbances[key] for key in disturbance_order
            ],
        )
