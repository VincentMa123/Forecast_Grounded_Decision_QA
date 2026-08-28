from __future__ import annotations

from dataclasses import dataclass, fields
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional

from .answer_limits import (
    ENGLISH_COMPARISON_MAX_CHARS,
    ENGLISH_MAX_WORDS,
    chinese_comparison_max_chars,
)
from .decision_policy import (
    RISK_RANK,
    collect_objective_evidence,
    decision_policy_source_has_priority_signal,
    llm_policy_excerpts,
    nested_value,
    number_value,
    normalize_decision_policy,
    rank_candidate_groups,
)
from .decision_trace_state import (
    ENGINEERING_METRIC_FIELDS,
    VerifiedDecisionState,
    canonical_json,
    has_boundary_action,
)
from .evidence.tool import attach_tool_arguments, classify_tool_evidence, requested_artifacts


INTERVENTION_RANK = {
    "no_intervention": 0,
    "monitoring_only": 1,
    "operator_attention_required": 2,
    "immediate_intervention_required": 3,
}


def forecast_views(output: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    return (
        dict(output.get("prediction") or output.get("prediction_summary") or {}),
        dict(output.get("verification") or output.get("constraint_check") or {}),
    )


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def successful_pipeformer_results(
    tool_results: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in tool_results
        if item.get("name") == "run_pipeformer_forecast"
        and dict(item.get("output") or {}).get("success") is True
    ]


def latest_decision_policy(
    results: Iterable[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for item in reversed(list(results)):
        output = dict(item.get("output") or {})
        if (
            item.get("name") == "set_decision_policy"
            and output.get("success") is True
            and isinstance(output.get("decision_policy"), dict)
        ):
            return dict(output["decision_policy"])
    return None


def _candidate_role(item: Dict[str, Any]) -> str:
    output = dict(item.get("output") or {})
    arguments = dict(item.get("arguments") or {})
    return str(
        output.get("candidate_role") or arguments.get("candidate_role") or ""
    ).casefold()


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    tool_call_id: str
    action: Dict[str, Any]
    failure_count: int
    warning_count: int
    risk_level: str
    manual_intervention_label: str
    dispatch_recommendation: str
    failed_rule_ids: List[str]
    warning_rule_ids: List[str]
    energy_consumption: Optional[float]
    nonzero_impacted_variable_count: Optional[int]
    pressure_metrics: Dict[str, Any]
    linepack_metrics: Dict[str, Any]
    flow_metrics: Dict[str, Any]
    compressor_metrics: Dict[str, Any]
    energy_metrics: Dict[str, Any]
    baseline_reference: Optional[str]
    category_status: Dict[str, Any]

    def compact(self) -> Dict[str, Any]:
        compact = {item.name: getattr(self, item.name) for item in fields(self)}
        compact["elimination_reasons"] = self.failed_rule_ids
        return compact

    @classmethod
    def from_compact(cls, value: Dict[str, Any]) -> "CandidateResult":
        item = dict(value or {})
        return cls(
            candidate_id=str(item.get("candidate_id") or ""),
            tool_call_id=str(item.get("tool_call_id") or ""),
            action=dict(item.get("action") or {}),
            failure_count=int(item.get("failure_count") or 0),
            warning_count=int(item.get("warning_count") or 0),
            risk_level=str(item.get("risk_level") or "low"),
            manual_intervention_label=str(
                item.get("manual_intervention_label") or "no_intervention"
            ),
            dispatch_recommendation=str(
                item.get("dispatch_recommendation") or ""
            ),
            failed_rule_ids=[
                str(rule) for rule in item.get("failed_rule_ids") or []
            ],
            warning_rule_ids=[
                str(rule) for rule in item.get("warning_rule_ids") or []
            ],
            energy_consumption=(
                float(item["energy_consumption"])
                if item.get("energy_consumption") is not None
                else None
            ),
            nonzero_impacted_variable_count=(
                int(item["nonzero_impacted_variable_count"])
                if item.get("nonzero_impacted_variable_count") is not None
                else None
            ),
            pressure_metrics=dict(item.get("pressure_metrics") or {}),
            linepack_metrics=dict(item.get("linepack_metrics") or {}),
            flow_metrics=dict(item.get("flow_metrics") or {}),
            compressor_metrics=dict(item.get("compressor_metrics") or {}),
            energy_metrics=dict(item.get("energy_metrics") or {}),
            baseline_reference=(
                str(item["baseline_reference"])
                if item.get("baseline_reference")
                else None
            ),
            category_status=dict(item.get("category_status") or {}),
        )


class GroundingContractBuilder:
    """Build compact evidence and deterministic decisions from saved tool results."""

    def build(
        self,
        question: str,
        tool_results: Iterable[Dict[str, Any]],
        *,
        decision_policy: Optional[Dict[str, Any]] = None,
        require_decision_policy: bool = False,
        prior_candidate_results: Optional[Iterable[Dict[str, Any]]] = None,
        prior_decision_policy: Optional[Dict[str, Any]] = None,
        prior_decision_policy_source_question: Optional[str] = None,
        prior_applied_disturbances: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        results = [dict(item) for item in tool_results]
        current_policy = latest_decision_policy(results)
        uses_prior_policy = (
            decision_policy is None
            and current_policy is None
            and prior_decision_policy is not None
        )
        resolved_decision_policy = (
            dict(decision_policy)
            if decision_policy is not None
            else current_policy
            if current_policy is not None
            else dict(prior_decision_policy)
            if prior_decision_policy is not None
            else None
        )
        decision_policy_question = (
            str(prior_decision_policy_source_question or "")
            if uses_prior_policy
            else question
        )
        pipeformer = successful_pipeformer_results(results)
        prior_candidates = [
            dict(item)
            for item in prior_candidate_results or []
            if dict(item or {}).get("candidate_id")
        ]
        prior_applied = [
            dict(item)
            for item in prior_applied_disturbances or []
            if isinstance(item, dict)
        ]
        if pipeformer or prior_candidates or prior_applied:
            return self._pipeformer_contract(
                pipeformer,
                question,
                decision_policy=resolved_decision_policy,
                decision_policy_question=decision_policy_question,
                require_decision_policy=require_decision_policy,
                prior_candidate_results=prior_candidates,
                prior_applied_disturbances=prior_applied,
            )
        return self._generic_contract(question, results)

    def _pipeformer_contract(
        self,
        results: List[Dict[str, Any]],
        question: str,
        *,
        decision_policy: Optional[Dict[str, Any]],
        decision_policy_question: str,
        require_decision_policy: bool,
        prior_candidate_results: List[Dict[str, Any]],
        prior_applied_disturbances: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        baselines = self._deduplicate_candidate_results(
            [
                item
                for item in results
                if _candidate_role(item) == "baseline"
            ]
        )
        candidates = self._deduplicate_candidate_results(
            [
                item
                for item in results
                if _candidate_role(item) != "baseline"
            ]
        )

        current_action_candidates = any(
            self._candidate_action(dict(item.get("arguments") or {}))
            for item in candidates
        )
        parsed_by_id: Dict[str, CandidateResult] = {}
        parsed_order: List[str] = []
        for value in prior_candidate_results:
            candidate = CandidateResult.from_compact(value)
            key = candidate.candidate_id.casefold()
            if not key:
                continue
            if (
                current_action_candidates
                and not has_boundary_action({"action": candidate.action})
            ):
                continue
            if key not in parsed_by_id:
                parsed_order.append(key)
            parsed_by_id[key] = candidate
        for index, item in enumerate(candidates, 1):
            candidate = self._candidate(index, item)
            key = candidate.candidate_id.casefold()
            if key not in parsed_by_id:
                parsed_order.append(key)
            parsed_by_id[key] = candidate
        parsed = self._deduplicate_candidate_actions(
            [parsed_by_id[key] for key in parsed_order]
        )
        contract: Dict[str, Any] = {
            "answer_mode": "dispatch_comparison" if len(parsed) > 1 else "single_forecast",
            "current_candidate_forecast_count": len(candidates),
            "current_decision_policy_call_count": sum(
                item.get("name") == "set_decision_policy"
                for item in results
            ),
            "candidate_results": [item.compact() for item in parsed],
            "worst_case_risk_level": self._worst(
                (item.risk_level for item in parsed), RISK_RANK, "low"
            ),
            "worst_case_intervention_label": self._worst(
                (item.manual_intervention_label for item in parsed),
                INTERVENTION_RANK,
                "no_intervention",
            ),
        }
        if baselines:
            contract["baseline_tool_call_id"] = baselines[0].get("tool_call_id")
        if len(parsed) > 1:
            contract["decision_summary"] = self._decision(
                parsed,
                question=decision_policy_question,
                decision_policy=decision_policy,
                require_decision_policy=require_decision_policy,
            )
            ranking_policy = dict(
                contract["decision_summary"].get("ranking_policy") or {}
            )
            if ranking_policy.get("source") == "llm_tool":
                contract["decision_policy"] = ranking_policy
            contract["comparison_leaders"] = self._comparison_leaders(parsed)
        assumptions = self._provisional_assumptions(results)
        if assumptions:
            contract["provisional_assumptions"] = assumptions
        applied_disturbances: List[Dict[str, Any]] = []
        applied_seen = set()
        for disturbance in [
            *prior_applied_disturbances,
            *self._applied_disturbances(results),
        ]:
            variable = str(disturbance.get("variable") or "")
            mode = str(disturbance.get("mode") or "")
            if not variable or not mode:
                continue
            key = canonical_json(disturbance)
            if key in applied_seen:
                continue
            applied_seen.add(key)
            applied_disturbances.append(dict(disturbance))
        if applied_disturbances:
            contract["applied_disturbances"] = applied_disturbances
        return contract

    @staticmethod
    def _deduplicate_candidate_results(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep the latest successful result for each stable candidate identifier."""
        order: List[str] = []
        latest: Dict[str, Dict[str, Any]] = {}
        for index, item in enumerate(results, 1):
            output = dict(item.get("output") or {})
            arguments = dict(item.get("arguments") or {})
            candidate_id = str(
                arguments.get("candidate_id")
                or output.get("candidate_id")
                or item.get("tool_call_id")
                or f"candidate_{index}"
            )
            key = candidate_id.casefold()
            if key not in latest:
                order.append(key)
            latest[key] = item
        return [latest[key] for key in order]

    @staticmethod
    def _deduplicate_candidate_actions(
        candidates: List[CandidateResult],
    ) -> List[CandidateResult]:
        """Keep the latest candidate for each canonical boundary action."""
        order: List[str] = []
        latest: Dict[str, CandidateResult] = {}
        for candidate in candidates:
            action = dict(candidate.action or {})
            payload = {
                key: {
                    str(variable): value
                    for variable, value in sorted(
                        dict(action.get(key) or {}).items()
                    )
                }
                for key in ("setpoints", "percentage_changes")
                if action.get(key)
            }
            key = (
                "action:" + canonical_json(payload)
                if payload
                else "candidate:" + candidate.candidate_id.casefold()
            )
            if key not in latest:
                order.append(key)
            latest[key] = candidate
        return [latest[key] for key in order]

    @staticmethod
    def _provisional_assumptions(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        assumptions: Dict[str, Dict[str, Any]] = {}
        for item in results:
            output = dict(item.get("output") or {})
            arguments = dict(item.get("arguments") or {})
            prediction, _ = forecast_views(output)
            parsed_task = dict(output.get("parsed_task") or {})
            sources = [parsed_task, prediction, output]
            for source in sources:
                assumption = source.get("disturbance_assumption")
                if not isinstance(assumption, dict) or assumption.get("source") != "llm_assumption":
                    continue
                variable = (
                    source.get("disturbance_variable")
                    or prediction.get("disturbance_variable")
                    or parsed_task.get("disturbance_variable")
                    or arguments.get("disturbance_variable")
                )
                setpoint = None
                for candidate in (source, parsed_task, arguments):
                    setpoints = dict(
                        dict(candidate.get("boundary_conditions") or {}).get(
                            "setpoints"
                        )
                        or {}
                    )
                    if variable in setpoints:
                        setpoint = setpoints[variable]
                        break
                value = {
                    "variable": variable,
                    "direction": (
                        source.get("disturbance_direction")
                        or prediction.get("disturbance_direction")
                        or parsed_task.get("disturbance_direction")
                        or arguments.get("disturbance_direction")
                    ),
                    "magnitude_percent": (
                        source.get("disturbance_magnitude_percent")
                        if source.get("disturbance_magnitude_percent") is not None
                        else prediction.get("disturbance_magnitude_percent")
                        if prediction.get("disturbance_magnitude_percent") is not None
                        else parsed_task.get("disturbance_magnitude_percent")
                        if parsed_task.get("disturbance_magnitude_percent") is not None
                        else arguments.get("disturbance_magnitude_percent")
                    ),
                    "setpoint": setpoint,
                    "statement": assumption.get("statement"),
                }
                key = json.dumps(value, ensure_ascii=False, sort_keys=True)
                assumptions[key] = value
        return list(assumptions.values())

    @staticmethod
    def _applied_disturbances(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        disturbances: Dict[tuple[str, str], Dict[str, Any]] = {}
        order: List[tuple[str, str]] = []
        for item in results:
            if item.get("name") != "run_pipeformer_forecast":
                continue
            output = dict(item.get("output") or {})
            arguments = dict(item.get("arguments") or {})
            prediction, _ = forecast_views(output)
            variable = str(
                arguments.get("disturbance_variable")
                or prediction.get("disturbance_variable")
                or ""
            )
            if not variable:
                continue
            evidence = dict(output.get("evidence") or {})
            application = next(
                (
                    dict(value)
                    for value in evidence.get(
                        "boundary_application_evidence"
                    )
                    or []
                    if str(dict(value or {}).get("variable") or "") == variable
                ),
                {},
            )
            if application:
                mode = str(application.get("mode") or "")
                requested_value = application.get("requested_value")
                before = list(application.get("input_values_before") or [])
                applied = list(application.get("input_values_applied") or [])
                verified = application.get("verified") is True
            elif arguments.get("disturbance_setpoint") is not None:
                mode = "setpoint"
                requested_value = arguments.get("disturbance_setpoint")
                before = []
                applied = []
                verified = False
            elif arguments.get("disturbance_magnitude_percent") is not None:
                mode = "percent_change"
                requested_value = arguments.get(
                    "disturbance_magnitude_percent"
                )
                before = []
                applied = []
                verified = False
            else:
                continue
            no_op = bool(before and applied and len(before) == len(applied))
            if no_op:
                try:
                    requested_number = float(requested_value)
                    no_op = all(
                        float(old) == requested_number
                        and float(new) == requested_number
                        for old, new in zip(before, applied)
                    )
                except (TypeError, ValueError):
                    no_op = all(
                        old == requested_value and new == requested_value
                        for old, new in zip(before, applied)
                    )
            value = {
                "variable": variable,
                "mode": mode,
                "requested_value": requested_value,
                "direction": (
                    arguments.get("disturbance_direction")
                    or prediction.get("disturbance_direction")
                    or dict(output.get("parsed_task") or {}).get(
                        "disturbance_direction"
                    )
                ),
                "input_values_before": before,
                "input_values_applied": applied,
                "verified": verified,
                "no_op": no_op,
            }
            key = (variable.casefold(), mode.casefold())
            if key not in disturbances:
                order.append(key)
            disturbances[key] = value
        return [disturbances[key] for key in order]

    def _candidate(self, index: int, item: Dict[str, Any]) -> CandidateResult:
        output = dict(item.get("output") or {})
        prediction, verification = forecast_views(output)
        arguments = dict(item.get("arguments") or {})
        metrics = dict(verification.get("comparable_metrics") or {})
        engineering = dict(verification.get("engineering_evidence") or {})
        comparison = dict(prediction.get("counterfactual_comparison") or {})
        category_metrics = {
            category: self._compact_category_metrics(
                engineering.get(category), fields
            )
            for category, fields in ENGINEERING_METRIC_FIELDS.items()
        }
        return CandidateResult(
            candidate_id=str(
                arguments.get("candidate_id")
                or output.get("candidate_id")
                or f"candidate_{index}"
            ),
            tool_call_id=str(item.get("tool_call_id") or ""),
            action=self._candidate_action(arguments),
            failure_count=self._integer(verification.get("failure_count")),
            warning_count=self._integer(verification.get("warning_count")),
            risk_level=str(
                verification.get("risk_level") or output.get("risk_level") or "low"
            ),
            manual_intervention_label=str(
                verification.get("human_intervention_label")
                or output.get("manual_intervention_label")
                or "no_intervention"
            ),
            dispatch_recommendation=str(
                verification.get("dispatch_recommendation")
                or output.get("dispatch_recommendation")
                or ""
            ),
            failed_rule_ids=[str(value) for value in verification.get("failed_rule_ids") or []],
            warning_rule_ids=[str(value) for value in verification.get("warning_rule_ids") or []],
            energy_consumption=(
                number_value(metrics.get("energy_consumption_delta"))
                if metrics.get("energy_consumption_delta") is not None
                else number_value(metrics.get("energy_consumption"))
            ),
            nonzero_impacted_variable_count=(
                self._integer(comparison.get("nonzero_impacted_variable_count"))
                if comparison.get("nonzero_impacted_variable_count") is not None
                else None
            ),
            **{
                f"{category}_metrics": category_metrics[category]
                for category in ENGINEERING_METRIC_FIELDS
            },
            energy_metrics={
                "total": number_value(metrics.get("energy_consumption")),
                "delta_vs_baseline": number_value(metrics.get("energy_consumption_delta")),
                "unit": metrics.get("energy_unit"),
                "variable_count": self._integer(metrics.get("energy_variable_count")),
                "evaluation_status": metrics.get("energy_evaluation_status"),
            },
            baseline_reference=(
                str(metrics.get("baseline_reference"))
                if metrics.get("baseline_reference")
                else None
            ),
            category_status=dict(verification.get("category_status") or {}),
        )

    @staticmethod
    def _compact_category_metrics(value: Any, keys: Iterable[str]) -> Dict[str, Any]:
        source = dict(value or {})
        return {
            key: source[key]
            for key in keys
            if key in source and source[key] is not None
        }

    def _comparison_leaders(self, candidates: List[CandidateResult]) -> Dict[str, List[str]]:
        return {
            "pressure_preservation": self._leaders(
                candidates,
                lambda item: number_value(
                    nested_value(
                        item.pressure_metrics,
                        ("minimum_operating_window_margin", "value"),
                    )
                ),
                prefer="maximum",
            ),
            "slowest_linepack_decline": [
                item.candidate_id
                for item in self._linepack_best_candidates(candidates)
            ],
            "lowest_energy_consumption": self._leaders(
                candidates,
                lambda item: item.energy_consumption,
                prefer="minimum",
            ),
        }

    def _linepack_best_candidates(
        self, candidates: List[CandidateResult]
    ) -> List[CandidateResult]:
        values = [
            (
                item,
                (
                    number_value(
                        nested_value(
                            item.linepack_metrics,
                            ("maximum_decline_from_start", "value"),
                        )
                    ),
                    number_value(
                        item.linepack_metrics.get("maximum_continuous_decline_minutes")
                    ),
                    number_value(
                        item.linepack_metrics.get("insufficient_recovery_count")
                    ),
                ),
            )
            for item in candidates
        ]
        comparable = [
            (item, metrics)
            for item, metrics in values
            if all(value is not None for value in metrics)
        ]
        if not comparable:
            return []
        best = min(metrics for _, metrics in comparable)
        return [item for item, metrics in comparable if metrics == best]

    @staticmethod
    def _leaders(
        candidates: List[CandidateResult], value_getter, *, prefer: str
    ) -> List[str]:
        values = [(item, value_getter(item)) for item in candidates]
        values = [(item, value) for item, value in values if value is not None]
        if not values:
            return []
        target = (
            max(value for _, value in values)
            if prefer == "maximum"
            else min(value for _, value in values)
        )
        return [item.candidate_id for item, value in values if value == target]

    @staticmethod
    def _candidate_action(arguments: Dict[str, Any]) -> Dict[str, Any]:
        boundary = dict(arguments.get("boundary_conditions") or {})
        if boundary.get("setpoints") or boundary.get("percentage_changes"):
            return boundary
        if str(arguments.get("disturbance_source") or "").casefold() != (
            "operator_action"
        ):
            return boundary
        variable = arguments.get("disturbance_variable") or boundary.get(
            "disturbance_variable"
        )
        magnitude = arguments.get("disturbance_magnitude_percent")
        if magnitude is None:
            magnitude = boundary.get("disturbance_magnitude_percent")
        direction = str(
            arguments.get("disturbance_direction")
            or boundary.get("disturbance_direction")
            or ""
        ).casefold()
        if not variable or magnitude is None:
            return boundary
        signed = float(magnitude)
        if direction == "down":
            signed = -abs(signed)
        elif direction == "up":
            signed = abs(signed)
        return {"percentage_changes": {str(variable): signed}}

    def _decision(
        self,
        candidates: List[CandidateResult],
        *,
        question: str,
        decision_policy: Optional[Dict[str, Any]],
        require_decision_policy: bool,
    ) -> Dict[str, Any]:
        if require_decision_policy and decision_policy is None:
            return self._unsupported_decision(
                [],
                ["llm_decision_policy_tool_call"],
                ranking_policy={},
            )
        policy, policy_errors = normalize_decision_policy(decision_policy)
        if policy.get("source") == "llm_tool":
            normalized_question = " ".join(str(question).split()).casefold()
            objectives = list(policy.get("objectives") or [])
            legacy_excerpt = " ".join(
                str(policy.get("source_excerpt") or "").split()
            ).casefold()
            for objective, excerpt in zip(objectives, llm_policy_excerpts(policy)):
                item = dict(objective or {})
                if len(excerpt) < 4 or excerpt not in normalized_question:
                    if len(objectives) == 1 and legacy_excerpt:
                        policy_errors.append(
                            "decision_policy_source_not_in_user_request"
                        )
                    else:
                        policy_errors.append(
                            "decision_policy_objective_source_not_in_user_request:"
                            + str(item.get("metric") or "missing")
                        )
                elif not decision_policy_source_has_priority_signal(excerpt):
                    policy_errors.append(
                        "decision_policy_objective_not_a_priority:"
                        + str(item.get("metric") or "missing")
                    )
        eliminated = [
            {"candidate_id": item.candidate_id, "failed_rules": item.failed_rule_ids}
            for item in candidates
            if item.failure_count > 0
        ]
        viable = [item for item in candidates if item.failure_count == 0]
        if len(viable) < 1:
            return self._unsupported_decision(
                eliminated,
                ["constraint_compliant_candidate"],
                ranking_policy=policy,
            )

        if all(item.nonzero_impacted_variable_count == 0 for item in viable):
            return self._unsupported_decision(
                eliminated,
                ["action_sensitive_forecast"],
                ranking_policy=policy,
            )

        compact_viable = [item.compact() for item in viable]
        objective_evidence, missing_metrics = collect_objective_evidence(
            compact_viable,
            policy,
        )
        if policy_errors or missing_metrics:
            return self._unsupported_decision(
                eliminated,
                [*policy_errors, *missing_metrics],
                ranking_policy=policy,
                objective_evidence=objective_evidence,
            )

        ranked_groups = rank_candidate_groups(
            (item.candidate_id for item in viable),
            policy,
            objective_evidence,
        )
        ranked_ids = [
            candidate_id
            for group in ranked_groups
            for candidate_id in group
        ]
        by_id = {item.candidate_id: item for item in viable}
        selected = by_id[ranked_ids[0]]
        return {
            "status": "selected",
            "selected_candidate_id": selected.candidate_id,
            "selected_dispatch_recommendation": selected.dispatch_recommendation,
            "ranking_basis": objective_evidence[selected.candidate_id],
            "ranking_policy": policy,
            "objective_evidence": objective_evidence,
            "ranked_candidate_ids": ranked_ids,
            "ranked_candidate_groups": ranked_groups,
            "eliminated_candidates": eliminated,
            "missing_metrics": [],
        }

    @staticmethod
    def _unsupported_decision(
        eliminated: List[Dict[str, Any]],
        missing_metrics: List[str],
        *,
        ranking_policy: Optional[Dict[str, Any]] = None,
        objective_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "insufficient_evidence",
            "selected_candidate_id": None,
            "selected_dispatch_recommendation": "",
            "ranking_basis": {},
            "ranking_policy": ranking_policy or {},
            "objective_evidence": objective_evidence or {},
            "eliminated_candidates": eliminated,
            "missing_metrics": list(dict.fromkeys(missing_metrics)),
        }

    @staticmethod
    def _generic_contract(
        question: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        requested = requested_artifacts(question)
        sources = []
        for item in results:
            assessment = classify_tool_evidence(item, requested=requested)
            artifacts = sorted(assessment.matched_artifacts)
            if not artifacts:
                artifacts = sorted(requested_artifacts(str(item.get("arguments") or "")))
            if not artifacts:
                artifacts = [""]
            sources.extend(
                {
                    "tool_call_id": str(item.get("tool_call_id") or ""),
                    "artifact": artifact,
                    "state": assessment.state.value,
                    "reason": assessment.reason,
                }
                for artifact in artifacts
            )
        return {
            "answer_mode": "file_grounded" if requested else "generic_tool",
            "evidence_sources": sources,
        }

    @staticmethod
    def _worst(
        values: Iterable[str], ranking: Dict[str, int], default: str
    ) -> str:
        candidates = list(values)
        return max(candidates, key=lambda value: ranking.get(value, -1)) if candidates else default

    @staticmethod
    def _integer(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

def _policy_unavailable_at_generation(record: Dict[str, Any]) -> bool:
    """Preserve a deliberately unsupported saved policy decision."""

    if record.get("decision_policy"):
        return False
    stored_decision = dict(record.get("decision_summary") or {})
    if stored_decision.get("status") != "insufficient_evidence":
        return False
    return "llm_decision_policy_tool_call" in list(
        stored_decision.get("missing_metrics") or []
    )


def record_grounding_contract(
    record: Dict[str, Any],
    tool_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Rebuild a current-turn contract from tools and verified prior state."""

    outputs = (
        list(tool_outputs)
        if tool_outputs is not None
        else attach_tool_arguments(
            record.get("tool_outputs") or [],
            record.get("tool_calls") or [],
        )
    )
    state = VerifiedDecisionState.from_dict(dict(record.get("state_before") or {}))
    stored_policy = dict(record.get("decision_policy") or {}) or None
    return GroundingContractBuilder().build(
        str(record.get("user_input") or ""),
        outputs,
        decision_policy=stored_policy if not state.decision_policy else None,
        require_decision_policy=_policy_unavailable_at_generation(record),
        prior_candidate_results=state.candidates,
        prior_decision_policy=state.decision_policy,
        prior_decision_policy_source_question=state.decision_policy_source_question,
        prior_applied_disturbances=state.applied_disturbances,
    )


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    formatted = f"{number:.9g}"
    return format(Decimal(formatted), "f") if "e" in formatted.casefold() else formatted


_NUMBER_TOKEN = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?(?!\w)",
    re.IGNORECASE,
)


def _number_disclosed(answer: str, value: Any) -> bool:
    try:
        expected = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return format_number(value) in answer
    for match in _NUMBER_TOKEN.finditer(answer):
        token = match.group(0)
        try:
            reported = Decimal(token)
        except (InvalidOperation, ValueError):
            continue
        if reported == expected:
            return True
        mantissa = token.casefold().split("e", 1)[0]
        if "e" not in token.casefold() and "." in mantissa:
            precision = Decimal(1).scaleb(-len(mantissa.rsplit(".", 1)[1]))
            try:
                if expected.quantize(precision) == reported:
                    return True
            except InvalidOperation:
                continue
    return False


def _format_action(action: Dict[str, Any], chinese: bool) -> str:
    parts: List[str] = []
    for variable, value in dict(action.get("percentage_changes") or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            parts.append(f"{variable}={value}")
            continue
        if chinese:
            direction = "\u4e0a\u8c03" if number >= 0 else "\u4e0b\u8c03"
            parts.append(f"{variable} {direction} {format_number(abs(number))}%")
        else:
            direction = "increase" if number >= 0 else "decrease"
            parts.append(f"{direction} {variable} by {format_number(abs(number))}%")
    for variable, value in dict(action.get("setpoints") or {}).items():
        parts.append(f"{variable}={format_number(value)}")
    if parts:
        return ", ".join(parts)
    return "\u672a\u8bb0\u5f55" if chinese else "not recorded"


def _candidate_line(candidate: Dict[str, Any], chinese: bool) -> str:
    candidate_id = str(candidate.get("candidate_id") or "candidate")
    action = _format_action(dict(candidate.get("action") or {}), chinese)
    failed_rules = ", ".join(candidate.get("failed_rule_ids") or []) or "none"
    warning_rules = ", ".join(candidate.get("warning_rule_ids") or []) or "none"
    pressure = number_value(
        nested_value(
            dict(candidate.get("pressure_metrics") or {}),
            ("minimum_operating_window_margin", "value"),
        )
    )
    linepack = dict(candidate.get("linepack_metrics") or {})
    decline = number_value(
        nested_value(linepack, ("maximum_decline_from_start", "value"))
    )
    duration = linepack.get("maximum_continuous_decline_minutes")
    recovery = linepack.get("insufficient_recovery_count")
    energy = dict(candidate.get("energy_metrics") or {}).get("delta_vs_baseline")
    metrics = []
    if pressure is not None:
        metrics.append(
            f"\u538b\u88d5{format_number(pressure)}"
            if chinese
            else f"pressure margin {format_number(pressure)}"
        )
    if decline is not None:
        linepack_text = (
            f"\u7ba1\u5b58\u964d{format_number(decline)}"
            if chinese
            else f"linepack decline {format_number(decline)}"
        )
        if duration is not None:
            linepack_text += f"/{format_number(duration)}min"
        if recovery is not None:
            linepack_text += (
                f"/\u6062\u590d\u4e0d\u8db3{format_number(recovery)}"
                if chinese
                else f"/insufficient recovery {format_number(recovery)}"
            )
        metrics.append(linepack_text)
    if energy is not None:
        metrics.append(
            f"\u80fd\u8017\u0394{format_number(energy)}"
            if chinese
            else f"energy delta {format_number(energy)}"
        )
    metric_text = ("\uff1b" if chinese else "; ").join(metrics)
    metric_text = metric_text or ("\u65e0\u53ef\u6bd4\u6307\u6807" if chinese else "no comparable metrics")
    if chinese:
        return (
            f"- {candidate_id}\uff08{action}\uff09\uff1aF{candidate.get('failure_count', 0)}/"
            f"W{candidate.get('warning_count', 0)}\uff1b{metric_text}\uff1b"
            f"\u89c4\u5219{failed_rules if failed_rules != 'none' else warning_rules}\u3002"
        )

    return (
        f"- {candidate_id}: action {action}; "
        f"{candidate.get('failure_count', 0)} failures ({failed_rules}); "
        f"{candidate.get('warning_count', 0)} warnings ({warning_rules}); "
        f"risk {candidate.get('risk_level', 'unknown')}; "
        f"intervention {candidate.get('manual_intervention_label', 'unknown')}."
    )
def _outcome_key(candidate: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate.get("failure_count", 0),
        tuple(candidate.get("failed_rule_ids") or []),
        candidate.get("warning_count", 0),
        tuple(candidate.get("warning_rule_ids") or []),
        candidate.get("risk_level", "unknown"),
        candidate.get("manual_intervention_label", "unknown"),
    )


_CANDIDATE_REFERENCE = re.compile(r"\bcandidate_[A-Za-z0-9_-]+\b", re.IGNORECASE)
_SELECTED_CANDIDATE = re.compile(
    r"selected_candidate_id\s*[:=]\s*([A-Za-z][A-Za-z0-9_-]*|none|null)",
    re.IGNORECASE,
)


def _contract_action_variables(contract: Dict[str, Any]) -> set[str]:
    return {
        str(variable)
        for candidate in contract.get("candidate_results") or []
        for key in ("percentage_changes", "setpoints")
        for variable in dict(dict(candidate.get("action") or {}).get(key) or {})
        if str(variable)
    }


def _contains_bare_action_prefix(answer: str, action_variables: Iterable[str]) -> bool:
    """Detect `T_002` when the registered action ID is `T_002:SNQ`."""
    for variable in action_variables:
        if ":" not in variable:
            continue
        prefix = variable.split(":", 1)[0]
        pattern = (
            rf"(?<![A-Za-z0-9_]){re.escape(prefix)}"
            rf"(?![A-Za-z0-9_:])"
        )
        if re.search(pattern, answer, re.IGNORECASE):
            return True
    return False


def _canonical_number(value: Any) -> str:
    try:
        number = Decimal(str(value).strip())
    except Exception:
        return str(value)
    if not number.is_finite():
        return str(value)
    if number == 0:
        return "0"
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


_CANONICAL_DISCLOSURE_PREFIXES = (
    "Applied disturbance:",
    "Applied setpoint:",
    "Application status:",
)
_CANONICAL_ASSUMPTION_PREFIX = "Assumption source:"
_PROVISIONAL_ASSUMPTION_DISCLOSURE = re.compile(
    r"(?:假设|临时假设|暂按|暂定|暂设|本次按|LLM假设|LLM暂定|LLM暂设|LLM临时假设"
    r"|assum(?:e|ed|ption)|provisional)",
    re.IGNORECASE,
)
_NOT_EVALUATED_DISCLOSURE = "not_evaluated（未执行该项校核，不能判定 pass/fail）"
_AMBIGUOUS_NOT_EVALUATED = re.compile(
    r"(?<![A-Za-z0-9_])not_evaluated(?![A-Za-z0-9_])(?:"
    r"\s*[（(][^）)\r\n]{0,60}未通过[^）)\r\n]{0,60}[）)]"
    r"|[，,]\s*(?:即|属|视为|按)\s*[“\"]?未通过"
    r"(?:校核项|校核|评估|处理|对待|项)?[”\"]?"
    r"(?:\s*(?:处理|看待|对待))?[）)]?"
    r")"
)


def normalize_not_evaluated_wording(answer: str) -> str:
    """Keep not-evaluated evidence distinct from an actual failed check."""
    return _AMBIGUOUS_NOT_EVALUATED.sub(_NOT_EVALUATED_DISCLOSURE, answer)


def _canonical_sequence_value(values: Any, fallback: Any) -> str:
    items = list(values or [])
    if not items:
        return _canonical_number(fallback)
    rendered = [_canonical_number(item) for item in items]
    if len(set(rendered)) == 1:
        return rendered[0]
    return json.dumps(rendered, ensure_ascii=False, separators=(",", ":"))


def canonical_applied_disturbance_lines(
    contract: Dict[str, Any],
) -> List[str]:
    lines: List[str] = []
    seen = set()
    for raw_item in contract.get("applied_disturbances") or []:
        item = dict(raw_item or {})
        variable = str(item.get("variable") or "")
        mode = str(item.get("mode") or "")
        if not variable or not mode or item.get("verified") is not True:
            continue
        requested = item.get("requested_value")
        if mode == "percent_change":
            try:
                number = Decimal(str(requested).strip())
            except Exception:
                number = None
            direction = str(item.get("direction") or "").casefold()
            if number is not None and number.is_finite():
                if direction == "down":
                    number = -abs(number)
                elif direction == "up":
                    number = abs(number)
                value = _canonical_number(number)
                if number > 0:
                    value = f"+{value}"
            else:
                value = _canonical_number(requested)
            primary = f"Applied disturbance: {variable}={value}%"
        elif mode == "setpoint":
            primary = (
                f"Applied setpoint: {variable}="
                f"{_canonical_number(requested)}"
            )
        else:
            primary = (
                f"Applied disturbance: {variable}="
                f"{_canonical_number(requested)}"
            )
        if primary not in seen:
            seen.add(primary)
            lines.append(primary)

        if item.get("verified") is True and item.get("no_op") is True:
            prior = _canonical_sequence_value(
                item.get("input_values_before"), requested
            )
            applied = _canonical_sequence_value(
                item.get("input_values_applied"), requested
            )
            status = (
                f"Application status: {variable}=no-op; "
                f"prior={prior}; applied={applied}"
            )
            if status not in seen:
                seen.add(status)
                lines.append(status)
    return lines


def _canonical_disclosure_lines_in_answer(answer: str) -> List[str]:
    return [
        line.strip()
        for line in answer.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip().startswith(_CANONICAL_DISCLOSURE_PREFIXES)
    ]


def answer_without_machine_disclosure(answer: str) -> str:
    """Return natural answer prose without deterministic metadata lines."""
    return "\n".join(
        line
        for line in answer.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if not line.strip().startswith(
            (*_CANONICAL_DISCLOSURE_PREFIXES, _CANONICAL_ASSUMPTION_PREFIX)
        )
    ).strip()


def _without_embedded_required_disclosures(
    prose: str,
    required_lines: Sequence[str],
) -> str:
    cleaned: List[str] = []
    for line in prose.split("\n"):
        positions = [
            line.find(required)
            for required in required_lines
            if required and required in line
        ]
        if positions:
            line = line[: min(positions)].rstrip()
        if line.strip():
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def _canonical_assumption_source_lines(
    contract: Dict[str, Any],
) -> List[str]:
    assignments: Dict[str, str] = {}
    for line in canonical_applied_disturbance_lines(contract):
        if not line.startswith(("Applied disturbance:", "Applied setpoint:")):
            continue
        assignment = line.split(":", 1)[1].strip()
        variable = assignment.split("=", 1)[0]
        assignments.setdefault(variable, assignment)

    lines: List[str] = []
    for raw_item in contract.get("provisional_assumptions") or []:
        item = dict(raw_item or {})
        variable = str(item.get("variable") or "")
        assignment = assignments.get(variable)
        if not assignment:
            continue
        line = (
            f"{_CANONICAL_ASSUMPTION_PREFIX} "
            f"LLM provisional; {assignment}"
        )
        if line not in lines:
            lines.append(line)
    return lines


def provisional_assumption_disclosed(
    answer: str,
    pipeformer: Dict[str, Any],
    numeric_values: List[float],
) -> bool:
    """Check canonical disclosures while retaining legacy teacher-trace wording."""
    forecasts = [pipeformer] + [item for item in pipeformer.get("candidate_forecasts") or [] if isinstance(item, dict)]
    assumptions = GroundingContractBuilder._provisional_assumptions(
        [{"output": forecast} for forecast in forecasts]
    )
    magnitudes = list(dict.fromkeys(
        abs(float(item["magnitude_percent"]))
        for item in assumptions
        if item.get("magnitude_percent") is not None
    ))
    if not magnitudes:
        return True
    disclosed = any(
        line.strip().startswith(f"{_CANONICAL_ASSUMPTION_PREFIX} LLM provisional;")
        for line in answer.splitlines()
    ) or _PROVISIONAL_ASSUMPTION_DISCLOSURE.search(answer)
    return bool(disclosed) and all(
        any(abs(abs(value) - expected) < 1e-6 for value in numeric_values)
        for expected in magnitudes
    )


def finalize_applied_disturbance_disclosure(
    answer: str,
    contract: Dict[str, Any],
) -> str:
    """Serialize the canonical application block without rewriting prose."""
    required = [
        *canonical_applied_disturbance_lines(contract),
        *_canonical_assumption_source_lines(contract),
    ]
    prose = normalize_not_evaluated_wording(
        answer_without_machine_disclosure(answer)
    )
    prose = _without_embedded_required_disclosures(prose, required)
    disclosure = "\n".join(required)
    if disclosure and prose:
        return f"{disclosure}\n{prose}"
    return disclosure or prose


def comparison_requirements_active(
    answer: str,
    contract: Dict[str, Any],
) -> bool:
    """Escalate comparison validation only for operational turns or claims."""
    return bool(
        int(contract.get("current_candidate_forecast_count") or 0) > 0
        or int(contract.get("current_decision_policy_call_count") or 0) > 0
        or _SELECTED_CANDIDATE.search(answer)
        or re.search(
            r"\bcandidate[_-]?\d+\s*(?:>|<|>=|<=)\s*candidate[_-]?\d+\b",
            answer,
            re.IGNORECASE,
        )
        or re.search(
            r"(?:recommend|select|choose|建议|推荐|选择).{0,24}\bcandidate[_-]?\d+\b",
            answer,
            re.IGNORECASE,
        )
    )


_AUDIT_CATEGORIES = (
    ("pressure", "pressure_metrics", "压力", "pressure", r"压力|压裕|pressure"),
    ("flow", "flow_metrics", "流量", "flow", r"流量|供需|flow|supply.?demand"),
    ("linepack", "linepack_metrics", "管存", "linepack", r"管存|linepack"),
    ("compressor", "compressor_metrics", "压缩机", "compressor", r"压缩机|负荷|compressor|load"),
    ("energy", "energy_metrics", "能耗", "energy", r"能耗|energy"),
)


def comparison_answer_issues(answer: str, contract: Dict[str, Any]) -> List[str]:
    """Validate semantic comparison coverage and the typed selection claim."""
    action_variables = _contract_action_variables(contract)
    issues: List[str] = []
    if _contains_bare_action_prefix(answer, action_variables):
        issues.append("canonical_action_variable_abbreviated")
    required_disclosure = canonical_applied_disturbance_lines(contract)
    actual_disclosure = _canonical_disclosure_lines_in_answer(answer)
    normalized_answer_lines = [
        line.strip()
        for line in answer.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]
    required_primary = [
        line
        for line in required_disclosure
        if not line.startswith("Application status:")
    ]
    required_status = [
        line
        for line in required_disclosure
        if line.startswith("Application status:")
    ]
    if any(line not in actual_disclosure for line in required_primary):
        issues.append("applied_disturbance_disclosure_missing")
    if any(line not in actual_disclosure for line in required_status):
        issues.append("disturbance_no_op_disclosure_missing")
    disclosure_occurrences = sum(
        answer.count(line)
        for line in required_disclosure
    )
    if (
        actual_disclosure != required_disclosure
        and actual_disclosure
    ) or disclosure_occurrences != len(required_disclosure):
        issues.append("unexpected_applied_disturbance_disclosure")
    if (
        required_disclosure
        and normalized_answer_lines[: len(required_disclosure)]
        != required_disclosure
    ):
        issues.append("canonical_disclosure_block_not_at_start")
    if contract.get("answer_mode") != "dispatch_comparison":
        return list(dict.fromkeys(issues))
    if not comparison_requirements_active(answer, contract):
        return list(dict.fromkeys(issues))
    candidates = [
        str(item.get("candidate_id") or "")
        for item in contract.get("candidate_results") or []
        if item.get("candidate_id")
    ]
    selection = _SELECTED_CANDIDATE.search(answer)
    comparison_text = _SELECTED_CANDIDATE.sub("", answer)
    answer_folded = comparison_text.casefold()
    if any(candidate.casefold() not in answer_folded for candidate in candidates):
        issues.append("candidate_comparison_incomplete")

    known = {candidate.casefold() for candidate in candidates}
    referenced = {match.group(0).casefold() for match in _CANDIDATE_REFERENCE.finditer(answer)}
    if referenced - known:
        issues.append("unknown_candidate_reference")

    if selection is None:
        issues.append("candidate_selection_missing")
    else:
        actual = selection.group(1).casefold()
        expected = str(
            (contract.get("decision_summary") or {}).get("selected_candidate_id")
            or "none"
        ).casefold()
        if actual == "null":
            actual = "none"
        if actual != expected:
            issues.append("candidate_selection_contradicts_contract")

    decision = dict(contract.get("decision_summary") or {})
    if decision.get("status") == "selected":
        objective_evidence = dict(decision.get("objective_evidence") or {})
        missing_objective_values = []
        for candidate_id, metrics in objective_evidence.items():
            if candidate_id.casefold() not in answer_folded:
                continue
            for metric, evidence in dict(metrics or {}).items():
                expected_value = dict(evidence or {}).get("value")
                if expected_value is not None and not _number_disclosed(answer, expected_value):
                    missing_objective_values.append(f"{candidate_id}:{metric}")
        if missing_objective_values:
            issues.append("decision_objective_evidence_incomplete")

        if any(variable.casefold() not in answer_folded for variable in action_variables):
            issues.append("candidate_action_mapping_incomplete")

        present_categories = {
            category
            for candidate in contract.get("candidate_results") or []
            for category, key, *_ in _AUDIT_CATEGORIES
            if dict(candidate.get(key) or {})
        }
        if any(
            not re.search(pattern, answer, re.IGNORECASE)
            for category, _, _, _, pattern in _AUDIT_CATEGORIES
            if category in present_categories
        ):
            issues.append("candidate_audit_evidence_incomplete")

        if not re.search(
            r"(?:F|失败|failure)\s*[:=]?\s*0|failure_count\s*[:=]\s*0|无(?:规则)?失败|"
            r"硬约束(?:均|全部|全都|已)?(?:通过|满足)|"
            r"(?:全部|所有|均).{0,16}(?<![未不])(?:通过|满足).{0,12}硬约束|"
            r"hard constraints?\s*(?:(?:all|are|were|have been)\s*)*"
            r"(?:pass(?:ed)?|satisf(?:ied|y))|"
            r"(?:all|every)\s+hard constraints?\s+"
            r"(?:pass(?:ed)?|are\s+satisfied)",
            answer,
            re.IGNORECASE,
        ):
            issues.append("hard_constraint_outcome_missing")
        if len(candidates) > 1 and not re.search(
            r"次优|未选|拒选|淘汰|lower-ranked|not selected|eliminated|rejected",
            answer,
            re.IGNORECASE,
        ):
            issues.append("candidate_rejection_reason_missing")
    if contract.get("answer_render_status") == "answer_budget_insufficient":
        issues.append("answer_budget_insufficient")
    return list(dict.fromkeys(issues))


def _finalize_comparison_answer(
    lines: List[str],
    selected_candidate_id: str,
    contract: Dict[str, Any],
    *,
    maximum_chars: int = 500,
    maximum_words: Optional[int] = None,
) -> str:
    body = "\n".join(str(line).strip() for line in lines if str(line).strip())
    answer = body + f"\nselected_candidate_id: {selected_candidate_id or 'none'}"
    over_budget = len(answer) > maximum_chars
    if maximum_words is not None and len(answer.split()) > maximum_words:
        over_budget = True
    if over_budget:
        contract["answer_render_status"] = "answer_budget_insufficient"
    else:
        contract.pop("answer_render_status", None)
    return answer


def _shared_outcome_line(candidate: Dict[str, Any], chinese: bool) -> str:
    failed_rules = ", ".join(candidate.get("failed_rule_ids") or []) or "none"
    warning_rules = ", ".join(candidate.get("warning_rule_ids") or []) or "none"
    if chinese:
        return (
            f"\u5171\u540c\u6821\u6838\u7ed3\u679c\uff1a\u5931\u8d25 {candidate.get('failure_count', 0)}\uff08{failed_rules}\uff09\uff1b"
            f"\u544a\u8b66 {candidate.get('warning_count', 0)}\uff08{warning_rules}\uff09\uff1b"
            f"\u98ce\u9669 {candidate.get('risk_level', 'unknown')}\uff1b"
            f"\u4eba\u5de5\u5e72\u9884 {candidate.get('manual_intervention_label', 'unknown')}\u3002"
        )
    return (
        f"Shared result: {candidate.get('failure_count', 0)} failures ({failed_rules}); "
        f"{candidate.get('warning_count', 0)} warnings ({warning_rules}); "
        f"risk {candidate.get('risk_level', 'unknown')}; "
        f"intervention {candidate.get('manual_intervention_label', 'unknown')}."
    )


def _compact_action(action: Dict[str, Any]) -> str:
    parts: List[str] = []
    for variable, raw_value in dict(action.get("percentage_changes") or {}).items():
        try:
            value = float(raw_value)
            parts.append(f"{variable}{value:+g}%")
        except (TypeError, ValueError):
            parts.append(f"{variable}={raw_value}")
    for variable, raw_value in dict(action.get("setpoints") or {}).items():
        parts.append(f"{variable}={format_number(raw_value)}")
    return ",".join(parts) or "未记录"


def applied_disturbance_disclosure(
    question: str,
    contract: Dict[str, Any],
) -> str:
    """Return the exact machine-verifiable application evidence block."""
    del question
    return "\n".join(canonical_applied_disturbance_lines(contract))


def _compact_assumption_lines(
    contract: Dict[str, Any], chinese: bool
) -> List[str]:
    lines: List[str] = []
    seen = set()
    for item in contract.get("provisional_assumptions") or []:
        variable = str(item.get("variable") or "disturbance")
        direction = str(item.get("direction") or "unknown")
        magnitude = item.get("magnitude_percent")
        setpoint = item.get("setpoint")
        key = (
            variable.casefold(),
            direction.casefold(),
            format_number(magnitude),
            format_number(setpoint),
        )
        if key in seen:
            continue
        seen.add(key)
        if variable.endswith(":ST") and setpoint is not None:
            if chinese:
                lines.append(f"LLM暂设：{variable}={format_number(setpoint)}。")
            else:
                lines.append(
                    f"Provisional LLM assumption: {variable}={format_number(setpoint)}."
                )
            continue
        if chinese:
            direction_label = {"up": "上调", "down": "下调"}.get(direction.casefold(), direction)
            if magnitude is None:
                lines.append(f"LLM临时假设：{variable}{direction_label}。")
            else:
                lines.append(f"LLM临时假设：{variable}{direction_label}{format_number(magnitude)}%。")
        else:
            if magnitude is None:
                lines.append(f"Provisional LLM assumption: {variable} {direction}.")
            else:
                lines.append(
                    f"Provisional LLM assumption: {variable} {direction} {format_number(magnitude)}%."
                )
    return lines


def _objective_label(objective: Dict[str, Any], chinese: bool) -> str:
    label = str(
        objective.get("label_zh" if chinese else "label_en")
        or objective.get("metric")
        or "metric"
    )
    proxy = str(objective.get("proxy_for") or "")
    if proxy:
        if chinese:
            proxy_label = "末压代理" if proxy == "terminal_pressure_preservation" else "代理"
            label += f"({proxy_label})"
        else:
            label += f" (proxy for {proxy.replace('_', ' ')})"
    return label


def _objective_token(
    objective: Dict[str, Any],
    evidence: Dict[str, Any],
    chinese: bool,
) -> str:
    label = _objective_label(objective, chinese)
    value = format_number(evidence.get("value"))
    variable = str(evidence.get("variable") or "")
    unit = str(evidence.get("unit") or "")
    detail = value
    if variable:
        detail += f"({variable})"
    if unit:
        detail += f" {unit}"
    return f"{label}={detail}"


def _audit_category_summary(
    candidates: List[Dict[str, Any]],
    chinese: bool,
) -> str:
    present = [
        (category, zh, en)
        for category, key, zh, en, _ in _AUDIT_CATEGORIES
        if any(dict(candidate.get(key) or {}) for candidate in candidates)
    ]
    labels = [zh if chinese else en for _, zh, en in present]
    statuses = {
        str(dict(candidate.get("category_status") or {}).get(category) or "evidence")
        for candidate in candidates
        for category, _, _ in present
    }
    status = next(iter(statuses)) if len(statuses) == 1 else "mixed"
    if chinese:
        status_label = {"evidence": "有据", "mixed": "状态见候选"}.get(status, status)
        return f"共同审核：{'/'.join(labels)}均{status_label}。"
    status_label = {"evidence": "evidenced", "mixed": "candidate-specific"}.get(
        status,
        status,
    )
    return f"Shared audit: {'/'.join(labels)}={status_label}."


def _selected_comparison_answer(
    question: str,
    contract: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    decision: Dict[str, Any],
) -> str:
    chinese = is_chinese(question)
    ranking = [str(value) for value in decision.get("ranked_candidate_ids") or []]
    by_id = {str(item.get("candidate_id") or ""): item for item in candidates}
    ranked = [by_id[candidate_id] for candidate_id in ranking if candidate_id in by_id]
    ranked.extend(item for item in candidates if item not in ranked)
    policy = dict(decision.get("ranking_policy") or {})
    objectives = [dict(item) for item in policy.get("objectives") or []]
    objective_evidence = dict(decision.get("objective_evidence") or {})
    selected_id = str(decision.get("selected_candidate_id") or "none")
    direction_labels = {"minimize": "↓", "maximize": "↑"}
    policy_line = (
        ("目标：" if chinese else "Objectives: ")
        + " > ".join(
            _objective_label(objective, chinese)
            + direction_labels.get(str(objective.get("direction") or ""), "")
            for objective in objectives
        )
        + ("；硬约束F=0。" if chinese else "; hard constraint: zero failures.")
    )
    candidate_lines: List[str] = []
    for candidate in ranked:
        candidate_id = str(candidate.get("candidate_id") or "")
        action = _compact_action(dict(candidate.get("action") or {}))
        evidence = dict(objective_evidence.get(candidate_id) or {})
        metric_tokens = [
            _objective_token(objective, dict(evidence.get(objective["metric"]) or {}), chinese)
            for objective in objectives
        ]
        if chinese:
            candidate_lines.append(
                f"{candidate_id}[{action}]：F{candidate.get('failure_count', 0)}/"
                f"W{candidate.get('warning_count', 0)}；{'；'.join(metric_tokens)}。"
            )
        else:
            candidate_lines.append(
                f"{candidate_id} [{action}]: F{candidate.get('failure_count', 0)}/"
                f"W{candidate.get('warning_count', 0)}; {'; '.join(metric_tokens)}."
            )

    ranked_groups = [
        [str(candidate_id) for candidate_id in group if str(candidate_id) in by_id]
        for group in decision.get("ranked_candidate_groups") or []
    ]
    ranked_groups = [group for group in ranked_groups if group]
    if not ranked_groups:
        ranked_groups = [[candidate_id] for candidate_id in ranking]
    ranking_text = ">".join("=".join(group) for group in ranked_groups)
    ranking_line = (
        ("排序：" if chinese else "Ranking: ")
        + ranking_text
        + ("。" if chinese else ".")
    )
    eliminated = list(decision.get("eliminated_candidates") or [])
    eliminated_ids = {
        str(item.get("candidate_id") or "") for item in eliminated
    }
    rejected = [
        candidate_id
        for candidate_id in ranking
        if candidate_id != selected_id and candidate_id not in eliminated_ids
    ]
    selected_group = next(
        (group for group in ranked_groups if selected_id in group),
        [selected_id],
    )
    tied_with_selected = [
        candidate_id
        for candidate_id in rejected
        if candidate_id in selected_group
    ]
    lower_ranked = [
        candidate_id
        for candidate_id in rejected
        if candidate_id not in selected_group
    ]
    eliminated_text = ";".join(
        f"{item.get('candidate_id')}={','.join(item.get('failed_rules') or ['rule_failure'])}"
        for item in eliminated
    )
    if chinese:
        rejection_parts = [f"淘汰：{eliminated_text}" if eliminated_text else "淘汰：无"]
        if lower_ranked:
            rejection_parts.append(f"未选：{','.join(lower_ranked)}仅因目标次优")
        if tied_with_selected:
            rejection_parts.append(
                f"并列未选：{','.join(tied_with_selected)}与首选目标相同，"
                "仅按候选ID稳定排序"
            )
        rejection_line = "；".join(rejection_parts) + "。"
    else:
        rejection_parts = [
            f"Eliminated: {eliminated_text}" if eliminated_text else "Eliminated: none"
        ]
        if lower_ranked:
            rejection_parts.append(
                "not selected because lower-ranked on objectives: "
                + ", ".join(lower_ranked)
            )
        if tied_with_selected:
            rejection_parts.append(
                "tied with the selected candidate on requested objectives; "
                "stable candidate-ID tie-break: "
                + ", ".join(tied_with_selected)
            )
        rejection_line = "; ".join(rejection_parts) + "."

    lines = [
        *canonical_applied_disturbance_lines(contract),
        *_compact_assumption_lines(contract, chinese),
        policy_line,
        *candidate_lines,
        _audit_category_summary(ranked, chinese),
        ranking_line,
        rejection_line,
    ]
    return _finalize_comparison_answer(
        lines,
        selected_id,
        contract,
        maximum_chars=(
            chinese_comparison_max_chars(len(ranked))
            if chinese
            else ENGLISH_COMPARISON_MAX_CHARS
        ),
        maximum_words=None if chinese else ENGLISH_MAX_WORDS,
    )

def grounded_fallback_answer(question: str, contract: Dict[str, Any]) -> str:
    """Render a compact, instruction-complete answer from deterministic facts."""
    decision = dict(contract.get("decision_summary") or {})
    candidates = list(contract.get("candidate_results") or [])
    chinese = is_chinese(question)
    ranking = [str(value) for value in decision.get("ranked_candidate_ids") or []]
    if ranking:
        positions = {candidate_id: index for index, candidate_id in enumerate(ranking)}
        candidates.sort(
            key=lambda item: positions.get(
                str(item.get("candidate_id") or ""), len(positions)
            )
        )
    if decision.get("status") == "selected":
        return _selected_comparison_answer(question, contract, candidates, decision)

    assumption_lines = [
        *canonical_applied_disturbance_lines(contract),
        *_compact_assumption_lines(contract, chinese),
    ]
    shared_outcome = not ranking and len(candidates) > 1 and len({
        _outcome_key(candidate) for candidate in candidates
    }) == 1
    lines = [
        *assumption_lines,
        (
            "\u5019\u9009\u52a8\u4f5c\uff1a" if chinese else "Candidate actions:"
        ) if shared_outcome else (
            "\u5019\u9009\u52a8\u4f5c\u6bd4\u8f83\uff1a" if chinese else "Candidate comparison:"
        ),
    ]
    if shared_outcome:
        lines.extend(
            f"- {candidate.get('candidate_id')}: "
            f"{_format_action(dict(candidate.get('action') or {}), chinese)}"
            for candidate in candidates
        )
        lines.append(_shared_outcome_line(candidates[0], chinese))
    else:
        lines.extend(_candidate_line(candidate, chinese) for candidate in candidates)
    eliminated = list(decision.get("eliminated_candidates") or [])
    missing = [str(value) for value in decision.get("missing_metrics") or []]

    all_eliminated = bool(candidates) and len(eliminated) == len(candidates)
    failed_rules = sorted({
        str(rule)
        for item in eliminated
        for rule in item.get("failed_rules") or []
    })
    if chinese:
        conclusion = (
            "\u7ed3\u8bba\uff1a\u6ca1\u6709\u6ee1\u8db3\u7ea6\u675f\u7684\u5019\u9009\u52a8\u4f5c\uff0c"
            "\u5f53\u524d\u4e0d\u80fd\u8fdb\u5165\u6267\u884c\u9636\u6bb5\u3002"
            if all_eliminated
            else "\u7ed3\u8bba\uff1a\u73b0\u6709\u53ef\u6bd4\u8bc1\u636e\u4e0d\u8db3\uff0c\u4e0d\u80fd\u9009\u51fa\u6392\u540d\u7b2c\u4e00\u7684\u52a8\u4f5c\u3002"
        )
        conditions = []
        if failed_rules:
            conditions.append(
                "\u5fc5\u987b\u5148\u8c03\u6574\u52a8\u4f5c\u5e76\u6d88\u9664\u4e0a\u8ff0\u5931\u8d25\u89c4\u5219"
                if shared_outcome
                else "\u5fc5\u987b\u5148\u8c03\u6574\u52a8\u4f5c\u5e76\u6d88\u9664\u5931\u8d25\u89c4\u5219 "
                + ", ".join(failed_rules)
            )
        if missing and not all_eliminated:
            conditions.append("\u9700\u8865\u5145\u53ef\u6bd4\u8bc1\u636e " + ", ".join(missing))
        lines.extend([
            conclusion,
            "\u9002\u7528\u524d\u63d0\uff1a" + "\uff1b".join(conditions) + "\u3002",
        ])
    else:
        conclusion = (
            "Conclusion: no candidate satisfies the constraints, so none can proceed to execution."
            if all_eliminated
            else "Conclusion: the comparable evidence is insufficient to select a first-ranked action."
        )
        conditions = []
        if failed_rules:
            conditions.append(
                "adjust the actions and clear the failed rules above"
                if shared_outcome
                else "adjust the actions and clear failed rules " + ", ".join(failed_rules)
            )
        if missing and not all_eliminated:
            conditions.append("provide comparable evidence " + ", ".join(missing))
        lines.extend([conclusion, "Applicability: " + "; ".join(conditions) + "."])
    return _finalize_comparison_answer(
        lines,
        "none",
        contract,
        maximum_chars=(
            chinese_comparison_max_chars(len(candidates))
            if chinese
            else ENGLISH_COMPARISON_MAX_CHARS
        ),
        maximum_words=None if chinese else ENGLISH_MAX_WORDS,
    )
