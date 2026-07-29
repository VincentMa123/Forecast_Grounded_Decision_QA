from __future__ import annotations

from dataclasses import dataclass
import json
import re
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional

from .answer_limits import (
    CHINESE_SINGLE_FORECAST_MAX_CHARS,
    ENGLISH_COMPARISON_MAX_CHARS,
    ENGLISH_MAX_WORDS,
    GENERIC_MAX_CHARS,
    chinese_comparison_max_chars,
)
from .decision_policy import (
    collect_objective_evidence,
    normalize_decision_policy,
    rank_candidate_groups,
)
from .tool_evidence import attach_tool_arguments, classify_tool_evidence, requested_artifacts


RISK_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
INTERVENTION_RANK = {
    "no_intervention": 0,
    "monitoring_only": 1,
    "operator_attention_required": 2,
    "immediate_intervention_required": 3,
}

_DECISION_PRIORITY_SIGNAL = re.compile(
    r"(?:\b(?:priority|prioritize|first|primary|secondary|most|least|"
    r"focus|reduce|increase|maintain|preserve|avoid|minimi[sz]e|maximi[sz]e)\b"
    r"|优先|首先|第一|最(?:大|小|低|高|少|多)|重点|关注|降低|减少|提高|增加|保持|避免)",
    re.IGNORECASE,
)


def decision_policy_source_has_priority_signal(source_excerpt: str) -> bool:
    """Return whether a source excerpt actually expresses a preference."""
    return bool(_DECISION_PRIORITY_SIGNAL.search(str(source_excerpt or "")))
@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    tool_call_id: str
    action: Dict[str, Any]
    prediction_summary: Dict[str, Any]
    constraint_check: Dict[str, Any]
    evidence: Dict[str, Any]
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
        return {
            "candidate_id": self.candidate_id,
            "tool_call_id": self.tool_call_id,
            "action": self.action,
            "failure_count": self.failure_count,
            "warning_count": self.warning_count,
            "risk_level": self.risk_level,
            "manual_intervention_label": self.manual_intervention_label,
            "dispatch_recommendation": self.dispatch_recommendation,
            "failed_rule_ids": self.failed_rule_ids,
            "warning_rule_ids": self.warning_rule_ids,
            "energy_consumption": self.energy_consumption,
            "nonzero_impacted_variable_count": self.nonzero_impacted_variable_count,
            "pressure_metrics": self.pressure_metrics,
            "linepack_metrics": self.linepack_metrics,
            "flow_metrics": self.flow_metrics,
            "compressor_metrics": self.compressor_metrics,
            "energy_metrics": self.energy_metrics,
            "baseline_reference": self.baseline_reference,
            "category_status": self.category_status,
            "elimination_reasons": self.failed_rule_ids,
        }

    @classmethod
    def from_compact(cls, value: Dict[str, Any]) -> "CandidateResult":
        item = dict(value or {})
        return cls(
            candidate_id=str(item.get("candidate_id") or ""),
            tool_call_id=str(item.get("tool_call_id") or ""),
            action=dict(item.get("action") or {}),
            prediction_summary={},
            constraint_check={},
            evidence={},
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
        current_policy = self._decision_policy_from_results(results)
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
        pipeformer = [
            item for item in results
            if item.get("name") == "run_pipeformer_forecast"
            and dict(item.get("output") or {}).get("success") is True
        ]
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
        baselines = self._deduplicate_candidate_results([
            item
            for item in results
            if str(
                dict(item.get("output") or {}).get("candidate_role")
                or dict(item.get("arguments") or {}).get("candidate_role")
                or ""
            ).casefold()
            == "baseline"
        ])
        candidates = self._deduplicate_candidate_results(
            [
                item
                for item in results
                if str(
                    dict(item.get("output") or {}).get("candidate_role")
                    or dict(item.get("arguments") or {}).get("candidate_role")
                    or ""
                ).casefold()
                != "baseline"
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
                and not self._candidate_has_boundary_action(candidate)
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
            key = json.dumps(
                disturbance,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if key in applied_seen:
                continue
            applied_seen.add(key)
            applied_disturbances.append(dict(disturbance))
        if applied_disturbances:
            contract["applied_disturbances"] = applied_disturbances
        return contract

    @staticmethod
    def _decision_policy_from_results(
        results: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        for item in reversed(results):
            output = dict(item.get("output") or {})
            if (
                item.get("name") == "set_decision_policy"
                and output.get("success") is True
                and isinstance(output.get("decision_policy"), dict)
            ):
                return dict(output["decision_policy"])
        return None

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
                "action:"
                + json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
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
            prediction = dict(
                output.get("prediction") or output.get("prediction_summary") or {}
            )
            parsed_task = dict(output.get("parsed_task") or {})
            sources = [parsed_task, prediction]
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
            variable = str(
                arguments.get("disturbance_variable")
                or dict(output.get("prediction") or {}).get(
                    "disturbance_variable"
                )
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
                    or dict(output.get("prediction") or {}).get(
                        "disturbance_direction"
                    )
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
        prediction = dict(output.get("prediction") or output.get("prediction_summary") or {})
        verification = dict(output.get("verification") or output.get("constraint_check") or {})
        evidence = dict(output.get("evidence") or {})
        arguments = dict(item.get("arguments") or {})
        metrics = dict(verification.get("comparable_metrics") or {})
        engineering = dict(verification.get("engineering_evidence") or {})
        comparison = dict(prediction.get("counterfactual_comparison") or {})
        return CandidateResult(
            candidate_id=str(
                arguments.get("candidate_id")
                or output.get("candidate_id")
                or f"candidate_{index}"
            ),
            tool_call_id=str(item.get("tool_call_id") or ""),
            action=self._candidate_action(arguments),
            prediction_summary=prediction,
            constraint_check=verification,
            evidence=evidence,
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
                self._number(metrics.get("energy_consumption_delta"))
                if metrics.get("energy_consumption_delta") is not None
                else self._number(metrics.get("energy_consumption"))
            ),
            nonzero_impacted_variable_count=(
                self._integer(comparison.get("nonzero_impacted_variable_count"))
                if comparison.get("nonzero_impacted_variable_count") is not None
                else None
            ),
            pressure_metrics=self._compact_category_metrics(
                engineering.get("pressure"),
                (
                    "minimum_pressure",
                    "maximum_pressure",
                    "minimum_lower_bound_margin",
                    "minimum_upper_bound_margin",
                    "minimum_operating_window_margin",
                    "violation_node_count",
                    "warning_node_count",
                    "maximum_continuous_pressure_violation_minutes",
                ),
            ),
            linepack_metrics=self._compact_category_metrics(
                engineering.get("linepack"),
                (
                    "minimum_linepack",
                    "maximum_decline_from_start",
                    "maximum_continuous_decline_minutes",
                    "minimum_peak_shaving_reserve",
                    "insufficient_recovery_count",
                    "linepack_warning_status",
                ),
            ),
            flow_metrics=self._compact_category_metrics(
                engineering.get("flow"),
                (
                    "maximum_segment_flow_change",
                    "maximum_boundary_flow_change_rate",
                    "flow_capacity_excursion_count",
                    "supply_demand_balance_status",
                    "supply_demand_balance",
                ),
            ),
            compressor_metrics=self._compact_category_metrics(
                engineering.get("compressor"),
                (
                    "operating_envelope_status",
                    "maximum_load",
                    "maximum_compression_ratio",
                    "maximum_rotational_speed",
                    "maximum_power_change",
                ),
            ),
            energy_metrics={
                "total": self._number(metrics.get("energy_consumption")),
                "delta_vs_baseline": self._number(metrics.get("energy_consumption_delta")),
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
                lambda item: self._nested_number(
                    item.pressure_metrics, "minimum_operating_window_margin", "value"
                ),
                prefer="maximum",
            ),
            "slowest_linepack_decline": self._leaders(
                self._linepack_best_candidates(candidates),
                lambda item: 0.0,
                prefer="minimum",
            ),
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
                    self._nested_number(
                        item.linepack_metrics,
                        "maximum_decline_from_start",
                        "value",
                    ),
                    self._number(
                        item.linepack_metrics.get("maximum_continuous_decline_minutes")
                    ),
                    self._number(
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
    def _nested_number(value: Dict[str, Any], *path: str) -> Optional[float]:
        current: Any = value
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return GroundingContractBuilder._number(current)

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

    @staticmethod
    def _candidate_has_boundary_action(candidate: CandidateResult) -> bool:
        action = dict(candidate.action or {})
        return bool(
            dict(action.get("percentage_changes") or {})
            or dict(action.get("setpoints") or {})
        )

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
            for objective in objectives:
                item = dict(objective or {})
                excerpt = " ".join(
                    str(item.get("source_excerpt") or "").split()
                ).casefold()
                if not excerpt and len(objectives) == 1:
                    excerpt = legacy_excerpt
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

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    formatted = f"{number:.9g}"
    return format(Decimal(formatted), "f") if "e" in formatted.casefold() else formatted


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
            parts.append(f"{variable} {direction} {_format_number(abs(number))}%")
        else:
            direction = "increase" if number >= 0 else "decrease"
            parts.append(f"{direction} {variable} by {_format_number(abs(number))}%")
    for variable, value in dict(action.get("setpoints") or {}).items():
        parts.append(f"{variable}={_format_number(value)}")
    if parts:
        return ", ".join(parts)
    return "\u672a\u8bb0\u5f55" if chinese else "not recorded"


def _candidate_line(candidate: Dict[str, Any], chinese: bool) -> str:
    candidate_id = str(candidate.get("candidate_id") or "candidate")
    action = _format_action(dict(candidate.get("action") or {}), chinese)
    failed_rules = ", ".join(candidate.get("failed_rule_ids") or []) or "none"
    warning_rules = ", ".join(candidate.get("warning_rule_ids") or []) or "none"
    pressure = GroundingContractBuilder._nested_number(
        dict(candidate.get("pressure_metrics") or {}),
        "minimum_operating_window_margin",
        "value",
    )
    linepack = dict(candidate.get("linepack_metrics") or {})
    decline = GroundingContractBuilder._nested_number(
        linepack, "maximum_decline_from_start", "value"
    )
    duration = linepack.get("maximum_continuous_decline_minutes")
    recovery = linepack.get("insufficient_recovery_count")
    energy = dict(candidate.get("energy_metrics") or {}).get("delta_vs_baseline")
    metrics = []
    if pressure is not None:
        metrics.append(
            f"\u538b\u88d5{_format_number(pressure)}"
            if chinese
            else f"pressure margin {_format_number(pressure)}"
        )
    if decline is not None:
        linepack_text = (
            f"\u7ba1\u5b58\u964d{_format_number(decline)}"
            if chinese
            else f"linepack decline {_format_number(decline)}"
        )
        if duration is not None:
            linepack_text += f"/{_format_number(duration)}min"
        if recovery is not None:
            linepack_text += (
                f"/\u6062\u590d\u4e0d\u8db3{_format_number(recovery)}"
                if chinese
                else f"/insufficient recovery {_format_number(recovery)}"
            )
        metrics.append(linepack_text)
    if energy is not None:
        metrics.append(
            f"\u80fd\u8017\u0394{_format_number(energy)}"
            if chinese
            else f"energy delta {_format_number(energy)}"
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


def candidate_contract_message(
    question: str,
    tool_results: Iterable[Dict[str, Any]],
    *,
    decision_policy: Optional[Dict[str, Any]] = None,
    prior_candidate_results: Optional[Iterable[Dict[str, Any]]] = None,
    prior_decision_policy: Optional[Dict[str, Any]] = None,
    prior_decision_policy_source_question: Optional[str] = None,
    prior_applied_disturbances: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Render accumulated multi-candidate facts for the next model request."""
    results = [dict(item) for item in tool_results]
    contract = GroundingContractBuilder().build(
        question,
        results,
        decision_policy=decision_policy,
        require_decision_policy=True,
        prior_candidate_results=prior_candidate_results,
        prior_decision_policy=prior_decision_policy,
        prior_decision_policy_source_question=(
            prior_decision_policy_source_question
        ),
        prior_applied_disturbances=prior_applied_disturbances,
    )
    if contract.get("answer_mode") == "single_forecast":
        successful_forecasts = [
            item
            for item in results
            if item.get("name") == "run_pipeformer_forecast"
            and dict(item.get("output") or {}).get("success") is True
        ]
        if len(successful_forecasts) != 1:
            return None
        forecast = successful_forecasts[0]
        if dict(forecast.get("arguments") or {}).get("candidate_id"):
            return None
        output = dict(forecast.get("output") or {})
        verification = dict(
            output.get("verification") or output.get("constraint_check") or {}
        )
        has_prediction_contract = any(
            key in verification
            for key in (
                "priority_findings",
                "top_watch_variables",
                "safety_energy_comparison",
            )
        )
        if not has_prediction_contract:
            return None
        applied_disturbance = json.dumps(
            contract.get("applied_disturbances") or [],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        required_disclosure = applied_disturbance_disclosure(
            question,
            contract,
        )
        return (
            "CURRENT PIPEFORMER FORECAST ANSWER CONTRACT\n"
            f"Required disturbance disclosure (copy verbatim): {required_disclosure}\n"
            f"Structured application evidence: {applied_disturbance}\n"
            "Use only the successful forecast's structured fields. If the current request asks "
            "for operating result, intervention, watch variables, and the safety-energy "
            "comparison, return exactly four short bullets and no heading: (1) passing "
            "categories once plus every priority_findings warning/failure with values and "
            "thresholds; (2) risk_level and human_intervention_label; (3) top_watch_variables "
            "in returned order with variable, mean_prediction, and "
            "mean_abs_delta_vs_observed; (4) safety_energy_comparison. When its comparison is "
            "complete and consistent, say `安全侧与能耗侧结论一致`; when inconsistent, include "
            "the first priority audit constraint and numerical key_observation_variables. "
            "For a narrower follow-up, answer only requested slots. Never abbreviate a "
            "canonical variable ID or add unreturned physical meaning."
        )
    if contract.get("answer_mode") != "dispatch_comparison":
        return None
    candidates = list(contract.get("candidate_results") or [])
    decision_summary = contract.get("decision_summary") or {}
    payload = {
        "successful_candidate_count": len(candidates),
        "candidate_results": candidates,
        "decision_summary": decision_summary,
        "comparison_leaders": contract.get("comparison_leaders") or {},
        "required_application_disclosure": applied_disturbance_disclosure(
            question,
            contract,
        ),
        "worst_case_risk_level": contract.get("worst_case_risk_level"),
        "worst_case_intervention_label": contract.get(
            "worst_case_intervention_label"
        ),
    }
    policy_nudge = ""
    missing_metrics = list(decision_summary.get("missing_metrics") or [])
    if "llm_decision_policy_tool_call" in missing_metrics:
        policy_nudge = (
            " No decision policy is recorded yet; treat this as a call to action, not a "
            "failure to disclose in the answer. If the user's current or earlier wording "
            "states or implies any priority or objective (for example 优先, 主要, 尽量, "
            "minimize, or first consider), call set_decision_policy NOW with each objective "
            "grounded in an exact contiguous source_excerpt of that wording, then rank all "
            "viable candidates. End with `selected_candidate_id: none` only when no priority "
            "can be derived from the conversation at all."
        )
    return (
        "CURRENT PIPEFORMER CANDIDATE CONTRACT\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\nUse only these successful candidates and their recorded facts in the final comparison. "
        "Copy required_application_disclosure verbatim at the start of the answer. "
        "Mention every candidate_id and action, report the ordered objective evidence for every "
        "viable candidate, state hard-constraint and audit outcomes, and do not invent rankings "
        "or effects. Copy every canonical variable ID exactly; never abbreviate it. Continue "
        "calling PipeFormer if the user's requested candidate count has not yet been evaluated. End the "
        "answer with exactly `selected_candidate_id: <candidate_id>` or "
        "`selected_candidate_id: none`, matching decision_summary."
        + policy_nudge
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


def _canonical_sequence_value(values: Any, fallback: Any) -> str:
    items = list(values or [])
    if not items:
        return _canonical_number(fallback)
    rendered = [_canonical_number(item) for item in items]
    if len(set(rendered)) == 1:
        return rendered[0]
    return json.dumps(rendered, ensure_ascii=False, separators=(",", ":"))


def _canonical_applied_disturbance_lines(
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
    for line in _canonical_applied_disturbance_lines(contract):
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


def finalize_applied_disturbance_disclosure(
    answer: str,
    contract: Dict[str, Any],
) -> str:
    """Serialize the canonical application block without rewriting prose."""
    required = [
        *_canonical_applied_disturbance_lines(contract),
        *_canonical_assumption_source_lines(contract),
    ]
    prose = answer_without_machine_disclosure(answer)
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
    if int(contract.get("current_candidate_forecast_count") or 0) > 0:
        return True
    if int(contract.get("current_decision_policy_call_count") or 0) > 0:
        return True
    if _SELECTED_CANDIDATE.search(answer):
        return True
    if re.search(
        r"\bcandidate[_-]?\d+\s*(?:>|<|>=|<=)\s*candidate[_-]?\d+\b",
        answer,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(
            r"(?:recommend|select|choose|建议|推荐|选择).{0,24}\bcandidate[_-]?\d+\b",
            answer,
            re.IGNORECASE,
        )
    )


def comparison_answer_issues(answer: str, contract: Dict[str, Any]) -> List[str]:
    """Validate semantic comparison coverage and the typed selection claim."""
    action_variables = _contract_action_variables(contract)
    issues: List[str] = []
    if _contains_bare_action_prefix(answer, action_variables):
        issues.append("canonical_action_variable_abbreviated")
    required_disclosure = _canonical_applied_disturbance_lines(contract)
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
                expected_value = _format_number(dict(evidence or {}).get("value"))
                if expected_value and expected_value not in answer:
                    missing_objective_values.append(f"{candidate_id}:{metric}")
        if missing_objective_values:
            issues.append("decision_objective_evidence_incomplete")

        if any(variable.casefold() not in answer_folded for variable in action_variables):
            issues.append("candidate_action_mapping_incomplete")

        present_categories = {
            category
            for candidate in contract.get("candidate_results") or []
            for category, key in (
                ("pressure", "pressure_metrics"),
                ("flow", "flow_metrics"),
                ("linepack", "linepack_metrics"),
                ("compressor", "compressor_metrics"),
                ("energy", "energy_metrics"),
            )
            if dict(candidate.get(key) or {})
        }
        category_patterns = {
            "pressure": r"压力|压裕|pressure",
            "flow": r"流量|供需|flow|supply.?demand",
            "linepack": r"管存|linepack",
            "compressor": r"压缩机|负荷|compressor|load",
            "energy": r"能耗|energy",
        }
        if any(
            not re.search(category_patterns[category], answer, re.IGNORECASE)
            for category in present_categories
        ):
            issues.append("candidate_audit_evidence_incomplete")

        if not re.search(
            r"(?:F|失败|failure)\s*0|failure_count\s*[:=]\s*0|无(?:规则)?失败",
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
    suffix = "\n".join(_comparison_markers(selected_candidate_id, contract))
    body = "\n".join(str(line).strip() for line in lines if str(line).strip())
    answer = body + "\n" + suffix
    over_budget = len(answer) > maximum_chars
    if maximum_words is not None and len(answer.split()) > maximum_words:
        over_budget = True
    if over_budget:
        contract["answer_render_status"] = "answer_budget_insufficient"
    else:
        contract.pop("answer_render_status", None)
    return answer


def _comparison_markers(
    selected_candidate_id: str,
    contract: Dict[str, Any],
) -> List[str]:
    return [f"selected_candidate_id: {selected_candidate_id or 'none'}"]


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
        parts.append(f"{variable}={_format_number(raw_value)}")
    return ",".join(parts) or "未记录"


def _action_variables(action: Dict[str, Any]) -> str:
    variables = [
        str(variable)
        for key in ("percentage_changes", "setpoints")
        for variable in dict(action.get(key) or {})
    ]
    return ",".join(variables) or "未记录"


def _applied_disturbance_lines(
    contract: Dict[str, Any],
    chinese: bool,
) -> List[str]:
    del chinese
    return _canonical_applied_disturbance_lines(contract)


def applied_disturbance_disclosure(
    question: str,
    contract: Dict[str, Any],
) -> str:
    """Return the exact machine-verifiable application evidence block."""
    del question
    return "\n".join(_canonical_applied_disturbance_lines(contract))


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
            _format_number(magnitude),
            _format_number(setpoint),
        )
        if key in seen:
            continue
        seen.add(key)
        if variable.endswith(":ST") and setpoint is not None:
            if chinese:
                lines.append(f"LLM暂设：{variable}={_format_number(setpoint)}。")
            else:
                lines.append(
                    f"Provisional LLM assumption: {variable}={_format_number(setpoint)}."
                )
            continue
        if chinese:
            direction_label = {"up": "上调", "down": "下调"}.get(direction.casefold(), direction)
            if magnitude is None:
                lines.append(f"LLM临时假设：{variable}{direction_label}。")
            else:
                lines.append(f"LLM临时假设：{variable}{direction_label}{_format_number(magnitude)}%。")
        else:
            if magnitude is None:
                lines.append(f"Provisional LLM assumption: {variable} {direction}.")
            else:
                lines.append(
                    f"Provisional LLM assumption: {variable} {direction} {_format_number(magnitude)}%."
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
    value = _format_number(evidence.get("value"))
    variable = str(evidence.get("variable") or "")
    unit = str(evidence.get("unit") or "")
    detail = value
    if variable:
        detail += f"({variable})"
    if unit:
        detail += unit
    return f"{label}{detail}" if chinese else f"{label}={detail}"


def _audit_category_summary(
    candidates: List[Dict[str, Any]],
    chinese: bool,
) -> str:
    category_definitions = (
        ("pressure", "pressure_metrics", "压力", "pressure"),
        ("flow", "flow_metrics", "流量", "flow"),
        ("linepack", "linepack_metrics", "管存", "linepack"),
        ("compressor", "compressor_metrics", "压缩机", "compressor"),
        ("energy", "energy_metrics", "能耗", "energy"),
    )
    present = [
        (category, zh, en)
        for category, key, zh, en in category_definitions
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
    chinese = any("\u4e00" <= character <= "\u9fff" for character in question)
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
        *_applied_disturbance_lines(contract, chinese),
        *_compact_assumption_lines(contract, chinese),
        policy_line,
        *candidate_lines,
        _audit_category_summary(ranked, chinese),
        ranking_line,
        rejection_line,
        *_comparison_markers(selected_id, contract),
    ]
    answer = "\n".join(lines)
    over_budget = len(answer) > (
        chinese_comparison_max_chars(len(ranked))
        if chinese
        else ENGLISH_COMPARISON_MAX_CHARS
    )
    if not chinese and len(answer.split()) > ENGLISH_MAX_WORDS:
        over_budget = True
    if over_budget:
        contract["answer_render_status"] = "answer_budget_insufficient"
    else:
        contract.pop("answer_render_status", None)
    return answer

def grounded_fallback_answer(question: str, contract: Dict[str, Any]) -> str:
    """Render a compact, instruction-complete answer from deterministic facts."""
    decision = dict(contract.get("decision_summary") or {})
    candidates = list(contract.get("candidate_results") or [])
    chinese = any("\u4e00" <= character <= "\u9fff" for character in question)
    ranking = [str(value) for value in decision.get("ranked_candidate_ids") or []]
    if ranking:
        positions = {candidate_id: index for index, candidate_id in enumerate(ranking)}
        candidates.sort(
            key=lambda item: positions.get(
                str(item.get("candidate_id") or ""), len(positions)
            )
        )
    assumption_lines = [
        *_applied_disturbance_lines(contract, chinese),
        *_compact_assumption_lines(contract, chinese),
    ]
    shared_outcome = not ranking and len(candidates) > 1 and len({
        _outcome_key(candidate) for candidate in candidates
    }) == 1
    if shared_outcome:
        lines = [
            *assumption_lines,
            "\u5019\u9009\u52a8\u4f5c\uff1a" if chinese else "Candidate actions:",
            *[
                f"- {candidate.get('candidate_id')}: "
                f"{_format_action(dict(candidate.get('action') or {}), chinese)}"
                for candidate in candidates
            ],
            _shared_outcome_line(candidates[0], chinese),
        ]
    else:
        lines = [
            *assumption_lines,
            "\u5019\u9009\u52a8\u4f5c\u6bd4\u8f83\uff1a" if chinese else "Candidate comparison:",
            *[_candidate_line(candidate, chinese) for candidate in candidates],
        ]
    selected_id = str(decision.get("selected_candidate_id") or "")
    eliminated = list(decision.get("eliminated_candidates") or [])
    missing = [str(value) for value in decision.get("missing_metrics") or []]

    if decision.get("status") == "selected":
        return _selected_comparison_answer(question, contract, candidates, decision)

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

_UNIT_CLAIM = re.compile(
    r"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>万方/日|万立方米/日|立方米/秒|m³/d|m3/d|m³/s|m3/s|MPa|kPa|bar|MW|kW)",
    re.IGNORECASE,
)
_LINEPACK_DECLINE_REQUEST = re.compile(
    r"管存.{0,12}(?:持续|下降|走低|消耗)|(?:持续|下降|走低).{0,12}管存|linepack.{0,12}(?:declin|fall|decreas)",
    re.IGNORECASE,
)


def _successful_pipeformer_results(tool_results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in tool_results
        if item.get("name") == "run_pipeformer_forecast"
        and dict(item.get("output") or {}).get("success") is True
    ]


def _status_summary(category_status: Dict[str, Any], chinese: bool) -> str:
    groups: Dict[str, List[str]] = {}
    for category, status in category_status.items():
        groups.setdefault(str(status), []).append(str(category))
    if chinese:
        labels = {"pass": "通过", "warning": "告警", "fail": "失败", "not_evaluated": "未评估"}
        return "；".join(
            f"{'/'.join(values)}{labels.get(status, status)}"
            for status, values in groups.items()
        )
    return "; ".join(f"{'/'.join(values)}={status}" for status, values in groups.items())


def _finding_summary(findings: List[Dict[str, Any]], chinese: bool) -> str:
    parts: List[str] = []
    for finding in findings[:2]:
        name = str(finding.get("name") or "unknown_rule")
        status = str(finding.get("status") or "unknown")
        variables = ", ".join(str(value) for value in finding.get("affected_variables") or [])
        detail = ""
        values = list(finding.get("evaluated_values") or [])
        if values:
            evaluated = dict(values[0])
            metric = str(evaluated.get("metric") or "value")
            detail = f"; {metric}={_format_number(evaluated.get('value'))}"
        parts.append(f"{name}({status}{'; ' + variables if variables else ''}{detail})")
    return "，".join(parts) if chinese else ", ".join(parts)


def _watch_summary(output: Dict[str, Any]) -> List[str]:
    evidence = dict(output.get("evidence") or {})
    watch = list(evidence.get("top_watch_variables") or [])
    if not watch:
        prediction = dict(output.get("prediction") or output.get("prediction_summary") or {})
        summaries = dict(prediction.get("output_forecast_summary") or {})
        watch = [{"variable": variable, **dict(summary or {})} for variable, summary in list(summaries.items())[:3]]
    result = []
    for item in watch[:3]:
        variable = str(item.get("variable") or "")
        value = item.get("mean_prediction")
        result.append(f"{variable}={_format_number(value)}" if value is not None else variable)
    return [value for value in result if value]


def _single_forecast_answer(question: str, tool_result: Dict[str, Any]) -> str:
    output = dict(tool_result.get("output") or {})
    prediction = dict(output.get("prediction") or output.get("prediction_summary") or {})
    verification = dict(output.get("verification") or output.get("constraint_check") or {})
    chinese = any("\u4e00" <= character <= "\u9fff" for character in question)
    contract = GroundingContractBuilder().build(question, [tool_result])
    lines: List[str] = _applied_disturbance_lines(contract, chinese)
    comparison = dict(prediction.get("counterfactual_comparison") or {})
    impact_count = comparison.get("nonzero_impacted_variable_count")
    linepack = dict(dict(verification.get("engineering_evidence") or {}).get("linepack") or {})
    linepack_status = dict(verification.get("category_status") or {}).get("linepack")
    decline_minutes = linepack.get("maximum_continuous_decline_minutes")
    if _LINEPACK_DECLINE_REQUEST.search(question):
        if linepack_status == "pass" and (decline_minutes is None or float(decline_minutes) <= 0):
            lines.append("持续管存下降未在当前预测中出现。" if chinese else "Sustained linepack decline did not appear in the current forecast.")
        elif linepack_status in {"warning", "fail"} or (decline_minutes is not None and float(decline_minutes) > 0):
            duration = _format_number(decline_minutes) if decline_minutes is not None else "unknown"
            lines.append(f"预测检测到管存下降，最长连续下降 {duration} 分钟。" if chinese else f"The forecast detected linepack decline; maximum continuous duration was {duration} minutes.")
    if impact_count is not None:
        lines.append(f"基线对比检出 {int(impact_count)} 个变化输出变量。" if chinese else f"Baseline comparison found {int(impact_count)} changed output variables.")
    category_status = dict(verification.get("category_status") or {})
    if category_status:
        summary = _status_summary(category_status, chinese)
        lines.append(f"校核：{summary}。" if chinese else f"Verification: {summary}.")
    risk = str(verification.get("risk_level") or output.get("risk_level") or "unknown")
    intervention = str(verification.get("human_intervention_label") or output.get("manual_intervention_label") or "unknown")
    lines.append(f"风险 {risk}；人工干预 {intervention}。" if chinese else f"Risk: {risk}; intervention: {intervention}.")
    findings = [dict(item) for item in verification.get("priority_findings") or []]
    if findings:
        summary = _finding_summary(findings, chinese)
        lines.append(f"优先发现：{summary}。" if chinese else f"Priority findings: {summary}.")
    watch = _watch_summary(output)
    if watch:
        lines.append(f"关注变量：{', '.join(watch)}。" if chinese else f"Watch variables: {', '.join(watch)}.")
    return "".join(lines) if chinese else " ".join(lines)


_UNSUPPORTED_UNIT_TOKEN = re.compile(
    r"(?:万方/日|万立方米/日|立方米/秒|m³/d|m3/d|m³/s|m3/s|MPa|kPa|bar|MW|kW)",
    re.IGNORECASE,
)


def _strip_unsupported_units(answer: str) -> str:
    stripped = _UNSUPPORTED_UNIT_TOKEN.sub("", answer)
    return re.sub(r"[ \t]+([，。；、,.!?])", r"\1", stripped)


def _compact_answer(answer: str, maximum_chars: int) -> str:
    """Extractively compact an answer without inventing replacement facts."""
    cleaned = re.sub(r"(?m)^\s*(?:#{1,6}\s*|---+\s*$)", "", answer)
    cleaned = cleaned.replace("**", "").replace("```", "")
    lines: List[str] = []
    seen = set()
    for raw_line in cleaned.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or re.fullmatch(r"\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?", line):
            continue
        if line not in seen:
            seen.add(line)
            lines.append(line)
    compact = " ".join(lines)
    if len(compact) <= maximum_chars:
        return compact
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])\s*|(?<=[A-Za-z])\.\s+", compact) if part.strip()]
    if not sentences:
        return compact[:maximum_chars].rstrip(" ，,；;")
    ending = sentences[-1]
    selected: List[str] = []
    reserve = len(ending) + 1 if len(ending) < maximum_chars // 2 else 0
    for sentence in sentences[:-1]:
        candidate = " ".join(selected + [sentence])
        if len(candidate) + reserve > maximum_chars:
            break
        selected.append(sentence)
    if ending not in selected and len(" ".join(selected + [ending])) <= maximum_chars:
        selected.append(ending)
    compacted = " ".join(selected)
    if compacted:
        return compacted
    return compact[:maximum_chars].rstrip(" ，,；;")


def repair_grounded_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Repair supported answers from stored evidence without an LLM call."""
    repaired = dict(record)
    tool_results = attach_tool_arguments(repaired.get("tool_outputs") or [], repaired.get("tool_calls") or [])
    contract = GroundingContractBuilder().build(
        str(repaired.get("user_input") or ""),
        tool_results,
        decision_policy=dict(repaired.get("decision_policy") or {}) or None,
    )
    repaired["answer_mode"] = contract.get("answer_mode")
    repaired["grounding_contract"] = contract
    repaired["decision_summary"] = dict(contract.get("decision_summary") or {})
    issues = {str(value) for value in repaired.get("quality_issues") or []}
    legacy_method = str(dict(repaired.get("repair_provenance") or {}).get("method") or "")
    pipeformer_results = _successful_pipeformer_results(tool_results)

    if contract.get("answer_mode") == "dispatch_comparison":
        decision = dict(contract.get("decision_summary") or {})
        repaired["final_answer"] = grounded_fallback_answer(str(repaired.get("user_input") or ""), contract)
        repaired["risk_level"] = contract.get("worst_case_risk_level")
        repaired["manual_intervention_label"] = contract.get("worst_case_intervention_label")
        repaired["dispatch_recommendation"] = str(decision.get("selected_dispatch_recommendation") or "")
        repaired["repair_provenance"] = {"method": "deterministic_grounding_contract", "external_llm_calls": 0, "reason": "Multi-candidate answer rebuilt from stored tool evidence."}

    should_render_single = bool(pipeformer_results) and (legacy_method == "offline_deterministic_repair" or bool(issues & {"answer_too_long", "unsupported_unit_claim"}))
    if contract.get("answer_mode") != "dispatch_comparison" and should_render_single:
        repaired["final_answer"] = _single_forecast_answer(str(repaired.get("user_input") or ""), pipeformer_results[0])
        output = dict(pipeformer_results[0].get("output") or {})
        verification = dict(output.get("verification") or output.get("constraint_check") or {})
        repaired["risk_level"] = verification.get("risk_level") or output.get("risk_level")
        repaired["manual_intervention_label"] = verification.get("human_intervention_label") or output.get("manual_intervention_label")
        repaired["dispatch_recommendation"] = str(verification.get("dispatch_recommendation") or "")
        repaired["repair_provenance"] = {"method": "scenario_aware_deterministic_repair", "external_llm_calls": 0, "reason": "Single-forecast answer rebuilt from stored tool evidence."}
    if "unsupported_unit_claim" in issues:
        stripped = _strip_unsupported_units(str(repaired.get("final_answer") or ""))
        if stripped != repaired.get("final_answer"):
            repaired["final_answer"] = stripped
            repaired["repair_provenance"] = {"method": "unsupported_unit_removal", "external_llm_calls": 0, "reason": "Unsupported unit labels removed without changing the numeric claims."}
    if "answer_too_long" in issues:
        maximum_chars = (
            chinese_comparison_max_chars(
                len(contract.get("candidate_results") or [])
            )
            if contract.get("answer_mode") == "dispatch_comparison"
            else CHINESE_SINGLE_FORECAST_MAX_CHARS
            if repaired.get("scenario_type") == "pipeformer"
            else GENERIC_MAX_CHARS
        )
        compacted = _compact_answer(str(repaired.get("final_answer") or ""), maximum_chars)
        if compacted != repaired.get("final_answer"):
            repaired["final_answer"] = compacted
            repaired["repair_provenance"] = {"method": "extractive_answer_compaction", "external_llm_calls": 0, "reason": "Formatting and redundant text removed; retained content is extractive."}
    return repaired
