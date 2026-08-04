from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Optional


VERIFIED_DECISION_STATE_SCHEMA_VERSION = "verified_decision_state_v1"
DEFAULT_STATE_MAX_CHARS = 16_000
DEFAULT_RECENT_TURNS_MAX_CHARS = 4_000
DEFAULT_RECENT_TURN_COUNT = 2
REGISTRY_STATE_FIELDS = (
    "variable",
    "role",
    "controllable",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _action_fingerprint(
    candidate: Dict[str, Any],
    scope: Dict[str, Any],
) -> str:
    action = dict(candidate.get("action") or {})
    candidate_disturbance = dict(candidate.get("disturbance") or {})
    payload = {
        "case_id": candidate.get("case_id") or scope.get("case_id"),
        "forecast_horizon_minutes": (
            candidate.get("forecast_horizon_minutes")
            or scope.get("forecast_horizon_minutes")
        ),
        "disturbance": _normalized_disturbance(
            candidate_disturbance or scope.get("disturbance") or {}
        ),
        "action": {
            "percentage_changes": dict(action.get("percentage_changes") or {}),
            "setpoints": dict(action.get("setpoints") or {}),
            "keep_other_boundary_controls": action.get(
                "keep_other_boundary_controls"
            ),
        },
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]


def _has_boundary_action(candidate: Dict[str, Any]) -> bool:
    action = dict(candidate.get("action") or {})
    return bool(
        dict(action.get("percentage_changes") or {})
        or dict(action.get("setpoints") or {})
    )


def _normalized_disturbance(value: Any) -> Dict[str, Any]:
    item = dict(value or {})
    if item.get("requested_value") is None:
        if item.get("setpoint") is not None:
            item["mode"] = item.get("mode") or "setpoint"
            item["requested_value"] = item["setpoint"]
        elif item.get("magnitude_percent") is not None:
            item["mode"] = item.get("mode") or "percent_change"
            item["requested_value"] = item["magnitude_percent"]
    result = {
        key: deepcopy(item[key])
        for key in (
            "variable",
            "mode",
            "requested_value",
            "direction",
        )
        if item.get(key) is not None
    }
    mode = str(result.get("mode") or "")
    requested = result.get("requested_value")
    if mode == "percent_change" and requested is not None:
        try:
            number = Decimal(str(requested))
        except InvalidOperation:
            pass
        else:
            direction = str(result.get("direction") or "").casefold()
            if direction == "down":
                number = -abs(number)
            elif direction == "up":
                number = abs(number)
            if number == 0:
                result["requested_value"] = "0"
            else:
                rendered = format(number, "f")
                if "." in rendered:
                    rendered = rendered.rstrip("0").rstrip(".")
                result["requested_value"] = rendered
    return result


def _scope_from_pipeformer(
    pipeformer: Dict[str, Any],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    candidates = [
        dict(item)
        for item in pipeformer.get("candidate_results") or []
        if isinstance(item, dict)
    ]
    first = candidates[0] if candidates else {}
    disturbances = [
        dict(item)
        for item in pipeformer.get("applied_disturbances") or []
        if isinstance(item, dict)
    ]
    disturbance = _normalized_disturbance(
        disturbances[-1]
        if disturbances
        else first.get("disturbance")
        or current.get("disturbance")
    )
    scope = {
        "case_id": first.get("case_id") or current.get("case_id"),
        "forecast_horizon_minutes": (
            first.get("forecast_horizon_minutes")
            or current.get("forecast_horizon_minutes")
        ),
        "disturbance": disturbance,
    }
    return {
        key: deepcopy(value)
        for key, value in scope.items()
        if value not in (None, "", [], {})
    }


def _scope_fingerprint(scope: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(scope).encode("utf-8")).hexdigest()[:20]


def _candidate_audit_status(candidate: Dict[str, Any]) -> str:
    try:
        failure_count = int(candidate.get("failure_count") or 0)
        warning_count = int(candidate.get("warning_count") or 0)
    except (TypeError, ValueError):
        return "unknown"
    return (
        "failed"
        if failure_count > 0
        else "warning"
        if warning_count > 0
        else "passed"
    )


def _policy_with_decision(
    policy: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    result = deepcopy(policy)
    ranking = (
        decision.get("ordered_viable_candidate_ids")
        or decision.get("ranking")
    )
    if ranking:
        result["ranking"] = deepcopy(ranking)
    if "selected_candidate_id" in decision:
        result["selection"] = {
            "candidate_id": decision.get("selected_candidate_id"),
            "status": decision.get("status"),
        }
    return result


def _compact_registry_state_item(
    item: Dict[str, Any],
    *,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    compact = {
        key: deepcopy(item[key])
        for key in REGISTRY_STATE_FIELDS
        if item.get(key) is not None
    }
    compact["context_only"] = True
    source = provenance or item.get("provenance")
    if isinstance(source, dict) and source:
        compact["provenance"] = {
            key: deepcopy(source[key])
            for key in ("tool_call_id",)
            if source.get(key) not in (None, "")
        }
    return compact


def _registry_state_projection(
    items: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    search_call_ids: List[str] = []
    search_positions: Dict[str, int] = {}
    returned_ids: List[Dict[str, Any]] = []
    for item in items:
        variable = str(item.get("variable") or "")
        if not variable:
            continue
        call_id = str(
            dict(item.get("provenance") or {}).get("tool_call_id") or ""
        )
        entry: Dict[str, Any] = {"variable": variable}
        if call_id:
            if call_id not in search_positions:
                search_positions[call_id] = len(search_call_ids)
                search_call_ids.append(call_id)
            entry["search"] = search_positions[call_id]
        returned_ids.append(entry)
    return {
        "context_only": True,
        "search_call_ids": search_call_ids,
        "returned_ids": returned_ids,
    }


def _registry_state_items(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [
            _compact_registry_state_item(dict(item))
            for item in value
            if isinstance(item, dict)
        ]
    if not isinstance(value, dict):
        return []
    call_ids = [str(item) for item in value.get("search_call_ids") or []]
    items: List[Dict[str, Any]] = []
    for entry in value.get("returned_ids") or []:
        if not isinstance(entry, dict) or not entry.get("variable"):
            continue
        search = entry.get("search")
        call_id = (
            call_ids[int(search)]
            if isinstance(search, int) and 0 <= search < len(call_ids)
            else ""
        )
        items.append(
            _compact_registry_state_item(
                {"variable": entry["variable"]},
                provenance={"tool_call_id": call_id},
            )
        )
    return items


def _successful_tool_result(item: Dict[str, Any]) -> bool:
    output = item.get("output")
    return (
        isinstance(output, dict)
        and output.get("success") is True
        and not output.get("error")
        and output.get("exit_code") in (None, 0)
    )


def _verified_state_tool_result(item: Dict[str, Any]) -> bool:
    if not _successful_tool_result(item):
        return False
    if item.get("name") != "run_pipeformer_forecast":
        return True
    output = dict(item.get("output") or {})
    verification = dict(
        output.get("verification") or output.get("constraint_check") or {}
    )
    applications = list(
        dict(output.get("evidence") or {}).get(
            "boundary_application_evidence"
        )
        or []
    )
    return (
        verification.get("verification_complete") is True
        and bool(applications)
        and all(
            isinstance(application, dict)
            and application.get("verified") is True
            for application in applications
        )
    )


def _forecast_scope(
    arguments: Dict[str, Any],
    applied_disturbances: List[Dict[str, Any]],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    disturbance = (
        _normalized_disturbance(applied_disturbances[-1])
        if applied_disturbances
        else _normalized_disturbance(
            {
                "variable": arguments.get("disturbance_variable"),
                "mode": (
                    "setpoint"
                    if arguments.get("disturbance_setpoint") is not None
                    else "percent_change"
                ),
                "requested_value": (
                    arguments.get("disturbance_setpoint")
                    if arguments.get("disturbance_setpoint") is not None
                    else arguments.get("disturbance_magnitude_percent")
                ),
                "direction": arguments.get("disturbance_direction"),
            }
        )
    )
    scope = {
        "case_id": arguments.get("case_id") or current.get("case_id"),
        "forecast_horizon_minutes": (
            arguments.get("forecast_horizon_minutes")
            or current.get("forecast_horizon_minutes")
        ),
        "disturbance": disturbance or current.get("disturbance"),
    }
    return {
        key: deepcopy(value)
        for key, value in scope.items()
        if value not in (None, "", [], {})
    }


_SINGLE_FORECAST_ENGINEERING_FIELDS = {
    "pressure": (
        "maximum_pressure",
        "minimum_pressure",
        "minimum_lower_bound_margin",
        "minimum_upper_bound_margin",
        "minimum_operating_window_margin",
        "violation_node_count",
        "warning_node_count",
        "maximum_continuous_pressure_violation_minutes",
    ),
    "flow": (
        "maximum_segment_flow_change",
        "maximum_boundary_flow_change_rate",
        "flow_capacity_excursion_count",
        "supply_demand_balance_status",
        "supply_demand_balance",
    ),
    "linepack": (
        "minimum_linepack",
        "maximum_decline_from_start",
        "maximum_continuous_decline_minutes",
        "minimum_peak_shaving_reserve",
        "insufficient_recovery_count",
        "linepack_warning_status",
    ),
    "compressor": (
        "operating_envelope_status",
        "maximum_load",
        "maximum_compression_ratio",
        "maximum_rotational_speed",
        "maximum_power_change",
    ),
}


def _compact_single_forecast_snapshot(
    call: Dict[str, Any],
    *,
    scope: Dict[str, Any],
    applied_disturbances: List[Dict[str, Any]],
    source_turn_id: str,
) -> Dict[str, Any]:
    """Project one verified forecast into bounded cross-turn evidence."""
    arguments = dict(call.get("arguments") or {})
    output = dict(call.get("output") or {})
    verification = dict(
        output.get("verification") or output.get("constraint_check") or {}
    )
    engineering = dict(verification.get("engineering_evidence") or {})
    comparable = dict(verification.get("comparable_metrics") or {})
    evidence = dict(output.get("evidence") or {})
    snapshot: Dict[str, Any] = {
        "scope": deepcopy(scope),
        "candidate_role": (
            output.get("candidate_role")
            or arguments.get("candidate_role")
            or "single_forecast"
        ),
        "applied_disturbance": (
            deepcopy(applied_disturbances[-1])
            if applied_disturbances
            else {}
        ),
        "audit": {
            "overall_status": verification.get("overall_status"),
            "failure_count": verification.get("failure_count"),
            "warning_count": verification.get("warning_count"),
            "failed_rule_ids": deepcopy(
                verification.get("failed_rule_ids") or []
            ),
            "warning_rule_ids": deepcopy(
                verification.get("warning_rule_ids") or []
            ),
            "category_status": deepcopy(
                dict(verification.get("category_status") or {})
            ),
        },
        "risk": {
            "risk_level": (
                output.get("risk_level") or verification.get("risk_level")
            ),
            "manual_intervention_label": (
                output.get("manual_intervention_label")
                or verification.get("human_intervention_label")
            ),
            "dispatch_recommendation": (
                output.get("dispatch_recommendation")
                or verification.get("dispatch_recommendation")
            ),
        },
        "energy": {
            "total": comparable.get("energy_consumption"),
            "delta_vs_baseline": comparable.get(
                "energy_consumption_delta"
            ),
            "unit": comparable.get("energy_unit"),
            "variable_count": comparable.get("energy_variable_count"),
            "evaluation_status": comparable.get(
                "energy_evaluation_status"
            ),
            "baseline_reference": comparable.get("baseline_reference"),
        },
        "top_watch_variables": deepcopy(
            list(evidence.get("top_watch_variables") or [])[:8]
        ),
        "key_observation_variables": deepcopy(
            list(evidence.get("key_observation_variables") or [])[:8]
        ),
        "provenance": {
            "source_turn_id": source_turn_id,
            "source_tool_call_id": str(call.get("tool_call_id") or ""),
        },
    }
    for category, fields in _SINGLE_FORECAST_ENGINEERING_FIELDS.items():
        category_evidence = dict(engineering.get(category) or {})
        snapshot[category] = {
            field: deepcopy(category_evidence[field])
            for field in fields
            if category_evidence.get(field) is not None
        }
    return snapshot


def bounded_recent_turns(
    turns: Iterable[Dict[str, Any]],
    *,
    max_turns: int = DEFAULT_RECENT_TURN_COUNT,
    max_chars: int = DEFAULT_RECENT_TURNS_MAX_CHARS,
) -> List[Dict[str, Any]]:
    """Return at most two recent dialogue turns without prior raw tool payloads."""
    if max_turns < 0 or max_chars < 1:
        raise ValueError("Recent-turn budgets must be positive.")
    projected: List[Dict[str, Any]] = []
    for turn in list(turns)[-max_turns:]:
        if not isinstance(turn, dict):
            continue
        item = {
            key: deepcopy(turn[key])
            for key in (
                "session_id",
                "turn_id",
                "user_input",
                "assistant_output",
            )
            if turn.get(key) not in (None, "")
        }
        projected.append(item)
    while projected and len(_canonical_json(projected)) > max_chars:
        if len(projected) > 1:
            projected.pop(0)
            continue
        only = projected[0]
        text_keys = [
            key for key in ("assistant_output", "user_input")
            if isinstance(only.get(key), str) and only[key]
        ]
        if not text_keys:
            raise ValueError("Recent-turn metadata exceeds the configured budget.")
        key = text_keys[0]
        overflow = len(_canonical_json(projected)) - max_chars
        keep = max(0, len(only[key]) - overflow - 1)
        only[key] = only[key][:keep]
        if keep == 0:
            only.pop(key, None)
    return projected


def serialize_verified_decision_state(
    state: "VerifiedDecisionState | Dict[str, Any]",
    *,
    max_chars: int = DEFAULT_STATE_MAX_CHARS,
    token_counter: Optional[Callable[[str], int]] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Serialize one state snapshot and fail closed instead of dropping evidence."""
    payload = (
        state.to_dict()
        if isinstance(state, VerifiedDecisionState)
        else VerifiedDecisionState.from_dict(dict(state)).to_dict()
    )
    rendered = _canonical_json(payload)
    if len(rendered) > max_chars:
        raise ValueError(
            "Verified decision state exceeds the configured state budget "
            f"({len(rendered)} > {max_chars}); required IDs, metrics, and "
            "provenance were not dropped."
        )
    if max_tokens is not None:
        if token_counter is None:
            raise ValueError(
                "A caller-supplied offline token_counter is required when "
                "max_tokens is configured."
            )
        token_count = int(token_counter(rendered))
        if token_count > max_tokens:
            raise ValueError(
                "Verified decision state exceeds the configured token budget "
                f"({token_count} > {max_tokens})."
            )
    return payload


@dataclass
class VerifiedDecisionState:
    """Bounded, verified cross-turn state shared by runtime and SFT export."""

    schema_version: str = VERIFIED_DECISION_STATE_SCHEMA_VERSION
    scope: Dict[str, Any] = field(default_factory=dict)
    verified_evidence: Dict[str, Any] = field(default_factory=dict)
    registry_variables: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    decision_policy: Optional[Dict[str, Any]] = None
    applied_disturbances: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_inputs: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(
        default_factory=lambda: {"turn_ids": [], "tool_call_ids": []}
    )

    @property
    def candidate_results(self) -> List[Dict[str, Any]]:
        return self.candidates

    @property
    def decision_policy_source_question(self) -> Optional[str]:
        value = dict(self.provenance).get("decision_policy_source_question")
        return str(value) if value else None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": VERIFIED_DECISION_STATE_SCHEMA_VERSION,
            "scope": deepcopy(self.scope),
            "verified_evidence": deepcopy(self.verified_evidence),
            "registry_variables": _registry_state_projection(
                self.registry_variables
            ),
            "candidates": deepcopy(self.candidates),
            "applied_disturbances": deepcopy(self.applied_disturbances),
            "unresolved_inputs": list(self.unresolved_inputs),
            "provenance": deepcopy(self.provenance),
        }
        if self.scope:
            payload["scope"]["fingerprint"] = _scope_fingerprint(
                {
                    key: value
                    for key, value in self.scope.items()
                    if key != "fingerprint"
                }
            )
        if self.decision_policy:
            payload["decision_policy"] = deepcopy(self.decision_policy)
        return payload

    def updated_from_tool_results(
        self,
        session_id: str,
        turn_id: int,
        question: str,
        tool_results: Iterable[Dict[str, Any]],
    ) -> "VerifiedDecisionState":
        """Reduce only successful verified tool results into a new state."""
        from .contract import GroundingContractBuilder

        successful = [
            deepcopy(dict(item))
            for item in tool_results
            if isinstance(item, dict) and _verified_state_tool_result(item)
        ]
        if not successful:
            return VerifiedDecisionState.from_dict(self.to_dict())

        state = VerifiedDecisionState.from_dict(self.to_dict())
        current_forecasts = [
            item
            for item in successful
            if item.get("name") == "run_pipeformer_forecast"
        ]
        if current_forecasts:
            preliminary_scope = _forecast_scope(
                dict(current_forecasts[-1].get("arguments") or {}),
                [],
                state.scope,
            )
            if (
                state.scope
                and preliminary_scope
                and _scope_fingerprint(preliminary_scope)
                != _scope_fingerprint(state.scope)
            ):
                state.candidates = []
                state.decision_policy = None
                state.applied_disturbances = []
                state.verified_evidence.pop(
                    "single_forecast_snapshot", None
                )
                state.provenance.pop("decision_policy_source_question", None)
        contract = GroundingContractBuilder().build(
            question,
            successful,
            require_decision_policy=True,
            prior_candidate_results=state.candidates,
            prior_decision_policy=state.decision_policy,
            prior_decision_policy_source_question=(
                state.decision_policy_source_question
            ),
            prior_applied_disturbances=state.applied_disturbances,
        )
        applied = [
            deepcopy(dict(item))
            for item in contract.get("applied_disturbances") or []
            if isinstance(item, dict)
        ]
        if current_forecasts:
            next_scope = _forecast_scope(
                dict(current_forecasts[-1].get("arguments") or {}),
                applied,
                state.scope,
            )
            if (
                state.scope
                and next_scope
                and _scope_fingerprint(next_scope)
                != _scope_fingerprint(state.scope)
            ):
                state.candidates = []
                state.decision_policy = None
                state.applied_disturbances = []
                state.verified_evidence.pop(
                    "single_forecast_snapshot", None
                )
                state.provenance.pop("decision_policy_source_question", None)
            state.scope = next_scope

        registry_by_id = {
            str(item.get("variable") or "").casefold(): index
            for index, item in enumerate(state.registry_variables)
            if item.get("variable")
        }
        for call in successful:
            if call.get("name") != "search_pipeformer_registry":
                continue
            arguments = dict(call.get("arguments") or {})
            output = dict(call.get("output") or {})
            for variable in output.get("variables") or []:
                if not isinstance(variable, dict):
                    continue
                variable_id = str(variable.get("variable") or "")
                if not variable_id:
                    continue
                item = _compact_registry_state_item(
                    dict(variable),
                    provenance={
                        "tool_call_id": call.get("tool_call_id"),
                        "query": arguments.get("query"),
                        "role": arguments.get("role"),
                        "controllable": arguments.get("controllable"),
                    },
                )
                key = variable_id.casefold()
                if key in registry_by_id:
                    state.registry_variables[registry_by_id[key]] = item
                else:
                    registry_by_id[key] = len(state.registry_variables)
                    state.registry_variables.append(item)

        if contract.get("candidate_results"):
            incoming_candidates = [
                dict(candidate)
                for candidate in contract.get("candidate_results") or []
                if isinstance(candidate, dict)
            ]
            if any(_has_boundary_action(candidate) for candidate in incoming_candidates):
                # An actionless snapshot is a prior forecast, not a third
                # action in a later explicit control comparison.
                state.candidates = [
                    candidate
                    for candidate in state.candidates
                    if _has_boundary_action(candidate)
                ]
                incoming_candidates = [
                    candidate
                    for candidate in incoming_candidates
                    if _has_boundary_action(candidate)
                ]
            current_by_id = {}
            for call in current_forecasts:
                arguments = dict(call.get("arguments") or {})
                output = dict(call.get("output") or {})
                candidate_id = str(
                    arguments.get("candidate_id")
                    or output.get("candidate_id")
                    or ""
                )
                if candidate_id:
                    current_by_id[candidate_id.casefold()] = call
            candidate_positions = {
                str(item.get("action_fingerprint") or _action_fingerprint(
                    item, state.scope
                )): index
                for index, item in enumerate(state.candidates)
            }
            for candidate in incoming_candidates:
                item = deepcopy(candidate)
                candidate_id = str(item.get("candidate_id") or "")
                item.setdefault(
                    "audit_status",
                    _candidate_audit_status(item),
                )
                call = current_by_id.get(candidate_id.casefold())
                if call:
                    arguments = dict(call.get("arguments") or {})
                    item["case_id"] = arguments.get("case_id")
                    item["forecast_horizon_minutes"] = arguments.get(
                        "forecast_horizon_minutes"
                    )
                    if applied:
                        item["disturbance"] = deepcopy(applied[-1])
                fingerprint = _action_fingerprint(item, state.scope)
                item["action_fingerprint"] = fingerprint
                prior_same_id = next(
                    (
                        candidate_value
                        for candidate_value in state.candidates
                        if str(
                            candidate_value.get("candidate_id") or ""
                        ).casefold()
                        == candidate_id.casefold()
                    ),
                    None,
                )
                if (
                    prior_same_id is not None
                    and str(
                        prior_same_id.get("action_fingerprint")
                        or _action_fingerprint(prior_same_id, state.scope)
                    )
                    != fingerprint
                ):
                    state.candidates = []
                    candidate_positions = {}
                    state.decision_policy = None
                    state.provenance.pop(
                        "decision_policy_source_question", None
                    )
                if fingerprint in candidate_positions:
                    position = candidate_positions[fingerprint]
                    stable_id = state.candidates[position].get("candidate_id")
                    merged = {**state.candidates[position], **item}
                    merged["candidate_id"] = stable_id or candidate_id
                    state.candidates[position] = merged
                else:
                    candidate_positions[fingerprint] = len(state.candidates)
                    state.candidates.append(item)

        current_policy = dict(contract.get("decision_policy") or {})
        if current_policy.get("source") == "llm_tool":
            state.decision_policy = _policy_with_decision(
                current_policy,
                dict(contract.get("decision_summary") or {}),
            )
            if question:
                state.provenance["decision_policy_source_question"] = question

        disturbance_positions = {
            _canonical_json(
                [
                    str(item.get("variable") or "").casefold(),
                    str(item.get("mode") or "").casefold(),
                    item.get("requested_value"),
                ]
            ): index
            for index, item in enumerate(state.applied_disturbances)
        }
        for disturbance in applied:
            key = _canonical_json(
                [
                    str(disturbance.get("variable") or "").casefold(),
                    str(disturbance.get("mode") or "").casefold(),
                    disturbance.get("requested_value"),
                ]
            )
            if key in disturbance_positions:
                state.applied_disturbances[disturbance_positions[key]] = disturbance
            else:
                disturbance_positions[key] = len(state.applied_disturbances)
                state.applied_disturbances.append(disturbance)

        current_candidate_calls = [
            call
            for call in current_forecasts
            if dict(call.get("arguments") or {}).get("candidate_id")
        ]
        if current_candidate_calls:
            state.verified_evidence.pop("single_forecast_snapshot", None)
        elif len(current_forecasts) == 1:
            turn_key = f"{session_id}::turn_{int(turn_id):03d}"
            state.verified_evidence["single_forecast_snapshot"] = (
                _compact_single_forecast_snapshot(
                    current_forecasts[0],
                    scope=state.scope,
                    applied_disturbances=state.applied_disturbances,
                    source_turn_id=turn_key,
                )
            )

        for call in successful:
            output = dict(call.get("output") or {})
            for key in ("csv_evidence", "topology_summary"):
                if isinstance(output.get(key), (dict, list)) and output[key]:
                    state.verified_evidence[key] = deepcopy(output[key])
            unresolved = output.get("unresolved_inputs") or output.get(
                "unresolved_task_vocabulary"
            )
            for value in unresolved or []:
                text = str(value)
                if text and text not in state.unresolved_inputs:
                    state.unresolved_inputs.append(text)

        turn_key = f"{session_id}::turn_{int(turn_id):03d}"
        turn_ids = state.provenance.setdefault("turn_ids", [])
        if turn_key not in turn_ids:
            turn_ids.append(turn_key)
        call_ids = state.provenance.setdefault("tool_call_ids", [])
        for call in successful:
            call_id = str(call.get("tool_call_id") or "")
            if call_id and call_id not in call_ids:
                call_ids.append(call_id)
        return state

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "VerifiedDecisionState":
        schema_version = str(
            value.get("schema_version") or VERIFIED_DECISION_STATE_SCHEMA_VERSION
        )
        if schema_version != VERIFIED_DECISION_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported verified decision state schema: {schema_version}"
            )
        scope = dict(value.get("scope") or {})
        scope.pop("fingerprint", None)
        return cls(
            schema_version=schema_version,
            scope=deepcopy(scope),
            verified_evidence=deepcopy(
                dict(value.get("verified_evidence") or {})
            ),
            registry_variables=_registry_state_items(
                value.get("registry_variables")
            ),
            candidates=[
                deepcopy(dict(item))
                for item in value.get("candidates") or []
                if isinstance(item, dict)
            ],
            decision_policy=(
                deepcopy(dict(value.get("decision_policy") or {})) or None
            ),
            applied_disturbances=[
                deepcopy(dict(item))
                for item in value.get("applied_disturbances") or []
                if isinstance(item, dict)
            ],
            unresolved_inputs=[
                str(item) for item in value.get("unresolved_inputs") or []
            ],
            provenance=deepcopy(
                dict(value.get("provenance") or {})
                or {"turn_ids": [], "tool_call_ids": []}
            ),
        )

    @classmethod
    def from_history(
        cls,
        conversation_context: Iterable[Dict[str, Any]],
    ) -> "VerifiedDecisionState":
        state = cls()
        candidate_positions: Dict[str, int] = {}
        disturbance_positions: Dict[str, int] = {}
        registry_positions: Dict[str, int] = {}

        for turn in conversation_context:
            if (
                "verified_state_eligible" in turn
                and turn.get("verified_state_eligible") is not True
            ):
                continue
            legacy_verified = (
                "tool_evidence_verified" not in turn
                and turn.get("grounding_verified") is True
            )
            if turn.get("tool_evidence_verified") is not True and not legacy_verified:
                continue
            summary = dict(turn.get("verified_evidence_summary") or {})
            pipeformer = dict(summary.get("pipeformer") or {})
            next_scope = _scope_from_pipeformer(pipeformer, state.scope)
            if (
                state.scope
                and next_scope
                and _scope_fingerprint(next_scope)
                != _scope_fingerprint(state.scope)
            ):
                state.candidates = []
                candidate_positions = {}
                state.decision_policy = None
                state.applied_disturbances = []
                disturbance_positions = {}
                state.provenance.pop("decision_policy_source_question", None)
            if next_scope:
                state.scope = next_scope

            for evidence_key in ("csv_evidence", "topology_summary"):
                if summary.get(evidence_key):
                    state.verified_evidence[evidence_key] = deepcopy(
                        summary[evidence_key]
                    )

            single_forecast_snapshot = dict(
                summary.get("single_forecast_snapshot") or {}
            )
            if single_forecast_snapshot:
                state.verified_evidence["single_forecast_snapshot"] = (
                    deepcopy(single_forecast_snapshot)
                )

            for registry in summary.get("registry_variables") or []:
                if not isinstance(registry, dict):
                    continue
                variable = str(registry.get("variable") or "")
                if not variable:
                    continue
                key = variable.casefold()
                item = _compact_registry_state_item(dict(registry))
                if key in registry_positions:
                    state.registry_variables[registry_positions[key]] = item
                else:
                    registry_positions[key] = len(state.registry_variables)
                    state.registry_variables.append(item)

            incoming_candidates = [
                dict(candidate)
                for candidate in pipeformer.get("candidate_results") or []
                if isinstance(candidate, dict)
            ]
            if any(_has_boundary_action(candidate) for candidate in incoming_candidates):
                state.candidates = [
                    candidate
                    for candidate in state.candidates
                    if _has_boundary_action(candidate)
                ]
                candidate_positions = {
                    str(
                        candidate.get("action_fingerprint")
                        or _action_fingerprint(candidate, state.scope)
                    ): index
                    for index, candidate in enumerate(state.candidates)
                }
                incoming_candidates = [
                    candidate
                    for candidate in incoming_candidates
                    if _has_boundary_action(candidate)
                ]

            for candidate in incoming_candidates:
                item = deepcopy(candidate)
                candidate_id = str(item.get("candidate_id") or "")
                if not candidate_id:
                    continue
                item.setdefault(
                    "audit_status",
                    _candidate_audit_status(item),
                )
                fingerprint = _action_fingerprint(item, state.scope)
                item["action_fingerprint"] = fingerprint
                prior_same_id = next(
                    (
                        candidate_value
                        for candidate_value in state.candidates
                        if str(
                            candidate_value.get("candidate_id") or ""
                        ).casefold()
                        == candidate_id.casefold()
                    ),
                    None,
                )
                if (
                    prior_same_id is not None
                    and str(
                        prior_same_id.get("action_fingerprint")
                        or _action_fingerprint(prior_same_id, state.scope)
                    )
                    != fingerprint
                ):
                    state.candidates = []
                    candidate_positions = {}
                    state.decision_policy = None
                    state.provenance.pop(
                        "decision_policy_source_question", None
                    )
                if fingerprint in candidate_positions:
                    position = candidate_positions[fingerprint]
                    stable_id = state.candidates[position].get("candidate_id")
                    stable_tool_call_id = state.candidates[position].get(
                        "tool_call_id"
                    )
                    merged = {**state.candidates[position], **item}
                    merged["candidate_id"] = stable_id
                    if stable_tool_call_id:
                        merged["tool_call_id"] = stable_tool_call_id
                    state.candidates[position] = merged
                else:
                    candidate_positions[fingerprint] = len(state.candidates)
                    state.candidates.append(item)

            if pipeformer.get("candidate_results"):
                state.verified_evidence.pop("single_forecast_snapshot", None)

            decision = dict(pipeformer.get("decision_summary") or {})
            missing = {
                str(value) for value in decision.get("missing_metrics") or []
            }
            raw_policy = (
                pipeformer.get("decision_policy")
                or decision.get("ranking_policy")
                or {}
            )
            policy = dict(raw_policy) if isinstance(raw_policy, dict) else {}
            if (
                policy.get("source") == "llm_tool"
                and not any(
                    "source_not_in_user_request" in value
                    for value in missing
                    if value.startswith("decision_policy_")
                )
            ):
                state.decision_policy = _policy_with_decision(
                    policy,
                    decision,
                )
                source_question = str(turn.get("user_input") or "")
                if source_question:
                    state.provenance[
                        "decision_policy_source_question"
                    ] = source_question

            for disturbance in pipeformer.get("applied_disturbances") or []:
                if not isinstance(disturbance, dict):
                    continue
                item = deepcopy(dict(disturbance))
                variable = str(item.get("variable") or "")
                mode = str(item.get("mode") or "")
                if not variable or not mode:
                    continue
                key = _canonical_json(
                    [variable.casefold(), mode.casefold(), item.get("requested_value")]
                )
                if key in disturbance_positions:
                    state.applied_disturbances[
                        disturbance_positions[key]
                    ] = item
                else:
                    disturbance_positions[key] = len(
                        state.applied_disturbances
                    )
                    state.applied_disturbances.append(item)

            turn_key = "::".join(
                [
                    str(turn.get("session_id") or ""),
                    f"turn_{int(turn.get('turn_id') or 0):03d}",
                ]
            ).strip(":")
            if turn_key:
                turn_ids = state.provenance.setdefault("turn_ids", [])
                if turn_key not in turn_ids:
                    turn_ids.append(turn_key)
            tool_call_ids = state.provenance.setdefault("tool_call_ids", [])
            for call in turn.get("tool_calls") or []:
                call_id = str(dict(call).get("tool_call_id") or "")
                if call_id and call_id not in tool_call_ids:
                    tool_call_ids.append(call_id)

        return state


# Backward-compatible name used by existing evaluator/generator imports.
DecisionTraceState = VerifiedDecisionState
