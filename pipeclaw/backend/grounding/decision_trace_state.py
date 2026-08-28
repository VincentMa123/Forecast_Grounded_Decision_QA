from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .decision_policy import llm_policy_excerpts
from .evidence.tool import tool_result_failed


VERIFIED_DECISION_STATE_SCHEMA_VERSION = "verified_decision_state_v1"
DEFAULT_STATE_MAX_CHARS = 16_000
DEFAULT_RECENT_TURNS_MAX_CHARS = 4_000
DEFAULT_RECENT_TURN_COUNT = 2
MAX_SINGLE_FORECAST_HISTORY_CHARS = 1_600
REGISTRY_STATE_FIELDS = (
    "variable",
    "role",
    "controllable",
)


def _explicit_scope_value(
    source: Dict[str, Any],
    key: str,
    current: Dict[str, Any],
) -> Any:
    value = source.get(key)
    return current.get(key) if value is None else value


def canonical_json(value: Any) -> str:
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
        "case_id": _explicit_scope_value(candidate, "case_id", scope),
        "forecast_horizon_minutes": _explicit_scope_value(
            candidate, "forecast_horizon_minutes", scope
        ),
        "disturbance": _normalized_disturbance(
            candidate_disturbance or scope.get("disturbance") or {}
        ),
        "action": {
            "percentage_changes": dict(action.get("percentage_changes") or {}),
            "setpoints": dict(action.get("setpoints") or {}),
            "keep_other_boundary_controls": action.get("keep_other_boundary_controls"),
        },
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:20]


def has_boundary_action(candidate: Dict[str, Any]) -> bool:
    action = dict(candidate.get("action") or {})
    return bool(
        dict(action.get("percentage_changes") or {})
        or dict(action.get("setpoints") or {})
    )


def _requested_action_fingerprint(
    arguments: Dict[str, Any],
    scope: Dict[str, Any],
) -> Optional[str]:
    candidate = {"action": dict(arguments.get("boundary_conditions") or {})}
    return (
        _action_fingerprint(candidate, scope)
        if has_boundary_action(candidate)
        else None
    )


def _failed_forecast_arguments(turn: Dict[str, Any]) -> Dict[str, Any]:
    request = turn.get("failed_forecast_request")
    return dict(request.get("arguments") or {}) if isinstance(request, dict) else {}


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
        else first.get("disturbance") or current.get("disturbance")
    )
    scope = {
        "case_id": _explicit_scope_value(first, "case_id", current),
        "forecast_horizon_minutes": _explicit_scope_value(
            first, "forecast_horizon_minutes", current
        ),
        "disturbance": disturbance,
    }
    return {
        key: deepcopy(value)
        for key, value in scope.items()
        if value not in (None, "", [], {})
    }


def _scope_fingerprint(scope: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(scope).encode("utf-8")).hexdigest()[:20]


def _candidate_audit_status(candidate: Dict[str, Any]) -> str:
    try:
        failure_count = int(candidate.get("failure_count") or 0)
        warning_count = int(candidate.get("warning_count") or 0)
    except (TypeError, ValueError):
        return "unknown"
    return "failed" if failure_count else "warning" if warning_count else "passed"


def _merge_candidate_results(
    prior: Iterable[Dict[str, Any]],
    incoming: Iterable[Dict[str, Any]],
    scope: Dict[str, Any],
    *,
    preserve_prior_tool_call_id: bool = False,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Merge candidate values without mutating either input collection."""
    candidates = [deepcopy(dict(item)) for item in prior if isinstance(item, dict)]
    incoming_candidates = [
        deepcopy(dict(item)) for item in incoming if isinstance(item, dict)
    ]
    if any(has_boundary_action(item) for item in incoming_candidates):
        candidates = [item for item in candidates if has_boundary_action(item)]
        incoming_candidates = [
            item for item in incoming_candidates if has_boundary_action(item)
        ]

    positions = {
        str(item.get("action_fingerprint") or _action_fingerprint(item, scope)): index
        for index, item in enumerate(candidates)
    }

    policy_reset = False
    for candidate in incoming_candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        item = deepcopy(candidate)
        item.setdefault("audit_status", _candidate_audit_status(item))
        fingerprint = _action_fingerprint(item, scope)
        item["action_fingerprint"] = fingerprint
        prior_same_id = next(
            (
                value
                for value in candidates
                if str(value.get("candidate_id") or "").casefold()
                == candidate_id.casefold()
            ),
            None,
        )
        if (
            prior_same_id is not None
            and str(
                prior_same_id.get("action_fingerprint")
                or _action_fingerprint(prior_same_id, scope)
            )
            != fingerprint
        ):
            candidates = []
            positions = {}
            policy_reset = True
        if fingerprint in positions:
            position = positions[fingerprint]
            stable_id = candidates[position].get("candidate_id")
            stable_call_id = candidates[position].get("tool_call_id")
            merged = {**candidates[position], **item}
            merged["candidate_id"] = stable_id or candidate_id
            if preserve_prior_tool_call_id and stable_call_id:
                merged["tool_call_id"] = stable_call_id
            candidates[position] = merged
        else:
            positions[fingerprint] = len(candidates)
            candidates.append(item)
    return candidates, policy_reset


def _scope_from_single_forecast_snapshot(
    snapshot: Dict[str, Any],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    """Recover the canonical scope when history stores only a state snapshot."""
    stored = snapshot.get("scope")
    return (
        {
            key: deepcopy(value)
            for key, value in dict(stored).items()
            if value not in (None, "", [], {})
        }
        if stored
        else deepcopy(current)
    )


def _accepted_llm_policy(
    policy: Dict[str, Any],
    decision: Dict[str, Any],
    *,
    source_question: str,
) -> Optional[Dict[str, Any]]:
    missing = {str(value) for value in decision.get("missing_metrics") or []}
    if policy.get("source") != "llm_tool" or any(
        "source_not_in_user_request" in value
        for value in missing
        if value.startswith("decision_policy_")
    ):
        return None
    question = " ".join(str(source_question).split()).casefold()
    if not question:
        return None
    if any(
        len(excerpt) < 4 or excerpt not in question
        for excerpt in llm_policy_excerpts(policy)
    ):
        return None
    result = deepcopy(policy)
    ranking = decision.get("ordered_viable_candidate_ids") or decision.get("ranking")
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
        call_id = str(dict(item.get("provenance") or {}).get("tool_call_id") or "")
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


def _merge_replaced_items(
    existing: Iterable[Dict[str, Any]],
    incoming: Iterable[Dict[str, Any]],
    key: Callable[[Dict[str, Any]], str],
) -> List[Dict[str, Any]]:
    merged = [deepcopy(dict(item)) for item in existing]
    positions = {key(item): index for index, item in enumerate(merged)}
    for item in incoming:
        item = deepcopy(dict(item))
        identity = key(item)
        if identity in positions:
            merged[positions[identity]] = item
        else:
            positions[identity] = len(merged)
            merged.append(item)
    return merged


def _merge_registry_state_items(
    existing: Iterable[Dict[str, Any]], incoming: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return _merge_replaced_items(
        existing,
        (item for item in incoming if item.get("variable")),
        lambda item: str(item.get("variable") or "").casefold(),
    )


def _merge_applied_disturbances(
    existing: Iterable[Dict[str, Any]], incoming: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    return _merge_replaced_items(
        existing,
        incoming,
        lambda item: canonical_json(
            [
                str(item.get("variable") or "").casefold(),
                str(item.get("mode") or "").casefold(),
                item.get("requested_value"),
            ]
        ),
    )


def _ordered_unique(existing: Iterable[Any], incoming: Iterable[Any]) -> List[Any]:
    merged = []
    seen = set()
    for item in [*existing, *incoming]:
        key = canonical_json(item)
        if key not in seen:
            seen.add(key)
            merged.append(deepcopy(item))
    return merged


def _verified_state_tool_result(item: Dict[str, Any]) -> bool:
    output = item.get("output")
    if not (
        isinstance(output, dict)
        and output.get("success") is True
        and not tool_result_failed(output)
    ):
        return False
    if item.get("name") != "run_pipeformer_forecast":
        return True
    output = dict(item.get("output") or {})
    verification = dict(
        output.get("verification") or output.get("constraint_check") or {}
    )
    applications = list(
        dict(output.get("evidence") or {}).get("boundary_application_evidence") or []
    )
    return (
        verification.get("verification_complete") is True
        and bool(applications)
        and all(
            isinstance(application, dict) and application.get("verified") is True
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
        "case_id": _explicit_scope_value(arguments, "case_id", current),
        "forecast_horizon_minutes": _explicit_scope_value(
            arguments, "forecast_horizon_minutes", current
        ),
        "disturbance": disturbance or current.get("disturbance"),
    }
    return {
        key: deepcopy(value)
        for key, value in scope.items()
        if value not in (None, "", [], {})
    }


ENGINEERING_METRIC_FIELDS = {
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
            deepcopy(applied_disturbances[-1]) if applied_disturbances else {}
        ),
        "audit": {
            "overall_status": verification.get("overall_status"),
            "failure_count": verification.get("failure_count"),
            "warning_count": verification.get("warning_count"),
            "failed_rule_ids": deepcopy(verification.get("failed_rule_ids") or []),
            "warning_rule_ids": deepcopy(verification.get("warning_rule_ids") or []),
            "category_status": deepcopy(
                dict(verification.get("category_status") or {})
            ),
        },
        "risk": {
            "risk_level": (output.get("risk_level") or verification.get("risk_level")),
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
            "delta_vs_baseline": comparable.get("energy_consumption_delta"),
            "unit": comparable.get("energy_unit"),
            "variable_count": comparable.get("energy_variable_count"),
            "evaluation_status": comparable.get("energy_evaluation_status"),
            "baseline_reference": comparable.get("baseline_reference"),
        },
    }
    watch_fields = (
        "variable",
        "role",
        "metric",
        "value",
        "status",
        "mean_prediction",
        "mean_abs_delta_vs_observed",
    )
    for key, limit in (("top_watch_variables", 3), ("key_observation_variables", 2)):
        snapshot[key] = [
            {
                name: deepcopy(item[name])
                for name in watch_fields
                if item.get(name) is not None
            }
            for item in list(evidence.get(key) or [])[:limit]
            if isinstance(item, dict)
        ]
    snapshot["provenance"] = {
        "source_turn_id": source_turn_id,
        "source_tool_call_id": str(call.get("tool_call_id") or ""),
    }
    for category, fields in ENGINEERING_METRIC_FIELDS.items():
        category_evidence = dict(engineering.get(category) or {})
        compact_category = {
            field: deepcopy(category_evidence[field])
            for field in fields
            if category_evidence.get(field) is not None
        }
        if compact_category:
            snapshot[category] = compact_category
    if len(canonical_json(snapshot)) <= MAX_SINGLE_FORECAST_HISTORY_CHARS:
        return snapshot
    # Preserve the facts used to compare an earlier single forecast while
    # staying below PromptBuilder's per-entry memory budget.
    return {
        key: snapshot[key]
        for key in (
            "scope",
            "candidate_role",
            "applied_disturbance",
            "audit",
            "risk",
            "energy",
            "top_watch_variables",
            "key_observation_variables",
            "provenance",
        )
        if snapshot.get(key) not in (None, {}, [])
    }


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
    while projected and len(canonical_json(projected)) > max_chars:
        if len(projected) > 1:
            projected.pop(0)
            continue
        only = projected[0]
        text_keys = [
            key
            for key in ("assistant_output", "user_input")
            if isinstance(only.get(key), str) and only[key]
        ]
        if not text_keys:
            raise ValueError("Recent-turn metadata exceeds the configured budget.")
        key = text_keys[0]
        overflow = len(canonical_json(projected)) - max_chars
        keep = max(0, len(only[key]) - overflow - 1)
        only[key] = only[key][:keep]
        if keep == 0:
            only.pop(key, None)
    return projected


def serialize_verified_decision_state(
    state: "VerifiedDecisionState | Dict[str, Any]",
    *,
    max_chars: int = DEFAULT_STATE_MAX_CHARS,
) -> Dict[str, Any]:
    """Serialize one state snapshot and fail closed instead of dropping evidence."""
    payload = (
        state.to_dict()
        if isinstance(state, VerifiedDecisionState)
        else VerifiedDecisionState.from_dict(dict(state)).to_dict()
    )
    rendered = canonical_json(payload)
    if len(rendered) > max_chars:
        raise ValueError(
            "Verified decision state exceeds the configured state budget "
            f"({len(rendered)} > {max_chars}); required IDs, metrics, and "
            "provenance were not dropped."
        )
    return payload


def transition_forecast_scope(
    state: "VerifiedDecisionState",
    arguments: Dict[str, Any],
    *,
    applied_disturbances: Iterable[Dict[str, Any]] = (),
) -> "VerifiedDecisionState":
    """Return state scoped to one observed forecast request."""
    requested_scope = _forecast_scope(
        dict(arguments),
        [
            deepcopy(dict(item))
            for item in applied_disturbances
            if isinstance(item, dict)
        ],
        state.scope,
    )
    return _apply_verified_state_delta(
        state,
        _VerifiedStateDelta(
            scope=requested_scope,
            requested_action_fingerprint=_requested_action_fingerprint(
                dict(arguments), requested_scope
            ),
        ),
    )


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
            "registry_variables": _registry_state_projection(self.registry_variables),
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
        """Reduce observed scope and successful verified results into state."""
        delta = _delta_from_verified_tool_results(
            session_id,
            turn_id,
            question,
            tool_results,
            self,
        )
        return _apply_verified_state_delta(self, delta)

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
            verified_evidence=deepcopy(dict(value.get("verified_evidence") or {})),
            registry_variables=_registry_state_items(value.get("registry_variables")),
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

        for turn in conversation_context:
            failed_forecast_arguments = _failed_forecast_arguments(turn)
            if (
                "verified_state_eligible" in turn
                and turn.get("verified_state_eligible") is not True
                and not failed_forecast_arguments
            ):
                continue
            legacy_verified = (
                "tool_evidence_verified" not in turn
                and turn.get("grounding_verified") is True
            )
            if (
                turn.get("tool_evidence_verified") is not True
                and not legacy_verified
                and not failed_forecast_arguments
            ):
                continue
            state = _apply_verified_state_delta(
                state,
                _delta_from_history_turn(turn, state),
            )

        return state


@dataclass(frozen=True)
class _VerifiedStateDelta:
    scope: Dict[str, Any] = field(default_factory=dict)
    requested_action_fingerprint: Optional[str] = None
    registry_variables: Tuple[Dict[str, Any], ...] = ()
    candidate_results: Tuple[Dict[str, Any], ...] = ()
    preserve_prior_tool_call_id: bool = False
    decision_policy: Optional[Dict[str, Any]] = None
    decision_policy_source_question: Optional[str] = None
    applied_disturbances: Tuple[Dict[str, Any], ...] = ()
    verified_evidence: Dict[str, Any] = field(default_factory=dict)
    single_forecast_snapshot: Optional[Dict[str, Any]] = None
    clear_single_forecast_snapshot: bool = False
    unresolved_inputs: Tuple[str, ...] = ()
    turn_ids: Tuple[str, ...] = ()
    tool_call_ids: Tuple[str, ...] = ()


def _delta_from_verified_tool_results(
    session_id: str,
    turn_id: int,
    question: str,
    tool_results: Iterable[Dict[str, Any]],
    current: "VerifiedDecisionState",
) -> _VerifiedStateDelta:
    from .contract import GroundingContractBuilder, latest_decision_policy

    observed = [deepcopy(dict(item)) for item in tool_results if isinstance(item, dict)]
    forecasts = [
        item for item in observed if item.get("name") == "run_pipeformer_forecast"
    ]
    successful = [item for item in observed if _verified_state_tool_result(item)]
    if not successful:
        scope = (
            _forecast_scope(
                dict(forecasts[-1].get("arguments") or {}), [], current.scope
            )
            if forecasts
            else {}
        )
        return _VerifiedStateDelta(
            scope=scope,
            requested_action_fingerprint=(
                _requested_action_fingerprint(
                    dict(forecasts[-1].get("arguments") or {}), scope
                )
                if forecasts
                else None
            ),
        )

    current_forecasts = [
        item for item in successful if item.get("name") == "run_pipeformer_forecast"
    ]
    scope: Dict[str, Any] = {}
    requested_action_fingerprint = None
    contract = GroundingContractBuilder().build(
        question,
        successful,
        require_decision_policy=True,
        prior_candidate_results=current.candidates,
        prior_decision_policy=current.decision_policy,
        prior_decision_policy_source_question=current.decision_policy_source_question,
        prior_applied_disturbances=current.applied_disturbances,
    )
    applied = tuple(
        deepcopy(dict(item))
        for item in contract.get("applied_disturbances") or []
        if isinstance(item, dict)
    )
    if current_forecasts:
        scope = _forecast_scope(
            dict(current_forecasts[-1].get("arguments") or {}),
            list(applied),
            current.scope,
        )
        requested_action_fingerprint = _requested_action_fingerprint(
            dict(current_forecasts[-1].get("arguments") or {}), scope
        )

    registry = [
        _compact_registry_state_item(
            dict(variable),
            provenance={"tool_call_id": call.get("tool_call_id")},
        )
        for call in successful
        if call.get("name") == "search_pipeformer_registry"
        for variable in dict(call.get("output") or {}).get("variables") or []
        if isinstance(variable, dict) and variable.get("variable")
    ]

    candidates = [
        deepcopy(dict(candidate))
        for candidate in contract.get("candidate_results") or []
        if isinstance(candidate, dict)
    ]
    current_by_id = {}
    for call in current_forecasts:
        candidate_id = str(
            dict(call.get("arguments") or {}).get("candidate_id")
            or dict(call.get("output") or {}).get("candidate_id")
            or ""
        )
        if candidate_id:
            current_by_id[candidate_id.casefold()] = call
    for candidate in candidates:
        call = current_by_id.get(str(candidate.get("candidate_id") or "").casefold())
        if call:
            arguments = dict(call.get("arguments") or {})
            for key in ("case_id", "forecast_horizon_minutes"):
                if arguments.get(key) is not None:
                    candidate[key] = arguments[key]
            if applied:
                candidate["disturbance"] = deepcopy(applied[-1])

    policy = latest_decision_policy(successful) or dict(
        contract.get("decision_policy") or {}
    )
    accepted_policy = _accepted_llm_policy(
        policy,
        dict(contract.get("decision_summary") or {}),
        source_question=question,
    )
    evidence: Dict[str, Any] = {}
    unresolved = []
    for call in successful:
        output = dict(call.get("output") or {})
        for key in ("csv_evidence", "topology_summary"):
            if isinstance(output.get(key), (dict, list)) and output[key]:
                evidence[key] = deepcopy(output[key])
        values = output.get("unresolved_inputs") or output.get(
            "unresolved_task_vocabulary"
        )
        unresolved.extend(str(value) for value in values or [] if str(value))

    turn_key = f"{session_id}::turn_{int(turn_id):03d}"
    candidate_forecasts = [
        call
        for call in current_forecasts
        if dict(call.get("arguments") or {}).get("candidate_id")
    ]
    snapshot = None
    if len(current_forecasts) == 1 and not candidate_forecasts:
        snapshot = _compact_single_forecast_snapshot(
            current_forecasts[0],
            scope=scope,
            applied_disturbances=list(applied),
            source_turn_id=turn_key,
        )

    return _VerifiedStateDelta(
        scope=scope,
        requested_action_fingerprint=requested_action_fingerprint,
        registry_variables=tuple(registry),
        candidate_results=tuple(candidates),
        decision_policy=accepted_policy,
        decision_policy_source_question=question
        if accepted_policy and question
        else None,
        applied_disturbances=applied,
        verified_evidence=evidence,
        single_forecast_snapshot=snapshot,
        clear_single_forecast_snapshot=bool(candidate_forecasts),
        unresolved_inputs=tuple(unresolved),
        turn_ids=(turn_key,),
        tool_call_ids=tuple(
            str(call.get("tool_call_id") or "")
            for call in successful
            if call.get("tool_call_id")
        ),
    )


def _delta_from_history_turn(
    turn: Dict[str, Any],
    current: "VerifiedDecisionState",
) -> _VerifiedStateDelta:
    summary = dict(turn.get("verified_evidence_summary") or {})
    pipeformer = dict(turn.get("comparison_state") or summary.get("pipeformer") or {})
    single_forecast_snapshot = dict(summary.get("single_forecast_snapshot") or {})
    decision = dict(pipeformer.get("decision_summary") or {})
    raw_policy = (
        pipeformer.get("decision_policy") or decision.get("ranking_policy") or {}
    )
    policy = dict(raw_policy) if isinstance(raw_policy, dict) else {}
    accepted_policy = _accepted_llm_policy(
        policy,
        decision,
        source_question=str(turn.get("user_input") or ""),
    )
    failed_forecast_arguments = _failed_forecast_arguments(turn)
    if failed_forecast_arguments:
        scope = _forecast_scope(failed_forecast_arguments, [], current.scope)
    elif pipeformer:
        scope = _scope_from_pipeformer(pipeformer, current.scope)
    else:
        scope = _scope_from_single_forecast_snapshot(
            single_forecast_snapshot,
            current.scope,
        )
    turn_key = "::".join(
        [
            str(turn.get("session_id") or ""),
            f"turn_{int(turn.get('turn_id') or 0):03d}",
        ]
    ).strip(":")
    tool_call_ids = tuple(
        str(dict(call).get("tool_call_id") or "")
        for call in turn.get("tool_calls") or []
        if isinstance(call, dict) and dict(call).get("tool_call_id")
    )
    return _VerifiedStateDelta(
        scope=scope,
        requested_action_fingerprint=(
            _requested_action_fingerprint(failed_forecast_arguments, scope)
            if failed_forecast_arguments
            else None
        ),
        registry_variables=tuple(
            _compact_registry_state_item(dict(registry))
            for registry in (
                turn.get("registry_variables")
                or summary.get("registry_variables")
                or []
            )
            if isinstance(registry, dict) and registry.get("variable")
        ),
        candidate_results=tuple(
            deepcopy(dict(candidate))
            for candidate in pipeformer.get("candidate_results") or []
            if isinstance(candidate, dict) and candidate.get("candidate_id")
        ),
        preserve_prior_tool_call_id=True,
        decision_policy=accepted_policy,
        decision_policy_source_question=(
            str(turn.get("user_input") or "") if accepted_policy else None
        ),
        applied_disturbances=tuple(
            deepcopy(dict(disturbance))
            for disturbance in (
                pipeformer.get("applied_disturbances")
                or [single_forecast_snapshot.get("applied_disturbance")]
            )
            if isinstance(disturbance, dict)
            and disturbance.get("variable")
            and disturbance.get("mode")
        ),
        verified_evidence={
            key: deepcopy(summary[key])
            for key in ("csv_evidence", "topology_summary")
            if summary.get(key)
        },
        single_forecast_snapshot=deepcopy(single_forecast_snapshot) or None,
        clear_single_forecast_snapshot=bool(pipeformer.get("candidate_results")),
        turn_ids=(turn_key,) if turn_key and not failed_forecast_arguments else (),
        tool_call_ids=tool_call_ids if not failed_forecast_arguments else (),
    )


def _apply_verified_state_delta(
    state: "VerifiedDecisionState",
    delta: _VerifiedStateDelta,
) -> "VerifiedDecisionState":
    result = VerifiedDecisionState.from_dict(state.to_dict())
    scope_changed = bool(
        delta.scope
        and result.scope
        and _scope_fingerprint(delta.scope) != _scope_fingerprint(result.scope)
    )
    action_changed = bool(
        delta.requested_action_fingerprint
        and result.candidates
        and not any(
            str(
                candidate.get("action_fingerprint")
                or _action_fingerprint(candidate, result.scope)
            )
            == delta.requested_action_fingerprint
            for candidate in result.candidates
        )
    )
    if scope_changed or action_changed:
        result.candidates = []
        result.decision_policy = None
        result.applied_disturbances = []
        result.verified_evidence.pop("single_forecast_snapshot", None)
        result.provenance = {"turn_ids": [], "tool_call_ids": []}
    if delta.scope:
        result.scope = deepcopy(delta.scope)

    result.registry_variables = _merge_registry_state_items(
        result.registry_variables, delta.registry_variables
    )
    if delta.candidate_results:
        result.candidates, policy_reset = _merge_candidate_results(
            result.candidates,
            delta.candidate_results,
            result.scope,
            preserve_prior_tool_call_id=delta.preserve_prior_tool_call_id,
        )
        if policy_reset:
            result.decision_policy = None
            result.provenance.pop("decision_policy_source_question", None)
    if delta.decision_policy is not None:
        result.decision_policy = deepcopy(delta.decision_policy)
        if delta.decision_policy_source_question:
            result.provenance["decision_policy_source_question"] = (
                delta.decision_policy_source_question
            )
    result.applied_disturbances = _merge_applied_disturbances(
        result.applied_disturbances, delta.applied_disturbances
    )
    result.verified_evidence.update(deepcopy(delta.verified_evidence))
    if delta.single_forecast_snapshot is not None:
        result.verified_evidence["single_forecast_snapshot"] = deepcopy(
            delta.single_forecast_snapshot
        )
    if delta.clear_single_forecast_snapshot:
        result.verified_evidence.pop("single_forecast_snapshot", None)
    result.unresolved_inputs = _ordered_unique(
        result.unresolved_inputs, delta.unresolved_inputs
    )
    result.provenance["turn_ids"] = _ordered_unique(
        result.provenance.get("turn_ids") or [], delta.turn_ids
    )
    result.provenance["tool_call_ids"] = _ordered_unique(
        result.provenance.get("tool_call_ids") or [], delta.tool_call_ids
    )
    return result


DecisionTraceState = VerifiedDecisionState
