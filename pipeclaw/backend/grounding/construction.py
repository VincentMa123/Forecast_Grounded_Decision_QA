from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

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
from .evidence.tool import (
    attach_tool_arguments,
    classify_tool_evidence,
    requested_artifacts,
)

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
    return any("一" <= character <= "鿿" for character in text)

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


_CANDIDATE_TEXT_DEFAULTS = {
    "candidate_id": "",
    "tool_call_id": "",
    "risk_level": "low",
    "manual_intervention_label": "no_intervention",
    "dispatch_recommendation": "",
}
_CANDIDATE_DICT_FIELDS = (
    "action", "pressure_metrics", "linepack_metrics", "flow_metrics",
    "compressor_metrics", "energy_metrics", "category_status",
)
_CANDIDATE_LIST_FIELDS = ("failed_rule_ids", "warning_rule_ids")
_DISTURBANCE_ARGUMENTS = (
    ("disturbance_setpoint", "setpoint"),
    ("disturbance_magnitude_percent", "percent_change"),
)


def _candidate_from_compact(value: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(value or {})
    candidate = {
        name: str(item.get(name) or default)
        for name, default in _CANDIDATE_TEXT_DEFAULTS.items()
    }
    candidate.update({name: dict(item.get(name) or {}) for name in _CANDIDATE_DICT_FIELDS})
    candidate.update({
        name: [str(entry) for entry in item.get(name) or []]
        for name in _CANDIDATE_LIST_FIELDS
    })
    candidate.update({
        "failure_count": int(item.get("failure_count") or 0),
        "warning_count": int(item.get("warning_count") or 0),
        "energy_consumption": _optional_number(item, "energy_consumption", float),
        "nonzero_impacted_variable_count": _optional_number(
            item, "nonzero_impacted_variable_count", int
        ),
        "baseline_reference": (
            str(item["baseline_reference"]) if item.get("baseline_reference") else None
        ),
    })
    return {name: candidate[name] for name in _CANDIDATE_FIELDS}


def _optional_number(item: Dict[str, Any], key: str, converter) -> Any:
    return converter(item[key]) if item.get(key) is not None else None


def _candidate_compact(candidate: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(candidate)
    compact["elimination_reasons"] = compact["failed_rule_ids"]
    return compact


# Field order of the compact candidate schema.  ``_candidate_from_compact``
# emits and the grounding contract persists exactly these keys in this order.
_CANDIDATE_FIELDS = (
    "candidate_id",
    "tool_call_id",
    "action",
    "failure_count",
    "warning_count",
    "risk_level",
    "manual_intervention_label",
    "dispatch_recommendation",
    "failed_rule_ids",
    "warning_rule_ids",
    "energy_consumption",
    "nonzero_impacted_variable_count",
    "pressure_metrics",
    "linepack_metrics",
    "flow_metrics",
    "compressor_metrics",
    "energy_metrics",
    "baseline_reference",
    "category_status",
)


def build_grounding_contract(
    question: str,
    tool_results: Iterable[Dict[str, Any]],
    *,
    decision_policy: Optional[Dict[str, Any]] = None,
    require_decision_policy: bool = False,
    prior_state: Optional[VerifiedDecisionState] = None,
    prior_candidate_results: Optional[Iterable[Dict[str, Any]]] = None,
    prior_decision_policy: Optional[Dict[str, Any]] = None,
    prior_decision_policy_source_question: Optional[str] = None,
    prior_applied_disturbances: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    state = prior_state if prior_state is not None else VerifiedDecisionState()
    resolved_prior_candidates = (
        state.candidates
        if prior_candidate_results is None
        else prior_candidate_results
    )
    resolved_prior_policy = (
        state.decision_policy
        if prior_decision_policy is None
        else prior_decision_policy
    )
    resolved_prior_source_question = (
        state.decision_policy_source_question
        if prior_decision_policy_source_question is None
        else prior_decision_policy_source_question
    )
    resolved_prior_disturbances = (
        state.applied_disturbances
        if prior_applied_disturbances is None
        else prior_applied_disturbances
    )
    results = [dict(item) for item in tool_results]
    current_policy = latest_decision_policy(results)
    uses_prior_policy = (
        decision_policy is None
        and current_policy is None
        and resolved_prior_policy is not None
    )
    selected_policy = next(
        (
            value
            for value in (decision_policy, current_policy, resolved_prior_policy)
            if value is not None
        ),
        None,
    )
    resolved_decision_policy = (
        dict(selected_policy) if selected_policy is not None else None
    )
    decision_policy_question = (
        str(resolved_prior_source_question or "") if uses_prior_policy else question
    )
    pipeformer = successful_pipeformer_results(results)
    prior_candidates = [
        dict(item) for item in resolved_prior_candidates or []
        if dict(item or {}).get("candidate_id")
    ]
    prior_applied = [
        dict(item) for item in resolved_prior_disturbances or [] if isinstance(item, dict)
    ]
    if pipeformer or prior_candidates or prior_applied:
        return _pipeformer_contract(
            pipeformer,
            question,
            decision_policy=resolved_decision_policy,
            decision_policy_question=decision_policy_question,
            require_decision_policy=require_decision_policy,
            prior_candidate_results=prior_candidates,
            prior_applied_disturbances=prior_applied,
        )
    return _generic_contract(question, results)

def _pipeformer_contract(
    results: List[Dict[str, Any]],
    question: str,
    *,
    decision_policy: Optional[Dict[str, Any]],
    decision_policy_question: str,
    require_decision_policy: bool,
    prior_candidate_results: List[Dict[str, Any]],
    prior_applied_disturbances: List[Dict[str, Any]],
) -> Dict[str, Any]:
    baselines = _deduplicate_candidate_results(
        [item for item in results if _candidate_role(item) == "baseline"]
    )
    candidates = _deduplicate_candidate_results(
        [item for item in results if _candidate_role(item) != "baseline"]
    )

    current_action_candidates = any(
        _candidate_action(dict(item.get("arguments") or {}))
        for item in candidates
    )
    parsed = [_candidate_from_compact(value) for value in prior_candidate_results]
    parsed = [
        candidate for candidate in parsed
        if candidate["candidate_id"]
        and (not current_action_candidates or has_boundary_action(candidate))
    ]
    parsed.extend(_candidate(index, item) for index, item in enumerate(candidates, 1))
    parsed = list({item["candidate_id"].casefold(): item for item in parsed}.values())
    parsed = _deduplicate_candidate_actions(parsed)
    contract: Dict[str, Any] = {
        "answer_mode": "dispatch_comparison"
        if len(parsed) > 1
        else "single_forecast",
        "current_candidate_forecast_count": len(candidates),
        "current_decision_policy_call_count": sum(
            item.get("name") == "set_decision_policy" for item in results
        ),
        "candidate_results": [_candidate_compact(item) for item in parsed],
        "worst_case_risk_level": _worst(
            (item["risk_level"] for item in parsed), RISK_RANK, "low"
        ),
        "worst_case_intervention_label": _worst(
            (item["manual_intervention_label"] for item in parsed),
            INTERVENTION_RANK,
            "no_intervention",
        ),
    }
    if baselines:
        contract["baseline_tool_call_id"] = baselines[0].get("tool_call_id")
    if len(parsed) > 1:
        contract["decision_summary"] = _decision(
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
        contract["comparison_leaders"] = _comparison_leaders(parsed)
    assumptions = provisional_assumptions(results)
    if assumptions:
        contract["provisional_assumptions"] = assumptions
    applied_disturbances: List[Dict[str, Any]] = []
    applied_seen = set()
    for disturbance in [
        *prior_applied_disturbances,
        *_applied_disturbances(results),
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

def _latest_by_key(items: Iterable[Dict[str, Any]], key) -> List[Dict[str, Any]]:
    order: List[str] = []
    latest: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(items, 1):
        identity = key(item, index)
        if identity not in latest:
            order.append(identity)
        latest[identity] = item
    return [latest[identity] for identity in order]


def _deduplicate_candidate_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep the latest successful result for each stable candidate identifier."""
    return _latest_by_key(
        results,
        lambda item, index: str(
            dict(item.get("arguments") or {}).get("candidate_id")
            or dict(item.get("output") or {}).get("candidate_id")
            or item.get("tool_call_id")
            or f"candidate_{index}"
        ).casefold(),
    )


def _candidate_action_key(candidate: Dict[str, Any]) -> str:
    action = dict(candidate.get("action") or {})
    payload = {
        key: {
            str(variable): value
            for variable, value in sorted(dict(action.get(key) or {}).items())
        }
        for key in ("setpoints", "percentage_changes")
        if action.get(key)
    }
    return (
        "action:" + canonical_json(payload)
        if payload
        else "candidate:" + str(candidate.get("candidate_id") or "").casefold()
    )

def _deduplicate_candidate_actions(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep the latest candidate for each canonical boundary action."""
    return _latest_by_key(candidates, lambda candidate, _: _candidate_action_key(candidate))

def _first_value(
    sources: Iterable[Dict[str, Any]], key: str, *, allow_none: bool = False
) -> Any:
    for source in sources:
        value = source.get(key)
        if value is not None if allow_none else value:
            return value
    return None


def provisional_assumptions(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assumptions: Dict[str, Dict[str, Any]] = {}
    for item in results:
        output = dict(item.get("output") or {})
        arguments = dict(item.get("arguments") or {})
        prediction, _ = forecast_views(output)
        parsed_task = dict(output.get("parsed_task") or {})
        sources = [parsed_task, prediction, output]
        for source in sources:
            assumption = source.get("disturbance_assumption")
            if (
                not isinstance(assumption, dict)
                or assumption.get("source") != "llm_assumption"
            ):
                continue
            sources = (source, prediction, parsed_task, arguments)
            variable = _first_value(sources, "disturbance_variable")
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
                "direction": _first_value(sources, "disturbance_direction"),
                "magnitude_percent": _first_value(
                    sources, "disturbance_magnitude_percent", allow_none=True
                ),
                "setpoint": setpoint,
                "statement": assumption.get("statement"),
            }
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            assumptions[key] = value
    return list(assumptions.values())

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
        variable = str(_first_value((arguments, prediction), "disturbance_variable") or "")
        if not variable:
            continue
        evidence = dict(output.get("evidence") or {})
        application = next(
            (
                dict(value)
                for value in evidence.get("boundary_application_evidence") or []
                if str(dict(value or {}).get("variable") or "") == variable
            ),
            {},
        )
        fallback = next((
            (mode, arguments[field]) for field, mode in _DISTURBANCE_ARGUMENTS
            if arguments.get(field) is not None
        ), None)
        if application:
            mode = str(application.get("mode") or "")
            requested_value = application.get("requested_value")
            before = list(application.get("input_values_before") or [])
            applied = list(application.get("input_values_applied") or [])
            verified = application.get("verified") is True
        elif fallback:
            mode, requested_value = fallback
            before = applied = []
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
            "direction": _first_value(
                (arguments, prediction, dict(output.get("parsed_task") or {})),
                "disturbance_direction",
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

def _candidate(index: int, item: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(item.get("output") or {})
    prediction, verification = forecast_views(output)
    arguments = dict(item.get("arguments") or {})
    metrics = dict(verification.get("comparable_metrics") or {})
    engineering = dict(verification.get("engineering_evidence") or {})
    comparison = dict(prediction.get("counterfactual_comparison") or {})
    category_metrics = {
        category: _compact_category_metrics(engineering.get(category), fields)
        for category, fields in ENGINEERING_METRIC_FIELDS.items()
    }
    return _candidate_from_compact(dict(
        candidate_id=str(
            arguments.get("candidate_id")
            or output.get("candidate_id")
            or f"candidate_{index}"
        ),
        tool_call_id=str(item.get("tool_call_id") or ""),
        action=_candidate_action(arguments),
        failure_count=_integer(verification.get("failure_count")),
        warning_count=_integer(verification.get("warning_count")),
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
        failed_rule_ids=list(map(str, verification.get("failed_rule_ids") or [])),
        warning_rule_ids=list(map(str, verification.get("warning_rule_ids") or [])),
        energy_consumption=(
            number_value(metrics.get("energy_consumption_delta"))
            if metrics.get("energy_consumption_delta") is not None
            else number_value(metrics.get("energy_consumption"))
        ),
        nonzero_impacted_variable_count=(
            _integer(comparison.get("nonzero_impacted_variable_count"))
            if comparison.get("nonzero_impacted_variable_count") is not None
            else None
        ),
        **{
            f"{category}_metrics": category_metrics[category]
            for category in ENGINEERING_METRIC_FIELDS
        },
        energy_metrics={
            "total": number_value(metrics.get("energy_consumption")),
            "delta_vs_baseline": number_value(
                metrics.get("energy_consumption_delta")
            ),
            "unit": metrics.get("energy_unit"),
            "variable_count": _integer(metrics.get("energy_variable_count")),
            "evaluation_status": metrics.get("energy_evaluation_status"),
        },
        baseline_reference=(
            str(metrics.get("baseline_reference"))
            if metrics.get("baseline_reference")
            else None
        ),
        category_status=dict(verification.get("category_status") or {}),
    ))

def _compact_category_metrics(value: Any, keys: Iterable[str]) -> Dict[str, Any]:
    source = dict(value or {})
    return {
        key: source[key]
        for key in keys
        if key in source and source[key] is not None
    }

def _comparison_leaders(
    candidates: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    return {
        "pressure_preservation": _leaders(
            candidates,
            lambda item: number_value(
                nested_value(
                    item.get("pressure_metrics"),
                    ("minimum_operating_window_margin", "value"),
                )
            ),
            prefer="maximum",
        ),
        "slowest_linepack_decline": [
            item["candidate_id"] for item in _linepack_best_candidates(candidates)
        ],
        "lowest_energy_consumption": _leaders(
            candidates,
            lambda item: item.get("energy_consumption"),
            prefer="minimum",
        ),
    }


def _linepack_score(item: Dict[str, Any]) -> tuple[Optional[float], ...]:
    metrics = item.get("linepack_metrics", {})
    return (
        number_value(nested_value(metrics, ("maximum_decline_from_start", "value"))),
        *(number_value(metrics.get(key)) for key in (
            "maximum_continuous_decline_minutes", "insufficient_recovery_count"
        )),
    )


def _linepack_best_candidates(
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    values = [(item, _linepack_score(item)) for item in candidates]
    comparable = [
        (item, metrics)
        for item, metrics in values
        if all(value is not None for value in metrics)
    ]
    if not comparable:
        return []
    best = min(metrics for _, metrics in comparable)
    return [item for item, metrics in comparable if metrics == best]

def _leaders(
    candidates: List[Dict[str, Any]], value_getter, *, prefer: str
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
    return [item["candidate_id"] for item, value in values if value == target]

def _candidate_action(arguments: Dict[str, Any]) -> Dict[str, Any]:
    boundary = dict(arguments.get("boundary_conditions") or {})
    if boundary.get("setpoints") or boundary.get("percentage_changes"):
        return boundary
    if str(arguments.get("disturbance_source") or "").casefold() != (
        "operator_action"
    ):
        return boundary
    variable = arguments.get("disturbance_variable") or boundary.get("disturbance_variable")
    magnitude = arguments.get("disturbance_magnitude_percent")
    if magnitude is None:
        magnitude = boundary.get("disturbance_magnitude_percent")
    direction = str(arguments.get("disturbance_direction") or boundary.get("disturbance_direction") or "").casefold()
    if not variable or magnitude is None:
        return boundary
    signed = float(magnitude)
    if direction in {"down", "up"}:
        signed = abs(signed) * (-1 if direction == "down" else 1)
    return {"percentage_changes": {str(variable): signed}}

def _decision(
    candidates: List[Dict[str, Any]],
    *,
    question: str,
    decision_policy: Optional[Dict[str, Any]],
    require_decision_policy: bool,
) -> Dict[str, Any]:
    if require_decision_policy and decision_policy is None:
        return _unsupported_decision(
            [],
            ["llm_decision_policy_tool_call"],
            ranking_policy={},
        )
    policy, policy_errors = normalize_decision_policy(decision_policy)
    if policy.get("source") == "llm_tool":
        normalized_question = " ".join(str(question).split()).casefold()
        objectives = list(policy.get("objectives") or [])
        legacy_excerpt = " ".join(str(policy.get("source_excerpt") or "").split()).casefold()
        for objective, excerpt in zip(objectives, llm_policy_excerpts(policy)):
            item = dict(objective or {})
            metric = str(item.get("metric") or "missing")
            if len(excerpt) < 4 or excerpt not in normalized_question:
                error = (
                    "decision_policy_source_not_in_user_request"
                    if len(objectives) == 1 and legacy_excerpt
                    else "decision_policy_objective_source_not_in_user_request:" + metric
                )
            elif not decision_policy_source_has_priority_signal(excerpt):
                error = "decision_policy_objective_not_a_priority:" + metric
            else:
                continue
            policy_errors.append(error)
    eliminated = [
        {"candidate_id": item["candidate_id"], "failed_rules": item["failed_rule_ids"]}
        for item in candidates
        if item["failure_count"] > 0
    ]
    viable = [item for item in candidates if item["failure_count"] == 0]
    if not viable:
        return _unsupported_decision(
            eliminated,
            ["constraint_compliant_candidate"],
            ranking_policy=policy,
        )

    if all(item["nonzero_impacted_variable_count"] == 0 for item in viable):
        return _unsupported_decision(
            eliminated,
            ["action_sensitive_forecast"],
            ranking_policy=policy,
        )

    compact_viable = [_candidate_compact(item) for item in viable]
    objective_evidence, missing_metrics = collect_objective_evidence(compact_viable, policy)
    if policy_errors or missing_metrics:
        return _unsupported_decision(
            eliminated,
            [*policy_errors, *missing_metrics],
            ranking_policy=policy,
            objective_evidence=objective_evidence,
        )

    ranked_groups = rank_candidate_groups(
        (item["candidate_id"] for item in viable), policy, objective_evidence
    )
    ranked_ids = [candidate_id for group in ranked_groups for candidate_id in group]
    by_id = {item["candidate_id"]: item for item in viable}
    selected = by_id[ranked_ids[0]]
    return {
        "status": "selected",
        "selected_candidate_id": selected["candidate_id"],
        "selected_dispatch_recommendation": selected["dispatch_recommendation"],
        "ranking_basis": objective_evidence[selected["candidate_id"]],
        "ranking_policy": policy,
        "objective_evidence": objective_evidence,
        "ranked_candidate_ids": ranked_ids,
        "ranked_candidate_groups": ranked_groups,
        "eliminated_candidates": eliminated,
        "missing_metrics": [],
    }

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

def _generic_contract(
    question: str, results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    requested = requested_artifacts(question)
    sources = []
    for item in results:
        assessment = classify_tool_evidence(item, requested=requested)
        artifacts = (
            sorted(assessment.matched_artifacts)
            or sorted(requested_artifacts(str(item.get("arguments") or "")))
            or [""]
        )
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

def _worst(values: Iterable[str], ranking: Dict[str, int], default: str) -> str:
    candidates = list(values)
    return (
        max(candidates, key=lambda value: ranking.get(value, -1))
        if candidates
        else default
    )

def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0

def _policy_unavailable_at_generation(record: Dict[str, Any]) -> bool:
    """Preserve a deliberately unsupported saved policy decision."""
    stored_decision = dict(record.get("decision_summary") or {})
    return (
        not record.get("decision_policy")
        and stored_decision.get("status") == "insufficient_evidence"
        and "llm_decision_policy_tool_call" in list(stored_decision.get("missing_metrics") or [])
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
    return build_grounding_contract(
        str(record.get("user_input") or ""),
        outputs,
        decision_policy=stored_policy if not state.decision_policy else None,
        require_decision_policy=_policy_unavailable_at_generation(record),
        prior_state=state,
    )
