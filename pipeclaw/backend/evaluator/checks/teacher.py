from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pipeclaw.backend.grounding.evidence.tool import (
    attach_tool_arguments,
    classify_tool_evidence,
    requested_artifacts,
)
from pipeclaw.backend.evaluator.numeric_grounding import (
    numeric_claims_are_grounded,
    numeric_grounding_evidence,
)
from pipeclaw.backend.evaluator.quality_references import numeric_claim_values

from ..models import EvaluationContext, MetricResult
from .common import (
    PIPEFORMER_TOOL,
    calls,
    checkpoint_inference_used,
    disturbance_was_applied,
    forecast_registry_order,
    horizon_is_consistent,
    mapping,
    metric,
    ordered_canonical_metrics,
    output_wrappers,
    requested_constraints_executed,
    sequence,
    task_views,
    verification_is_complete,
)
from .evidence import evidence_consistency


REQUIRED_RECORD_FIELDS = (
    "sample_id",
    "scenario_id",
    "scenario_type",
    "user_input",
    "parsed_task",
    "tool_calls",
    "tool_outputs",
    "prediction_summary",
    "constraint_check",
    "evidence",
    "risk_level",
    "manual_intervention_label",
    "dispatch_recommendation",
    "final_answer",
    "quality_flag",
)
TEACHER_TRACE_REQUIRED_FIELDS = (
    *REQUIRED_RECORD_FIELDS[:3],
    "state_before",
    "recent_turns",
    *REQUIRED_RECORD_FIELDS[3:],
)
TEACHER_TRACE_EXPECTED_TYPES = {
    "sample_id": str,
    "scenario_id": str,
    "scenario_type": str,
    "state_before": dict,
    "recent_turns": list,
    "user_input": str,
    "parsed_task": dict,
    "tool_calls": list,
    "tool_outputs": list,
    "prediction_summary": dict,
    "constraint_check": dict,
    "evidence": dict,
    "final_answer": str,
    "quality_flag": str,
}
ENTIRELY_SAFE_CLAIM = re.compile(
    r"完全安全|无任何风险|没有任何风险|所有(?:校核|规则|约束)均?通过|各项(?:校核|规则|约束)均?通过"
    r"|\b(?:entirely|completely|fully)\s+safe\b"
    r"|\ball\s+(?:requested\s+)?(?:checks|constraints|rules)\s+pass(?:ed)?\b"
    r"|\bno\s+(?:operational\s+)?risk\b",
    re.IGNORECASE,
)
REDUCE_UPSTREAM_INJECTION = re.compile(
    r"(?:减少|降低|下调|削减).{0,24}(?:上游|气源).{0,16}(?:注气|供气|供给|流量)"
    r"|(?:上游|气源).{0,16}(?:注气|供气|供给|流量).{0,24}(?:减少|降低|下调|削减)"
    r"|\b(?:reduce|decrease|lower|cut)\b.{0,40}\b(?:upstream\s+)?(?:injection|supply|inflow)\b",
    re.IGNORECASE,
)
RAISE_COMPRESSOR_LOAD = re.compile(
    r"(?:提高|增加|上调).{0,24}压缩机.{0,12}负荷|压缩机.{0,12}负荷.{0,24}(?:提高|增加|上调)"
    r"|\b(?:raise|increase|boost)\b.{0,40}\bcompressor\s+load\b",
    re.IGNORECASE,
)


def _pipeformer_outputs(
    record: Mapping[str, Any],
    referenced_call_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    forecast_call_ids = {
        str(call.get("tool_call_id") or "")
        for call in calls(record)
        if call.get("name") == PIPEFORMER_TOOL
    }
    wrappers = [
        item
        for item in output_wrappers(record)
        if item.get("name") == PIPEFORMER_TOOL
        or str(item.get("tool_call_id") or "") in forecast_call_ids
    ]
    by_id = {
        str(item.get("tool_call_id") or ""): mapping(item.get("output"))
        for item in wrappers
    }
    if referenced_call_ids:
        return [by_id[call_id] for call_id in referenced_call_ids if call_id in by_id]
    return [mapping(item.get("output")) for item in wrappers]


def _task_is_complete(task: Mapping[str, Any]) -> bool:
    variable = str(task.get("disturbance_variable") or "")
    boundary = mapping(task.get("boundary_conditions"))
    has_change = (
        task.get("disturbance_magnitude_percent") is not None
        or variable in mapping(boundary.get("setpoints"))
        or variable in mapping(boundary.get("percentage_changes"))
    )
    resolved_output_count = task.get("resolved_output_variable_count")
    return bool(
        task.get("case_id")
        and variable
        and has_change
        and task.get("forecast_horizon_minutes")
        and task.get("constraint_verification_types")
        and not sequence(task.get("unresolved_attention_targets"))
        and not sequence(task.get("unresolved_output_state_variables"))
        and not sequence(task.get("invalid_normalized_variables"))
        and (resolved_output_count is None or int(resolved_output_count or 0) > 0)
    )


def _record_contract(
    context: EvaluationContext,
    *,
    teacher_variant: str,
    maximum_chars: int,
) -> MetricResult:
    missing = [name for name in REQUIRED_RECORD_FIELDS if name not in context.record]
    size = len(
        json.dumps(
            dict(context.record),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return metric(
        context,
        "record_contract",
        applicable=True,
        passed=not missing and size <= maximum_chars,
        details={
            "record_chars": size,
            "maximum_chars": maximum_chars,
            "missing_fields": missing,
        },
        teacher_variant=teacher_variant,
    )


def teacher_trace_diagnostics(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return Task 1 compatibility diagnostics from the canonical evaluator."""
    missing = [name for name in TEACHER_TRACE_REQUIRED_FIELDS if name not in record]
    invalid_types = [
        name
        for name, expected in TEACHER_TRACE_EXPECTED_TYPES.items()
        if name in record and not isinstance(record[name], expected)
    ]
    invalid_types.extend(
        name
        for name in (
            "risk_level",
            "manual_intervention_label",
            "dispatch_recommendation",
        )
        if name in record
        and record[name] is not None
        and not isinstance(record[name], str)
    )
    schema_issues = [f"missing:{name}" for name in missing]
    schema_issues.extend(f"invalid_type:{name}" for name in invalid_types)

    answer = str(record.get("final_answer") or "")
    constraint = dict(record.get("constraint_check") or {})
    category_status = dict(constraint.get("category_status") or {})
    rule_status = dict(constraint.get("rule_status") or {})
    rule_issues: list[str] = []
    if constraint:
        if constraint.get("risk_level") is not None and record.get(
            "risk_level"
        ) != constraint.get("risk_level"):
            rule_issues.append("risk_level_disagrees_with_constraint_check")
        if constraint.get("human_intervention_label") is not None and record.get(
            "manual_intervention_label"
        ) != constraint.get("human_intervention_label"):
            rule_issues.append("intervention_label_disagrees_with_constraint_check")
        nonpass = any(
            value in {"warning", "fail"} for value in category_status.values()
        ) or any(value in {"warning", "fail"} for value in rule_status.values())
        if nonpass and ENTIRELY_SAFE_CLAIM.search(answer):
            rule_issues.append("final_answer_claims_entirely_safe_despite_nonpass_rule")
        pressure_fail = category_status.get("pressure") == "fail" or any(
            str(flag).startswith("pressure_violation")
            for flag in constraint.get("triggered_flags") or []
        )
        if pressure_fail and record.get("risk_level") == "low":
            rule_issues.append("pressure_violation_cannot_have_low_risk")
        if (
            pressure_fail
            and record.get("manual_intervention_label") == "no_intervention"
        ):
            rule_issues.append("pressure_violation_cannot_require_no_intervention")

    dispatch_issues: list[str] = []
    if constraint:
        flags = {str(value) for value in constraint.get("triggered_flags") or []}
        pressure_fail = (
            category_status.get("pressure") == "fail"
            or any(value.startswith("pressure_violation") for value in flags)
            or rule_status.get("node_pressure_operating_window") == "fail"
        )
        compressor_overload = (
            "compressor_overload" in flags
            or rule_status.get("compressor_load_limit") == "fail"
        )
        dispatch = "\n".join(
            str(value).strip()
            for value in (record.get("dispatch_recommendation") or "", answer)
            if str(value).strip()
        )
        if pressure_fail and REDUCE_UPSTREAM_INJECTION.search(dispatch):
            dispatch_issues.append(
                "pressure_violation_recommends_reducing_upstream_injection"
            )
        if compressor_overload and RAISE_COMPRESSOR_LOAD.search(dispatch):
            dispatch_issues.append(
                "compressor_overload_recommends_raising_compressor_load"
            )
        if (
            constraint.get("dispatch_recommendation")
            and record.get("dispatch_recommendation")
            and str(constraint["dispatch_recommendation"]).strip()
            != str(record["dispatch_recommendation"]).strip()
        ):
            dispatch_issues.append(
                "dispatch_recommendation_disagrees_with_constraint_check"
            )
    grounded = numeric_claims_are_grounded(
        answer,
        str(record.get("user_input") or ""),
        numeric_grounding_evidence(dict(record)),
    )
    rule_check = (
        {"status": "not_applicable", "issues": []}
        if not constraint
        else {
            "status": "pass" if not rule_issues else "fail",
            "issues": rule_issues,
            "overall_constraint_status": constraint.get("overall_status"),
        }
    )
    dispatch_check = (
        {"status": "not_applicable", "issues": []}
        if not constraint
        else {
            "status": "pass" if not dispatch_issues else "fail",
            "issues": dispatch_issues,
            "pressure_failure_present": pressure_fail,
            "compressor_overload_present": compressor_overload,
        }
    )
    return {
        "schema": {
            "status": "pass" if not schema_issues else "fail",
            "issues": schema_issues,
        },
        "numerical_consistency": {
            "status": "pass" if grounded else "fail",
            "claimed_numeric_value_count": len(numeric_claim_values(answer)),
            "issues": [] if grounded else ["unsupported_numerical_claim"],
        },
        "rule_consistency": rule_check,
        "dispatch_consistency": dispatch_check,
    }


def _pipeformer_checks(
    context: EvaluationContext,
    issues: Sequence[str],
    *,
    maximum_chars: int,
) -> list[MetricResult]:
    record = context.record
    tool_calls = calls(record)
    tasks = task_views(record)
    referenced_ids = [
        str(task.get("tool_call_id")) for task in tasks if task.get("tool_call_id")
    ]
    all_outputs = _pipeformer_outputs(record, referenced_ids)
    outputs = [item for item in all_outputs if item.get("success") is True]
    trace_status = record.get("trace_status")
    successful = bool(
        trace_status in (None, "completed") and outputs and len(tasks) == len(outputs)
    )
    outputs_by_id = {
        str(item.get("tool_call_id") or ""): item.get("output")
        for item in output_wrappers(record)
    }
    registry_pass, unauthorized = forecast_registry_order(tool_calls, outputs_by_id)
    metrics = [
        metric(
            context,
            "task_parsing",
            applicable=True,
            passed=bool(tasks) and all(_task_is_complete(task) for task in tasks),
            details={"task_count": len(tasks)},
        ),
        metric(
            context,
            "tool_call",
            applicable=True,
            passed=successful,
            details={
                "successful_output_count": len(outputs),
                "trace_status": trace_status,
            },
        ),
        metric(
            context,
            "checkpoint_inference",
            applicable=True,
            passed=successful
            and all(checkpoint_inference_used(output) for output in outputs),
        ),
        metric(
            context,
            "disturbance_application",
            applicable=True,
            passed=successful
            and all(
                disturbance_was_applied(output, task)
                for output, task in zip(outputs, tasks)
            ),
        ),
        metric(
            context,
            "forecast_horizon",
            applicable=True,
            passed=successful
            and all(horizon_is_consistent(output) for output in outputs),
        ),
        metric(
            context,
            "constraint_execution",
            applicable=True,
            passed=bool(outputs)
            and all(requested_constraints_executed(output) for output in outputs),
        ),
        metric(
            context,
            "verification_completeness",
            applicable=True,
            passed=bool(outputs)
            and all(verification_is_complete(output) for output in outputs),
        ),
        metric(
            context,
            "registry_ordering",
            applicable=True,
            passed=registry_pass,
            details={"unauthorized_forecast_call_ids": unauthorized},
        ),
        evidence_consistency(context, issues=issues),
        _record_contract(
            context,
            teacher_variant="pipeformer",
            maximum_chars=maximum_chars,
        ),
    ]
    return ordered_canonical_metrics(context, metrics)


def _generic_checks(
    context: EvaluationContext,
    issues: Sequence[str],
    *,
    maximum_chars: int,
) -> list[MetricResult]:
    record = context.record
    outputs = attach_tool_arguments(
        [dict(item) for item in output_wrappers(record)],
        [dict(item) for item in calls(record)],
    )
    requested = requested_artifacts(str(record.get("user_input") or ""))
    assessments = [
        classify_tool_evidence(item, requested=requested) for item in outputs
    ]
    failed_count = sum(not item.evidence_found for item in assessments)
    successful_count = sum(item.evidence_found for item in assessments)
    requested_ok = "requested_evidence_not_retrieved" not in issues
    unresolved = bool(failed_count and not successful_count) or not requested_ok
    completed = record.get("trace_status") in (None, "completed")
    metrics = [
        metric(
            context,
            "task_parsing",
            applicable=True,
            passed=completed,
            details={"trace_status": record.get("trace_status")},
            teacher_variant="generic",
        ),
        metric(
            context,
            "answer_completeness",
            applicable=True,
            passed=bool(str(record.get("final_answer") or "").strip()),
            teacher_variant="generic",
        ),
        metric(
            context,
            "tool_call",
            applicable=True,
            passed=not unresolved,
            details={
                "failed_tool_count": failed_count,
                "successful_tool_count": successful_count,
                "requested_artifacts": list(requested),
                "evidence_states": [item.state.value for item in assessments],
                "evidence_reasons": [item.reason for item in assessments],
                "recovered": bool(failed_count) and not unresolved,
            },
            teacher_variant="generic",
        ),
        evidence_consistency(
            context,
            issues=issues,
            teacher_variant="generic",
        ),
        _record_contract(
            context,
            teacher_variant="generic",
            maximum_chars=maximum_chars,
        ),
    ]
    return ordered_canonical_metrics(
        context,
        metrics,
        teacher_variant="generic",
    )


def evaluate_teacher_checks(
    context: EvaluationContext,
    *,
    derive_hard_issues: bool,
    maximum_chars: int,
) -> tuple[list[MetricResult], tuple[str, ...], dict[str, Any]]:
    """Return canonical teacher metrics and the single grounding issue set."""

    teacher_diagnostics = teacher_trace_diagnostics(context.record)
    if derive_hard_issues:
        from ..answer_quality import record_answer_quality_issues

        issues = tuple(record_answer_quality_issues(dict(context.record)))
    else:
        issues = tuple(context.hard_issues)
    has_pipeformer = any(
        call.get("name") == PIPEFORMER_TOOL for call in calls(context.record)
    )
    if has_pipeformer:
        metrics = _pipeformer_checks(
            context,
            issues,
            maximum_chars=maximum_chars,
        )
        variant = "pipeformer"
    else:
        metrics = _generic_checks(
            context,
            issues,
            maximum_chars=maximum_chars,
        )
        variant = "generic"
    return (
        metrics,
        issues,
        {
            "teacher_variant": variant,
            "teacher_trace_checks": teacher_diagnostics,
        },
    )
