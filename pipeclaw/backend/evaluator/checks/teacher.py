"""Canonical metrics for native teacher traces."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from pipeclaw.backend.grounding.evidence.tool import (
        attach_tool_arguments,
        classify_tool_evidence,
        requested_artifacts,
    )
    from pipeclaw.backend.pipeline.forecast_registry_contract import (
        authorize_forecast_registry,
    )
except ImportError:  # pragma: no cover - direct backend execution
    from grounding.evidence.tool import (
        attach_tool_arguments,
        classify_tool_evidence,
        requested_artifacts,
    )
    from pipeline.forecast_registry_contract import authorize_forecast_registry

from ..models import EvaluationContext, MetricResult
from .assumptions import prediction_view
from .common import (
    CANONICAL_METRIC_NAMES,
    PIPEFORMER_TOOL,
    disturbance_was_applied,
    horizon_is_consistent,
    mapping,
    metric,
    requested_constraints_executed,
    sequence,
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


def _calls(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in sequence(record.get("tool_calls")) if isinstance(item, Mapping)]


def _output_wrappers(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in sequence(record.get("tool_outputs")) if isinstance(item, Mapping)]


def _task_views(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    parsed = mapping(record.get("parsed_task"))
    candidates = sequence(parsed.get("candidate_forecasts"))
    if candidates:
        return [item for item in candidates if isinstance(item, Mapping)]
    return [parsed] if parsed else []


def _pipeformer_outputs(
    record: Mapping[str, Any],
    referenced_call_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    forecast_call_ids = {
        str(call.get("tool_call_id") or "")
        for call in _calls(record)
        if call.get("name") == PIPEFORMER_TOOL
    }
    wrappers = [
        item
        for item in _output_wrappers(record)
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
        and (
            resolved_output_count is None
            or int(resolved_output_count or 0) > 0
        )
    )


def _registry_ordering(record: Mapping[str, Any]) -> tuple[bool, list[str]]:
    calls = _calls(record)
    outputs_by_id = {
        str(item.get("tool_call_id") or ""): item.get("output")
        for item in _output_wrappers(record)
    }
    forecast_indices = [
        index for index, call in enumerate(calls) if call.get("name") == PIPEFORMER_TOOL
    ]
    unauthorized: list[str] = []
    for index in forecast_indices:
        completed = [
            {
                "tool_call_id": str(call.get("tool_call_id") or ""),
                "name": call.get("name"),
                "arguments": dict(mapping(call.get("arguments"))),
                "output": outputs_by_id.get(str(call.get("tool_call_id") or "")),
            }
            for call in calls[:index]
        ]
        result = authorize_forecast_registry(
            dict(mapping(calls[index].get("arguments"))),
            completed,
        )
        if not result["authorized"]:
            unauthorized.append(str(calls[index].get("tool_call_id") or ""))
    return bool(forecast_indices) and not unauthorized, unauthorized


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
        passed=not missing,
        details={
            "record_chars": size,
            "maximum_chars": maximum_chars,
            "missing_fields": missing,
        },
        teacher_variant=teacher_variant,
    )


def _pipeformer_checks(
    context: EvaluationContext,
    issues: Sequence[str],
    *,
    maximum_chars: int,
) -> list[MetricResult]:
    record = context.record
    tasks = _task_views(record)
    referenced_ids = [
        str(task.get("tool_call_id")) for task in tasks if task.get("tool_call_id")
    ]
    all_outputs = _pipeformer_outputs(record, referenced_ids)
    outputs = [item for item in all_outputs if item.get("success") is True]
    trace_status = record.get("trace_status")
    successful = bool(
        trace_status in (None, "completed")
        and outputs
        and len(tasks) == len(outputs)
    )
    registry_pass, unauthorized = _registry_ordering(record)
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
            passed=successful and all(
                prediction_view(output).get("forecast_mode") == "checkpoint_inference"
                and bool(mapping(output.get("provenance")).get("checkpoint_id"))
                for output in outputs
            ),
        ),
        metric(
            context,
            "disturbance_application",
            applicable=True,
            passed=successful and all(
                disturbance_was_applied(output, task)
                for output, task in zip(outputs, tasks)
            ),
        ),
        metric(
            context,
            "forecast_horizon",
            applicable=True,
            passed=successful and all(horizon_is_consistent(output) for output in outputs),
        ),
        metric(
            context,
            "constraint_execution",
            applicable=True,
            passed=bool(outputs) and all(requested_constraints_executed(output) for output in outputs),
        ),
        metric(
            context,
            "verification_completeness",
            applicable=True,
            passed=bool(outputs) and all(verification_is_complete(output) for output in outputs),
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
    present = {item.name for item in metrics}
    metrics.extend(
        metric(context, name, applicable=False)
        for name in CANONICAL_METRIC_NAMES
        if name not in present
    )
    by_name = {item.name: item for item in metrics}
    return [by_name[name] for name in CANONICAL_METRIC_NAMES]


def _generic_checks(
    context: EvaluationContext,
    issues: Sequence[str],
    *,
    maximum_chars: int,
) -> list[MetricResult]:
    record = context.record
    outputs = attach_tool_arguments(
        [dict(item) for item in _output_wrappers(record)],
        [dict(item) for item in _calls(record)],
    )
    requested = requested_artifacts(str(record.get("user_input") or ""))
    assessments = [classify_tool_evidence(item, requested=requested) for item in outputs]
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
    present = {item.name for item in metrics}
    metrics.extend(
        metric(
            context,
            name,
            applicable=False,
            teacher_variant="generic",
        )
        for name in CANONICAL_METRIC_NAMES
        if name not in present
    )
    by_name = {item.name: item for item in metrics}
    return [by_name[name] for name in CANONICAL_METRIC_NAMES]


def evaluate_teacher_checks(
    context: EvaluationContext,
    *,
    derive_hard_issues: bool,
    maximum_chars: int,
) -> tuple[list[MetricResult], tuple[str, ...], dict[str, Any]]:
    """Return canonical teacher metrics and the single grounding issue set."""

    if derive_hard_issues:
        from ..teacher_quality import record_quality_issues

        issues = tuple(record_quality_issues(dict(context.record)))
    else:
        issues = tuple(context.hard_issues)
    has_pipeformer = any(
        call.get("name") == PIPEFORMER_TOOL for call in _calls(context.record)
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
    return metrics, issues, {"teacher_variant": variant}
