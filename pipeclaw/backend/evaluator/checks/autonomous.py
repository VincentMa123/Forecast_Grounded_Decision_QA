"""Canonical metrics for held-out autonomous rollouts."""

from __future__ import annotations

import json
import re
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
from ..teacher_quality import VARIABLE_REFERENCE
from .assumptions import assumption_consistency, inferred_task_fields, prediction_view
from .common import (
    CANONICAL_METRIC_NAMES,
    PIPEFORMER_TOOL,
    disturbance_was_applied,
    horizon_is_consistent,
    mapping,
    metric,
    requested_constraints_executed,
    sequence,
    task_field_comparison,
    verification_is_complete,
    verification_view,
)
from .evidence import evidence_consistency


def _calls(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in sequence(record.get("tool_calls")) if isinstance(item, Mapping)]


def _output_wrappers(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in sequence(record.get("tool_outputs")) if isinstance(item, Mapping)]


def _output(wrapper: Mapping[str, Any]) -> Mapping[str, Any]:
    value = wrapper.get("output", wrapper)
    return value if isinstance(value, Mapping) else {}


def _successful_forecast_pairs(
    record: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    calls_by_id = {
        str(call.get("tool_call_id") or ""): call
        for call in _calls(record)
    }
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for wrapper in _output_wrappers(record):
        value = _output(wrapper)
        call = calls_by_id.get(str(wrapper.get("tool_call_id") or ""), {})
        is_forecast = wrapper.get("name") == PIPEFORMER_TOOL or call.get("name") == PIPEFORMER_TOOL
        if is_forecast and value.get("success", True) is not False:
            pairs.append((call, value))
    return pairs


def _student_tasks(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tasks = [
        mapping(call.get("arguments"))
        for call in _calls(record)
        if call.get("name") == PIPEFORMER_TOOL
        and isinstance(call.get("arguments"), Mapping)
    ]
    if tasks:
        return tasks
    pairs = _successful_forecast_pairs(record)
    return [prediction_view(pairs[0][1])] if pairs else []


def _task_parsing(context: EvaluationContext) -> MetricResult:
    expected_tasks = [
        item for item in sequence(context.oracle.get("tasks")) if isinstance(item, Mapping)
    ]
    if not expected_tasks:
        task = mapping(context.oracle.get("task"))
        expected_tasks = [task] if task else []
    actual_tasks = _student_tasks(context.record)
    unmatched = list(expected_tasks)
    mismatches: list[str] = []
    matched_fields: list[str] = []
    matched_count = 0
    for actual in actual_tasks:
        matched_index = next(
            (
                index
                for index, expected in enumerate(unmatched)
                if task_field_comparison(expected, actual)[0]
            ),
            None,
        )
        if matched_index is not None:
            matched_count += 1
            matched_fields.extend(
                task_field_comparison(unmatched[matched_index], actual)[2]
            )
            unmatched.pop(matched_index)
        elif unmatched:
            mismatches.extend(task_field_comparison(unmatched[0], actual)[1])
    assumed_fields = sorted(
        {
            field
            for task in expected_tasks
            for field in inferred_task_fields(task)
        }
    )
    applicable = bool(expected_tasks)
    return metric(
        context,
        "task_parsing",
        applicable=applicable,
        passed=applicable and matched_count == len(expected_tasks),
        details={
            "expected_task_count": len(expected_tasks),
            "matched_task_count": matched_count,
            "matched_fields": sorted(set(matched_fields)),
            "mismatched_fields": sorted(set(mismatches)),
            "assumed_fields": assumed_fields,
        },
    )


def _tool_metrics(
    context: EvaluationContext,
) -> tuple[MetricResult, MetricResult]:
    calls = _calls(context.record)
    teacher_names = {str(item) for item in sequence(context.oracle.get("teacher_tool_names"))}
    required = set()
    if PIPEFORMER_TOOL in teacher_names:
        required.add(PIPEFORMER_TOOL)
    if "set_decision_policy" in teacher_names:
        required.add("set_decision_policy")
    if not required:
        required = set(teacher_names)
    valid_calls = [
        call
        for call in calls
        if call.get("schema_valid") is not False
        and call.get("execution_success") is not False
    ]
    failed_calls = [call for call in calls if call not in valid_calls]
    emitted_valid = {str(call.get("name")) for call in valid_calls if call.get("name")}
    applicable = bool(teacher_names)
    last_required: dict[str, Mapping[str, Any]] = {}
    for call in calls:
        name = str(call.get("name") or "")
        if name in required:
            last_required[name] = call
    tool_result = metric(
        context,
        "tool_call",
        applicable=applicable,
        passed=(
            applicable
            and bool(valid_calls)
            and required <= emitted_valid
            and len(valid_calls) == len(calls)
        ),
        details={
            "expected_tool_names": sorted(teacher_names),
            "required_tool_names": sorted(required),
            "emitted_tool_names": [str(call.get("name")) for call in calls if call.get("name")],
            "failed_call_count": len(failed_calls),
        },
    )
    recovered = bool(failed_calls) and required <= set(last_required) and all(
        call.get("schema_valid") is not False
        and call.get("execution_success") is not False
        for call in last_required.values()
    )
    recovery_result = metric(
        context,
        "tool_recovery",
        applicable=applicable and bool(failed_calls),
        passed=recovered,
        details={
            "failed_call_count": len(failed_calls),
            "recovered_tool_names": sorted(
                name
                for name, call in last_required.items()
                if call.get("schema_valid") is not False
                and call.get("execution_success") is not False
            ),
        },
    )
    return tool_result, recovery_result


def _assumption_metric(context: EvaluationContext) -> MetricResult:
    expected_tasks = [
        item for item in sequence(context.oracle.get("tasks")) if isinstance(item, Mapping)
    ]
    assumed = sorted(
        {field for task in expected_tasks for field in inferred_task_fields(task)}
    )
    pairs = _successful_forecast_pairs(context.record)
    actual_task = mapping(pairs[0][0].get("arguments")) if pairs else {}
    output = pairs[0][1] if pairs else {}
    passed, mismatches = assumption_consistency(expected_tasks, actual_task, output)
    return metric(
        context,
        "assumption_consistency",
        applicable=bool(assumed),
        passed=passed,
        details={"assumed_fields": assumed, "mismatched_fields": mismatches},
    )


def _pipeformer_metrics(context: EvaluationContext) -> list[MetricResult]:
    pairs = _successful_forecast_pairs(context.record)
    applicable = PIPEFORMER_TOOL in {
        str(item) for item in sequence(context.oracle.get("teacher_tool_names"))
    }
    outputs = [output for _, output in pairs]
    tasks = [mapping(call.get("arguments")) for call, _ in pairs]
    checkpoint = metric(
        context,
        "checkpoint_inference",
        applicable=applicable,
        passed=applicable and bool(outputs) and all(
            prediction_view(output).get("forecast_mode") == "checkpoint_inference"
            and bool(mapping(output.get("provenance")).get("checkpoint_id"))
            for output in outputs
        ),
    )
    disturbance = metric(
        context,
        "disturbance_application",
        applicable=applicable,
        passed=applicable and bool(outputs) and all(
            disturbance_was_applied(output, task)
            for output, task in zip(outputs, tasks)
        ),
    )
    horizon = metric(
        context,
        "forecast_horizon",
        applicable=applicable,
        passed=applicable and bool(outputs) and all(horizon_is_consistent(output) for output in outputs),
    )
    constraint_execution = metric(
        context,
        "constraint_execution",
        applicable=applicable,
        passed=applicable and bool(outputs) and all(requested_constraints_executed(output) for output in outputs),
    )
    complete = metric(
        context,
        "verification_completeness",
        applicable=applicable,
        passed=applicable and bool(outputs) and all(verification_is_complete(output) for output in outputs),
    )

    expected_statuses: Mapping[str, Any] = {}
    for item in sequence((context.reference or {}).get("tool_outputs")):
        if not isinstance(item, Mapping):
            continue
        value = _output(item)
        if item.get("name") == PIPEFORMER_TOOL or verification_view(value):
            expected_statuses = mapping(verification_view(value).get("category_status"))
            if expected_statuses:
                break
    expected_constraints = {
        str(item) for item in sequence(context.oracle.get("required_constraints"))
    }
    actual_statuses = mapping(
        verification_view(outputs[0]).get("category_status") if outputs else {}
    )
    judgment_applicable = applicable and bool(expected_constraints or expected_statuses)
    judgment_pass = bool(actual_statuses) and all(
        str(actual_statuses.get(key)) == str(value)
        for key, value in expected_statuses.items()
    )
    if expected_constraints:
        judgment_pass = judgment_pass and expected_constraints <= {
            str(key) for key in actual_statuses
        }
    judgment = metric(
        context,
        "constraint_judgment",
        applicable=judgment_applicable,
        passed=judgment_pass,
        details={
            "expected_constraints": sorted(expected_constraints),
            "actual_constraints": sorted(str(key) for key in actual_statuses),
        },
    )
    return [checkpoint, disturbance, horizon, constraint_execution, judgment, complete]


def _registry_metric(context: EvaluationContext) -> MetricResult:
    calls = _calls(context.record)
    outputs_by_id = {
        str(item.get("tool_call_id") or ""): _output(item)
        for item in _output_wrappers(context.record)
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
        authorization = authorize_forecast_registry(
            dict(mapping(calls[index].get("arguments"))),
            completed,
        )
        if not authorization["authorized"]:
            unauthorized.append(str(calls[index].get("tool_call_id") or ""))
    applicable = bool(forecast_indices)
    return metric(
        context,
        "registry_ordering",
        applicable=applicable,
        passed=applicable and not unauthorized,
        details={"unauthorized_forecast_call_ids": unauthorized},
    )


# The data-file branch must precede the identifier branch: ``ghost_station.csv``
# would otherwise match as the stem ``ghost_station``, so a fabricated file name
# could be satisfied by an unrelated variable of the same name in the corpus.
#
# The identifier branch is the project's canonical variable pattern rather than
# "any snake_case token".  The looser form matched schema field names the answer
# names in prose while explaining its own output -- ``forecast_id``,
# ``recovery_ratio``, ``selected_candidate_id`` -- and SFT compaction strips
# those keys from ``tool_outputs``, so the check scored compaction rather than
# fabrication (9 of 24 PipeFormer teacher records, every one a false positive).
_IDENTIFIER_CLAIM = re.compile(
    r"[\w\-]+\.(?:csv|xlsx?|json|txt)|" + VARIABLE_REFERENCE.pattern
)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _claim_support_metric(context: EvaluationContext) -> MetricResult:
    """Are the identifiers named in the answer present in observed evidence?

    ``evidence_consistency`` only inspects numbers, so an answer that cites no
    figures is never checked.  Identifiers (canonical variable IDs, CSV file
    names) are the other thing a student can fabricate, and they appear in the
    records numbers do not reach.
    """

    answer = str(context.record.get("final_answer") or "")
    claims = {match.group(0) for match in _IDENTIFIER_CLAIM.finditer(answer)}
    haystack: list[str] = []
    for source in (context.record, context.reference or {}):
        for key in ("tool_outputs", "tool_calls", "state_before", "recent_turns"):
            value = source.get(key)
            if value:
                haystack.append(_json_text(value))
    haystack.append(_json_text(context.oracle))
    corpus = "\n".join(haystack)
    unsupported = sorted(claim for claim in claims if claim not in corpus)
    applicable = bool(claims) and bool(corpus.strip())
    return metric(
        context,
        "answer_claim_support",
        applicable=applicable,
        passed=applicable and not unsupported,
        details={
            "claim_count": len(claims),
            "unsupported_count": len(unsupported),
            "unsupported_identifiers": unsupported[:20],
        },
    )


def _label_metric(
    context: EvaluationContext,
    name: str,
    oracle_key: str,
    output_key: str,
) -> MetricResult:
    expected = context.oracle.get(oracle_key)
    pairs = _successful_forecast_pairs(context.record)
    output = pairs[0][1] if pairs else {}
    verification = verification_view(output)
    actual = verification.get(output_key, output.get(output_key))
    if output_key == "human_intervention_label":
        actual = verification.get(
            output_key,
            output.get(output_key, output.get("manual_intervention_label")),
        )
    applicable = expected is not None
    return metric(
        context,
        name,
        applicable=applicable,
        passed=applicable and actual == expected,
        details={"expected": expected, "actual": actual},
    )


# The shortest genuine ``final_answer`` in the frozen 1,140-record release is 10
# characters, so a floor below that rejects degenerate stubs without failing any
# real record.  This is a non-degeneracy floor, not a semantic judgement: for the
# tool-less OpenClaw records ``answer_completeness`` is the only always-scored
# deliverable, and accepting ``"x"`` let an empty rollout score as a pass.
_MIN_SUBSTANTIVE_ANSWER_CHARS = 8


def _answer_metric(context: EvaluationContext) -> MetricResult:
    answer = str(context.record.get("final_answer") or "")
    stripped = answer.strip()
    applicable = bool(str((context.reference or {}).get("final_answer") or "").strip()) or bool(stripped)
    return metric(
        context,
        "answer_completeness",
        applicable=applicable,
        passed=applicable and len(stripped) >= _MIN_SUBSTANTIVE_ANSWER_CHARS,
        details={
            "answer_length": len(answer),
            "minimum_length": _MIN_SUBSTANTIVE_ANSWER_CHARS,
        },
    )


def _json_metric(context: EvaluationContext) -> MetricResult:
    calls = _calls(context.record)
    errors = sequence(context.record.get("json_errors"))
    applicable = bool(sequence(context.oracle.get("teacher_tool_names")))
    return metric(
        context,
        "json_validity",
        applicable=applicable,
        passed=applicable and bool(calls) and not errors and all(
            call.get("schema_valid") is not False for call in calls
        ),
        details={"error_count": len(errors)},
    )


def _artifact_metric(context: EvaluationContext) -> MetricResult:
    requested = requested_artifacts(str((context.reference or {}).get("user_input") or ""))
    if not requested:
        return metric(
            context,
            "artifact_evidence",
            applicable=False,
            details={"requested_artifacts": []},
        )
    enriched = attach_tool_arguments(
        [dict(item) for item in _output_wrappers(context.record)],
        [dict(item) for item in _calls(context.record)],
    )
    assessments = [
        classify_tool_evidence(item, requested=requested) for item in enriched
    ]
    matched = sorted(
        {
            artifact
            for assessment in assessments
            for artifact in assessment.matched_artifacts
            if assessment.evidence_found
        }
    )
    missing = sorted(set(requested) - set(matched))
    return metric(
        context,
        "artifact_evidence",
        applicable=True,
        passed=not missing and any(item.evidence_found for item in assessments),
        details={
            "requested_artifacts": list(requested),
            "matched_artifacts": matched,
            "missing_artifacts": missing,
            "evidence_states": [item.state.value for item in assessments],
            "evidence_reasons": [item.reason for item in assessments],
        },
    )


def _record_contract(context: EvaluationContext) -> MetricResult:
    required = ("tool_calls", "tool_outputs", "final_answer")
    missing = [name for name in required if name not in context.record]
    return metric(
        context,
        "record_contract",
        applicable=True,
        passed=not missing,
        details={"missing_fields": missing},
    )


def _portability_diagnostics(record: Mapping[str, Any]) -> dict[str, int]:
    calls = _calls(record)
    rebased = [call for call in calls if call.get("cwd_rebased")]
    return {
        "cwd_rebased_calls": len(rebased),
        "records_with_cwd_rebased": int(bool(rebased)),
        "rebased_execution_successes": sum(
            call.get("execution_success") is not False for call in rebased
        ),
        "rebased_execution_failures": sum(
            call.get("execution_success") is False for call in rebased
        ),
        "portable_path_normalization_calls": sum(
            bool(call.get("portable_path_normalization")) for call in calls
        ),
    }


def evaluate_autonomous_checks(
    context: EvaluationContext,
) -> tuple[list[MetricResult], dict[str, Any]]:
    """Return every canonical autonomous metric plus unscored diagnostics."""

    tool_call, tool_recovery = _tool_metrics(context)
    claim_support = _claim_support_metric(context)
    metrics = [
        _task_parsing(context),
        _assumption_metric(context),
        tool_call,
        *_pipeformer_metrics(context),
        _registry_metric(context),
        _label_metric(context, "risk", "risk_level", "risk_level"),
        _label_metric(
            context,
            "manual_intervention",
            "manual_intervention_label",
            "human_intervention_label",
        ),
        _label_metric(
            context,
            "dispatch",
            "dispatch_recommendation",
            "dispatch_recommendation",
        ),
        evidence_consistency(context),
        _answer_metric(context),
        _json_metric(context),
        _artifact_metric(context),
        _record_contract(context),
        tool_recovery,
    ]
    by_name = {item.name: item for item in metrics}
    ordered = [by_name[name] for name in CANONICAL_METRIC_NAMES]
    ordered.append(tool_recovery)
    ordered.append(claim_support)
    return ordered, {"portability": _portability_diagnostics(context.record)}
