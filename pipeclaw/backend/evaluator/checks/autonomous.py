from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from typing import Any, NamedTuple

from pipeclaw.backend.grounding.evidence.tool import (
    attach_tool_arguments,
    classify_tool_evidence,
    requested_artifacts,
)
from ..models import EvaluationContext, MetricResult
from ..quality_references import VARIABLE_REFERENCE
from .assumptions import assumption_consistency, inferred_task_fields, prediction_view
from .common import (
    PIPEFORMER_TOOL,
    calls,
    case_identity_matches,
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
    task_field_comparison,
    verification_is_complete,
    verification_view,
)
from .evidence import evidence_consistency


def _output(wrapper: Mapping[str, Any]) -> Mapping[str, Any]:
    value = wrapper.get("output", wrapper)
    return value if isinstance(value, Mapping) else {}


def _successful_forecast_pairs(
    record: Mapping[str, Any],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    calls_by_id = {str(call.get("tool_call_id") or ""): call for call in calls(record)}
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for wrapper in output_wrappers(record):
        value = _output(wrapper)
        call = calls_by_id.get(str(wrapper.get("tool_call_id") or ""), {})
        is_forecast = (
            wrapper.get("name") == PIPEFORMER_TOOL
            or call.get("name") == PIPEFORMER_TOOL
        )
        if is_forecast and value.get("success", True) is not False:
            pairs.append((call, value))
    return pairs


class _ResolvedForecast(NamedTuple):
    call: Mapping[str, Any]
    output: Mapping[str, Any]
    arguments: dict[str, Any]


class _ForecastIndex(NamedTuple):
    successful: tuple[_ResolvedForecast, ...]
    tasks: tuple[dict[str, Any], ...]


def _resolve_forecast_arguments(
    call: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **prediction_view(output),
        **mapping(output.get("parsed_task")),
        **mapping(call.get("arguments")),
    }


def _resolved_actual_call(
    actual_call: Mapping[str, Any],
    forecast_index: _ForecastIndex,
    actual_arguments: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if actual_arguments is not None:
        return {**actual_call, "arguments": actual_arguments}
    call_id = str(actual_call.get("tool_call_id") or "")
    entry = next(
        (
            entry
            for entry in forecast_index.successful
            if str(entry.call.get("tool_call_id") or "") == call_id
        ),
        None,
    )
    return {
        **actual_call,
        "arguments": (
            _resolve_forecast_arguments(actual_call, entry.output)
            if entry is not None
            else mapping(actual_call.get("arguments"))
        ),
    }


def _resolved_forecast_index(record: Mapping[str, Any]) -> _ForecastIndex:
    """Resolve one episode's forecast calls once for all matching checks."""

    pairs = _successful_forecast_pairs(record)
    first_outputs: dict[str, Mapping[str, Any]] = {}
    for call, output in pairs:
        first_outputs.setdefault(str(call.get("tool_call_id") or ""), output)
    successful = tuple(
        _ResolvedForecast(
            call,
            output,
            _resolve_forecast_arguments(
                call,
                first_outputs[str(call.get("tool_call_id") or "")],
            ),
        )
        for call, output in pairs
    )
    tasks = tuple(
        _resolve_forecast_arguments(
            call,
            first_outputs.get(str(call.get("tool_call_id") or ""), {}),
        )
        for call in calls(record)
        if call.get("name") == PIPEFORMER_TOOL
        and isinstance(call.get("arguments"), Mapping)
    )
    if not tasks:
        tasks = tuple(
            dict(prediction_view(entry.output)) for entry in successful[:1]
        )
    return _ForecastIndex(successful, tasks)


def _same_forecast_action(
    expected_call: Mapping[str, Any],
    actual_call: Mapping[str, Any],
    *,
    ignored_fields: frozenset[str] = frozenset(),
) -> bool:
    """Match legacy partial teacher calls without equating different actions."""
    expected = mapping(expected_call.get("arguments"))
    actual = mapping(actual_call.get("arguments"))
    scalar_keys = (
        "case_id",
        "disturbance_variable",
        "disturbance_direction",
        "disturbance_magnitude_percent",
        "disturbance_setpoint",
        "forecast_horizon_minutes",
    )
    if not expected:
        return False
    for key in scalar_keys:
        if key not in expected or key in ignored_fields:
            continue
        if key == "case_id":
            if not case_identity_matches(expected, actual):
                return False
        elif actual.get(key) != expected[key]:
            return False
    if "boundary_conditions" not in expected:
        return True
    expected_boundary = mapping(expected.get("boundary_conditions"))
    actual_boundary = mapping(actual.get("boundary_conditions"))
    return all(
        dict(mapping(expected_boundary.get(key)))
        == dict(mapping(actual_boundary.get(key)))
        for key in ("percentage_changes", "setpoints")
    )


def _matching_reference_output(
    context: EvaluationContext,
    actual_call: Mapping[str, Any],
    *,
    forecast_index: _ForecastIndex | None = None,
    reference_index: _ForecastIndex | None = None,
    actual_arguments: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    forecast_index = forecast_index or _resolved_forecast_index(context.record)
    reference_index = reference_index or _resolved_forecast_index(
        context.reference or {}
    )
    resolved_call = _resolved_actual_call(
        actual_call,
        forecast_index,
        actual_arguments,
    )
    for entry in reference_index.successful:
        expected_call = {**entry.call, "arguments": entry.arguments}
        if _same_forecast_action(expected_call, resolved_call):
            return entry.output
    return None


def _matches_reference_contract(
    context: EvaluationContext,
    actual_call: Mapping[str, Any],
    *,
    forecast_index: _ForecastIndex | None = None,
    reference_index: _ForecastIndex | None = None,
    actual_arguments: Mapping[str, Any] | None = None,
) -> bool:
    """Accept a different action only when its underlying task still matches."""

    forecast_index = forecast_index or _resolved_forecast_index(context.record)
    reference_index = reference_index or _resolved_forecast_index(
        context.reference or {}
    )
    resolved_call = _resolved_actual_call(
        actual_call,
        forecast_index,
        actual_arguments,
    )
    return any(
        task_field_comparison(entry.arguments, resolved_call["arguments"])[0]
        for entry in reference_index.successful
    )


def _oracle_tasks(context: EvaluationContext) -> list[Mapping[str, Any]]:
    return [
        item
        for item in sequence(context.oracle.get("tasks"))
        if isinstance(item, Mapping)
    ]


def _assumed_task_fields(tasks: list[Mapping[str, Any]]) -> list[str]:
    return sorted({field for task in tasks for field in inferred_task_fields(task)})


def _task_parsing(
    context: EvaluationContext,
    forecast_index: _ForecastIndex,
) -> MetricResult:
    expected_tasks = _oracle_tasks(context)
    if not expected_tasks:
        task = mapping(context.oracle.get("task"))
        expected_tasks = [task] if task else []
    teacher_task_count = len(expected_tasks)
    distinct_tasks: list[Mapping[str, Any]] = []
    for task in expected_tasks:
        if not any(
            task_field_comparison(task, existing)[0]
            and task_field_comparison(existing, task)[0]
            for existing in distinct_tasks
        ):
            distinct_tasks.append(task)
    expected_tasks = distinct_tasks
    actual_tasks = list(forecast_index.tasks)
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
    assumed_fields = _assumed_task_fields(expected_tasks)
    applicable = bool(expected_tasks)
    return metric(
        context,
        "task_parsing",
        applicable=applicable,
        passed=applicable and matched_count == len(expected_tasks),
        details={
            "teacher_task_count": teacher_task_count,
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
    tool_calls = calls(context.record)
    teacher_names = {
        str(item) for item in sequence(context.oracle.get("teacher_tool_names"))
    }
    is_openclaw = str(
        (context.reference or {}).get("scenario_type") or ""
    ).casefold() in {
        "openclaw",
        "pipeclaw",
    }

    def capabilities(names: set[str]) -> set[str]:
        result = names - {"write_file", "edit_file"}
        if names & {"write_file", "edit_file"}:
            result.add("workspace_mutation")
        return result

    if is_openclaw:
        required = capabilities(teacher_names)
        state_evidence = mapping(
            mapping((context.reference or {}).get("state_before")).get(
                "verified_evidence"
            )
        )
        question = str((context.reference or {}).get("user_input") or "")
        if (
            required == {"read_file"}
            and state_evidence
            and not requested_artifacts(question)
        ):
            required.clear()
    else:
        required = {
            name
            for name in (PIPEFORMER_TOOL, "set_decision_policy")
            if name in teacher_names
        } or set(teacher_names)
    valid_calls = [
        call
        for call in tool_calls
        if call.get("schema_valid") is not False
        and call.get("execution_success") is not False
    ]
    failed_calls = [call for call in tool_calls if call not in valid_calls]
    emitted_names = {str(call.get("name")) for call in valid_calls if call.get("name")}
    emitted_valid = capabilities(emitted_names) if is_openclaw else emitted_names
    applicable = bool(required or tool_calls)
    failed_names = {str(call.get("name")) for call in failed_calls if call.get("name")}
    failed_capabilities = capabilities(failed_names) if is_openclaw else failed_names
    recovery_targets = required | failed_capabilities
    last_required: dict[str, Mapping[str, Any]] = {}
    for call in tool_calls:
        name = str(call.get("name") or "")
        call_capabilities = capabilities({name}) if is_openclaw else {name}
        for capability in call_capabilities & recovery_targets:
            last_required[capability] = call
    # Keep recovery separate from clean tool-call correctness: a successful
    # retry passes ``tool_recovery`` but does not erase the failed call.
    failure_signatures: dict[str, int] = {}
    for call in failed_calls:
        signature = json.dumps(
            [call.get("name"), call.get("arguments")], sort_keys=True, default=str
        )
        failure_signatures[signature] = failure_signatures.get(signature, 0) + 1
    repeated_failures = [s for s, count in failure_signatures.items() if count > 1]
    success_signatures: dict[str, int] = {}
    for call in valid_calls:
        signature = json.dumps(
            [call.get("name"), call.get("arguments")],
            sort_keys=True,
            default=str,
        )
        success_signatures[signature] = success_signatures.get(signature, 0) + 1
    duplicate_successes = sum(
        count - 1 for count in success_signatures.values() if count > 1
    )
    tool_result = metric(
        context,
        "tool_call",
        applicable=applicable,
        passed=(
            applicable
            and required <= emitted_valid
            and not failed_calls
            and not duplicate_successes
        ),
        details={
            "expected_tool_names": sorted(teacher_names),
            "required_tool_names": sorted(required),
            "emitted_tool_names": [
                str(call.get("name")) for call in tool_calls if call.get("name")
            ],
            "failed_call_count": len(failed_calls),
            "repeated_failure_signatures": len(repeated_failures),
            "successful_call_count": len(valid_calls),
            "total_call_count": len(tool_calls),
            "call_success_rate": (
                len(valid_calls) / len(tool_calls) if tool_calls else None
            ),
            "duplicate_successful_call_count": duplicate_successes,
        },
    )
    recovered = (
        bool(failed_calls)
        and recovery_targets <= set(last_required)
        and all(
            call.get("schema_valid") is not False
            and call.get("execution_success") is not False
            for call in last_required.values()
        )
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


def _assumption_metric(
    context: EvaluationContext,
    forecast_index: _ForecastIndex,
) -> MetricResult:
    expected_tasks = _oracle_tasks(context)
    assumed = _assumed_task_fields(expected_tasks)
    forecast = forecast_index.successful[0] if forecast_index.successful else None
    actual_task = mapping(forecast.call.get("arguments")) if forecast else {}
    output = forecast.output if forecast else {}
    passed, mismatches = assumption_consistency(expected_tasks, actual_task, output)
    return metric(
        context,
        "assumption_consistency",
        applicable=bool(assumed),
        passed=passed,
        details={"assumed_fields": assumed, "mismatched_fields": mismatches},
    )


def _pipeformer_metrics(
    context: EvaluationContext,
    forecast_index: _ForecastIndex,
    reference_index: _ForecastIndex,
) -> list[MetricResult]:
    pairs = forecast_index.successful
    applicable = PIPEFORMER_TOOL in {
        str(item) for item in sequence(context.oracle.get("teacher_tool_names"))
    }
    outputs = [entry.output for entry in pairs]
    tasks = [mapping(entry.call.get("arguments")) for entry in pairs]
    predicates = {
        "checkpoint_inference": lambda out, _task: checkpoint_inference_used(out),
        "disturbance_application": disturbance_was_applied,
        "forecast_horizon": lambda out, _task: horizon_is_consistent(out),
        "constraint_execution": lambda out, _task: requested_constraints_executed(out),
        "verification_completeness": lambda out, _task: verification_is_complete(out),
    }
    checkpoint, disturbance, horizon, constraint_execution, complete = [
        metric(
            context,
            name,
            applicable=applicable,
            passed=applicable
            and bool(outputs)
            and all(pred(out, task) for out, task in zip(outputs, tasks)),
        )
        for name, pred in predicates.items()
    ]

    expected_constraints = {
        str(item) for item in sequence(context.oracle.get("required_constraints"))
    }
    actual_statuses = [
        mapping(verification_view(output).get("category_status")) for output in outputs
    ]
    matched_statuses = [
        (
            actual,
            mapping(verification_view(reference).get("category_status")),
        )
        for entry, actual in zip(pairs, actual_statuses)
        for reference in [
            _matching_reference_output(
                context,
                entry.call,
                forecast_index=forecast_index,
                reference_index=reference_index,
                actual_arguments=entry.arguments,
            )
        ]
        if reference is not None
    ]
    judgment_applicable = applicable and bool(expected_constraints or matched_statuses)
    judgment_pass = (
        bool(actual_statuses)
        and all(
            expected_constraints <= {str(key) for key in statuses}
            for statuses in actual_statuses
        )
        and all(
            all(str(actual.get(key)) == str(value) for key, value in expected.items())
            for actual, expected in matched_statuses
        )
    )
    judgment = metric(
        context,
        "constraint_judgment",
        applicable=judgment_applicable,
        passed=judgment_pass,
        details={
            "expected_constraints": sorted(expected_constraints),
            "actual_constraints": sorted(
                {str(key) for statuses in actual_statuses for key in statuses}
            ),
            "matched_reference_forecast_count": len(matched_statuses),
        },
    )
    return [checkpoint, disturbance, horizon, constraint_execution, judgment, complete]


def _registry_metric(context: EvaluationContext) -> MetricResult:
    tool_calls = calls(context.record)
    outputs_by_id = {
        str(item.get("tool_call_id") or ""): _output(item)
        for item in output_wrappers(context.record)
    }
    _, unauthorized = forecast_registry_order(tool_calls, outputs_by_id)
    applicable = bool(any(call.get("name") == PIPEFORMER_TOOL for call in tool_calls))
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


# Missing-resource behaviour contract: when a tool output tells the agent the
# asked resource does not exist, the final answer must still NAME that exact
# resource — token from the failed output's own payload, never inferred from
# question text. Failing to name it is evidence substitution: answering for
# some other date or station the user never asked about.
_CJK = re.compile(r"[一-龥]")
_BARE_DATE = re.compile(r"\d{8}")


def _missing_resource_anchors(record: Mapping[str, Any]) -> list[str]:
    # order-independent lookup into the matching call when the output payload
    # drops the structured details (e.g. compacted projections).
    calls_by_id = {
        str(call.get("tool_call_id") or ""): mapping(call.get("arguments"))
        for call in calls(record)
    }
    anchors: list[str] = []
    for item in output_wrappers(record):
        if (
            not (output := item.get("output"))
            or not isinstance(output, Mapping)
            or output.get("success", True) is not False
        ):
            continue
        error_code = str(output.get("error_code") or "")
        if not (str(output.get("error") or "").strip() or error_code):
            continue
        name = str(item.get("name") or "")
        details = mapping(output.get("details"))
        station = (
            details.get("requested_target_station")
            or details.get("canonical_target_station")
            or ""
        )
        station = station or calls_by_id.get(
            str(item.get("tool_call_id") or ""), {}
        ).get("target_station", "")
        if isinstance(station, str) and station.strip():
            anchors.append(station.strip())
        error_text = str(output.get("error") or "")
        if name == "read_file" and "not found" in error_text.casefold():
            path = str(
                details.get("requested_path") or output.get("path") or error_text
            )
            anchors.extend(_BARE_DATE.findall(path))
    return sorted(dict.fromkeys(anchors))


def _covers(answer: str, compact_answer: str, anchor: str) -> bool:
    token = str(anchor)
    if _CJK.search(token):
        # \w in Python re lumps CJK with word chars, so the boundary guard can
        # never match an honest anchor glued to a following Chinese character.
        return token in answer or token.replace("-", "") in compact_answer
    for text, key in ((answer, token), (compact_answer, token.replace("-", ""))):
        if re.search(r"(?<![\w-])" + re.escape(key) + r"(?![\w-])", text):
            return True
    return False


def _question_anchor_metric(context: EvaluationContext) -> MetricResult:
    answer = str(context.record.get("final_answer") or "")
    compact_answer = answer.replace("-", "")
    anchors = _missing_resource_anchors(context.record)
    # Forgive a failure only when the question never asked for that resource
    # (a misprobe), or when some successful output produced it (healed). Every
    # other failure stays anchored: answering from a substitute target is the
    # signal being priced. Dates and station names share this rule.
    question = str((context.reference or {}).get("user_input") or "")
    success_corpus = "\n".join(
        _json_text(item.get("output"))
        for item in sequence(context.record.get("tool_outputs"))
        if isinstance(item, Mapping)
        and isinstance(item.get("output"), Mapping)
        and item["output"].get("success", True) is not False
    )
    question_compact = question.replace("-", "")
    success_compact = success_corpus.replace("-", "")
    retained = [
        anchor
        for anchor in anchors
        if (key := anchor.replace("-", "")) in question_compact
        and key not in success_compact
    ]
    covered = sorted(
        anchor for anchor in retained if _covers(answer, compact_answer, anchor)
    )
    applicable = bool(retained)
    return metric(
        context,
        "question_anchor",
        applicable=applicable,
        passed=applicable and len(covered) == len(retained),
        details={
            "missing_resource_anchors": anchors,
            "retained_anchors": retained,
            "covered_anchors": covered,
        },
    )


_ENTITY_COUNT_CLAIM = re.compile(
    r"([一-鿿A-Za-z0-9_]{2,20})[：:\s，、]*[,，]?\s*(\d+(?:\.\d+)?)\s*(?:个|点|条|名|位)"
)
_CLAIM_SUPPORT_WINDOW = 30


def _entity_count_claims(text: str) -> set[tuple[str, float]]:
    return {
        (m.group(1), float(m.group(2)))
        for m in _ENTITY_COUNT_CLAIM.finditer(text)
        # Digits inside the entity token mean regex segmentation glued two
        # numbers ("每条线约1:4") or units together; drop rather than guess.
        if not re.search(r"\d", m.group(1))
    }


def _count_near_entity(count: float, entity: str, text: str) -> bool:
    # Bounded digit match: claim "1" must not draw support from "1132" or "0.1".
    count_pattern = re.compile(rf"(?<![\d.]){re.escape(f'{count:g}')}(?![\d.])")
    for occurrence in re.finditer(re.escape(entity), text):
        window = text[
            max(0, occurrence.start() - _CLAIM_SUPPORT_WINDOW) : occurrence.end()
            + _CLAIM_SUPPORT_WINDOW
        ]
        if count_pattern.search(window):
            return True
    return False


def _evidence_header_fields(context: EvaluationContext) -> set[str]:
    """Column headers from successful read_file CSV outputs seen in the episode.

    An (entity, count) claim naming one of these is the header-row-counting bug
    ("用户 1个"), not a data value — deterministic from tool output, no NLP.
    """
    fields: set[str] = set()
    for source in (context.record, context.reference or {}):
        for wrapper in sequence(source.get("tool_outputs")):
            if not isinstance(wrapper, Mapping) or wrapper.get("name") != "read_file":
                continue
            payload = _output(wrapper)
            if payload.get("success", True) is False:
                continue
            content_lines = [
                line
                for line in str(payload.get("content") or "").splitlines()
                if line.strip()
            ]
            for row in csv.reader(content_lines[:1]):
                fields.update(cell.strip() for cell in row if cell.strip())
    return fields


def _claim_alignment_metric(context: EvaluationContext) -> MetricResult:
    """(entity, count) claims in the student answer vs. the reference answer.

    The GRPO prompt records always carry the teacher's ``final_answer``, so a
    fabricated ranking ("上海管网 1个") contradicting the reference is a
    deterministic cheat signal — no LLM needed.  Support = entity named by the
    teacher with the claimed count nearby; entity-tag artifacts glued to a
    real number (header-as-user "用户 1个") still match and are a known
    blind spot, priced cheaper than count/entity fabrication.
    """
    reference_answer = str((context.reference or {}).get("final_answer") or "")
    claims = _entity_count_claims(str(context.record.get("final_answer") or ""))
    header_fields = _evidence_header_fields(context)
    question = str((context.reference or {}).get("user_input") or "")
    # The header bug is ranked-dimension-specific: the question ranks by a
    # dimension column ("按用户统计…前 5 个用户") and the answer names THAT
    # column as a data value. A claim naming a different concept word which
    # merely shares the header's spelling ("供气点 1个") is legitimate.
    ranked_dimensions = {
        field
        for field in header_fields
        if f"按{field}" in question or f"个{field}" in question
    }
    unsupported = sorted(
        (entity, count)
        for entity, count in claims
        if not _count_near_entity(count, entity, reference_answer)
    )
    header_claims = sorted(
        (entity, count) for entity, count in claims if entity in ranked_dimensions
    )
    applicable = bool(claims) and bool(reference_answer.strip())
    passed = applicable and not unsupported and not header_claims
    return metric(
        context,
        "claim_alignment",
        applicable=applicable,
        passed=passed,
        details={
            "claim_count": len(claims),
            "unsupported_claims": [f"{e}:{g:g}" for (e, g) in unsupported[:10]],
            "column_header_claims": [f"{e}:{g:g}" for (e, g) in header_claims[:10]],
        },
    )


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
    reference_answer = (context.reference or {}).get("final_answer")
    if reference_answer:
        haystack.append(str(reference_answer))
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
    output_key: str,
    forecast_index: _ForecastIndex,
    reference_index: _ForecastIndex,
) -> MetricResult:
    matched: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    student_evidence: list[Mapping[str, Any]] = []
    for entry in forecast_index.successful:
        reference = _matching_reference_output(
            context,
            entry.call,
            forecast_index=forecast_index,
            reference_index=reference_index,
            actual_arguments=entry.arguments,
        )
        if reference is not None:
            matched.append((reference, entry.output))
        elif _matches_reference_contract(
            context,
            entry.call,
            forecast_index=forecast_index,
            reference_index=reference_index,
            actual_arguments=entry.arguments,
        ):
            student_evidence.append(entry.output)

    def value(output: Mapping[str, Any]) -> Any:
        verification = verification_view(output)
        fallback = output.get(output_key)
        if output_key == "human_intervention_label":
            fallback = output.get(output_key, output.get("manual_intervention_label"))
        return verification.get(output_key, fallback)

    comparisons = [
        (expected, value(output))
        for reference, output in matched
        if (expected := value(reference)) is not None
    ]
    expected = [item[0] for item in comparisons]
    actual = [item[1] for item in comparisons]
    own_values = [value(output) for output in student_evidence]
    state_candidates = [
        item
        for item in sequence(
            mapping((context.reference or {}).get("state_before")).get("candidates")
        )
        if isinstance(item, Mapping)
    ]
    state_values = [value(item) for item in state_candidates]
    own_values.extend(state_values)
    applicable = bool(comparisons or own_values)
    return metric(
        context,
        name,
        applicable=applicable,
        passed=applicable
        and actual == expected
        and all(item is not None for item in own_values),
        details={
            "expected": expected[0] if len(expected) == 1 else expected,
            "actual": actual[0] if len(actual) == 1 else actual,
            "matched_reference_forecast_count": len(matched),
            "student_evidence_forecast_count": len(student_evidence),
            "state_candidate_count": len(state_candidates),
        },
    )


# Length rejects degenerate stubs; reference-token overlap also keeps tool-less
# follow-up answers relevant without requiring exact teacher wording.
_MIN_SUBSTANTIVE_ANSWER_CHARS = 8

_ANSWER_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}")
_ANSWER_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "that",
        "this",
        "with",
        "from",
        "answer",
        "result",
        "现在",
        "记得",
        "回答",
        "结果",
        "可以",
        "需要",
        "这个",
        "那个",
    }
)


def _answer_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for value in _ANSWER_WORD.findall(text):
        value = value.casefold()
        if value in _ANSWER_STOPWORDS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", value):
            tokens.update(
                value[index : index + 2]
                for index in range(len(value) - 1)
                if value[index : index + 2] not in _ANSWER_STOPWORDS
            )
        else:
            tokens.add(value)
    return tokens


def _answer_metric(context: EvaluationContext) -> MetricResult:
    answer = str(context.record.get("final_answer") or "")
    stripped = answer.strip()
    reference_answer = str((context.reference or {}).get("final_answer") or "").strip()
    applicable = bool(reference_answer or stripped)
    reference_tokens = _answer_tokens(reference_answer)
    answer_tokens = _answer_tokens(stripped)
    matched_tokens = sorted(reference_tokens & answer_tokens)
    requires_relevance = not bool(context.oracle.get("has_tool_target")) and bool(
        reference_tokens
    )
    trace_completed = context.record.get("trace_status") == "completed"
    return metric(
        context,
        "answer_completeness",
        applicable=applicable,
        passed=(
            applicable
            and trace_completed
            and len(stripped) >= _MIN_SUBSTANTIVE_ANSWER_CHARS
            and (not requires_relevance or bool(matched_tokens))
        ),
        details={
            "answer_length": len(answer),
            "minimum_length": _MIN_SUBSTANTIVE_ANSWER_CHARS,
            "trace_completed": trace_completed,
            "relevance_required": requires_relevance,
            "matched_reference_tokens": matched_tokens[:20],
        },
    )


def _json_metric(context: EvaluationContext) -> MetricResult:
    tool_calls = calls(context.record)
    errors = sequence(context.record.get("json_errors"))
    applicable = bool(tool_calls or errors)
    return metric(
        context,
        "json_validity",
        applicable=applicable,
        passed=applicable
        and not errors
        and all(call.get("schema_valid") is not False for call in tool_calls),
        details={"error_count": len(errors), "scope": "tool_call_json_and_schema"},
    )


def _hallucination_metric(
    context: EvaluationContext,
    components: list[MetricResult],
) -> MetricResult:
    evidence, *other_components = components
    evidence_claims = any(
        evidence.details.get(key)
        for key in (
            "claimed_numeric_values",
            "unsupported_row_claims",
            "candidate_contract_issues",
        )
    )
    applicable = [item for item in other_components if item.applicable]
    applicable_names = [item.name for item in applicable]
    failed = [item.name for item in applicable if not item.passed]
    if evidence.applicable:
        applicable_names.insert(0, evidence.name)
        if evidence_claims and any(
            evidence.details.get(key)
            for key in (
                "unsupported_numeric_values",
                "unsupported_row_claims",
                "candidate_contract_issues",
            )
        ):
            failed.insert(0, evidence.name)
    return metric(
        context,
        "hallucination",
        applicable=bool(applicable_names),
        passed=bool(applicable_names) and not failed,
        details={
            "applicable_components": applicable_names,
            "failed_components": failed,
        },
    )


def _artifact_metric(context: EvaluationContext) -> MetricResult:
    requested = requested_artifacts(
        str((context.reference or {}).get("user_input") or "")
    )
    if not requested:
        return metric(
            context,
            "artifact_evidence",
            applicable=False,
            details={"requested_artifacts": []},
        )
    enriched = attach_tool_arguments(
        [dict(item) for item in output_wrappers(context.record)],
        [dict(item) for item in calls(context.record)],
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
    tool_calls = calls(record)
    rebased = [call for call in tool_calls if call.get("cwd_rebased")]
    failures = sum(call.get("execution_success") is False for call in rebased)
    return {
        "cwd_rebased_calls": len(rebased),
        "records_with_cwd_rebased": int(bool(rebased)),
        "rebased_execution_successes": len(rebased) - failures,
        "rebased_execution_failures": failures,
        "portable_path_normalization_calls": sum(
            bool(call.get("portable_path_normalization")) for call in tool_calls
        ),
    }


def evaluate_autonomous_checks(
    context: EvaluationContext,
) -> tuple[list[MetricResult], dict[str, Any]]:
    """Return every canonical autonomous metric plus unscored diagnostics."""

    forecast_index = _resolved_forecast_index(context.record)
    reference_index = _resolved_forecast_index(context.reference or {})
    tool_call, tool_recovery = _tool_metrics(context)
    claim_support = _claim_support_metric(context)
    evidence = evidence_consistency(context)
    claim_alignment = _claim_alignment_metric(context)
    question_anchor = _question_anchor_metric(context)
    metrics = [
        _task_parsing(context, forecast_index),
        _assumption_metric(context, forecast_index),
        tool_call,
        *_pipeformer_metrics(context, forecast_index, reference_index),
        _registry_metric(context),
        *(
            _label_metric(
                context,
                name,
                key,
                forecast_index,
                reference_index,
            )
            for name, key in (
                ("risk", "risk_level"),
                ("manual_intervention", "human_intervention_label"),
                ("dispatch", "dispatch_recommendation"),
            )
        ),
        evidence,
        _answer_metric(context),
        _json_metric(context),
        _artifact_metric(context),
        _record_contract(context),
        tool_recovery,
        claim_support,
        claim_alignment,
        question_anchor,
        _hallucination_metric(
            context,
            [evidence, claim_support, claim_alignment, question_anchor],
        ),
    ]
    ordered = ordered_canonical_metrics(context, metrics)
    return ordered, {"portability": _portability_diagnostics(context.record)}
