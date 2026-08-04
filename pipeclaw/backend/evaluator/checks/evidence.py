"""Single canonical implementation of evidence-consistency evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..models import EvaluationContext, EvaluationProfile, MetricResult
from .common import mapping, metric, sequence


_NUMERIC_CLAIM = re.compile(
    r"(?<![A-Za-z0-9_])[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9_])"
)


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, Mapping):
        return [
            number
            for item in value.values()
            for number in _numeric_values(item)
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [number for item in value for number in _numeric_values(item)]
    return []


def _numeric_claims(text: str) -> list[float]:
    claims: list[float] = []
    for match in _NUMERIC_CLAIM.finditer(text):
        remainder = text[match.end() :]
        raw_value = match.group(0)
        is_integer = raw_value.lstrip("+-−").isdigit()
        if is_integer and remainder[:1] in {".", ")", "、"} and (
            len(remainder) == 1 or remainder[1].isspace()
        ):
            continue
        prefix = text[max(0, match.start() - 24) : match.start()]
        suffix = text[match.end() : match.end() + 24]
        if is_integer and re.search(
            r"(?:candidate|候选(?:动作)?|option|choice)\s*$",
            prefix,
            re.IGNORECASE,
        ):
            continue
        if is_integer and re.match(
            r"\s*(?:candidates?|候选(?:动作)?|options?|choices?|个候选)",
            suffix,
            re.IGNORECASE,
        ):
            continue
        try:
            claims.append(float(raw_value.replace("−", "-")))
        except ValueError:
            continue
    return claims


def _source_outputs(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for item in sequence(source.get("tool_outputs")):
        if not isinstance(item, Mapping):
            continue
        output = item.get("output", item)
        if isinstance(output, Mapping) and output.get("success", True) is not False:
            result.append(output)
    return result


def _autonomous_evidence(
    context: EvaluationContext,
) -> MetricResult:
    source = context.reference or {}
    oracle = context.oracle
    rollout = context.record
    evidence_numbers = _numeric_values(oracle.get("verified_evidence", {}))
    for output in _source_outputs(source):
        evidence_numbers.extend(_numeric_values(output))
    state_before = mapping(source.get("state_before"))
    evidence_numbers.extend(_numeric_values(state_before.get("verified_evidence")))
    for task in sequence(oracle.get("tasks")):
        evidence_numbers.extend(_numeric_values(task))
    # Successful student tool output is independently verified evidence.  It
    # need not reproduce a different valid teacher forecast sample.
    for output in _source_outputs(rollout):
        evidence_numbers.extend(_numeric_values(output))

    unique_numbers: list[float] = []
    for number in evidence_numbers:
        if not any(
            math.isclose(number, existing, rel_tol=1e-6, abs_tol=1e-6)
            for existing in unique_numbers
        ):
            unique_numbers.append(number)
    final_answer = str(rollout.get("final_answer") or "")
    claims = _numeric_claims(final_answer)
    unsupported = [
        value
        for value in claims
        if not any(
            math.isclose(value, evidence, rel_tol=1e-5, abs_tol=1e-5)
            for evidence in unique_numbers
        )
    ]
    applicable = bool(unique_numbers) and bool(final_answer.strip())
    return metric(
        context,
        "evidence_consistency",
        applicable=applicable,
        passed=applicable and not unsupported,
        details={
            "claimed_numeric_values": claims,
            "unsupported_numeric_values": unsupported,
        },
    )


def _teacher_evidence(
    context: EvaluationContext,
    issues: Sequence[str],
    *,
    teacher_variant: str,
) -> MetricResult:
    return metric(
        context,
        "evidence_consistency",
        applicable=True,
        passed=not issues,
        details={"issues": list(issues)},
        teacher_variant=teacher_variant,
    )


def evidence_consistency(
    context: EvaluationContext,
    *,
    issues: Sequence[str] = (),
    teacher_variant: str = "pipeformer",
) -> MetricResult:
    """Evaluate evidence once for either canonical profile."""

    if context.profile is EvaluationProfile.AUTONOMOUS_ROLLOUT:
        return _autonomous_evidence(context)
    return _teacher_evidence(
        context,
        issues,
        teacher_variant=teacher_variant,
    )
