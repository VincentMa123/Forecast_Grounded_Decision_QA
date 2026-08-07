"""Single canonical implementation of evidence-consistency evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from pipeclaw.backend.grounding.contract import comparison_answer_issues
except ImportError:  # pragma: no cover - direct backend execution
    from grounding.contract import comparison_answer_issues

from ..models import EvaluationContext, EvaluationProfile, MetricResult
from ..teacher_quality import record_grounding_contract
from .common import mapping, metric, sequence


# ``14,514.122`` is one number.  The grouped alternative must come first so the
# separator is consumed here rather than splitting the value into ``14`` and
# ``514.122``, neither of which matches the ``14514.122`` the tool reported.
_NUMERIC_CLAIM = re.compile(
    r"(?<![A-Za-z0-9_])[+\-−]?"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?![A-Za-z0-9_])"
)
# ``2019-07-13`` and ``2019年7月13日`` are timestamps.  Their components are not
# measurements, and the ``-`` would otherwise read as a negative sign.
_DATE_SPAN = re.compile(
    r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?"
)
# ``第 3 段`` indexes a pipeline segment and ``第5名`` a rank; the digit is an
# ordinal, not a data claim, so it must not be checked against the evidence pool.
_CHINESE_ORDINAL_SUFFIX = re.compile(r"\s*[段条章节次项名位类级]")
# ``至19日`` abbreviates the end of a date range whose head ``_DATE_SPAN``
# already consumed, so the trailing day number arrives here unattached.
_DATE_COMPONENT_SUFFIX = re.compile(r"\s*[日号月]")
_ROW_CLAUSE_SEPARATOR = re.compile(r"(?:\r?\n|[；;。，])")
_LEADING_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _is_compact_date(raw: str) -> bool:
    """``20190114`` is a date literal, not a quantity."""

    if len(raw) != 8 or not raw.isdigit():
        return False
    year, month, day = int(raw[:4]), int(raw[4:6]), int(raw[6:])
    return 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31


def _numeric_claims(text: str) -> list[float]:
    claims: list[float] = []
    date_spans = [match.span() for match in _DATE_SPAN.finditer(text)]
    for match in _NUMERIC_CLAIM.finditer(text):
        if any(start <= match.start() < end for start, end in date_spans):
            continue
        remainder = text[match.end() :]
        raw_value = match.group(0)
        digits = raw_value.lstrip("+-−").replace(",", "")
        is_integer = digits.isdigit()
        if is_integer and _is_compact_date(digits):
            continue
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
        if (
            is_integer
            and prefix.rstrip().endswith("第")
            and _CHINESE_ORDINAL_SUFFIX.match(suffix)
        ):
            continue
        if is_integer and prefix.rstrip().endswith("至") and _DATE_COMPONENT_SUFFIX.match(suffix):
            continue
        try:
            claims.append(float(raw_value.replace("−", "-").replace(",", "")))
        except ValueError:
            continue
    return claims


def _observed_numbers(value: Any) -> list[float]:
    """Numbers the model could read, including those inside text payloads.

    Numeric types and numeric claims inside strings are both evidence. OpenClaw tools return
    their payload as a serialized string under ``evidence_excerpt`` (file
    content, command stdout), so the rows the model actually read are invisible
    to a type-only walk and every figure quoted from them scores as
    hallucination.  Text the tool returned is observed evidence.
    """

    if isinstance(value, str):
        return _numeric_claims(value)
    if isinstance(value, Mapping):
        return [
            number
            for item in value.values()
            for number in _observed_numbers(item)
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [number for item in value for number in _observed_numbers(item)]
    return [float(value)] if isinstance(value, (int, float)) and not isinstance(value, bool) else []


def _source_outputs(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for item in sequence(source.get("tool_outputs")):
        if not isinstance(item, Mapping):
            continue
        output = item.get("output", item)
        if isinstance(output, Mapping) and output.get("success", True) is not False:
            result.append(output)
    return result


def _answer_rows(evidence: Any) -> list[Mapping[str, Any]]:
    """Return typed CSV row values already produced by the evidence reducer."""

    csv_evidence = mapping(mapping(evidence).get("csv_evidence"))
    return [
        mapping(item.get("values"))
        for item in sequence(csv_evidence.get("answer_rows"))
        if isinstance(item, Mapping) and mapping(item.get("values"))
    ]


def _unsupported_row_claims(
    answer: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Find numbers attached to the wrong entity within a CSV-row claim."""

    row_strings = [
        {
            str(value).strip().casefold()
            for value in row.values()
            if isinstance(value, str) and value.strip()
        }
        for row in rows
    ]
    string_counts = {
        value: sum(value in values for values in row_strings)
        for values in row_strings
        for value in values
    }
    discriminative = {
        value
        for value, count in string_counts.items()
        if count < len(row_strings)
    }
    issues: list[dict[str, Any]] = []
    for clause in _ROW_CLAUSE_SEPARATOR.split(answer):
        normalized = clause.casefold()
        mentioned = sorted(value for value in discriminative if value in normalized)
        numbers = _numeric_claims(clause)
        if not mentioned or not numbers:
            continue
        unsupported = [
            number
            for number in numbers
            if not any(
                set(mentioned).issubset(strings)
                and any(
                    math.isclose(number, row_number, rel_tol=1e-5, abs_tol=1e-5)
                    for row_number in _observed_numbers(row)
                )
                for row, strings in zip(rows, row_strings)
            )
        ]
        if unsupported:
            issues.append(
                {
                    "claim": clause.strip(),
                    "identifiers": mentioned,
                    "numeric_values": unsupported,
                }
            )
    return issues


def _prior_turn_evidence(source: Mapping[str, Any]) -> list[float]:
    """Numbers a previous turn in this session already established.

    ``rollout/prompting.py`` renders ``recent_turns`` into the student's system
    prompt, so a later turn can legitimately restate those values without
    calling a tool again.  Treating them as ungrounded scores correct recall as
    hallucination.
    """

    numbers: list[float] = []
    for turn in sequence(source.get("recent_turns")):
        if not isinstance(turn, Mapping):
            continue
        numbers.extend(_numeric_claims(str(turn.get("assistant_output") or "")))
    return numbers


def _autonomous_evidence(
    context: EvaluationContext,
) -> MetricResult:
    source = context.reference or {}
    oracle = context.oracle
    rollout = context.record
    evidence_numbers = _observed_numbers(oracle.get("verified_evidence", {}))
    for output in _source_outputs(source):
        evidence_numbers.extend(_observed_numbers(output))
    state_before = mapping(source.get("state_before"))
    evidence_numbers.extend(_observed_numbers(state_before.get("verified_evidence")))
    for task in sequence(oracle.get("tasks")):
        evidence_numbers.extend(_observed_numbers(task))
    # Successful student tool output is independently verified evidence.  It
    # need not reproduce a different valid teacher forecast sample.
    for output in _source_outputs(rollout):
        evidence_numbers.extend(_observed_numbers(output))
    evidence_numbers.extend(_prior_turn_evidence(rollout))
    evidence_numbers.extend(_prior_turn_evidence(source))

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
    row_evidence = [
        *_answer_rows(oracle.get("verified_evidence", {})),
        *_answer_rows(state_before.get("verified_evidence")),
    ]
    unsupported_rows = _unsupported_row_claims(final_answer, row_evidence)
    contract = record_grounding_contract({**source, **rollout})
    candidate_issues = (
        comparison_answer_issues(_LEADING_THINK.sub("", final_answer), contract)
        if contract.get("answer_mode") == "dispatch_comparison"
        else []
    )
    applicable = bool(unique_numbers or candidate_issues) and bool(final_answer.strip())
    return metric(
        context,
        "evidence_consistency",
        applicable=applicable,
        passed=(
            applicable
            and not unsupported
            and not unsupported_rows
            and not candidate_issues
        ),
        details={
            "claimed_numeric_values": claims,
            "unsupported_numeric_values": unsupported,
            "unsupported_row_claims": unsupported_rows,
            "candidate_contract_issues": candidate_issues,
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
