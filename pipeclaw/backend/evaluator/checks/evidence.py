from __future__ import annotations

import math
import re
from datetime import datetime
from collections.abc import Mapping, Sequence
from typing import Any

from pipeclaw.backend.grounding.contract import (
    comparison_answer_issues,
    record_grounding_contract,
)

from ..models import EvaluationContext, EvaluationProfile, MetricResult
from ..numeric_grounding import derived_numeric_values, numeric_values_match
from ..quality_references import (
    observed_numeric_claim_items,
    observed_numeric_claim_values,
    observed_numeric_values,
)
from .common import mapping, metric, sequence

_ROW_CLAUSE_SEPARATOR = re.compile(r"(?:\r?\n|[；;。，])")
_LEADING_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_BARE_DATE = re.compile(r"\d{8}|\d{4}-\d{2}-\d{2}")
_IDENTIFIER_TOKEN = re.compile(
    r"[\w.\u4e00-\u9fa5-]+\.(?:csv|xlsx?|json|txt)|[A-Z]_[A-Za-z0-9]+[::][A-Za-z0-9_:-]+|[\u4e00-\u9fa5]{1,7}?站"
)


_DATE_SPAN = re.compile(
    r"(\d{4}-\d{2}-\d{2}|\d{8})\s*(?:至|-)\s*(\d{4}-\d{2}-\d{2}|\d{8})"
)


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
        value for value, count in string_counts.items() if count < len(row_strings)
    }
    issues: list[dict[str, Any]] = []
    for clause in _ROW_CLAUSE_SEPARATOR.split(answer):
        normalized = clause.casefold()
        occurrences = [
            (match.start(), match.end(), value)
            for value in discriminative
            for match in re.finditer(re.escape(value), normalized)
        ]
        numeric_items = observed_numeric_claim_items(clause)
        if not occurrences or not numeric_items:
            continue
        unsupported: list[float] = []
        issue_identifiers: set[str] = set()
        previous_end = 0
        for number, start, end in numeric_items:
            identifiers = {
                value
                for item_start, item_end, value in occurrences
                if previous_end <= item_start and item_end <= start
            }
            if not identifiers:
                preceding = [item for item in occurrences if item[1] <= start]
                if preceding:
                    identifiers = {max(preceding, key=lambda item: item[1])[2]}
            previous_end = end
            if not identifiers:
                continue
            matching_rows = [
                row
                for row, strings in zip(rows, row_strings)
                if identifiers.issubset(strings)
            ]
            supported_numbers = [
                value for row in matching_rows for value in observed_numeric_values(row)
            ]
            if len(identifiers) == 1 and len(matching_rows) > 1:
                numeric_keys = {
                    key
                    for row in matching_rows
                    for key, value in row.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                supported_numbers.extend(
                    sum(float(row[key]) for row in matching_rows if key in row)
                    for key in numeric_keys
                )
            if not any(
                math.isclose(number, supported, rel_tol=1e-5, abs_tol=1e-5)
                for supported in supported_numbers
            ):
                unsupported.append(number)
                issue_identifiers.update(identifiers)
        if unsupported:
            issues.append(
                {
                    "claim": clause.strip(),
                    "identifiers": sorted(issue_identifiers),
                    "numeric_values": unsupported,
                }
            )
    return issues


def _ranking_row_issues(
    answer: str,
    rows: Sequence[Mapping[str, Any]],
    question: str,
) -> list[dict[str, Any]]:
    if not re.search(r"(?:前\s*\d+\s*名|top\s*\d+|升序|降序)", question, re.IGNORECASE):
        return []
    ranked: list[tuple[str, float]] = []
    for line in answer.splitlines():
        if not re.match(r"\s*\d+\s*[.)、]\s*", line):
            continue
        normalized = line.casefold()
        for row in rows:
            identifiers = [
                str(value).strip()
                for value in row.values()
                if isinstance(value, str)
                and str(value).strip().casefold() in normalized
            ]
            if not identifiers:
                continue
            row_numbers = observed_numeric_values(row)
            number = next(
                (
                    claim
                    for claim in observed_numeric_claim_values(line)
                    if any(
                        math.isclose(claim, observed, rel_tol=1e-5, abs_tol=1e-5)
                        for observed in row_numbers
                    )
                ),
                None,
            )
            if number is not None:
                ranked.append((identifiers[-1], number))
                break
    if len(ranked) < 2:
        return []
    ascending = bool(
        re.search(r"(?:升序|最小|lowest|ascending)", question, re.IGNORECASE)
    )
    ordered = all(
        left <= right if ascending else left >= right
        for (_, left), (_, right) in zip(ranked, ranked[1:])
    )
    if ordered:
        return []
    return [
        {
            "claim": "ranked rows are not in the requested order",
            "identifiers": [identifier for identifier, _ in ranked],
            "numeric_values": [number for _, number in ranked],
        }
    ]


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
        numbers.extend(
            observed_numeric_claim_values(str(turn.get("assistant_output") or ""))
        )
    return numbers


def _autonomous_evidence(
    context: EvaluationContext,
) -> MetricResult:
    source = context.reference or {}
    oracle = context.oracle
    rollout = context.record
    evidence_numbers = observed_numeric_values(oracle.get("verified_evidence", {}))
    evidence_numbers.extend(derived_numeric_values(oracle.get("verified_evidence", {})))
    # Successful student tool output is independently verified evidence.  It
    # need not reproduce a different valid teacher forecast sample.
    for output in [*_source_outputs(source), *_source_outputs(rollout)]:
        evidence_numbers.extend(observed_numeric_values(output))
        evidence_numbers.extend(derived_numeric_values(output))
    state_before = mapping(source.get("state_before"))
    evidence_numbers.extend(
        observed_numeric_values(state_before.get("verified_evidence"))
    )
    evidence_numbers.extend(
        derived_numeric_values(state_before.get("verified_evidence"))
    )
    evidence_numbers.extend(observed_numeric_values(state_before.get("candidates")))
    evidence_numbers.extend(derived_numeric_values(state_before.get("candidates")))
    for task in sequence(oracle.get("tasks")):
        evidence_numbers.extend(observed_numeric_values(task))
    evidence_numbers.extend(_prior_turn_evidence(rollout))
    evidence_numbers.extend(_prior_turn_evidence(source))
    user_input = str(source.get("user_input") or "")
    evidence_numbers.extend(observed_numeric_claim_values(user_input))
    # Calendar arithmetic is really verified evidence: a date span named in the
    # question makes its inclusive day count citable (otherwise the hard gate
    # punishes the honest "窗口共 N 天" deliverable).
    for match in _DATE_SPAN.finditer(user_input):
        start_date = datetime.strptime(match.group(1).replace("-", ""), "%Y%m%d")
        end_date = datetime.strptime(match.group(2).replace("-", ""), "%Y%m%d")
        if start_date <= end_date:
            evidence_numbers.append(float((end_date - start_date).days + 1))
    row_evidence = [
        *_answer_rows(oracle.get("verified_evidence", {})),
        *_answer_rows(state_before.get("verified_evidence")),
    ]

    unique_numbers: list[float] = []
    for number in evidence_numbers:
        if not any(
            math.isclose(number, existing, rel_tol=1e-6, abs_tol=1e-6)
            for existing in unique_numbers
        ):
            unique_numbers.append(number)
    final_answer = str(rollout.get("final_answer") or "")
    claims = observed_numeric_claim_values(final_answer)
    unsupported = [
        value
        for value in claims
        if not any(numeric_values_match(value, evidence) for evidence in unique_numbers)
    ]
    unsupported_rows = [
        *_unsupported_row_claims(final_answer, row_evidence),
        *_ranking_row_issues(
            final_answer,
            row_evidence,
            user_input,
        ),
    ]
    contract = record_grounding_contract({**source, **rollout})
    candidate_issues = (
        comparison_answer_issues(_LEADING_THINK.sub("", final_answer), contract)
        if contract.get("answer_mode") == "dispatch_comparison"
        else []
    )
    applicable = bool(unique_numbers or candidate_issues) and bool(final_answer.strip())
    # A template answer is vacuously grounded: no numeric claim, no identifier,
    # no date token — the hard gate must not score it. (Numeric-only guards
    # flip honest recall answers; identifier+date tri-guard is measured-safe.)
    # Template-cheat guard is scoped to episodes that USED the evidence plane:
    # "ran tools but produced no grounded claim" is the cheat shape; recall-only
    # episodes routinely (and legitimately) have no numeric/identifier/date in
    # their answer — they keep their existing gates, no new ones.
    vacuous = (
        bool(sequence(rollout.get("tool_calls")))
        and not claims
        and not _BARE_DATE.search(final_answer)
        and not _IDENTIFIER_TOKEN.search(final_answer)
    )
    return metric(
        context,
        "evidence_consistency",
        applicable=applicable,
        passed=(
            applicable
            and not unsupported
            and not unsupported_rows
            and not candidate_issues
            and not vacuous
        ),
        details={
            "claimed_numeric_values": claims,
            "unsupported_numeric_values": unsupported,
            "unsupported_row_claims": unsupported_rows,
            "candidate_contract_issues": candidate_issues,
            "vacuous_answer": vacuous,
        },
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
    return metric(
        context,
        "evidence_consistency",
        applicable=True,
        passed=not issues,
        details={"issues": list(issues)},
        teacher_variant=teacher_variant,
    )
