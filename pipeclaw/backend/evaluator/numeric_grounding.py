from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Dict, Iterable, List, Optional

from pipeclaw.backend.grounding.decision_trace_state import VerifiedDecisionState
from pipeclaw.backend.grounding.evidence.csv import record_csv_evidence
from pipeclaw.backend.grounding.evidence.tool import (
    attach_tool_arguments,
    classify_tool_evidence,
    requested_artifacts,
)

from .quality_references import (
    CASE_IDENTIFIER_NUMBER,
    numeric_claim_values,
    time_values_in_minutes,
    walk_numeric_values,
)


def trusted_conversation_context(
    items: Iterable[Dict[str, Any]],
    *,
    verified_evidence_only: bool = False,
) -> List[Dict[str, Any]]:
    """Retain trusted context, optionally projecting only verified evidence."""

    if not verified_evidence_only:
        return [
            dict(item)
            if item.get("grounding_verified") is True
            else {
                key: value for key, value in item.items() if key != "assistant_output"
            }
            for item in items
        ]
    context = []
    for item in items:
        if item.get("grounding_verified") is True:
            context.append(dict(item))
            continue
        summary = item.get("verified_evidence_summary")
        legacy_verified_summary = (
            "tool_evidence_verified" not in item
            and isinstance(summary, dict)
            and bool(summary)
        )
        if item.get("tool_evidence_verified") is True or legacy_verified_summary:
            context.append(
                {
                    "tool_evidence_verified": True,
                    "evidence_artifacts": list(item.get("evidence_artifacts") or []),
                    "verified_evidence_summary": summary,
                }
            )
    return context


def _csv_line_totals_from_record(record: Dict[str, Any]) -> List[float]:
    """Per-(file, pipeline) and window sums/diffs parsed from raw read_file content.

    The record's evidence.answer_rows slice is deliberately narrow (top-N scenario
    rows), so derived group-sums for multi-pipeline comparisons can't rebuild the
    other line's totals. The read_file outputs carry the full CSV text already —
    sum it here instead of reaching back to disk.
    """
    totals: List[float] = []
    flows_column = "管道流量"
    window_line_stats: Dict[str, float] = {}
    window_seg_stats: Dict[tuple[str, str, str], float] = {}
    daily_min_diffs: List[float] = []
    for wrapper in record.get("tool_outputs") or []:
        if not isinstance(wrapper, dict) or wrapper.get("name") != "read_file":
            continue
        payload = wrapper.get("output")
        if not isinstance(payload, dict) or payload.get("success") is False:
            continue
        content = str(payload.get("content") or "")
        if flows_column not in content:
            continue
        per_line_stats: Dict[str, tuple[float, int]] = {}
        try:
            for row in csv.DictReader(io.StringIO(content)):
                value = row.get(flows_column)
                if value is None:
                    continue
                try:
                    flow = abs(float(value))
                except ValueError:
                    continue
                pipeline = (
                    row.get("管道划分") or row.get("管线") or row.get("所属地") or "?"
                )
                start_st = str(row.get("起点站名") or "")
                end_st = str(row.get("终点站名") or "")
                total, count = per_line_stats.get(pipeline, (0.0, 0))
                per_line_stats[pipeline] = (total + flow, count + 1)
                window_line_stats[pipeline] = window_line_stats.get(pipeline, 0.0) + flow
                if start_st and end_st:
                    seg_key = (pipeline, start_st, end_st)
                    window_seg_stats[seg_key] = window_seg_stats.get(seg_key, 0.0) + flow
        except csv.Error:
            continue
        day_vals = [total for _, (total, _) in sorted(per_line_stats.items())]
        for _, (total, count) in sorted(per_line_stats.items()):
            totals.extend((round(total, 6), float(count)))
        for i, a in enumerate(day_vals):
            for b in day_vals[i + 1 :]:
                diff = abs(a - b)
                totals.extend((round(diff, 6), round(-diff, 6), round(diff, 0)))
                daily_min_diffs.append(diff)
    if window_line_stats:
        w_lines = [round(v, 6) for _, v in sorted(window_line_stats.items())]
        totals.extend(w_lines)
        for i, a in enumerate(w_lines):
            for b in w_lines[i + 1 :]:
                totals.extend((round(abs(a - b), 6), round(a - b, 6), round(b - a, 6)))
    if window_seg_stats:
        by_pipe: Dict[str, List[float]] = {}
        for (pipe, _, _), val in sorted(window_seg_stats.items()):
            rv = round(val, 6)
            totals.append(rv)
            by_pipe.setdefault(pipe, []).append(rv)
        two_seg_sums: List[float] = []
        for seg_vals in by_pipe.values():
            for i, a in enumerate(seg_vals):
                for b in seg_vals[i + 1 :]:
                    two_seg_sums.append(round(a + b, 6))
        totals.extend(two_seg_sums)
        seg_all = [round(v, 6) for v in window_seg_stats.values()] + two_seg_sums
        for i, a in enumerate(seg_all):
            for b in seg_all[i + 1 :]:
                totals.extend((round(abs(a - b), 6),))
    return totals


def grounded_numeric_claim_values(
    answer: str,
    question: str,
    evidence: Dict[str, Any],
) -> List[float]:
    """Return final-answer numbers supported by direct or derived evidence."""

    def leaf(text: str) -> List[float]:
        values = numeric_claim_values(text)
        values.extend(
            float(match.group(1)) for match in CASE_IDENTIFIER_NUMBER.finditer(text)
        )
        return values

    claimed = numeric_claim_values(answer)
    supported = numeric_claim_values(question)
    supported.extend(walk_numeric_values(evidence, leaf))
    supported.extend(derived_numeric_values(evidence))
    claimed_times = time_values_in_minutes(answer)
    supported_times = [
        minutes
        for _, minutes in (
            time_values_in_minutes(question)
            + time_values_in_minutes(json.dumps(evidence, ensure_ascii=False))
        )
    ]
    return [
        value
        for value in claimed
        if (
            _number_is_supported(value, supported)
            or _number_is_deterministically_derived(value, supported)
        )
        or any(
            numeric_values_match(value, raw_value)
            and _number_is_supported(minutes, supported_times)
            for raw_value, minutes in claimed_times
        )
    ]


def numeric_claims_are_grounded(
    answer: str,
    question: str,
    evidence: Dict[str, Any],
) -> bool:
    claimed = numeric_claim_values(answer)
    return len(grounded_numeric_claim_values(answer, question, evidence)) == len(
        claimed
    )


def numeric_grounding_evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build the one reusable numeric-evidence view for Task 1 and SFT export."""

    question = str(record.get("user_input") or "")
    requested = requested_artifacts(question)
    outputs = attach_tool_arguments(
        record.get("tool_outputs") or [],
        record.get("tool_calls") or [],
    )
    successful_outputs = [
        item
        for item in outputs
        if classify_tool_evidence(item, requested=requested).evidence_found
    ]
    successful_call_ids = {
        str(item.get("tool_call_id") or "")
        for item in successful_outputs
        if item.get("tool_call_id")
    }
    record_evidence = dict(record.get("evidence") or {})
    rebuilt_csv_evidence = record_csv_evidence(record, scope_text=question)
    if rebuilt_csv_evidence:
        record_evidence["csv_evidence"] = {
            **dict(record_evidence.get("csv_evidence") or {}),
            **rebuilt_csv_evidence,
        }
    derived_stats = _csv_line_totals_from_record(record)
    if derived_stats:
        record_evidence.setdefault("derived_line_stats", derived_stats)
    return {
        "prediction_summary": record.get("prediction_summary") or {},
        "constraint_check": record.get("constraint_check") or {},
        "evidence": record_evidence,
        "parsed_task": record.get("parsed_task") or {},
        "decision_summary": record.get("decision_summary") or {},
        "tool_calls": [
            item
            for item in record.get("tool_calls") or []
            if str(item.get("tool_call_id") or "") in successful_call_ids
        ],
        "tool_outputs": successful_outputs,
        "conversation_context": trusted_conversation_context(
            record.get("conversation_context") or [],
            verified_evidence_only=True,
        ),
        "verified_state": VerifiedDecisionState.from_dict(
            dict(record.get("state_before") or {})
        ).to_dict(),
    }


def derived_numeric_values(value: Any) -> List[float]:
    """Derive bounded counts, sums, and shares from typed or JSON evidence."""

    def number(item: Any) -> Optional[float]:
        if isinstance(item, bool):
            return None
        if isinstance(item, (int, float)):
            return float(item)
        if isinstance(item, str) and re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", item.strip()):
            return float(item)
        return None

    def rounded(item: float) -> List[float]:
        return [item, *(round(item, digits) for digits in range(7))]

    def collect(item: Any, totals: List[float]) -> List[float]:
        if isinstance(item, str):
            stripped = item.lstrip()
            if not stripped.startswith(("{", "[")):
                return []
            try:
                item, _ = json.JSONDecoder().raw_decode(stripped)
            except (TypeError, ValueError):
                return []
        if isinstance(item, dict):
            local_totals = [
                value
                for key, field in item.items()
                if "total" in str(key).casefold()
                and (value := number(field)) is not None
                and value != 0
            ]
            return [
                value
                for field in item.values()
                for value in collect(field, local_totals or totals)
            ]
        if not isinstance(item, list):
            return []

        derived = [float(len(item))]
        rows = [
            dict(row["values"]) if isinstance(row.get("values"), dict) else dict(row)
            for row in item
            if isinstance(row, dict)
        ]
        if rows and len(rows) <= 2_000:
            keys = sorted({key for row in rows for key in row})[:40]
            numeric_keys = [
                key
                for key in keys
                if any(number(row.get(key)) is not None for row in rows)
            ]
            category_keys = [
                key
                for key in keys
                if any(isinstance(row.get(key), str) for row in rows)
            ][:8]
            for numeric_key in numeric_keys:
                values = [
                    value
                    for row in rows
                    if (value := number(row.get(numeric_key))) is not None
                ]
                if not values:
                    continue
                derived.extend(rounded(sum(values)))
                derived.append(float(sum(value > 0 for value in values)))
                running = 0.0
                for value in values:
                    running += value
                    derived.extend(rounded(running))
                    for total in totals:
                        derived.extend(rounded(value / total * 100.0))
                        derived.extend(rounded(running / total * 100.0))
                for category_key in category_keys:
                    grouped: Dict[str, tuple[float, int]] = {}
                    for row in rows:
                        category = row.get(category_key)
                        value = number(row.get(numeric_key))
                        if isinstance(category, str) and value is not None:
                            total, count = grouped.get(category, (0.0, 0))
                            grouped[category] = (total + value, count + 1)
                    if len(grouped) <= 200:
                        derived.extend(
                            value
                            for total, _ in grouped.values()
                            for value in rounded(total)
                        )
                        derived.extend(float(count) for _, count in grouped.values())
        return derived + [value for field in item for value in collect(field, totals)]

    return collect(value, [])


def numeric_values_match(value: float, candidate: float) -> bool:
    return abs(value - candidate) <= max(0.01, abs(candidate) * 0.005)


def _number_is_supported(value: float, supported: List[float]) -> bool:
    return any(numeric_values_match(value, candidate) for candidate in supported)


def _number_is_deterministically_derived(value: float, supported: List[float]) -> bool:
    """Recognize sign changes and simple sums/differences using bounded evidence."""

    if _number_is_supported(-value, supported):
        return True
    bounded = supported[:2_000]
    rounded_counts: Dict[float, int] = {}
    for candidate in bounded:
        key = round(candidate, 6)
        rounded_counts[key] = rounded_counts.get(key, 0) + 1

    def has_two_operands(first: float, second: float) -> bool:
        first_key = round(first, 6)
        second_key = round(second, 6)
        return second_key in rounded_counts and (
            first_key != second_key or rounded_counts.get(first_key, 0) >= 2
        )

    for candidate in dict.fromkeys(bounded):
        if has_two_operands(candidate, value - candidate):
            return True
        if has_two_operands(candidate, candidate - value):
            return True
    for a in dict.fromkeys(bounded):
        for b in dict.fromkeys(bounded):
            if b:
                ratio = a / b
                if abs(value - ratio) <= max(0.005, abs(ratio) * 0.005):
                    return True
                pct_diff = abs(a - b) / abs(b) * 100.0
                if abs(value - pct_diff) <= max(0.5, pct_diff * 0.01):
                    return True
            if a + b > 0 and a > 0 and b > 0:
                share = a / (a + b) * 100.0
                if abs(value - share) <= max(0.5, share * 0.01):
                    return True
    return False
