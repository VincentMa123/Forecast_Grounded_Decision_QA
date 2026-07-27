from __future__ import annotations

import csv
import io
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .tool_evidence import attach_tool_arguments, tool_output_failed
from .pipeline_scope import PIPELINE_COLUMNS, filter_rows_by_named_pipeline


MAX_EVIDENCE_FILES = 12
MAX_EVIDENCE_ROWS = 12
MAX_COMPUTED_RESULTS = 1
MAX_EVIDENCE_FIELDS = 10
MAX_VALUE_CHARS = 160
MAX_COMPUTED_OUTPUT_CHARS = 8_000
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])")
CSV_FILE_RE = re.compile(r"(?i)(?<![\w.-])[\w.-]+\.csv(?![\w.-])")
ANSWER_TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z_]{2,}")
GENERIC_TEXT_VALUES = {
    "管段",
    "管线",
    "节点",
    "气源",
    "需求",
    "联络",
    "平衡",
    "pipeline",
    "node",
    "source",
}
AGGREGATION_RE = re.compile(
    r"累计|合计|总和|总(?:用户数|消耗量|流量|用量|数量)|求和|汇总"
    r"|\b(?:sum|total|aggregate|cumulative)\b",
    re.IGNORECASE,
)
ABSOLUTE_RE = re.compile(r"绝对|absolute|\|[^|]+\|", re.IGNORECASE)
START_COLUMNS = ("起点站名", "起点", "from", "start")
END_COLUMNS = ("终点站名", "终点", "to", "end")


def build_csv_evidence(
    tool_calls: Iterable[Dict[str, Any]],
    tool_outputs: Iterable[Dict[str, Any]],
    answer: str,
    *,
    scope_text: str = "",
) -> Dict[str, Any]:
    """Return bounded CSV rows whose values directly support the answer."""
    outputs = attach_tool_arguments(tool_outputs, tool_calls)
    answer_text = answer.casefold()
    answer_numbers = {_normalized_number(value) for value in NUMBER_RE.findall(answer)}
    folded_scope = scope_text.casefold()
    scope_numbers = {_normalized_number(value) for value in NUMBER_RE.findall(scope_text)}
    candidates: List[tuple[int, int, str, Dict[str, Any]]] = []
    source_files: List[str] = []
    seen_source_files = set()
    seen_rows = set()
    row_index = 0

    for item in outputs:
        if str(item.get("name") or "").casefold() != "read_file" or tool_output_failed(item):
            continue
        output = item.get("output") or {}
        if not isinstance(output, dict):
            continue
        source_file = _csv_filename(item, output)
        content = str(output.get("content") or "")
        rows, _ = filter_rows_by_named_pipeline(_parse_csv_rows(content), scope_text)
        if not source_file or not rows:
            continue
        if source_file not in seen_source_files:
            seen_source_files.add(source_file)
            if len(source_files) < MAX_EVIDENCE_FILES:
                source_files.append(source_file)
        for row in rows:
            signature = (source_file, tuple(row.items()))
            if signature in seen_rows:
                continue
            seen_rows.add(signature)
            answer_score = _row_relevance(row, answer_text, answer_numbers)
            scope_score = _row_relevance(row, folded_scope, scope_numbers)
            score = answer_score * 2 + scope_score
            if score:
                candidates.append((score, row_index, source_file, row))
            row_index += 1

    for item in outputs:
        if str(item.get("name") or "").casefold() != "run_command" or tool_output_failed(item):
            continue
        output = item.get("output") or {}
        if not isinstance(output, dict):
            continue
        for match in CSV_FILE_RE.findall(str(output.get("stdout") or "")):
            source_file = Path(match).name
            if source_file in seen_source_files:
                continue
            seen_source_files.add(source_file)
            if len(source_files) < MAX_EVIDENCE_FILES:
                source_files.append(source_file)

    computed_results = _structured_computation_results(
        outputs,
        answer_text,
        answer_numbers,
    )
    if not source_files and not computed_results:
        return {}

    maximum_score = max((item[0] for item in candidates), default=0)
    minimum_score = max(2, maximum_score - 1) if maximum_score >= 2 else 1
    selected = sorted(
        (item for item in candidates if item[0] >= minimum_score),
        key=lambda item: (-item[0], item[1]),
    )[:MAX_EVIDENCE_ROWS]
    evidence: Dict[str, Any] = {
        "source_files": source_files,
        "source_file_count": len(seen_source_files),
    }
    if computed_results:
        evidence["computed_results"] = computed_results
    elif selected:
        evidence["answer_rows"] = [
            {
                "source_file": source_file,
                "values": _compact_row(row),
            }
            for _, _, source_file, row in selected
        ]
        selected_rows = [
            (source_file, row)
            for _, _, source_file, row in selected
        ]
        selection_summary = _selection_summary(selected_rows)
        if selection_summary:
            evidence["selection_summary"] = selection_summary
        derived_results = _derived_aggregate_results(
            selected_rows,
            answer_text=answer_text,
            answer_numbers=answer_numbers,
            scope_text=folded_scope,
        )
        if derived_results:
            evidence["derived_results"] = derived_results
    return evidence


def _selection_summary(
    selected_rows: List[tuple[str, Dict[str, Any]]],
) -> Dict[str, int]:
    """Describe the bounded row set so count claims remain auditable."""
    source_files = {source_file for source_file, _ in selected_rows}
    pipelines = {
        value
        for _, row in selected_rows
        for column in PIPELINE_COLUMNS
        if (value := str(row.get(column) or "").strip())
    }
    segment_pairs = {
        (start, end)
        for _, row in selected_rows
        if (start := _first_row_value(row, START_COLUMNS))
        and (end := _first_row_value(row, END_COLUMNS))
    }
    summary = {
        "selected_row_count": len(selected_rows),
        "selected_source_file_count": len(source_files),
    }
    if pipelines:
        summary["pipeline_count"] = len(pipelines)
    if segment_pairs:
        summary["segment_pair_count"] = len(segment_pairs)
    return summary


def _derived_aggregate_results(
    selected_rows: List[tuple[str, Dict[str, Any]]],
    *,
    answer_text: str,
    answer_numbers: set[str],
    scope_text: str,
) -> List[Dict[str, Any]]:
    """Verify bounded aggregate claims from the query-scoped CSV rows.

    The derivation is intentionally conservative: it runs only for an explicit
    aggregation request, groups by a named pipeline column, and records only
    totals or pairwise differences that are actually repeated in the answer.
    """
    combined_text = f"{scope_text}\n{answer_text}"
    if not AGGREGATION_RE.search(combined_text):
        return []

    group_column = _named_group_column(selected_rows, combined_text)
    measure_column = _numeric_measure_column(selected_rows, combined_text)
    if not measure_column:
        return []

    use_absolute = bool(ABSOLUTE_RE.search(combined_text))
    if not group_column:
        values = [
            (source_file, number)
            for source_file, row in selected_rows
            if (number := _decimal_value(row.get(measure_column))) is not None
        ]
        total = sum(
            (abs(value) if use_absolute else value for _, value in values),
            Decimal("0"),
        )
        if not values or not _decimal_is_claimed(total, answer_numbers):
            return []
        return [
            {
                "operation": "sum_abs" if use_absolute else "sum",
                "measure": measure_column,
                "row_count": len(values),
                "nonzero_row_count": sum(value != 0 for _, value in values),
                "zero_row_count": sum(value == 0 for _, value in values),
                "source_files": list(
                    dict.fromkeys(source_file for source_file, _ in values)
                ),
                "value": _decimal_to_number(total),
            }
        ]

    grouped: Dict[str, List[tuple[str, Decimal]]] = {}
    for source_file, row in selected_rows:
        group_value = str(row.get(group_column) or "").strip()
        number = _decimal_value(row.get(measure_column))
        if not group_value or number is None:
            continue
        grouped.setdefault(group_value, []).append(
            (source_file, abs(number) if use_absolute else number)
        )

    results: List[Dict[str, Any]] = []
    totals: List[tuple[str, Decimal]] = []
    for group_value, values in grouped.items():
        total = sum((value for _, value in values), Decimal("0"))
        if not _decimal_is_claimed(total, answer_numbers):
            continue
        totals.append((group_value, total))
        results.append(
            {
                "operation": "sum_abs" if use_absolute else "sum",
                "group_by": group_column,
                "group_value": group_value,
                "measure": measure_column,
                "row_count": len(values),
                "source_files": list(dict.fromkeys(source for source, _ in values)),
                "value": _decimal_to_number(total),
            }
        )

    for left_index, (left_group, left_total) in enumerate(totals):
        for right_group, right_total in totals[left_index + 1 :]:
            difference = abs(left_total - right_total)
            if not _decimal_is_claimed(difference, answer_numbers):
                continue
            results.append(
                {
                    "operation": "absolute_difference",
                    "left_group": left_group,
                    "right_group": right_group,
                    "measure": measure_column,
                    "value": _decimal_to_number(difference),
                }
            )
    return results


def _named_group_column(
    selected_rows: List[tuple[str, Dict[str, Any]]],
    combined_text: str,
) -> str:
    for column in PIPELINE_COLUMNS:
        values = list(dict.fromkeys(
            str(row.get(column) or "").strip()
            for _, row in selected_rows
            if str(row.get(column) or "").strip()
        ))
        if len(values) >= 2 and all(value.casefold() in combined_text for value in values):
            return column
    return ""


def _numeric_measure_column(
    selected_rows: List[tuple[str, Dict[str, Any]]],
    combined_text: str,
) -> str:
    columns = list(dict.fromkeys(
        key
        for _, row in selected_rows
        for key in row
    ))
    candidates = []
    for column in columns:
        values = [_decimal_value(row.get(column)) for _, row in selected_rows]
        if not values or any(value is None for value in values):
            continue
        folded = column.casefold()
        score = int(folded in combined_text) * 2 + int(
            any(token in folded for token in ("流量", "flow", "value", "amount", "volume"))
        )
        candidates.append((score, column))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[1]


def _first_row_value(row: Dict[str, Any], columns: Iterable[str]) -> str:
    for column in columns:
        value = str(row.get(column) or "").strip()
        if value:
            return value
    return ""


def _decimal_value(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def _decimal_is_claimed(value: Decimal, answer_numbers: set[str]) -> bool:
    for raw_candidate in answer_numbers:
        try:
            candidate = Decimal(raw_candidate)
        except (InvalidOperation, ValueError):
            continue
        tolerance = max(Decimal("0.01"), abs(candidate) * Decimal("0.005"))
        if abs(value - candidate) <= tolerance:
            return True
    return False


def _decimal_to_number(value: Decimal) -> int | float:
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return int(normalized)
    return float(normalized)


def _structured_computation_results(
    outputs: Iterable[Dict[str, Any]],
    answer_text: str,
    answer_numbers: set[str],
) -> List[Dict[str, Any]]:
    """Keep bounded JSON results emitted by successful computation commands.

    Aggregate answers cannot be represented faithfully by a few source CSV rows.
    A structured command result is therefore preferred when at least one of its
    scalar values is repeated in the answer.
    """
    results = []
    seen_values = set()
    for item in reversed(list(outputs)):
        if len(results) >= MAX_COMPUTED_RESULTS:
            break
        if str(item.get("name") or "").casefold() != "run_command" or tool_output_failed(item):
            continue
        output = item.get("output") or {}
        if not isinstance(output, dict):
            continue
        stdout = str(output.get("stdout") or "").strip()
        if not stdout or len(stdout) > MAX_COMPUTED_OUTPUT_CHARS:
            continue
        try:
            value = json.loads(stdout)
        except (json.JSONDecodeError, TypeError):
            value = stdout
        if isinstance(value, (dict, list)):
            supports_answer = _structured_value_supports_answer(value, answer_text, answer_numbers)
            compact_value = _compact_json_value(value)
        else:
            supports_answer = _plain_computation_supports_answer(stdout, answer_text, answer_numbers)
            compact_value = stdout[:MAX_COMPUTED_OUTPUT_CHARS]
        if not supports_answer:
            continue
        signature = json.dumps(compact_value, ensure_ascii=False, sort_keys=True)
        if signature in seen_values:
            continue
        seen_values.add(signature)
        results.append({
            "tool_call_id": item.get("tool_call_id"),
            "value": compact_value,
        })
    return list(reversed(results))


def _plain_computation_supports_answer(
    value: str,
    answer_text: str,
    answer_numbers: set[str],
) -> bool:
    output_numbers = {_normalized_number(item) for item in NUMBER_RE.findall(value)}
    if any(number and number in answer_numbers for number in output_numbers):
        return True
    ignored = {"因此", "其中", "数据", "记录", "累计", "流量", "比较"}
    tokens = {
        token.casefold()
        for token in ANSWER_TOKEN_RE.findall(answer_text)
        if token not in ignored and len(token) >= 3
    }
    return sum(token in value.casefold() for token in tokens) >= 2


def _structured_value_supports_answer(
    value: Any,
    answer_text: str,
    answer_numbers: set[str],
) -> bool:
    for scalar in _iter_json_scalars(value):
        text = str(scalar).strip()
        number = _normalized_number(text)
        if number and number in answer_numbers:
            return True
        folded = text.casefold()
        if len(folded) >= 2 and folded in answer_text:
            return True
    return False


def _iter_json_scalars(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _iter_json_scalars(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_scalars(item)
    elif value is not None:
        yield value


def _compact_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return str(value)[:MAX_VALUE_CHARS]
    if isinstance(value, dict):
        return {
            str(key)[:MAX_VALUE_CHARS]: _compact_json_value(item, depth=depth + 1)
            for key, item in list(value.items())[:MAX_EVIDENCE_FIELDS]
        }
    if isinstance(value, list):
        return [
            _compact_json_value(item, depth=depth + 1)
            for item in value[:MAX_EVIDENCE_ROWS]
        ]
    if isinstance(value, str):
        return value[:MAX_VALUE_CHARS]
    return value


def _csv_filename(item: Dict[str, Any], output: Dict[str, Any]) -> str:
    arguments = item.get("arguments") or {}
    candidate = arguments.get("path") or output.get("path") or output.get("abs_path") or ""
    normalized = str(candidate).replace("\\", "/")
    name = Path(normalized).name
    return name if name.casefold().endswith(".csv") else ""


def _parse_csv_rows(content: str) -> List[Dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")))
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            return []
        rows = []
        for row in reader:
            cleaned = {
                str(key).strip(): str(value or "").strip()
                for key, value in row.items()
                if key is not None and str(key).strip()
            }
            if any(cleaned.values()):
                rows.append(cleaned)
        return rows
    except (csv.Error, UnicodeError):
        return []


def _row_relevance(
    row: Dict[str, str],
    answer_text: str,
    answer_numbers: set[str],
) -> int:
    matched = set()
    for value in row.values():
        stripped = value.strip()
        if not stripped:
            continue
        number = _normalized_number(stripped)
        if number and number in answer_numbers:
            matched.add(("number", number))
            continue
        folded = stripped.casefold()
        if (
            len(folded) >= 2
            and folded not in GENERIC_TEXT_VALUES
            and folded in answer_text
        ):
            matched.add(("text", folded))
    return len(matched)


def _compact_row(row: Dict[str, str]) -> Dict[str, Any]:
    compact = {}
    for key, value in row.items():
        if not value or len(compact) >= MAX_EVIDENCE_FIELDS:
            continue
        compact[key[:MAX_VALUE_CHARS]] = _typed_value(value[:MAX_VALUE_CHARS])
    return compact


def _typed_value(value: str) -> Any:
    if not NUMBER_RE.fullmatch(value):
        return value
    number = _normalized_number(value)
    if not number:
        return value
    return float(number) if "." in number else int(number)


def _normalized_number(value: str) -> str:
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        return ""
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", "+0"} else normalized
