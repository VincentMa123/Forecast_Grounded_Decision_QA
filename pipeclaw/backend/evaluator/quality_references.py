from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pipeclaw.backend.grounding.evidence.tool import DATA_FILE_REFERENCE


NUMERIC_SIGN_TRANSLATION = str.maketrans(
    {
        "\u2212": "-",
        "\ufe63": "-",
        "\uff0d": "-",
    }
)
NUMERIC_SPAN = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?"
)
CANDIDATE_IDENTIFIER = re.compile(
    r"\bcandidate_[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*\b",
    re.IGNORECASE,
)
CHINESE_ORDINAL_REFERENCE = re.compile(
    r"第\s*\d+(?:\s*[、,，/]\s*\d+)*\s*"
    r"(?:名|位|段|个|项|条|种|组|候选|方案|动作|管段|用户)?"
)
ENGLISH_ORDINAL_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_])\d+(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)
ENGLISH_RANK_REFERENCE = re.compile(
    r"\b(?:rank(?:ed)?|position)\s*(?:#|no\.?\s*)?"
    r"\d+(?:\s*[/,]\s*\d+)*",
    re.IGNORECASE,
)
NEGATED_NUMERIC_REFERENCE = re.compile(
    r"(?:不是|并非|不应为|\bnot\b|\bis\s+not\b)"
    r"[\s:*_`]{0,12}[-+]?\d+(?:\.\d+)?%?",
    re.IGNORECASE,
)
CASE_IDENTIFIER_NUMBER = re.compile(
    r"(?:mock_test|case)[_-]0*(\d+)\b",
    re.IGNORECASE,
)
TIME_RANGE = re.compile(
    r"(?P<first>\d+(?:\.\d+)?)\s*(?:-|–|—|~|～|to|through|到|至)\s*"
    r"(?P<second>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>hours?|hrs?|hr|h|小时|小時|minutes?|mins?|min|分钟|分鐘)",
    re.IGNORECASE,
)
TIME_VALUE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>hours?|hrs?|hr|h|小时|小時|minutes?|mins?|min|分钟|分鐘)",
    re.IGNORECASE,
)
_FULL_DATE_PATTERN = (
    r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"\d{4}年\d{1,2}月\d{1,2}日)"
)
_SHORT_DATE_PATTERN = r"(?:\d{1,2}[-/.]\d{1,2}|\d{1,2}(?:日|号))"
DATE_RANGE_REFERENCE = re.compile(
    rf"(?<!\d){_FULL_DATE_PATTERN}\s*"
    r"(?:-|–|—|~|～|to|through|到|至)\s*"
    rf"(?:{_FULL_DATE_PATTERN}|{_SHORT_DATE_PATTERN})(?!\d)",
    re.IGNORECASE,
)
DATE_REFERENCE = re.compile(rf"(?<!\d){_FULL_DATE_PATTERN}(?!\d)")
COMPACT_DATE_REFERENCE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?!\d)"
)
YEAR_REFERENCE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?=年|[-/.]\d)")
VARIABLE_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?(?![A-Za-z0-9_:])"
)
SFT_FILE_REFERENCE = re.compile(r"(?i)\b[\w.-]+\.(?:csv|jsonl?|xlsx?|parquet)\b")

# Tool-output parsing treats serialized rows as evidence.  It intentionally
# accepts leading-decimal values while final-answer parsing retains its stricter
# historical contract.
_OBSERVED_DATE_SPAN = re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?")
_OBSERVED_NUMERIC_SPAN = re.compile(
    r"(?<![A-Za-z0-9_])[+\-−]?"
    r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?![A-Za-z0-9_])"
)
_CHINESE_ORDINAL_SUFFIX = re.compile(r"\s*(?:[段条章节次项名位类级]|[、，,]\s*第)")
_DATE_COMPONENT_SUFFIX = re.compile(r"\s*[日号月]")


def variable_references(text: str) -> list[str]:
    """Return canonical PipeFormer variable IDs named in *text*."""

    return VARIABLE_REFERENCE.findall(text)


def file_references(text: str) -> list[str]:
    """Return data-artifact names using the grounding tool contract."""

    return DATA_FILE_REFERENCE.findall(text)


def sft_file_references(text: str) -> list[str]:
    """Return the historical SFT token set without broadening quality checks."""

    return SFT_FILE_REFERENCE.findall(text)


def numeric_claim_values(text: str) -> list[float]:
    """Return final-answer numbers while excluding metadata and list labels."""

    normalized = text.translate(NUMERIC_SIGN_TRANSLATION)
    ignored_spans = [
        match.span() for match in DATE_RANGE_REFERENCE.finditer(normalized)
    ]
    ignored_spans.extend(match.span() for match in DATE_REFERENCE.finditer(normalized))
    ignored_spans.extend(
        match.span() for match in COMPACT_DATE_REFERENCE.finditer(normalized)
    )
    ignored_spans.extend(
        match.span() for match in DATA_FILE_REFERENCE.finditer(normalized)
    )
    ignored_spans.extend(match.span() for match in YEAR_REFERENCE.finditer(normalized))
    ignored_spans.extend(
        match.span() for match in CANDIDATE_IDENTIFIER.finditer(normalized)
    )
    ignored_spans.extend(
        match.span() for match in CHINESE_ORDINAL_REFERENCE.finditer(normalized)
    )
    ignored_spans.extend(
        match.span() for match in ENGLISH_ORDINAL_REFERENCE.finditer(normalized)
    )
    ignored_spans.extend(
        match.span() for match in ENGLISH_RANK_REFERENCE.finditer(normalized)
    )
    ignored_spans.extend(
        match.span() for match in NEGATED_NUMERIC_REFERENCE.finditer(normalized)
    )
    values = []
    for match in NUMERIC_SPAN.finditer(normalized):
        raw_value = match.group(0)
        if raw_value.lstrip("+-").startswith("."):
            continue
        if any(
            start <= match.start() and match.end() <= end
            for start, end in ignored_spans
        ):
            continue
        previous = normalized[match.start() - 1 : match.start()]
        following = normalized[match.end() : match.end() + 1]
        if previous in {"(", "（"} and following in {")", "）"}:
            continue
        line_start = normalized.rfind("\n", 0, match.start()) + 1
        prefix = normalized[line_start : match.start()]
        remainder = normalized[match.end() :]
        if not prefix.strip(" \t-*") and remainder[:1] in {".", ")", "）", "、"}:
            continue
        if prefix.strip() == "|" and remainder.lstrip().startswith("|"):
            continue
        if re.match(r"\s*(?:个?字|characters?\b)", remainder, re.IGNORECASE):
            continue
        values.append(float(raw_value.replace(",", "")))
    return values


def time_values_in_minutes(text: str) -> list[tuple[float, float]]:
    """Return raw time claims paired with their minute-normalized values."""

    values: list[tuple[float, float]] = []

    def append(raw_value: str, unit: str) -> None:
        number = float(raw_value)
        multiplier = (
            60.0
            if unit.casefold() in {"hour", "hours", "hr", "hrs", "h", "小时", "小時"}
            else 1.0
        )
        values.append((number, number * multiplier))

    for match in TIME_RANGE.finditer(text):
        append(match.group("first"), match.group("unit"))
        append(match.group("second"), match.group("unit"))
    for match in TIME_VALUE.finditer(text):
        append(match.group("value"), match.group("unit"))
    return values


def observed_numeric_claim_items(text: str) -> list[tuple[float, int, int]]:
    """Return number spans that a tool or evaluator could observe as evidence."""

    items: list[tuple[float, int, int]] = []
    date_spans = [match.span() for match in _OBSERVED_DATE_SPAN.finditer(text)]
    for match in _OBSERVED_NUMERIC_SPAN.finditer(text):
        raw_value = match.group(0)
        if any(start <= match.start() < end for start, end in date_spans):
            continue
        remainder = text[match.end() :]
        digits = raw_value.lstrip("+-−").replace(",", "")
        is_integer = digits.isdigit()
        if is_integer and _is_compact_date(digits):
            continue
        if (
            is_integer
            and remainder[:1] in {".", ")", "、"}
            and (len(remainder) == 1 or remainder[1].isspace())
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
        if (
            is_integer
            and prefix.rstrip().endswith("至")
            and _DATE_COMPONENT_SUFFIX.match(suffix)
        ):
            continue
        try:
            value = float(raw_value.replace("−", "-").replace(",", ""))
        except ValueError:
            continue
        items.append((value, match.start(), match.end()))
    return items


def observed_numeric_claim_values(text: str) -> list[float]:
    return [value for value, _, _ in observed_numeric_claim_items(text)]


def walk_numeric_values(value: Any, leaf) -> list[float]:
    """Walk structured or serialized evidence, applying a leaf parser to strings."""

    if isinstance(value, str):
        return leaf(value)
    if isinstance(value, Mapping):
        return [
            number
            for item in value.values()
            for number in walk_numeric_values(item, leaf)
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [number for item in value for number in walk_numeric_values(item, leaf)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []


def observed_numeric_values(value: Any) -> list[float]:
    """Walk structured or serialized evidence without inventing conversions."""

    return walk_numeric_values(value, observed_numeric_claim_values)


def _is_compact_date(raw: str) -> bool:
    return (
        len(raw) == 8
        and raw.isdigit()
        and 1900 <= int(raw[:4]) <= 2100
        and 1 <= int(raw[4:6]) <= 12
        and 1 <= int(raw[6:]) <= 31
    )
