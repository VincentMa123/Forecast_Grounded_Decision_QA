from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


PIPEFORMER_TASK_SCHEMA_VERSION = "pipeformer_task"


CATEGORY_MARKERS: Dict[str, List[str]] = {
    "pressure": ["\u538b\u529b", "pressure"],
    "flow": ["\u6d41\u91cf", "flow"],
    "linepack": ["\u7ba1\u5b58", "linepack"],
    "compressor": ["\u538b\u7f29\u673a", "compressor"],
    "equipment_regulation": ["\u8bbe\u5907", "\u9600\u95e8", "\u8c03\u538b", "\u8fb9\u754c\u63a7\u5236", "equipment", "valve", "regulator", "boundary control"],
    "abnormality_warning": ["\u5f02\u5e38", "\u6cc4\u6f0f", "\u7a81\u53d8", "abnormal", "leak", "sudden"],
    "dispatch_priority": ["\u80fd\u8017", "\u6210\u672c", "\u4f18\u5148", "energy", "cost", "priority"],
}

CATEGORY_ATTENTION_TARGETS: Dict[str, List[str]] = {
    "pressure": ["nodes"],
    "flow": ["segments"],
    "linepack": ["linepack"],
    "compressor": ["compressors"],
    "equipment_regulation": ["valves", "pressure_regulators", "boundary_controls"],
    "abnormality_warning": ["nodes", "segments", "compressors"],
    "dispatch_priority": ["dispatch_priority_audit"],
}

CATEGORY_OUTPUT_STATE_VARIABLES: Dict[str, List[str]] = {
    "pressure": ["pressure"],
    "flow": ["flow"],
    "linepack": ["linepack"],
    "compressor": ["compressor_load", "compression_ratio", "compressor_speed", "compressor_power"],
    "equipment_regulation": ["valve_opening", "regulator_range"],
    "abnormality_warning": ["pressure", "flow", "compressor"],
    "dispatch_priority": ["energy_consumption", "operating_cost"],
}

DEFAULT_CONSTRAINT_VERIFICATION_TYPES = list(CATEGORY_MARKERS)
CASE_PATTERNS = [
    re.compile(r"mock_test\s*(?:\u7684)?\u7b2c\s*0*(\d+)\s*(?:\u4e2a)?\u7b97\u4f8b", re.I),
    re.compile(r"mock_test[_\s-]*0*(\d+)", re.I),
    re.compile(r"case[_\s-]*0*(\d+)", re.I),
]
OPERATING_CONDITION_PATTERNS = [
    re.compile(r"(?:operating[-\s]?condition|current\s+condition|condition)\s*(?:number|id|#|:)?\s*0*(\d+)", re.I),
    re.compile(r"(?:\u5de5\u51b5|\u8fd0\u884c\u6761\u4ef6)\s*(?:\u7f16\u53f7|id|#|:|\u4e3a|\u662f)?\s*0*(\d+)", re.I),
]
VARIABLE_RE = re.compile(r"\b[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?\b")
PERCENT_PATTERNS = [
    re.compile(r"(\u4e0a\u8c03|\u4e0b\u8c03|increase|decrease|raise|lower|up|down)\s*(\d+(?:\.\d+)?)\s*%", re.I),
    re.compile(r"(\u4e0a\u8c03|\u4e0b\u8c03|increase|decrease|raise|lower|up|down)[^%]{0,80}?(\d+(?:\.\d+)?)\s*%", re.I),
]
HORIZON_RE = re.compile(r"(?:\u672a\u6765|next|future|forecast\s*horizon)?\s*(\d+(?:\.\d+)?)\s*(\u5c0f\u65f6|\u5206\u949f|hours?|hrs?|minutes?|mins?)", re.I)
STATUS_TARGET_PATTERNS = [
    (re.compile(r"(?:\u5207\u6362\u4e3a|\u8bbe\u4e3a)\s*[\"'\u201c\u201d\u2018\u2019]*(?:\u505c\u673a|\u5173\u95ed)"), 0.0),
    (re.compile(r"(?:\u5207\u6362\u4e3a|\u8bbe\u4e3a)\s*[\"'\u201c\u201d\u2018\u2019]*(?:\u5f00\u673a|\u5f00\u542f)"), 1.0),
    (re.compile(r"\b(?:to|as)\s+(?:off|stopped|closed)\b", re.I), 0.0),
    (re.compile(r"\b(?:to|as)\s+(?:on|running|open)\b", re.I), 1.0),
]
STATUS_VALUE_PATTERNS = [
    (re.compile(r"(?:\u505c\u673a|\u5173\u95ed)|\b(?:off|stopped|closed)\b", re.I), 0.0),
    (re.compile(r"(?:\u5f00\u673a|\u5f00\u542f)|\b(?:on|running|open)\b", re.I), 1.0),
]
PREDICTION_MARKERS = ["\u9884\u6d4b", "forecast", "predict", "prediction"]
VERIFICATION_MARKERS = ["\u6821\u6838", "\u68c0\u67e5", "verify", "verification", "check"]


def parse_condition(question: str) -> Dict[str, Any]:
    case_number = _first_matched_int(question, CASE_PATTERNS)
    operating_condition_number = _first_matched_int(question, OPERATING_CONDITION_PATTERNS) or case_number
    variable_match = VARIABLE_RE.search(question)
    percent_match = _first_match(question, PERCENT_PATTERNS)
    horizon_match = HORIZON_RE.search(question)

    if not variable_match:
        raise ValueError("Could not parse disturbance variable from scenario question.")

    disturbance_direction = _parse_direction(percent_match.group(1) if percent_match else "")
    disturbance_magnitude_percent = float(percent_match.group(2)) if percent_match else None
    forecast_horizon_minutes = _parse_horizon_minutes(horizon_match)
    constraint_verification_types = _parse_constraint_verification_types(question)
    # Unspecified boundary controls are held at their observed values by default.
    # Callers can still explicitly override this through the structured tool argument.
    keep_other_boundary_controls = True
    disturbance_variable = variable_match.group(0)
    status_setpoint = _parse_status_setpoint(question) if disturbance_variable.endswith(":ST") else None
    setpoints = {disturbance_variable: status_setpoint} if status_setpoint is not None else {}

    task = {
        "case_id": f"mock_test_{case_number:03d}" if case_number is not None else None,
        "current_operating_condition_number": operating_condition_number,
        "boundary_conditions": {
            "keep_other_boundary_controls": keep_other_boundary_controls,
            "disturbance_variable": disturbance_variable,
            "disturbance_direction": disturbance_direction,
            "disturbance_magnitude_percent": disturbance_magnitude_percent,
            "setpoints": setpoints,
        },
        "disturbance_variable": disturbance_variable,
        "disturbance_direction": disturbance_direction,
        "disturbance_magnitude_percent": disturbance_magnitude_percent,
        "forecast_horizon_minutes": forecast_horizon_minutes,
        "attention_targets": _targets_for_checks(constraint_verification_types, CATEGORY_ATTENTION_TARGETS),
        "output_state_variables": _targets_for_checks(constraint_verification_types, CATEGORY_OUTPUT_STATE_VARIABLES),
        "constraint_verification_types": constraint_verification_types,
        "task_type": _parse_task_type(question),
        "parse_schema_version": PIPEFORMER_TASK_SCHEMA_VERSION,
    }
    task.update(_legacy_aliases(task))
    return task


def _first_match(text: str, patterns: Iterable[re.Pattern[str]]) -> Optional[re.Match[str]]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match
    return None


def _first_matched_int(text: str, patterns: Iterable[re.Pattern[str]]) -> Optional[int]:
    match = _first_match(text, patterns)
    return int(match.group(1)) if match else None


def _parse_direction(raw: str) -> str:
    value = raw.lower()
    if value in {"\u4e0a\u8c03", "increase", "raise", "up"}:
        return "up"
    if value in {"\u4e0b\u8c03", "decrease", "lower", "down"}:
        return "down"
    return "unknown"


def _parse_horizon_minutes(match: Optional[re.Match[str]]) -> Optional[int]:
    if not match:
        return None
    magnitude = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"\u5c0f\u65f6", "hour", "hours", "hr", "hrs"}:
        return int(magnitude * 60)
    return int(magnitude)


def _parse_status_setpoint(text: str) -> Optional[float]:
    for pattern, value in STATUS_TARGET_PATTERNS:
        if pattern.search(text):
            return value
    matched_values = {
        value
        for pattern, value in STATUS_VALUE_PATTERNS
        if pattern.search(text)
    }
    return next(iter(matched_values)) if len(matched_values) == 1 else None


def _parse_constraint_verification_types(question: str) -> List[str]:
    lowered = question.lower()
    result = []
    for category, markers in CATEGORY_MARKERS.items():
        if any(marker.lower() in lowered for marker in markers):
            result.append(category)
    return result


def _targets_for_checks(checks: List[str], mapping: Dict[str, List[str]]) -> List[str]:
    source = checks or DEFAULT_CONSTRAINT_VERIFICATION_TYPES
    result = []
    for check in source:
        for value in mapping.get(check, []):
            if value not in result:
                result.append(value)
    return result


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def _parse_task_type(question: str) -> str:
    has_prediction = _contains_any(question, PREDICTION_MARKERS)
    has_verification = _contains_any(question, VERIFICATION_MARKERS)
    if has_prediction and has_verification:
        return "prediction_and_verification"
    if has_prediction:
        return "prediction"
    if has_verification:
        return "verification"
    return "unknown"


def _legacy_aliases(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "changed_variable": task["disturbance_variable"],
        "change_direction": task["disturbance_direction"],
        "change_percent": task["disturbance_magnitude_percent"],
        "keep_other_boundary_controls": task["boundary_conditions"]["keep_other_boundary_controls"],
        "requested_checks": task["constraint_verification_types"],
    }
