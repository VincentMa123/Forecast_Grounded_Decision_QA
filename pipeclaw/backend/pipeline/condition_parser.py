from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


PIPEFORMER_TASK_SCHEMA_VERSION = "pipeformer_task"


CATEGORY_MARKERS: Dict[str, List[str]] = {
    "pressure": ["压力", "pressure"],
    "flow": ["流量", "flow"],
    "linepack": ["管存", "linepack"],
    "compressor": ["压缩机", "compressor"],
    "equipment_regulation": [
        "设备",
        "阀门",
        "调压",
        "边界控制",
        "equipment",
        "valve",
        "regulator",
        "boundary control",
    ],
    "abnormality_warning": ["异常", "泄漏", "突变", "abnormal", "leak", "sudden"],
    "dispatch_priority": ["能耗", "成本", "优先", "energy", "cost", "priority"],
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
    "compressor": [
        "compressor_load",
        "compression_ratio",
        "compressor_speed",
        "compressor_power",
    ],
    "equipment_regulation": ["valve_opening", "regulator_range"],
    "abnormality_warning": ["pressure", "flow", "compressor"],
    "dispatch_priority": ["energy_consumption", "operating_cost"],
}

DEFAULT_CONSTRAINT_VERIFICATION_TYPES = list(CATEGORY_MARKERS)
CASE_PATTERNS = [
    re.compile(r"mock_test\s*(?:的)?第\s*0*(\d+)\s*(?:个)?算例", re.I),
    re.compile(r"mock_test[_\s-]*0*(\d+)", re.I),
    re.compile(r"case[_\s-]*0*(\d+)", re.I),
]
OPERATING_CONDITION_PATTERNS = [
    re.compile(
        r"(?:operating[-\s]?condition|current\s+condition|condition)\s*(?:number|id|#|:)?\s*0*(\d+)",
        re.I,
    ),
    re.compile(r"(?:工况|运行条件)\s*(?:编号|id|#|:|为|是)?\s*0*(\d+)", re.I),
]
VARIABLE_RE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?"
    r"(?![A-Za-z0-9_])"
)
PERCENT_PATTERNS = [
    re.compile(
        r"(上调|下调|increase|decrease|raise|lower|up|down)\s*(\d+(?:\.\d+)?)\s*%", re.I
    ),
    re.compile(
        r"(上调|下调|increase|decrease|raise|lower|up|down)[^%]{0,80}?(\d+(?:\.\d+)?)\s*%",
        re.I,
    ),
]
DIRECTION_RE = re.compile(r"(上调|下调|increase|decrease|raise|lower|up|down)", re.I)
HORIZON_RE = re.compile(
    r"(?:未来|next|future|forecast\s*horizon)?\s*(\d+(?:\.\d+)?)\s*(小时|分钟|hours?|hrs?|minutes?|mins?)",
    re.I,
)
STATUS_TARGET_PATTERNS = [
    (re.compile(r"(?:切换为|设为)\s*[\"'“”‘’]*(?:停机|关闭)"), 0.0),
    (re.compile(r"(?:切换为|设为)\s*[\"'“”‘’]*(?:开机|开启)"), 1.0),
    (re.compile(r"\b(?:to|as)\s+(?:off|stopped|closed)\b", re.I), 0.0),
    (re.compile(r"\b(?:to|as)\s+(?:on|running|open)\b", re.I), 1.0),
]
STATUS_VALUE_PATTERNS = [
    (re.compile(r"(?:停机|关闭)|\b(?:off|stopped|closed)\b", re.I), 0.0),
    (re.compile(r"(?:开机|开启)|\b(?:on|running|open)\b", re.I), 1.0),
]
PREDICTION_MARKERS = ["预测", "forecast", "predict", "prediction"]
VERIFICATION_MARKERS = ["校核", "检查", "verify", "verification", "check"]


def parse_condition(question: str) -> Dict[str, Any]:
    """Parse bilingual scenario text using the repository's fixed grammar."""
    case_number = _first_matched_int(question, CASE_PATTERNS)
    operating_condition_number = (
        _first_matched_int(question, OPERATING_CONDITION_PATTERNS) or case_number
    )
    variable_match = VARIABLE_RE.search(question)
    percent_match = _first_match(question, PERCENT_PATTERNS)
    direction_match = percent_match or DIRECTION_RE.search(question)
    horizon_match = HORIZON_RE.search(question)

    if not variable_match:
        raise ValueError("Could not parse disturbance variable from scenario question.")

    disturbance_direction = _parse_direction(
        direction_match.group(1) if direction_match else ""
    )
    disturbance_magnitude_percent = (
        float(percent_match.group(2)) if percent_match else None
    )
    forecast_horizon_minutes = _parse_horizon_minutes(horizon_match)
    lowered = question.lower()
    constraint_verification_types = [
        category
        for category, markers in CATEGORY_MARKERS.items()
        if any(marker.lower() in lowered for marker in markers)
    ]
    # Unspecified boundary controls are held at their observed values by default.
    # Callers can still explicitly override this through the structured tool argument.
    keep_other_boundary_controls = True
    disturbance_variable = variable_match.group(0)
    status_setpoint = (
        _parse_status_setpoint(question, disturbance_variable)
        if disturbance_variable.endswith(":ST")
        else None
    )
    setpoints = (
        {disturbance_variable: status_setpoint} if status_setpoint is not None else {}
    )

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
        "attention_targets": targets_for_checks(
            constraint_verification_types,
            CATEGORY_ATTENTION_TARGETS,
        ),
        "output_state_variables": targets_for_checks(
            constraint_verification_types,
            CATEGORY_OUTPUT_STATE_VARIABLES,
        ),
        "constraint_verification_types": constraint_verification_types,
        "task_type": _parse_task_type(question),
        "parse_schema_version": PIPEFORMER_TASK_SCHEMA_VERSION,
    }
    return task


def _first_match(
    text: str, patterns: Iterable[re.Pattern[str]]
) -> Optional[re.Match[str]]:
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
    if value in {"上调", "increase", "raise", "up"}:
        return "up"
    if value in {"下调", "decrease", "lower", "down"}:
        return "down"
    return "unknown"


def _parse_horizon_minutes(match: Optional[re.Match[str]]) -> Optional[int]:
    if not match:
        return None
    magnitude = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"小时", "hour", "hours", "hr", "hrs"}:
        return int(magnitude * 60)
    return int(magnitude)


def _parse_status_setpoint(
    text: str,
    disturbance_variable: Optional[str] = None,
) -> Optional[float]:
    if disturbance_variable:
        assignment = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(disturbance_variable)}"
            r"(?![A-Za-z0-9_])\s*=\s*([01])(?:\.0+)?(?![\d.])",
            text,
            re.I,
        )
        if assignment:
            return float(assignment.group(1))
    for pattern, value in STATUS_TARGET_PATTERNS:
        if pattern.search(text):
            return value
    matched_values = {
        value for pattern, value in STATUS_VALUE_PATTERNS if pattern.search(text)
    }
    return next(iter(matched_values)) if len(matched_values) == 1 else None


def targets_for_checks(checks: List[str], mapping: Dict[str, List[str]]) -> List[str]:
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
