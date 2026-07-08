from __future__ import annotations

import re
from typing import Any, Dict, Optional


def parse_condition(question: str) -> Dict[str, Any]:
    case_match = re.search(r"mock_test\s*的?第\s*0*(\d+)\s*个?算例", question)
    variable_match = re.search(r"\b[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?\b", question)
    percent_match = re.search(r"(上调|下调|increase|decrease|raise|lower|up|down)\s*(\d+(?:\.\d+)?)\s*%", question, re.I)
    horizon_match = re.search(r"未来\s*(\d+(?:\.\d+)?)\s*(小时|分钟)", question)

    if not variable_match:
        raise ValueError("Could not parse changed variable from scenario question.")

    direction_raw = percent_match.group(1).lower() if percent_match else ""
    if direction_raw in {"上调", "increase", "raise", "up"}:
        direction = "up"
    elif direction_raw in {"下调", "decrease", "lower", "down"}:
        direction = "down"
    else:
        direction = "unknown"

    horizon_minutes: Optional[int] = None
    if horizon_match:
        magnitude = float(horizon_match.group(1))
        unit = horizon_match.group(2)
        horizon_minutes = int(magnitude * 60) if unit == "小时" else int(magnitude)

    requested_checks = []
    for key, aliases in {
        "pressure": ["压力", "pressure"],
        "flow": ["流量", "flow"],
        "linepack": ["管存", "linepack"],
        "compressor_load": ["压缩机负荷", "compressor"],
        "energy": ["能耗", "energy"],
    }.items():
        if any(alias.lower() in question.lower() for alias in aliases):
            requested_checks.append(key)

    return {
        "case_id": f"mock_test_{int(case_match.group(1)):03d}" if case_match else None,
        "changed_variable": variable_match.group(0),
        "change_direction": direction,
        "change_percent": float(percent_match.group(2)) if percent_match else None,
        "forecast_horizon_minutes": horizon_minutes,
        "keep_other_boundary_controls": "保持其它边界控制量不变" in question,
        "task_type": "prediction_and_verification" if "预测" in question and "校核" in question else "unknown",
        "requested_checks": requested_checks,
    }