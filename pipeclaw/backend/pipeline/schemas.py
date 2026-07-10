from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ForecastRow:
    label: str
    values: Dict[str, float]


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    category: str
    description: str
    priority: int
    metric: str
    prefixes: Tuple[str, ...] = ()
    suffixes: Tuple[str, ...] = ()
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    fail_low: Optional[float] = None
    fail_high: Optional[float] = None
    warning_threshold: Optional[float] = None
    fail_threshold: Optional[float] = None
    pass_flag: Optional[str] = None
    warning_flag: Optional[str] = None
    fail_flag: Optional[str] = None
