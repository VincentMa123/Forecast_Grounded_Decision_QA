from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ForecastRow:
    label: str
    values: Dict[str, float]