from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import ConstraintSpec
from .common import run_specs


PRESSURE_SPECS: Tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        name="node_pressure_operating_window",
        category="pressure",
        description="Pressure-like node variables must stay inside the configured operating window.",
        priority=10,
        metric="predicted_range",
        prefixes=("N_", "P_"),
        suffixes=("_v000",),
        warning_low=-2.5,
        warning_high=2.5,
        fail_low=-3.0,
        fail_high=3.0,
    ),
    ConstraintSpec(
        name="key_node_pressure_margin",
        category="pressure",
        description="Key pressure variables should keep normalized margin from the alarm boundary.",
        priority=11,
        metric="max_abs_prediction",
        prefixes=("N_", "P_"),
        suffixes=("_v000",),
        warning_threshold=2.0,
        fail_threshold=2.8,
    ),
)


def run_pressure_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_specs(PRESSURE_SPECS, summaries, parsed_task)