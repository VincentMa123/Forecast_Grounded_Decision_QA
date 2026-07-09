from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import ConstraintSpec
from .common import run_specs


ABNORMALITY_WARNING_SPECS: Tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        name="abnormal_pressure_drop_proxy",
        category="abnormality_warning",
        description="Large pressure-variable deviations are reviewed as possible abnormal pressure-drop signals.",
        priority=12,
        metric="mean_abs_delta_vs_observed",
        prefixes=("N_", "P_"),
        suffixes=("_v000",),
        warning_threshold=0.5,
        fail_threshold=1.0,
    ),
    ConstraintSpec(
        name="sudden_flow_change_proxy",
        category="abnormality_warning",
        description="Large flow-variable deviations are reviewed as sudden-flow-change anomaly signals.",
        priority=22,
        metric="mean_abs_delta_vs_observed",
        prefixes=("B_", "P_"),
        suffixes=("_v001",),
        warning_threshold=0.5,
        fail_threshold=1.0,
    ),
)


def run_abnormality_warning_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_specs(ABNORMALITY_WARNING_SPECS, summaries, parsed_task)