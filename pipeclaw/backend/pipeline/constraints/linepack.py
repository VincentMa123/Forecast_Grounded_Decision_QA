from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import ConstraintSpec
from .common import run_specs


LINEPACK_SPECS: Tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        name="linepack_decline_and_recovery_proxy",
        category="linepack",
        description="Linepack-like storage variables should not show a large short-term decline from baseline.",
        priority=30,
        metric="mean_abs_delta_vs_observed",
        prefixes=("R_", "N_"),
        suffixes=("_v000",),
        warning_threshold=0.45,
        fail_threshold=1.0,
    ),
    ConstraintSpec(
        name="linepack_warning_threshold",
        category="linepack",
        description="Linepack-like variables are checked against the normalized short-term warning threshold.",
        priority=31,
        metric="max_abs_prediction",
        prefixes=("R_", "N_"),
        suffixes=("_v000",),
        warning_threshold=2.0,
        fail_threshold=3.0,
    ),
)


def run_linepack_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_specs(LINEPACK_SPECS, summaries, parsed_task)