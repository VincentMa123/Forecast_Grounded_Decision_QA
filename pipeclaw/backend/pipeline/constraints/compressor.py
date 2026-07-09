from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import ConstraintSpec
from .common import run_specs


COMPRESSOR_SPECS: Tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        name="compressor_operating_envelope",
        category="compressor",
        description="Compressor proxy variables should remain inside the normalized operating envelope.",
        priority=40,
        metric="max_abs_prediction",
        prefixes=("C_",),
        suffixes=("_v000", "_v001"),
        warning_threshold=1.2,
        fail_threshold=2.0,
    ),
    ConstraintSpec(
        name="compressor_power_change_proxy",
        category="compressor",
        description="Compressor and energy proxy variables should not jump sharply from the observed baseline.",
        priority=41,
        metric="mean_abs_delta_vs_observed",
        prefixes=("C_", "TE_"),
        suffixes=("_v000", "_v001"),
        warning_threshold=0.5,
        fail_threshold=1.2,
    ),
)


def run_compressor_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_specs(COMPRESSOR_SPECS, summaries, parsed_task)