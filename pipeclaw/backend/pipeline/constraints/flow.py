from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import ConstraintSpec
from .common import run_specs


FLOW_SPECS: Tuple[ConstraintSpec, ...] = (
    ConstraintSpec(
        name="segment_flow_change",
        category="flow",
        description="Segment-flow variables should not deviate sharply from the observed baseline.",
        priority=20,
        metric="mean_abs_delta_vs_observed",
        prefixes=("B_", "P_"),
        suffixes=("_v001",),
        warning_threshold=0.35,
        fail_threshold=0.8,
    ),
    ConstraintSpec(
        name="maximum_transmission_capacity_proxy",
        category="flow",
        description="Flow-like variables are checked against a normalized transmission-capacity proxy.",
        priority=21,
        metric="max_abs_prediction",
        prefixes=("B_", "P_"),
        suffixes=("_v001",),
        warning_threshold=2.2,
        fail_threshold=3.0,
    ),
)


def run_flow_checks(summaries: Dict[str, Dict[str, Any]], parsed_task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return run_specs(FLOW_SPECS, summaries, parsed_task)