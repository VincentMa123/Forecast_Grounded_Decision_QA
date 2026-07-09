from __future__ import annotations

from typing import Any, Dict, Optional

from .constraints.common import variables_matching
from .engineering_constraints import run_engineering_constraint_checks


def run_constraint_checks(
    summaries: Dict[str, Dict[str, Any]],
    parsed_task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return run_engineering_constraint_checks(summaries, parsed_task=parsed_task)


__all__ = ["run_constraint_checks", "variables_matching"]