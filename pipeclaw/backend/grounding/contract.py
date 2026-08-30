"""Stable function imports for grounding contract operations.

Implementation lives in the focused construction, validation, and rendering
modules; this facade keeps existing function-level imports stable.
"""

from .construction import (
    build_grounding_contract,
    forecast_views,
    is_chinese,
    latest_decision_policy,
    provisional_assumptions,
    record_grounding_contract,
    successful_pipeformer_results,
)
from .rendering import applied_disturbance_disclosure, grounded_fallback_answer
from .validation import (
    answer_without_machine_disclosure,
    canonical_applied_disturbance_lines,
    comparison_answer_issues,
    comparison_requirements_active,
    finalize_applied_disturbance_disclosure,
    format_number,
    normalize_not_evaluated_wording,
    provisional_assumption_disclosed,
)

__all__ = [
    "answer_without_machine_disclosure",
    "applied_disturbance_disclosure",
    "build_grounding_contract",
    "canonical_applied_disturbance_lines",
    "comparison_answer_issues",
    "comparison_requirements_active",
    "finalize_applied_disturbance_disclosure",
    "forecast_views",
    "format_number",
    "grounded_fallback_answer",
    "is_chinese",
    "latest_decision_policy",
    "normalize_not_evaluated_wording",
    "provisional_assumption_disclosed",
    "provisional_assumptions",
    "record_grounding_contract",
    "successful_pipeformer_results",
]
