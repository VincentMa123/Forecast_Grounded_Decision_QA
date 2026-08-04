"""Canonical check dispatch for both evaluation profiles."""

from __future__ import annotations

from typing import Any

from ..models import EvaluationContext, EvaluationProfile, MetricResult
from .assumptions import assumption_consistency, inferred_task_fields
from .autonomous import evaluate_autonomous_checks
from .common import CANONICAL_METRIC_NAMES
from .teacher import evaluate_teacher_checks


def evaluate_context(
    context: EvaluationContext,
    *,
    derive_hard_issues: bool = False,
    maximum_chars: int = 24_000,
) -> tuple[list[MetricResult], tuple[str, ...], dict[str, Any]]:
    if context.profile is EvaluationProfile.TEACHER_TRACE:
        return evaluate_teacher_checks(
            context,
            derive_hard_issues=derive_hard_issues,
            maximum_chars=maximum_chars,
        )
    metrics, diagnostics = evaluate_autonomous_checks(context)
    return metrics, tuple(context.hard_issues), diagnostics


__all__ = [
    "CANONICAL_METRIC_NAMES",
    "assumption_consistency",
    "evaluate_context",
    "inferred_task_fields",
]
