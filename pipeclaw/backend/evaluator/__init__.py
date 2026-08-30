"""Canonical evaluation API for teacher traces and autonomous rollouts."""

from .adapters import build_evaluation_context
from .aggregation import summarize
from .checks import assumption_consistency, inferred_task_fields
from .engine import build_report, evaluate
from .models import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationContext,
    EvaluationInputError,
    EvaluationProfile,
    EvaluationReport,
    MetricResult,
)
from .oracle import build_teacher_oracle

__all__ = [
    "build_evaluation_context",
    "build_report",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationContext",
    "EvaluationInputError",
    "EvaluationProfile",
    "EvaluationReport",
    "MetricResult",
    "build_teacher_oracle",
    "assumption_consistency",
    "evaluate",
    "inferred_task_fields",
    "summarize",
]
