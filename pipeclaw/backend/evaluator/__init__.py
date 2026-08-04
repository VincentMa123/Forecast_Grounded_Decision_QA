"""Canonical evaluation API for teacher traces and autonomous rollouts."""

from .adapters import AutonomousRolloutAdapter, TeacherTraceAdapter
from .aggregation import summarize
from .checks import assumption_consistency, inferred_task_fields
from .engine import evaluate
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
    "AutonomousRolloutAdapter",
    "EVALUATION_SCHEMA_VERSION",
    "EvaluationContext",
    "EvaluationInputError",
    "EvaluationProfile",
    "EvaluationReport",
    "MetricResult",
    "TeacherTraceAdapter",
    "build_teacher_oracle",
    "assumption_consistency",
    "evaluate",
    "inferred_task_fields",
    "summarize",
]
