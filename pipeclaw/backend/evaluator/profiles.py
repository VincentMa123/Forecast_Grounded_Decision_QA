from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .models import EvaluationProfile


DEFAULT_TEACHER_MINIMUM_SCORE = 85.0


@dataclass(frozen=True)
class MetricPolicy:
    weight: float
    critical: bool
    included_in_score: bool


@dataclass(frozen=True)
class ProfilePolicy:
    minimum_score: float | None
    metrics: Mapping[str, MetricPolicy]
    critical_metrics: frozenset[str] = frozenset()
    diagnostic_metrics: frozenset[str] = frozenset()
    equal_weight_deliverables: bool = False

    def metric(self, name: str) -> MetricPolicy:
        if name in self.diagnostic_metrics:
            return MetricPolicy(0.0, False, False)
        if (
            name == "trace_completed"
            and "answer_completeness" in self.metrics
            and "task_parsing" in self.metrics
        ):
            # Read-only compatibility lookup for the retired generic name.
            return self.metrics["task_parsing"]
        configured = self.metrics.get(name)
        if configured is not None:
            return configured
        if self.equal_weight_deliverables:
            return MetricPolicy(1.0, name in self.critical_metrics, True)
        return MetricPolicy(0.0, False, False)


_TEACHER_PIPEFORMER_WEIGHTS = {
    "task_parsing": 10.0,
    "tool_call": 15.0,
    "checkpoint_inference": 10.0,
    "disturbance_application": 15.0,
    "forecast_horizon": 10.0,
    "constraint_execution": 10.0,
    "verification_completeness": 10.0,
    "registry_ordering": 5.0,
    "evidence_consistency": 10.0,
    "record_contract": 5.0,
}
_TEACHER_PIPEFORMER_CRITICAL = frozenset(
    name for name in _TEACHER_PIPEFORMER_WEIGHTS if name != "record_contract"
)
_TEACHER_GENERIC_WEIGHTS = {
    "task_parsing": 25.0,
    "answer_completeness": 25.0,
    "tool_call": 20.0,
    "evidence_consistency": 20.0,
    "record_contract": 10.0,
}
_TEACHER_GENERIC_CRITICAL = frozenset(
    {"task_parsing", "answer_completeness", "tool_call", "evidence_consistency"}
)

AUTONOMOUS_CRITICAL_METRICS = frozenset(
    {
        "task_parsing",
        "tool_call",
        "assumption_consistency",
        "checkpoint_inference",
        "disturbance_application",
        "forecast_horizon",
        "constraint_execution",
        "constraint_judgment",
        "verification_completeness",
        "registry_ordering",
        "risk",
        "manual_intervention",
        "dispatch",
        "evidence_consistency",
        "answer_completeness",
        "json_validity",
        "artifact_evidence",
        "question_anchor",
        "claim_alignment",
        "answer_claim_support",
    }
)
AUTONOMOUS_DIAGNOSTIC_METRICS = frozenset(
    {
        "tool_recovery",
        "portability",
        "raw_capture_metadata",
        "model_loading_metadata",
        "hallucination",
    }
)


def _metric_map(
    weights: Mapping[str, float],
    critical: frozenset[str],
) -> Mapping[str, MetricPolicy]:
    return MappingProxyType(
        {
            name: MetricPolicy(weight, name in critical, True)
            for name, weight in weights.items()
        }
    )


_TEACHER_PIPEFORMER_POLICY = ProfilePolicy(
    minimum_score=DEFAULT_TEACHER_MINIMUM_SCORE,
    metrics=_metric_map(_TEACHER_PIPEFORMER_WEIGHTS, _TEACHER_PIPEFORMER_CRITICAL),
    critical_metrics=_TEACHER_PIPEFORMER_CRITICAL,
)
_TEACHER_GENERIC_POLICY = ProfilePolicy(
    minimum_score=DEFAULT_TEACHER_MINIMUM_SCORE,
    metrics=_metric_map(_TEACHER_GENERIC_WEIGHTS, _TEACHER_GENERIC_CRITICAL),
    critical_metrics=_TEACHER_GENERIC_CRITICAL,
)
_AUTONOMOUS_POLICY = ProfilePolicy(
    minimum_score=None,
    metrics=MappingProxyType({}),
    critical_metrics=AUTONOMOUS_CRITICAL_METRICS,
    diagnostic_metrics=AUTONOMOUS_DIAGNOSTIC_METRICS,
    equal_weight_deliverables=True,
)


def get_profile_policy(
    profile: EvaluationProfile,
    *,
    teacher_variant: str = "pipeformer",
) -> ProfilePolicy:
    profile = EvaluationProfile(profile)
    if profile is EvaluationProfile.AUTONOMOUS_ROLLOUT:
        return _AUTONOMOUS_POLICY
    if teacher_variant == "pipeformer":
        return _TEACHER_PIPEFORMER_POLICY
    if teacher_variant == "generic":
        return _TEACHER_GENERIC_POLICY
    raise ValueError(f"Unknown teacher profile variant: {teacher_variant}")
