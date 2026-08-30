from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .models import EvaluationProfile


DEFAULT_TEACHER_MINIMUM_SCORE = 85.0
DEFAULT_MAX_RECORD_CHARS = 24_000


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
    metric_order: tuple[str, ...] = ()
    diagnostic_order: tuple[str, ...] = ()
    legacy_aliases: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self,
            "metric_order",
            tuple(self.metric_order or self.metrics),
        )
        object.__setattr__(
            self,
            "diagnostic_order",
            tuple(
                self.diagnostic_order
                or (name for name in self.metric_order if name in self.diagnostic_metrics)
            ),
        )
        object.__setattr__(
            self,
            "legacy_aliases",
            MappingProxyType(dict(self.legacy_aliases)),
        )

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

    def legacy_name(self, name: str) -> str:
        """Return the released alias for a canonical metric name, if any."""

        return self.legacy_aliases.get(name, name)


TEACHER_METRIC_ORDER = (
    "task_parsing",
    "assumption_consistency",
    "tool_call",
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
    "record_contract",
)
AUTONOMOUS_METRIC_ORDER = TEACHER_METRIC_ORDER + (
    "tool_recovery",
    "answer_claim_support",
    "claim_alignment",
    "question_anchor",
    "hallucination",
)
AUTONOMOUS_DIAGNOSTIC_ORDER = (
    "tool_recovery",
    "portability",
    "raw_capture_metadata",
    "model_loading_metadata",
    "hallucination",
)

_PIPEFORMER_LEGACY_ALIASES = {
    "task_parsing": "parsed_task_correct",
    "tool_call": "forecast_tool_succeeded",
    "checkpoint_inference": "checkpoint_inference_used",
    "disturbance_application": "disturbance_applied_correctly",
    "forecast_horizon": "forecast_horizon_consistent",
    "constraint_execution": "requested_constraints_executed",
    "verification_completeness": "verification_complete",
    "registry_ordering": "registry_search_precedes_forecast",
    "evidence_consistency": "answer_grounded",
    "record_contract": "compact_record_contract",
}
_GENERIC_LEGACY_ALIASES = {
    "task_parsing": "trace_completed",
    "answer_completeness": "answer_present",
    "tool_call": "tool_trajectory_valid",
    "evidence_consistency": "answer_grounded",
    "record_contract": "compact_record_contract",
}


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
    AUTONOMOUS_DIAGNOSTIC_ORDER
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
    metric_order=TEACHER_METRIC_ORDER,
    legacy_aliases=_PIPEFORMER_LEGACY_ALIASES,
)
_TEACHER_GENERIC_POLICY = ProfilePolicy(
    minimum_score=DEFAULT_TEACHER_MINIMUM_SCORE,
    metrics=_metric_map(_TEACHER_GENERIC_WEIGHTS, _TEACHER_GENERIC_CRITICAL),
    critical_metrics=_TEACHER_GENERIC_CRITICAL,
    metric_order=TEACHER_METRIC_ORDER,
    legacy_aliases=_GENERIC_LEGACY_ALIASES,
)
_AUTONOMOUS_POLICY = ProfilePolicy(
    minimum_score=None,
    metrics=MappingProxyType({}),
    critical_metrics=AUTONOMOUS_CRITICAL_METRICS,
    diagnostic_metrics=AUTONOMOUS_DIAGNOSTIC_METRICS,
    equal_weight_deliverables=True,
    metric_order=AUTONOMOUS_METRIC_ORDER,
    diagnostic_order=AUTONOMOUS_DIAGNOSTIC_ORDER,
)

EVALUATOR_METRIC_REGISTRY: Mapping[
    EvaluationProfile, Mapping[str, ProfilePolicy]
] = MappingProxyType(
    {
        EvaluationProfile.TEACHER_TRACE: MappingProxyType(
            {
                "pipeformer": _TEACHER_PIPEFORMER_POLICY,
                "generic": _TEACHER_GENERIC_POLICY,
            }
        ),
        EvaluationProfile.AUTONOMOUS_ROLLOUT: MappingProxyType(
            {"default": _AUTONOMOUS_POLICY}
        ),
    }
)


def get_profile_policy(
    profile: EvaluationProfile,
    *,
    teacher_variant: str = "pipeformer",
) -> ProfilePolicy:
    profile = EvaluationProfile(profile)
    variants = EVALUATOR_METRIC_REGISTRY[profile]
    if profile is EvaluationProfile.AUTONOMOUS_ROLLOUT:
        return variants["default"]
    try:
        return variants[teacher_variant]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown teacher profile variant: {teacher_variant}") from exc


def legacy_boolean_aliases(
    profile: EvaluationProfile,
    *,
    teacher_variant: str = "pipeformer",
) -> Mapping[str, str]:
    """Return released boolean aliases derived from the canonical registry."""

    aliases = get_profile_policy(
        profile,
        teacher_variant=teacher_variant,
    ).legacy_aliases
    return MappingProxyType(
        {
            alias: canonical
            for canonical, alias in aliases.items()
            if alias != canonical
        }
    )
