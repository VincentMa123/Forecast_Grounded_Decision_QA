"""Public contracts for canonical PipeClaw evaluation results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any


EVALUATION_SCHEMA_VERSION = "pipeclaw_evaluation_v2"


class EvaluationInputError(ValueError):
    """Raised when an evaluation request cannot be normalized safely."""


class EvaluationProfile(str, Enum):
    TEACHER_TRACE = "teacher_trace"
    AUTONOMOUS_ROLLOUT = "autonomous_rollout"


def _json_compatible(value: Any) -> Any:
    """Return a recursively JSON-compatible representation."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_json_compatible(item) for item in value]
        return sorted(converted, key=lambda item: repr(item))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_compatible(getattr(value, item.name))
            for item in fields(value)
        }
    return str(value)


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class MetricResult:
    name: str
    applicable: bool
    passed: bool
    weight: float
    critical: bool
    included_in_score: bool
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _frozen_mapping(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "applicable": bool(self.applicable),
            "passed": bool(self.passed),
            "weight": _json_compatible(self.weight),
            "critical": bool(self.critical),
            "included_in_score": bool(self.included_in_score),
            "details": _json_compatible(self.details),
        }


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: str
    profile: EvaluationProfile
    overall_score: float | None
    hard_gate_passed: bool
    passed: bool
    metrics: Mapping[str, MetricResult]
    diagnostics: Mapping[str, Any]
    failed_checks: tuple[str, ...]
    critical_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _frozen_mapping(self.metrics))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))
        object.__setattr__(self, "failed_checks", tuple(self.failed_checks))
        object.__setattr__(self, "critical_failures", tuple(self.critical_failures))

    def to_dict(self) -> dict[str, Any]:
        metrics = {
            str(name): metric.to_dict()
            for name, metric in self.metrics.items()
        }
        evidence = self.metrics.get("evidence_consistency")
        if (
            self.profile is EvaluationProfile.AUTONOMOUS_ROLLOUT
            and evidence is not None
        ):
            hallucination = evidence.to_dict()
            hallucination["included_in_score"] = False
            hallucination["derived_from"] = "evidence_consistency"
            metrics["hallucination"] = hallucination
        return {
            "schema_version": str(self.schema_version),
            "profile": self.profile.value,
            "overall_score": _json_compatible(self.overall_score),
            "hard_gate_passed": bool(self.hard_gate_passed),
            "passed": bool(self.passed),
            "metrics": metrics,
            "diagnostics": _json_compatible(self.diagnostics),
            "failed_checks": list(self.failed_checks),
            "critical_failures": list(self.critical_failures),
        }


@dataclass(frozen=True)
class EvaluationContext:
    """Normalized inputs consumed by the shared checks added in Task 4."""

    profile: EvaluationProfile
    record: Mapping[str, Any]
    reference: Mapping[str, Any] | None = None
    oracle: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    hard_issues: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", _frozen_mapping(self.record))
        if self.reference is not None:
            object.__setattr__(self, "reference", _frozen_mapping(self.reference))
        object.__setattr__(self, "oracle", _frozen_mapping(self.oracle))
        object.__setattr__(self, "hard_issues", tuple(self.hard_issues))
        object.__setattr__(self, "diagnostics", _frozen_mapping(self.diagnostics))
