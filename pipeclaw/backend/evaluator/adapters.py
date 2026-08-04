"""Input adapters for teacher traces and autonomous rollouts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from .models import EvaluationContext, EvaluationInputError, EvaluationProfile
from .oracle import build_teacher_oracle


def _copy_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationInputError(f"{label} must be a mapping.")
    return deepcopy(dict(value))


def _issues(values: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in (values or ())))


class TeacherTraceAdapter:
    """Normalize a native teacher trace into an evaluation context."""

    def adapt(
        self,
        record: Mapping[str, Any],
        *,
        hard_issues: Iterable[str] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> EvaluationContext:
        return EvaluationContext(
            profile=EvaluationProfile.TEACHER_TRACE,
            record=_copy_mapping(record, "Teacher record"),
            hard_issues=_issues(hard_issues),
            diagnostics=_copy_mapping(diagnostics or {}, "Diagnostics"),
        )


class AutonomousRolloutAdapter:
    """Normalize a rollout and its required held-out teacher reference."""

    def adapt(
        self,
        rollout: Mapping[str, Any],
        *,
        reference: Mapping[str, Any] | None = None,
        hard_issues: Iterable[str] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> EvaluationContext:
        if reference is None:
            raise EvaluationInputError(
                "Autonomous rollout evaluation requires a teacher reference."
            )
        copied_reference = _copy_mapping(reference, "Teacher reference")
        return EvaluationContext(
            profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
            record=_copy_mapping(rollout, "Autonomous rollout"),
            reference=copied_reference,
            oracle=build_teacher_oracle(copied_reference),
            hard_issues=_issues(hard_issues),
            diagnostics=_copy_mapping(diagnostics or {}, "Diagnostics"),
        )
