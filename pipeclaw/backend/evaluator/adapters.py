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


def build_evaluation_context(
    profile: EvaluationProfile,
    record: Mapping[str, Any],
    *,
    reference: Mapping[str, Any] | None = None,
    hard_issues: Iterable[str] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> EvaluationContext:
    """Normalize one teacher trace or rollout for the canonical checks."""

    profile = EvaluationProfile(profile)
    if profile is EvaluationProfile.TEACHER_TRACE:
        return EvaluationContext(
            profile=profile,
            record=_copy_mapping(record, "Teacher record"),
            hard_issues=_issues(hard_issues),
            diagnostics=_copy_mapping(diagnostics or {}, "Diagnostics"),
        )
    if reference is None:
        raise EvaluationInputError(
            "Autonomous rollout evaluation requires a teacher reference."
        )
    copied_reference = _copy_mapping(reference, "Teacher reference")
    return EvaluationContext(
        profile=profile,
        record=_copy_mapping(record, "Autonomous rollout"),
        reference=copied_reference,
        oracle=build_teacher_oracle(copied_reference),
        hard_issues=_issues(hard_issues),
        diagnostics=_copy_mapping(diagnostics or {}, "Diagnostics"),
    )


__all__ = ["build_evaluation_context"]
