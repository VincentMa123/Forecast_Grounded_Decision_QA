"""One evaluation pass: normalize the input, run the profile's checks, build the report."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from .checks.autonomous import evaluate_autonomous_checks
from .checks.teacher import evaluate_teacher_checks
from .models import (
    EvaluationContext,
    EvaluationInputError,
    EvaluationProfile,
    EvaluationReport,
    MetricResult,
)
from .oracle import build_teacher_oracle
from .profiles import DEFAULT_MAX_RECORD_CHARS, get_profile_policy


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Profile check dispatch
# ---------------------------------------------------------------------------


def evaluate_context(
    context: EvaluationContext,
    *,
    derive_hard_issues: bool = False,
    maximum_chars: int = DEFAULT_MAX_RECORD_CHARS,
) -> tuple[list[MetricResult], tuple[str, ...], dict[str, Any]]:
    if context.profile is EvaluationProfile.TEACHER_TRACE:
        return evaluate_teacher_checks(
            context,
            derive_hard_issues=derive_hard_issues,
            maximum_chars=maximum_chars,
        )
    metrics, diagnostics = evaluate_autonomous_checks(context)
    return metrics, tuple(context.hard_issues), diagnostics


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------


def build_report(
    profile: EvaluationProfile,
    metrics: Iterable[MetricResult],
    *,
    hard_issues: Iterable[str] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    minimum_score: float | None = None,
) -> EvaluationReport:
    """Build one evaluation report using the canonical score formula."""

    profile = EvaluationProfile(profile)
    metric_values = list(metrics)
    if any(not isinstance(metric, MetricResult) for metric in metric_values):
        raise EvaluationInputError("Every metric must be a MetricResult.")
    issue_values = _issues(hard_issues)

    included = [
        metric
        for metric in metric_values
        if metric.applicable and metric.included_in_score
    ]
    denominator = sum(metric.weight for metric in included)
    overall_score = (
        round(
            100.0
            * sum(metric.weight for metric in included if metric.passed)
            / denominator,
            6,
        )
        if denominator
        else None
    )
    hard_gate_passed = not issue_values and all(
        metric.passed for metric in included if metric.critical
    )

    if profile is EvaluationProfile.TEACHER_TRACE:
        threshold = (
            get_profile_policy(profile).minimum_score
            if minimum_score is None
            else float(minimum_score)
        )
        passed = bool(
            hard_gate_passed
            and overall_score is not None
            and threshold is not None
            and overall_score >= threshold
        )
    else:
        passed = bool(hard_gate_passed and overall_score is not None)

    failed_checks = tuple(metric.name for metric in included if not metric.passed)
    critical_failures = tuple(
        metric.name for metric in included if metric.critical and not metric.passed
    )
    diagnostic_values = dict(diagnostics or {})
    if issue_values or "hard_issues" not in diagnostic_values:
        diagnostic_values["hard_issues"] = issue_values

    return EvaluationReport(
        profile=profile,
        overall_score=overall_score,
        hard_gate_passed=hard_gate_passed,
        passed=passed,
        metrics={metric.name: metric for metric in metric_values},
        diagnostics=diagnostic_values,
        failed_checks=failed_checks,
        critical_failures=critical_failures,
    )


def evaluate(
    record: Mapping[str, Any],
    *,
    profile: EvaluationProfile,
    reference: Mapping[str, Any] | None = None,
    hard_issues: Iterable[str] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    minimum_score: float | None = None,
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS,
) -> EvaluationReport:
    """Normalize one input, run canonical checks, and build one report."""

    context = build_evaluation_context(
        profile,
        record,
        reference=reference,
        hard_issues=hard_issues,
        diagnostics=diagnostics,
    )
    evaluated_diagnostics = dict(context.diagnostics)
    (
        evaluated_metrics,
        evaluated_issues,
        check_diagnostics,
    ) = evaluate_context(
        context,
        derive_hard_issues=(
            context.profile is EvaluationProfile.TEACHER_TRACE and hard_issues is None
        ),
        maximum_chars=max_record_chars,
    )
    evaluated_diagnostics.update(check_diagnostics)
    return build_report(
        context.profile,
        evaluated_metrics,
        hard_issues=evaluated_issues,
        diagnostics=evaluated_diagnostics,
        minimum_score=minimum_score,
    )
