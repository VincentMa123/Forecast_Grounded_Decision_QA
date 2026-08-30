from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .adapters import build_evaluation_context
from .checks import evaluate_context
from .models import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationInputError,
    EvaluationProfile,
    EvaluationReport,
    MetricResult,
)
from .profiles import DEFAULT_MAX_RECORD_CHARS, get_profile_policy


def build_report(
    profile: EvaluationProfile,
    metrics: Iterable[MetricResult],
    *,
    hard_issues: Iterable[str] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    minimum_score: float | None = None,
) -> EvaluationReport:
    """Build one schema-v3 report using the canonical score formula."""

    profile = EvaluationProfile(profile)
    metric_values = list(metrics)
    if any(not isinstance(metric, MetricResult) for metric in metric_values):
        raise EvaluationInputError("Every metric must be a MetricResult.")
    issue_values = tuple(dict.fromkeys(str(issue) for issue in (hard_issues or ())))

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
        schema_version=EVALUATION_SCHEMA_VERSION,
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

    profile = EvaluationProfile(profile)
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
            profile is EvaluationProfile.TEACHER_TRACE and hard_issues is None
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
