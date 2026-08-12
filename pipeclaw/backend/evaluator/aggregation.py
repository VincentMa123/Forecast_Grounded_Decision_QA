from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import EVALUATION_SCHEMA_VERSION, EvaluationReport


def _payload(report: EvaluationReport | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(report, EvaluationReport):
        return report.to_dict()
    if isinstance(report, Mapping):
        return report
    raise TypeError("Evaluation summaries require reports or report mappings.")


def _metric_summary(
    reports: Sequence[Mapping[str, Any]],
    name: str,
) -> dict[str, Any]:
    numerator = 0
    denominator = 0
    for report in reports:
        metrics = report.get("metrics")
        metric = metrics.get(name) if isinstance(metrics, Mapping) else None
        if not isinstance(metric, Mapping) or not metric.get("applicable", False):
            continue
        denominator += 1
        if metric.get("passed", False):
            numerator += 1
    return {
        "numerator": numerator,
        "denominator": denominator,
        "pass_rate": numerator / denominator if denominator else None,
        "failure_rate": (denominator - numerator) / denominator if denominator else None,
        "status": "ok" if denominator else "not_applicable",
    }


def summarize(
    reports: Sequence[EvaluationReport | Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate scores and denominator-aware metric outcomes."""

    payloads = [_payload(report) for report in reports]
    profiles = {str(report.get("profile")) for report in payloads}
    if profiles == {"autonomous_rollout"}:
        mode = "autonomous"
    elif profiles == {"teacher_trace"}:
        mode = "teacher"
    else:
        mode = "mixed" if profiles else None
    scores = [
        float(report["overall_score"])
        for report in payloads
        if report.get("overall_score") is not None
    ]
    metric_names = sorted(
        {
            str(name)
            for report in payloads
            for metrics in [report.get("metrics")]
            if isinstance(metrics, Mapping)
            for name in metrics
        }
    )
    diagnostic_names: set[str] = set()
    for name in metric_names:
        for report in payloads:
            metrics = report.get("metrics")
            metric = metrics.get(name) if isinstance(metrics, Mapping) else None
            if isinstance(metric, Mapping) and not metric.get("included_in_score", True):
                diagnostic_names.add(name)
                break
    metrics = {
        name: _metric_summary(payloads, name)
        for name in metric_names
        if name not in diagnostic_names
    }
    diagnostics = {
        name: _metric_summary(payloads, name)
        for name in metric_names
        if name in diagnostic_names
    }
    pass_count = sum(bool(report.get("passed")) for report in payloads)
    record_count = len(payloads)
    evidence_summary = metrics.get("evidence_consistency", {})
    portability = {
        key: sum(
            int(
                (
                    report.get("diagnostics", {}).get("portability", {})
                    if isinstance(report.get("diagnostics"), Mapping)
                    else {}
                ).get(key, 0)
            )
            for report in payloads
        )
        for key in (
            "cwd_rebased_calls",
            "records_with_cwd_rebased",
            "rebased_execution_successes",
            "rebased_execution_failures",
            "portable_path_normalization_calls",
        )
    }
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "mode": mode,
        "record_count": record_count,
        "overall": {
            "mean_score": round(sum(scores) / len(scores), 6) if scores else None,
            "minimum_score": min(scores) if scores else None,
            "maximum_score": max(scores) if scores else None,
            "pass_count": pass_count,
            "pass_rate": pass_count / record_count if record_count else None,
        },
        "metrics": metrics,
        "diagnostics": diagnostics,
        "hallucination_rate": evidence_summary.get("failure_rate"),
        "portability": portability,
        "by_scenario_type": {},
    }
