from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .models import EvaluationReport


def _payload(report: EvaluationReport | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(report, EvaluationReport):
        return report.to_dict()
    if isinstance(report, Mapping):
        return report
    raise TypeError("Evaluation summaries require reports or report mappings.")


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
    report_metrics = []
    for report in payloads:
        metrics_for_report = report.get("metrics")
        report_metrics.append(
            metrics_for_report if isinstance(metrics_for_report, Mapping) else {}
        )
    metric_names = sorted({str(name) for metrics in report_metrics for name in metrics})
    metrics: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        numerator = 0
        denominator = 0
        is_diagnostic = False
        for metrics_for_report in report_metrics:
            metric = metrics_for_report.get(name)
            if not isinstance(metric, Mapping):
                continue
            if not metric.get("included_in_score", True):
                is_diagnostic = True
            if not metric.get("applicable", False):
                continue
            denominator += 1
            numerator += bool(metric.get("passed", False))
        target = diagnostics if is_diagnostic else metrics
        target[name] = {
            "numerator": numerator,
            "denominator": denominator,
            "pass_rate": numerator / denominator if denominator else None,
            "failure_rate": (denominator - numerator) / denominator
            if denominator
            else None,
            "status": "ok" if denominator else "not_applicable",
        }
    pass_count = sum(bool(report.get("passed")) for report in payloads)
    record_count = len(payloads)
    hallucination_summary = diagnostics.get("hallucination", {})
    tool_successes = 0
    tool_calls = 0
    duplicate_successes = 0
    for metrics_for_report in report_metrics:
        tool_metric = metrics_for_report.get("tool_call")
        details = (
            tool_metric.get("details") if isinstance(tool_metric, Mapping) else None
        )
        if not isinstance(details, Mapping):
            continue
        tool_successes += int(details.get("successful_call_count") or 0)
        tool_calls += int(details.get("total_call_count") or 0)
        duplicate_successes += int(details.get("duplicate_successful_call_count") or 0)
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
        "hallucination_rate": hallucination_summary.get("failure_rate"),
        "tool_call_execution": {
            "successful_calls": tool_successes,
            "total_calls": tool_calls,
            "success_rate": tool_successes / tool_calls if tool_calls else None,
            "duplicate_successful_calls": duplicate_successes,
        },
        "portability": portability,
    }
