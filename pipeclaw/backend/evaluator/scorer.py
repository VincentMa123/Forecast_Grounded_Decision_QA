"""Legacy native-evaluator facade over the canonical schema-v2 engine."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .aggregation import summarize
from .engine import evaluate
from .models import EvaluationProfile, EvaluationReport


DEFAULT_MINIMUM_SCORE = 85.0
DEFAULT_MAX_RECORD_CHARS = 24_000
PIPEFORMER_TOOL = "run_pipeformer_forecast"

_PIPEFORMER_LEGACY_NAMES = {
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
_GENERIC_LEGACY_NAMES = {
    "task_parsing": "trace_completed",
    "answer_completeness": "answer_present",
    "tool_call": "tool_trajectory_valid",
    "evidence_consistency": "answer_grounded",
    "record_contract": "compact_record_contract",
}
_BOOLEAN_ALIASES = {
    "parsed_task_correct": "task_parsing",
    "forecast_tool_succeeded": "tool_call",
    "checkpoint_inference_used": "checkpoint_inference",
    "disturbance_applied_correctly": "disturbance_application",
    "forecast_horizon_consistent": "forecast_horizon",
    "requested_constraints_executed": "constraint_execution",
    "verification_complete": "verification_completeness",
    "registry_search_precedes_forecast": "registry_ordering",
    "answer_grounded": "evidence_consistency",
    "compact_record_contract": "record_contract",
}


@dataclass(frozen=True)
class NativeEvaluationConfig:
    minimum_score: float = DEFAULT_MINIMUM_SCORE
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS


def _record_with_trace_status(
    record: Dict[str, Any],
    trace_status: Optional[str],
) -> Dict[str, Any]:
    copied = dict(record)
    effective_status = trace_status
    if effective_status is None:
        effective_status = copied.get("trace_status")
    if effective_status is None and {
        "trace_completed",
        "forecast_tool_succeeded",
    } & set(copied.get("quality_failed_checks") or []):
        effective_status = "unknown"
    if effective_status is not None:
        copied["trace_status"] = effective_status
    return copied


def _legacy_name(name: str, variant: str) -> str:
    aliases = (
        _PIPEFORMER_LEGACY_NAMES
        if variant == "pipeformer"
        else _GENERIC_LEGACY_NAMES
    )
    return aliases.get(name, name)


def _serialize_legacy(report: EvaluationReport, minimum_score: float) -> Dict[str, Any]:
    payload = report.to_dict()
    variant = str(report.diagnostics.get("teacher_variant") or "pipeformer")
    canonical_failed = list(report.failed_checks)
    canonical_critical = list(report.critical_failures)
    failed = [_legacy_name(name, variant) for name in canonical_failed]
    failed_critical = [
        _legacy_name(name, variant) for name in canonical_critical
    ]
    checks = []
    for metric_result in report.metrics.values():
        if not metric_result.applicable:
            continue
        check = {
            "name": _legacy_name(metric_result.name, variant),
            "weight": metric_result.weight,
            "status": "pass" if metric_result.passed else "fail",
        }
        check.update(dict(metric_result.details))
        checks.append(check)

    payload.update(
        {
            "canonical_failed_checks": canonical_failed,
            "canonical_critical_failures": canonical_critical,
            "quality_score": report.overall_score,
            "quality_flag": "pass" if report.passed else "needs_review",
            "minimum_pass_score": float(minimum_score),
            "failed_checks": failed,
            "quality_failed_checks": failed,
            "failed_critical_checks": failed_critical,
            "quality_issues": list(report.diagnostics.get("hard_issues") or []),
            "checks": checks,
        }
    )
    for alias, canonical in _BOOLEAN_ALIASES.items():
        metric_result = report.metrics.get(canonical)
        payload[alias] = bool(
            metric_result
            and metric_result.applicable
            and metric_result.passed
        )
    return payload


TEACHER_QUALITY_ALIASES = {
    "quality_flag": "quality_flag",
    "quality_score": "quality_score",
    "quality_profile": "profile",
    "quality_failed_checks": "failed_checks",
    "quality_issues": "quality_issues",
}


def apply_quality_aliases(
    record: Dict[str, Any],
    native: Dict[str, Any],
    *,
    aliases: Iterable[str] = TEACHER_QUALITY_ALIASES,
) -> Dict[str, Any]:
    """Write the released ``quality_*`` teacher fields from one v2 report.

    Teacher generation and offline evaluation both persist these aliases.  They
    are copied from the canonical report rather than recomputed, so there is
    exactly one score formula in the repository.
    """

    for alias in aliases:
        source = TEACHER_QUALITY_ALIASES[alias]
        if source in native:
            record[alias] = native[source]
    return record


class NativeTraceEvaluator:
    """Compatibility API that delegates all evaluation to schema v2."""

    def __init__(self, config: Optional[NativeEvaluationConfig] = None) -> None:
        self.config = config or NativeEvaluationConfig()

    def evaluate(
        self,
        record: Dict[str, Any],
        *,
        hard_issues: Optional[Iterable[str]] = None,
        trace_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        return evaluate_native_record(
            record,
            hard_issues=hard_issues,
            trace_status=trace_status,
            minimum_score=self.config.minimum_score,
            max_record_chars=self.config.max_record_chars,
        )

    @staticmethod
    def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return summarize_evaluations(results)

    @staticmethod
    def load(path: Path) -> List[Dict[str, Any]]:
        return load_records(path)


def evaluate_native_record(
    record: Dict[str, Any],
    *,
    hard_issues: Optional[Iterable[str]] = None,
    trace_status: Optional[str] = None,
    minimum_score: float = DEFAULT_MINIMUM_SCORE,
    max_record_chars: int = DEFAULT_MAX_RECORD_CHARS,
) -> Dict[str, Any]:
    """Return schema-v2 data plus stable native compatibility aliases."""

    report = evaluate(
        _record_with_trace_status(record, trace_status),
        profile=EvaluationProfile.TEACHER_TRACE,
        hard_issues=hard_issues,
        minimum_score=minimum_score,
        max_record_chars=max_record_chars,
    )
    return _serialize_legacy(report, minimum_score)


def summarize_evaluations(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return canonical aggregation plus legacy dataset aliases."""

    summary = summarize(results)
    overall = dict(summary.get("overall") or {})
    issue_counts = Counter(
        str(issue)
        for result in results
        for issue in result.get("quality_issues") or []
    )
    profile_counts = Counter(str(result.get("profile")) for result in results)
    summary.update(
        {
            "pass_count": overall.get("pass_count", 0),
            "needs_review_count": len(results) - int(overall.get("pass_count", 0)),
            "average_quality_score": overall.get("mean_score"),
            "minimum_quality_score": overall.get("minimum_score"),
            "maximum_quality_score": overall.get("maximum_score"),
            "profile_counts": dict(sorted(profile_counts.items())),
            "quality_issue_counts": dict(sorted(issue_counts.items())),
        }
    )
    return summary


def load_records(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [
            value
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
            for value in [json.loads(line)]
            if isinstance(value, dict)
        ]
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    raise TypeError(
        "Teacher trace must contain a JSON object, list, or JSONL records: "
        f"{path}"
    )
