from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, median
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pipeclaw.backend.evaluator.checks.teacher import TEACHER_TRACE_REQUIRED_FIELDS
from pipeclaw.backend.grounding.evidence.tool import (
    attach_tool_arguments,
    tool_output_failed,
)


@dataclass(frozen=True)
class TeacherTraceAuditConfig:
    manual_sample_rate: float = 0.25
    manual_sample_seed: str = "task1-quality-v1"

    def __post_init__(self) -> None:
        if not 0.20 <= self.manual_sample_rate <= 0.30:
            raise ValueError("manual_sample_rate must be between 0.20 and 0.30.")


class TeacherTraceQualityAuditor:
    """Evaluate teacher-trace records and build dataset-level audit summaries."""

    def __init__(self, config: Optional[TeacherTraceAuditConfig] = None) -> None:
        self.config = config or TeacherTraceAuditConfig()

    @staticmethod
    def evidence_item_count(value: Any) -> int:
        if value is None or value == "" or value == [] or value == {}:
            return 0
        if isinstance(value, list):
            return sum(max(1, TeacherTraceQualityAuditor.evidence_item_count(item)) for item in value)
        if isinstance(value, dict):
            return sum(TeacherTraceQualityAuditor.evidence_item_count(item) for item in value.values())
        return 1

    @staticmethod
    def manual_review_queue(
        records: Sequence[Dict[str, Any]],
        evaluations: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return every record still requiring native or Task 1 review."""
        by_id = {str(item.get("sample_id")): item for item in evaluations}
        selected = [
            record
            for record in records
            if (
                (by_id.get(str(record.get("sample_id"))) or {}).get("native_quality_flag")
                != "pass"
                or (by_id.get(str(record.get("sample_id"))) or {}).get("task1_quality_flag")
                != "pass"
            )
        ]
        return sorted(selected, key=lambda item: str(item.get("sample_id")))

    @staticmethod
    def manual_evidence_summary(record: Dict[str, Any]) -> str:
        """Return compact rule and watch-variable evidence for human review."""
        constraint = dict(record.get("constraint_check") or {})
        constraint_items = [constraint]
        constraint_items.extend(
            dict(item) for item in constraint.get("candidate_forecasts") or []
        )
        findings = []
        for item in constraint_items:
            for finding in item.get("priority_findings") or []:
                finding = dict(finding)
                findings.append({
                    "name": finding.get("name"),
                    "status": finding.get("status"),
                    "affected_variables": list(finding.get("affected_variables") or [])[:5],
                })

        evidence = dict(record.get("evidence") or {})
        evidence_items = [evidence]
        evidence_items.extend(dict(item) for item in evidence.get("candidate_forecasts") or [])
        watch = []
        for item in evidence_items:
            for variable in item.get("top_watch_variables") or []:
                variable = dict(variable)
                watch.append({
                    "variable": variable.get("variable"),
                    "mean_prediction": variable.get("mean_prediction"),
                    "mean_abs_delta_vs_observed": variable.get("mean_abs_delta_vs_observed"),
                })
        return json.dumps(
            {
                "priority_findings": findings[:5],
                "top_watch_variables": watch[:5],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    def manual_sample(
        self,
        records: Sequence[Dict[str, Any]],
        evaluations: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_id = {str(item.get("sample_id")): item for item in evaluations}
        groups: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            result = by_id.get(str(record.get("sample_id"))) or {}
            key = (
                str(record.get("dataset_source") or "unknown"),
                str(record.get("scenario_type") or "unknown"),
                str(result.get("task1_quality_flag") or "unknown"),
            )
            groups[key].append(record)

        selected: List[Dict[str, Any]] = []
        for group_records in groups.values():
            count = max(1, round(len(group_records) * self.config.manual_sample_rate))
            ranked = sorted(
                group_records,
                key=lambda item: hashlib.sha256(
                    f"{self.config.manual_sample_seed}:{item.get('sample_id')}".encode("utf-8")
                ).hexdigest(),
            )
            selected.extend(ranked[:count])
        return sorted(selected, key=lambda item: str(item.get("sample_id")))

    @staticmethod
    def schema() -> Dict[str, Any]:
        nullable_string = {"type": ["string", "null"]}
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "teacher_trace_schema.json",
            "title": "PipeClaw Task 1 Teacher Trace Record",
            "type": "object",
            "required": list(TEACHER_TRACE_REQUIRED_FIELDS),
            "properties": {
                "sample_id": {"type": "string", "minLength": 1},
                "scenario_id": {"type": "string", "minLength": 1},
                "scenario_type": {"type": "string", "minLength": 1},
                "state_before": {
                    "type": "object",
                    "required": ["schema_version", "scope", "provenance"],
                    "properties": {
                        "schema_version": {
                            "const": "verified_decision_state_v1"
                        },
                        "scope": {"type": "object"},
                        "verified_evidence": {"type": "object"},
                        "registry_variables": {
                            "type": "object",
                            "required": [
                                "context_only",
                                "search_call_ids",
                                "returned_ids",
                            ],
                        },
                        "candidates": {"type": "array"},
                        "decision_policy": {"type": "object"},
                        "applied_disturbances": {"type": "array"},
                        "unresolved_inputs": {"type": "array"},
                        "provenance": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
                "recent_turns": {
                    "type": "array",
                    "maxItems": 2,
                    "items": {"type": "object"},
                },
                "user_input": {"type": "string"},
                "parsed_task": {"type": "object"},
                "tool_calls": {"type": "array", "items": {"type": "object"}},
                "tool_outputs": {"type": "array", "items": {"type": "object"}},
                "prediction_summary": {"type": "object"},
                "constraint_check": {"type": "object"},
                "evidence": {"type": "object"},
                "risk_level": {
                    "type": ["string", "null"],
                    "enum": ["low", "medium", "high", "unknown", None],
                },
                "manual_intervention_label": {
                    "type": ["string", "null"],
                    "enum": [
                        "no_intervention",
                        "monitoring_only",
                        "operator_attention_required",
                        "immediate_intervention_required",
                        "unknown",
                        None,
                    ],
                },
                "dispatch_recommendation": nullable_string,
                "final_answer": {"type": "string"},
                "quality_flag": {"type": "string", "enum": ["pass", "needs_review"]},
            },
            "additionalProperties": True,
        }


_CHECK_NAMES = ("schema", "numerical_consistency", "rule_consistency", "dispatch_consistency")
_CHECK_PURPOSES = {
    "schema": "Required Task 1.7 fields and JSON container types.",
    "numerical_consistency": "Final-answer numeric claims are present in trusted evidence.",
    "rule_consistency": "Risk and intervention conclusions agree with constraint_check.",
    "dispatch_consistency": "Recommendations obey pressure and compressor safety priorities.",
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def materialize_teacher_report_value(value: Any) -> Any:
    """Return plain JSON containers at the persistence boundary."""
    if isinstance(value, Mapping):
        return {key: materialize_teacher_report_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, frozenset)):
        return [materialize_teacher_report_value(item) for item in value]
    return value


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _is_sft_eligible(record: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    return bool(
        result.get("task1_quality_flag") == "pass"
        and not record.get("sft_exclusion_reason")
        and all(item.get("quality_flag") == "pass" for item in record.get("conversation_context") or [])
    )


def _distribution_rows(values: Mapping[str, int]) -> list[tuple[Any, ...]]:
    total = sum(values.values()) or 1
    return [(value, count, count / total) for value, count in sorted(values.items())]


def _evaluation_row(record: Mapping[str, Any], native: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        name: dict((native.get("teacher_trace_checks") or {}).get(name) or {
            "status": "not_applicable", "issues": []
        })
        for name in _CHECK_NAMES
    }
    failed = [name for name, check in checks.items() if check.get("status") == "fail"]
    outputs = attach_tool_arguments(
        [dict(item) for item in record.get("tool_outputs") or []],
        [dict(item) for item in record.get("tool_calls") or []],
    )
    return {
        "sample_id": record.get("sample_id"),
        "scenario_id": record.get("scenario_id"),
        "dataset_source": record.get("dataset_source"),
        "scenario_type": record.get("scenario_type"),
        "split": record.get("split"),
        "native_profile": native.get("profile"),
        "native_quality_flag": native.get("quality_flag"),
        "native_quality_score": native.get("quality_score"),
        "native_failed_checks": list(native.get("failed_checks") or []),
        "native_failed_critical_checks": list(native.get("failed_critical_checks") or []),
        "native_quality_issues": list(native.get("quality_issues") or []),
        "native_checks": list(native.get("checks") or []),
        "task1_quality_flag": "pass" if native.get("quality_flag") == "pass" and not failed else "needs_review",
        "task1_failed_checks": failed,
        "evidence_item_count": TeacherTraceQualityAuditor.evidence_item_count(record.get("evidence") or {}),
        "final_answer_chars": len(str(record.get("final_answer") or "")),
        "successful_tool_output_count": sum(not tool_output_failed(item) for item in outputs),
        "failed_tool_output_count": sum(tool_output_failed(item) for item in outputs),
        "checks": checks,
    }


def build_teacher_report_facts(
    records: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Build the deeply immutable read model rendered by both workbooks."""
    record_values = tuple(records)
    if len(record_values) != len(reports):
        raise ValueError("Each teacher record requires one native evaluation report.")
    evaluations = tuple(_evaluation_row(record, report) for record, report in zip(record_values, reports))
    by_id = {str(row.get("sample_id") or ""): row for row in evaluations}
    total = len(record_values)
    evidence_counts = [int(row["evidence_item_count"]) for row in evaluations]
    answer_lengths = [len(str(record.get("final_answer") or "")) for record in record_values]
    tool_counts = [len(record.get("tool_calls") or []) for record in record_values]
    eligible = [_is_sft_eligible(record, by_id[str(record.get("sample_id") or "")]) for record in record_values]

    requested = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    category_scenarios: dict[str, set[str]] = defaultdict(set)
    rule_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in record_values:
        constraint = dict(record.get("constraint_check") or {})
        category_status = dict(constraint.get("category_status") or {})
        for category in constraint.get("requested_categories") or []:
            category = str(category)
            requested[category] += 1
            category_counts[category][str(category_status.get(category) or "missing")] += 1
            category_scenarios[category].add(str(record.get("scenario_id") or ""))
        for rule, status in dict(constraint.get("rule_status") or {}).items():
            rule_counts[str(rule)][str(status or "missing")] += 1

    task_types = Counter(
        str((record.get("parsed_task") or {}).get("task_type") or "unspecified")
        for record in record_values if record.get("scenario_type") == "pipeformer"
    )
    task1_flags = Counter(str(row.get("task1_quality_flag") or "unknown") for row in evaluations)
    native_flags = Counter(str(row.get("native_quality_flag") or "unknown") for row in evaluations)
    statistics = {
        "total_sample_count": total,
        "task_type_distribution": dict(sorted(task_types.items())),
        "scenario_type_distribution": dict(sorted(Counter(str(record.get("scenario_type") or "unknown") for record in record_values).items())),
        "dataset_source_distribution": dict(sorted(Counter(str(record.get("dataset_source") or "unknown") for record in record_values).items())),
        "constraint_type_distribution": dict(sorted(requested.items())),
        "risk_level_distribution": dict(sorted(Counter(str(record.get("risk_level") or "not_applicable") for record in record_values).items())),
        "human_intervention_distribution": dict(sorted(Counter(str(record.get("manual_intervention_label") or "not_applicable") for record in record_values).items())),
        "native_quality_distribution": dict(sorted(native_flags.items())),
        "task1_quality_distribution": dict(sorted(task1_flags.items())),
        "average_evidence_item_count": round(sum(evidence_counts) / total, 6) if total else 0.0,
        "native_quality_pass_rate": round(native_flags.get("pass", 0) / total, 6) if total else 0.0,
        "task1_quality_pass_rate": round(task1_flags.get("pass", 0) / total, 6) if total else 0.0,
    }
    check_counts = {
        name: dict(Counter(str(row["checks"][name].get("status")) for row in evaluations))
        for name in _CHECK_NAMES
    }

    category_rows = []
    quality_category_rows = []
    for category in sorted(requested):
        counts = category_counts[category]
        evaluated = sum(counts.get(value, 0) for value in ("pass", "warning", "fail"))
        nonpass = counts.get("warning", 0) + counts.get("fail", 0)
        missing = requested[category] - evaluated
        category_rows.append((category, requested[category], evaluated, counts.get("pass", 0), counts.get("warning", 0), counts.get("fail", 0), missing, evaluated / requested[category], counts.get("pass", 0) / evaluated if evaluated else None, nonpass / evaluated if evaluated else None))
        quality_category_rows.append((category, requested[category], evaluated, counts.get("pass", 0), counts.get("warning", 0), counts.get("fail", 0), missing, evaluated / requested[category], nonpass / evaluated if evaluated else None, len(category_scenarios[category])))

    rule_rows = []
    for rule in sorted(rule_counts):
        counts = rule_counts[rule]
        evaluated = sum(counts.get(value, 0) for value in ("pass", "warning", "fail"))
        nonpass = counts.get("warning", 0) + counts.get("fail", 0)
        rule_rows.append((rule, evaluated, counts.get("pass", 0), counts.get("warning", 0), counts.get("fail", 0), sum(counts.values()) - evaluated, counts.get("pass", 0) / evaluated if evaluated else None, nonpass / evaluated if evaluated else None))

    coverage_groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    evidence_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    source_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in record_values:
        sample_id = str(record.get("sample_id") or "")
        source = str(record.get("dataset_source") or "unknown")
        scenario_type = str(record.get("scenario_type") or "unknown")
        split = str(record.get("split") or "unknown")
        coverage_groups[(source, scenario_type, split)].append(record)
        evidence_groups[(source, scenario_type)].append(by_id[sample_id]["evidence_item_count"])
        source_groups[source].append(record)

    coverage_rows = []
    for key, group in sorted(coverage_groups.items()):
        rows = [by_id[str(record.get("sample_id") or "")] for record in group]
        passed = sum(row.get("task1_quality_flag") == "pass" for row in rows)
        coverage_rows.append((*key, len(group), len({str(record.get("scenario_id") or "") for record in group}), len({str(record.get("session_id") or "") for record in group}), passed, passed / len(group), mean(row["evidence_item_count"] for row in rows), mean(row["final_answer_chars"] for row in rows), mean(len(record.get("tool_calls") or []) for record in group)))

    source_statistics = {}
    for source, group in sorted(source_groups.items()):
        rows = [by_id[str(record.get("sample_id") or "")] for record in group]
        evidence = [row["evidence_item_count"] for row in rows]
        passed = sum(row.get("task1_quality_flag") == "pass" for row in rows)
        source_statistics[source] = {
            "sample_count": len(group), "percentage": len(group) / total if total else 0.0,
            "scenario_count": len({str(record.get("scenario_id") or "") for record in group}),
            "session_count": len({str(record.get("session_id") or "") for record in group}),
            "openclaw": sum(record.get("scenario_type") == "openclaw" for record in group),
            "pipeformer": sum(record.get("scenario_type") == "pipeformer" for record in group),
            "train": sum(record.get("split") == "train" for record in group),
            "valid": sum(record.get("split") == "valid" for record in group),
            "test": sum(record.get("split") == "test" for record in group),
            "quality_pass": passed, "needs_review": len(group) - passed,
            "quality_pass_rate": passed / len(group),
            "average_evidence_items": mean(evidence) if evidence else 0.0,
            "median_evidence_items": median(evidence) if evidence else 0.0,
            "p95_evidence_items": _percentile(evidence, 0.95),
            "average_answer_chars": mean(len(str(record.get("final_answer") or "")) for record in group),
            "average_tool_calls": mean(len(record.get("tool_calls") or []) for record in group),
        }
    source_columns = tuple(sorted(source_statistics))
    source_rows = [
        (metric, *(source_statistics[source][metric] for source in source_columns))
        for metric in next(iter(source_statistics.values()), {})
    ]

    split_rows = []
    for split in ("train", "valid", "test"):
        group = [record for record in record_values if record.get("split") == split]
        rows = [by_id[str(record.get("sample_id") or "")] for record in group]
        passed = sum(row.get("task1_quality_flag") == "pass" for row in rows)
        split_eligible = sum(_is_sft_eligible(record, by_id[str(record.get("sample_id") or "")]) for record in group)
        split_rows.append((split, len(group), len({str(record.get("scenario_id") or "") for record in group}), len({str(record.get("session_id") or "") for record in group}), passed, len(group) - passed, split_eligible, split_eligible / len(group) if group else 0.0, mean(row["evidence_item_count"] for row in rows) if rows else 0.0, mean(len(str(record.get("final_answer") or "")) for record in group) if group else 0.0))

    evidence_rows = [
        (*key, len(values), sum(value == 0 for value in values), min(values), _percentile(values, 0.25), median(values), mean(values), _percentile(values, 0.75), _percentile(values, 0.90), _percentile(values, 0.95), max(values))
        for key, values in sorted(evidence_groups.items())
    ]
    quality_rows = []
    for name in _CHECK_NAMES:
        counts = check_counts[name]
        applicable = counts.get("pass", 0) + counts.get("fail", 0)
        quality_rows.append((name, counts.get("pass", 0), counts.get("fail", 0), counts.get("not_applicable", 0), applicable, counts.get("pass", 0) / applicable if applicable else None))
    for name, distribution in (("native_quality", native_flags), ("task1_quality", task1_flags)):
        applicable = sum(distribution.values())
        quality_rows.append((name, distribution.get("pass", 0), distribution.get("needs_review", 0), 0, applicable, distribution.get("pass", 0) / applicable if applicable else None))

    record_check_rows = []
    needs_review_rows = []
    manual_rows_by_id = {}
    issue_counts = Counter()
    issue_samples: dict[str, list[str]] = defaultdict(list)
    issue_datasets: dict[str, set[str]] = defaultdict(set)
    issue_scenarios: dict[str, set[str]] = defaultdict(set)
    for record in record_values:
        sample_id = str(record.get("sample_id") or "")
        result = by_id[sample_id]
        checks = result["checks"]
        check_issues = [issue for check in checks.values() for issue in check.get("issues") or []]
        record_check_rows.append((record.get("sample_id"), record.get("dataset_source"), record.get("scenario_id"), record.get("session_id"), record.get("turn_id"), record.get("scenario_type"), record.get("split"), (record.get("parsed_task") or {}).get("task_type") or record.get("scenario_type"), len(record.get("tool_calls") or []), result["successful_tool_output_count"], result["failed_tool_output_count"], result.get("native_profile"), result.get("native_quality_flag"), result.get("native_quality_score"), result.get("task1_quality_flag"), result.get("evidence_item_count"), result.get("final_answer_chars"), checks["schema"]["status"], checks["numerical_consistency"]["status"], checks["rule_consistency"]["status"], checks["dispatch_consistency"]["status"], (record.get("constraint_check") or {}).get("overall_status"), record.get("risk_level"), record.get("manual_intervention_label"), ", ".join(result.get("task1_failed_checks") or []), ", ".join([*result.get("native_quality_issues", []), *check_issues])))
        if result.get("task1_quality_flag") != "pass":
            needs_review_rows.append((record.get("sample_id"), record.get("dataset_source"), record.get("scenario_id"), record.get("session_id"), record.get("turn_id"), record.get("scenario_type"), record.get("split"), result.get("native_quality_score"), ", ".join(result.get("native_failed_checks") or []), ", ".join(result.get("task1_failed_checks") or []), ", ".join(result.get("native_quality_issues") or []), record.get("user_input"), record.get("final_answer"), "", "pending"))
        record_issues = {str(issue) for issue in result.get("native_quality_issues") or []}
        record_issues.update(str(issue) for issue in check_issues)
        for issue in sorted(record_issues):
            issue_counts[issue] += 1
            issue_samples[issue].append(sample_id)
            issue_datasets[issue].add(str(record.get("dataset_source") or "unknown"))
            issue_scenarios[issue].add(str(record.get("scenario_id") or "unknown"))
        remaining = sorted(record_issues | {f"native_check:{name}" for name in result.get("native_failed_checks") or []} | {f"task1_check:{name}" for name in result.get("task1_failed_checks") or []})
        manual_rows_by_id[sample_id] = (record.get("sample_id"), record.get("dataset_source"), record.get("scenario_id"), record.get("session_id"), record.get("turn_id"), record.get("scenario_type"), record.get("split"), result.get("task1_quality_flag"), result.get("native_quality_score"), ", ".join(remaining), checks["schema"]["status"], checks["numerical_consistency"]["status"], checks["rule_consistency"]["status"], checks["dispatch_consistency"]["status"], record.get("user_input"), record.get("final_answer"), json.dumps(record.get("parsed_task") or {}, ensure_ascii=False, separators=(",", ":")), ", ".join(str(item.get("name") or "") for item in record.get("tool_calls") or []), json.dumps((record.get("constraint_check") or {}).get("category_status") or {}, ensure_ascii=False, separators=(",", ":")), TeacherTraceQualityAuditor.manual_evidence_summary(dict(record)), record.get("risk_level"), record.get("manual_intervention_label"), record.get("dispatch_recommendation"), "pending", "pending", "pending", "pending", "", "", "pending")

    issue_rows = [(issue, count, count / total if total else 0.0, ", ".join(sorted(issue_datasets[issue])), len(issue_scenarios[issue]), ", ".join(issue_samples[issue])) for issue, count in sorted(issue_counts.items())]
    distribution_rows = [(dimension, value, count, percentage) for dimension in ("dataset_source_distribution", "task_type_distribution", "scenario_type_distribution", "constraint_type_distribution", "risk_level_distribution", "human_intervention_distribution", "native_quality_distribution", "task1_quality_distribution") for value, count, percentage in _distribution_rows(statistics[dimension])]
    check_summary_rows = []
    for name in _CHECK_NAMES:
        counts = check_counts[name]
        applicable = counts.get("pass", 0) + counts.get("fail", 0)
        check_summary_rows.append((name, counts.get("pass", 0), counts.get("fail", 0), counts.get("not_applicable", 0), applicable, counts.get("pass", 0) / applicable if applicable else None, _CHECK_PURPOSES[name]))

    constraint_records = [record for record in record_values if record.get("constraint_check")]
    statistics_overview_rows = (
        ("Total samples", total),
        ("Dataset sources", len(source_groups)),
        ("Scenarios", len({str(record.get("scenario_id") or "") for record in record_values})),
        ("Sessions", len({str(record.get("session_id") or "") for record in record_values})),
        ("SFT-eligible samples", sum(eligible)),
        ("SFT-eligible rate", sum(eligible) / total if total else 0.0),
        ("Native quality pass rate", statistics["native_quality_pass_rate"]),
        ("Task 1 quality pass rate", statistics["task1_quality_pass_rate"]),
        ("Average evidence items", mean(evidence_counts) if evidence_counts else 0.0),
        ("Median evidence items", median(evidence_counts) if evidence_counts else 0.0),
        ("Average final-answer characters", mean(answer_lengths) if answer_lengths else 0.0),
        ("Average tool calls", mean(tool_counts) if tool_counts else 0.0),
        ("Records with constraint verification", len(constraint_records)),
        ("Records with warning/fail constraints", sum((record.get("constraint_check") or {}).get("overall_status") in {"warning", "fail"} for record in constraint_records)),
    )
    quality_summary_rows = (
        ("Total sample count", total),
        ("Native quality pass count", native_flags.get("pass", 0)),
        ("Native quality pass rate", statistics["native_quality_pass_rate"]),
        ("Task 1 verification pass count", task1_flags.get("pass", 0)),
        ("Task 1 verification pass rate", statistics["task1_quality_pass_rate"]),
        ("Schema check pass count", check_counts["schema"].get("pass", 0)),
        ("Numerical consistency pass count", check_counts["numerical_consistency"].get("pass", 0)),
        ("Numerical consistency failure count", check_counts["numerical_consistency"].get("fail", 0)),
        ("Applicable rule-consistency checks", check_counts["rule_consistency"].get("pass", 0) + check_counts["rule_consistency"].get("fail", 0)),
        ("Rule-consistency failure count", check_counts["rule_consistency"].get("fail", 0)),
        ("Applicable dispatch-consistency checks", check_counts["dispatch_consistency"].get("pass", 0) + check_counts["dispatch_consistency"].get("fail", 0)),
        ("Dispatch-consistency failure count", check_counts["dispatch_consistency"].get("fail", 0)),
        ("Average evidence item count", statistics["average_evidence_item_count"]),
    )
    intervention_values = sorted(statistics["human_intervention_distribution"])
    risk_intervention_rows = []
    for risk in sorted(statistics["risk_level_distribution"]):
        counts = [sum(str(record.get("risk_level") or "not_applicable") == risk and str(record.get("manual_intervention_label") or "not_applicable") == intervention for record in record_values) for intervention in intervention_values]
        risk_intervention_rows.append((risk, *counts, sum(counts)))

    facts = {
        "records": record_values, "evaluations": evaluations, "by_id": by_id,
        "statistics": statistics, "check_counts": check_counts,
        "requested_categories": dict(requested),
        "category_counts": {key: dict(value) for key, value in category_counts.items()},
        "category_scenarios": {key: set(value) for key, value in category_scenarios.items()},
        "rule_counts": {key: dict(value) for key, value in rule_counts.items()},
        "statistics_overview_rows": statistics_overview_rows,
        "quality_summary_rows": quality_summary_rows,
        "source_columns": source_columns, "source_rows": source_rows,
        "distribution_tables": {dimension: _distribution_rows(statistics[dimension]) for dimension in statistics if dimension.endswith("_distribution")},
        "statistics_category_rows": category_rows,
        "quality_category_rows": quality_category_rows,
        "rule_rows": rule_rows,
        "risk_intervention_columns": intervention_values,
        "risk_intervention_rows": risk_intervention_rows,
        "split_rows": split_rows, "evidence_rows": evidence_rows,
        "quality_rows": quality_rows, "check_summary_rows": check_summary_rows,
        "record_check_rows": record_check_rows, "coverage_rows": coverage_rows,
        "distribution_rows": distribution_rows, "issue_rows": issue_rows,
        "needs_review_rows": needs_review_rows, "manual_rows_by_id": manual_rows_by_id,
        "evidence_groups": evidence_groups,
    }
    return _freeze(facts)
