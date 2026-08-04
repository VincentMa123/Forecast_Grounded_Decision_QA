from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from evaluator.teacher_quality import (
    numeric_claim_values,
    numeric_claims_are_grounded,
    numeric_grounding_evidence,
    tool_output_failed,
)

from grounding.evidence.tool import attach_tool_arguments

TEACHER_TRACE_REQUIRED_FIELDS = (
    "sample_id",
    "scenario_id",
    "scenario_type",
    "state_before",
    "recent_turns",
    "user_input",
    "parsed_task",
    "tool_calls",
    "tool_outputs",
    "prediction_summary",
    "constraint_check",
    "evidence",
    "risk_level",
    "manual_intervention_label",
    "dispatch_recommendation",
    "final_answer",
    "quality_flag",
)

EXPECTED_TYPES = {
    "sample_id": str,
    "scenario_id": str,
    "scenario_type": str,
    "state_before": dict,
    "recent_turns": list,
    "user_input": str,
    "parsed_task": dict,
    "tool_calls": list,
    "tool_outputs": list,
    "prediction_summary": dict,
    "constraint_check": dict,
    "evidence": dict,
    "final_answer": str,
    "quality_flag": str,
}

ENTIRELY_SAFE_CLAIM = re.compile(
    r"完全安全|无任何风险|没有任何风险|所有(?:校核|规则|约束)均?通过|各项(?:校核|规则|约束)均?通过"
    r"|\b(?:entirely|completely|fully)\s+safe\b"
    r"|\ball\s+(?:requested\s+)?(?:checks|constraints|rules)\s+pass(?:ed)?\b"
    r"|\bno\s+(?:operational\s+)?risk\b",
    re.IGNORECASE,
)
REDUCE_UPSTREAM_INJECTION = re.compile(
    r"(?:减少|降低|下调|削减).{0,24}(?:上游|气源).{0,16}(?:注气|供气|供给|流量)"
    r"|(?:上游|气源).{0,16}(?:注气|供气|供给|流量).{0,24}(?:减少|降低|下调|削减)"
    r"|\b(?:reduce|decrease|lower|cut)\b.{0,40}\b(?:upstream\s+)?(?:injection|supply|inflow)\b",
    re.IGNORECASE,
)
RAISE_COMPRESSOR_LOAD = re.compile(
    r"(?:提高|增加|上调).{0,24}压缩机.{0,12}负荷"
    r"|压缩机.{0,12}负荷.{0,24}(?:提高|增加|上调)"
    r"|\b(?:raise|increase|boost)\b.{0,40}\bcompressor\s+load\b",
    re.IGNORECASE,
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

    def evaluate(
        self,
        record: Dict[str, Any],
        native_result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        schema = self._schema_check(record)
        numerical = self._numerical_check(record)
        rule = self._rule_check(record)
        dispatch = self._dispatch_check(record)
        checks = {
            "schema": schema,
            "numerical_consistency": numerical,
            "rule_consistency": rule,
            "dispatch_consistency": dispatch,
        }
        failed = [name for name, value in checks.items() if value["status"] == "fail"]
        task1_flag = (
            "pass"
            if native_result.get("quality_flag") == "pass" and not failed
            else "needs_review"
        )
        return {
            "sample_id": record.get("sample_id"),
            "scenario_id": record.get("scenario_id"),
            "dataset_source": record.get("dataset_source"),
            "scenario_type": record.get("scenario_type"),
            "split": record.get("split"),
            "native_quality_flag": native_result.get("quality_flag"),
            "native_quality_score": native_result.get("quality_score"),
            "native_failed_checks": native_result.get("failed_checks") or [],
            "native_quality_issues": native_result.get("quality_issues") or [],
            "task1_quality_flag": task1_flag,
            "task1_failed_checks": failed,
            "evidence_item_count": self.evidence_item_count(record.get("evidence") or {}),
            "final_answer_chars": len(str(record.get("final_answer") or "")),
            "checks": checks,
        }

    @staticmethod
    def _schema_check(record: Dict[str, Any]) -> Dict[str, Any]:
        missing = [field for field in TEACHER_TRACE_REQUIRED_FIELDS if field not in record]
        invalid_types = [
            field
            for field, expected in EXPECTED_TYPES.items()
            if field in record and not isinstance(record[field], expected)
        ]
        nullable_type_issues = [
            field
            for field in ("risk_level", "manual_intervention_label", "dispatch_recommendation")
            if field in record and record[field] is not None and not isinstance(record[field], str)
        ]
        issues = [f"missing:{field}" for field in missing]
        issues.extend(f"invalid_type:{field}" for field in invalid_types + nullable_type_issues)
        return {
            "status": "pass" if not issues else "fail",
            "issues": issues,
        }

    @staticmethod
    def _numerical_check(record: Dict[str, Any]) -> Dict[str, Any]:
        answer = str(record.get("final_answer") or "")
        question = str(record.get("user_input") or "")
        evidence = numeric_grounding_evidence(record)
        claims = numeric_claim_values(answer)
        grounded = numeric_claims_are_grounded(answer, question, evidence)
        return {
            "status": "pass" if grounded else "fail",
            "claimed_numeric_value_count": len(claims),
            "issues": [] if grounded else ["unsupported_numerical_claim"],
        }

    @staticmethod
    def _rule_check(record: Dict[str, Any]) -> Dict[str, Any]:
        constraint = dict(record.get("constraint_check") or {})
        if not constraint:
            return {"status": "not_applicable", "issues": []}

        issues: List[str] = []
        expected_risk = constraint.get("risk_level")
        expected_intervention = constraint.get("human_intervention_label")
        if expected_risk is not None and record.get("risk_level") != expected_risk:
            issues.append("risk_level_disagrees_with_constraint_check")
        if (
            expected_intervention is not None
            and record.get("manual_intervention_label") != expected_intervention
        ):
            issues.append("intervention_label_disagrees_with_constraint_check")

        category_status = dict(constraint.get("category_status") or {})
        rule_status = dict(constraint.get("rule_status") or {})
        nonpass = any(value in {"warning", "fail"} for value in category_status.values())
        nonpass = nonpass or any(value in {"warning", "fail"} for value in rule_status.values())
        if nonpass and ENTIRELY_SAFE_CLAIM.search(str(record.get("final_answer") or "")):
            issues.append("final_answer_claims_entirely_safe_despite_nonpass_rule")

        pressure_fail = category_status.get("pressure") == "fail" or any(
            str(flag).startswith("pressure_violation")
            for flag in constraint.get("triggered_flags") or []
        )
        if pressure_fail and record.get("risk_level") == "low":
            issues.append("pressure_violation_cannot_have_low_risk")
        if pressure_fail and record.get("manual_intervention_label") == "no_intervention":
            issues.append("pressure_violation_cannot_require_no_intervention")
        return {
            "status": "pass" if not issues else "fail",
            "issues": issues,
            "overall_constraint_status": constraint.get("overall_status"),
        }

    @staticmethod
    def _dispatch_check(record: Dict[str, Any]) -> Dict[str, Any]:
        constraint = dict(record.get("constraint_check") or {})
        if not constraint:
            return {"status": "not_applicable", "issues": []}

        issues: List[str] = []
        category_status = dict(constraint.get("category_status") or {})
        rule_status = dict(constraint.get("rule_status") or {})
        flags = {str(value) for value in constraint.get("triggered_flags") or []}
        pressure_fail = (
            category_status.get("pressure") == "fail"
            or any(value.startswith("pressure_violation") for value in flags)
            or rule_status.get("node_pressure_operating_window") == "fail"
        )
        compressor_overload = (
            "compressor_overload" in flags
            or rule_status.get("compressor_load_limit") == "fail"
        )
        dispatch = "\n".join(
            value
            for value in (
                str(record.get("dispatch_recommendation") or "").strip(),
                str(record.get("final_answer") or "").strip(),
            )
            if value
        )
        if pressure_fail and REDUCE_UPSTREAM_INJECTION.search(dispatch):
            issues.append("pressure_violation_recommends_reducing_upstream_injection")
        if compressor_overload and RAISE_COMPRESSOR_LOAD.search(dispatch):
            issues.append("compressor_overload_recommends_raising_compressor_load")

        expected = constraint.get("dispatch_recommendation")
        actual = record.get("dispatch_recommendation")
        if expected and actual and str(expected).strip() != str(actual).strip():
            issues.append("dispatch_recommendation_disagrees_with_constraint_check")
        return {
            "status": "pass" if not issues else "fail",
            "issues": issues,
            "pressure_failure_present": pressure_fail,
            "compressor_overload_present": compressor_overload,
        }

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
    def statistics(
        records: Sequence[Dict[str, Any]],
        evaluations: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        by_id = {str(item.get("sample_id")): item for item in evaluations}
        constraint_types = Counter()
        task_types = Counter()
        evidence_counts = []
        for record in records:
            parsed = dict(record.get("parsed_task") or {})
            if record.get("scenario_type") == "pipeformer":
                task_types[str(parsed.get("task_type") or "unspecified")] += 1
            for category in (record.get("constraint_check") or {}).get("requested_categories") or []:
                constraint_types[str(category)] += 1
            evidence_counts.append(TeacherTraceQualityAuditor.evidence_item_count(record.get("evidence") or {}))

        task1_flags = Counter(
            str(by_id.get(str(record.get("sample_id")), {}).get("task1_quality_flag") or "unknown")
            for record in records
        )
        native_flags = Counter(
            str(
                by_id.get(str(record.get("sample_id")), {}).get("native_quality_flag")
                or "unknown"
            )
            for record in records
        )
        total = len(records)
        return {
            "total_sample_count": total,
            "task_type_distribution": dict(sorted(task_types.items())),
            "scenario_type_distribution": dict(sorted(Counter(str(r.get("scenario_type") or "unknown") for r in records).items())),
            "dataset_source_distribution": dict(sorted(Counter(str(r.get("dataset_source") or "unknown") for r in records).items())),
            "constraint_type_distribution": dict(sorted(constraint_types.items())),
            "risk_level_distribution": dict(sorted(Counter(str(r.get("risk_level") or "not_applicable") for r in records).items())),
            "human_intervention_distribution": dict(sorted(Counter(str(r.get("manual_intervention_label") or "not_applicable") for r in records).items())),
            "native_quality_distribution": dict(sorted(native_flags.items())),
            "task1_quality_distribution": dict(sorted(task1_flags.items())),
            "average_evidence_item_count": round(sum(evidence_counts) / total, 6) if total else 0.0,
            "native_quality_pass_rate": round(native_flags.get("pass", 0) / total, 6) if total else 0.0,
            "task1_quality_pass_rate": round(task1_flags.get("pass", 0) / total, 6) if total else 0.0,
        }

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
