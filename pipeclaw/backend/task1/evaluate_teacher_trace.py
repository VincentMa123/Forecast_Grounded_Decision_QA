from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from pipeclaw.backend.grounding.evidence.csv import record_csv_evidence
from pipeclaw.backend.grounding.record_repair import repair_grounded_record
from pipeclaw.backend.evaluator.scorer import (
    DEFAULT_MINIMUM_SCORE,
    NativeEvaluationConfig,
    NativeTraceEvaluator,
    apply_quality_aliases,
)
from pipeclaw.backend.reporting.teacher_trace_audit import (
    TeacherTraceAuditConfig,
    TeacherTraceQualityAuditor,
    build_teacher_report_facts,
    materialize_teacher_report_value,
)
from pipeclaw.backend.reporting.reviewer_annotations import (
    export_reviewer_annotations,
    load_reviewer_annotations,
    load_sample_id_set,
)
from pipeclaw.backend.evaluator.numeric_grounding import (
    numeric_claims_are_grounded,
    numeric_grounding_evidence,
)
from pipeclaw.backend.task1.trace_export import write_split_records
from pipeclaw.backend.task1.trace_history import (
    build_history_turn,
    summarize_record_tool_evidence,
)
from pipeclaw.backend.pipeline.io_utils import write_json, write_jsonl
from pipeclaw.backend.reporting.statistics_report import Task1StatisticsWorkbook
from pipeclaw.backend.reporting.teacher_trace_quality_report import (
    TeacherTraceQualityReportWriter,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BACKEND_ROOT.parents[1]
DEFAULT_TRACE = BACKEND_ROOT / "generated_teacher_traces" / "teacher_trace.json"
GENERATED_ROOT = BACKEND_ROOT / "generated_teacher_traces"
DEFAULT_DELIVERABLE_DIR = GENERATED_ROOT / "task1_deliverables"
DEFAULT_OUTPUT = DEFAULT_DELIVERABLE_DIR / "quality_evaluation.jsonl"
DEFAULT_SUMMARY = DEFAULT_DELIVERABLE_DIR / "quality_evaluation_summary.json"
DEFAULT_SCHEMA = DEFAULT_DELIVERABLE_DIR / "teacher_trace_schema.json"
DEFAULT_REPORT = DEFAULT_DELIVERABLE_DIR / "teacher_trace_quality_report.xlsx"
DEFAULT_STATISTICS = DEFAULT_DELIVERABLE_DIR / "teacher_trace_statistics.xlsx"
DEFAULT_REVIEWER_ANNOTATIONS = DEFAULT_DELIVERABLE_DIR / "manual_quality_decisions.jsonl"
DEFAULT_COMPACT_SPLITS = GENERATED_ROOT / "splits"
CONSTRAINT_LIBRARY = BACKEND_ROOT / "pipeline" / "constraint_library"
REQUIRED_RULE_FILES = (
    "pipeline_constraints.json",
    "pressure_rules.json",
    "flow_rules.json",
    "linepack_rules.json",
    "compressor_rules.json",
    "intervention_rules.json",
    "dispatch_priority_rules.json",
)


def display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate compact teacher-trace records using the native PipeClaw workflow."
    )
    parser.add_argument("--teacher-trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--scenario-id")
    parser.add_argument("--sample-id")
    parser.add_argument("--minimum-score", type=float, default=DEFAULT_MINIMUM_SCORE)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--schema-json", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--report-xlsx", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--statistics-xlsx", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument(
        "--reviewer-annotations",
        type=Path,
        default=DEFAULT_REVIEWER_ANNOTATIONS,
        help="Durable JSONL copy of reviewer-entered workbook columns.",
    )
    parser.add_argument(
        "--reset-review-sample-ids",
        type=Path,
        help="JSON, JSONL, or text file of repaired sample IDs to reset to pending review.",
    )
    parser.add_argument("--deliverable-dir", type=Path, default=DEFAULT_DELIVERABLE_DIR)
    parser.add_argument("--compact-split-dir", type=Path, default=DEFAULT_COMPACT_SPLITS)
    parser.add_argument("--manual-sample-rate", type=float, default=0.25)
    parser.add_argument("--manual-sample-seed", default="task1-quality-v1")
    parser.add_argument(
        "--repair-grounded-records",
        action="store_true",
        help="Deterministically repair multi-candidate answers from stored tool evidence.",
    )
    parser.add_argument(
        "--repair-output",
        type=Path,
        help="Separate JSON or JSONL destination for repaired records.",
    )
    return parser


@dataclass(frozen=True)
class _ReportInputs:
    facts: Dict[str, object]
    evaluations: list[dict[str, object]]
    quality_sample_ids: set[str]


@dataclass(frozen=True)
class _ReportArtifacts:
    statistics: Dict[str, object]
    manual_records: list[dict[str, object]]
    artifacts: Dict[str, object]
    report_source: Path
    workbook_verification: Dict[str, object]
    statistics_workbook_verification: Dict[str, object]


class TeacherTraceEvaluationRunner:
    """Load, filter, evaluate, and persist one teacher-trace evaluation run."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.evaluator = NativeTraceEvaluator(
            NativeEvaluationConfig(minimum_score=args.minimum_score)
        )
        self.auditor = TeacherTraceQualityAuditor(
            TeacherTraceAuditConfig(
                manual_sample_rate=args.manual_sample_rate,
                manual_sample_seed=args.manual_sample_seed,
            )
        )
        self.report_writer = TeacherTraceQualityReportWriter(self.auditor)

    def run(self) -> Dict[str, object]:
        self._validate_options()
        annotation_export, reviewer_annotations, reset_review_sample_ids = self._load_review_inputs()
        records, native_results = self._load_filter_and_normalize_records()
        if self.args.repair_grounded_records:
            assert self.args.repair_output is not None
            if self.args.repair_output.suffix.casefold() == ".jsonl":
                write_jsonl(self.args.repair_output, records, force=True)
            else:
                write_json(self.args.repair_output, records, force=True)
        report_inputs = self._build_report_inputs(records, native_results)
        report_artifacts = self._write_report_artifacts(
            records,
            report_inputs,
            reviewer_annotations,
            reset_review_sample_ids,
        )
        summary = self._build_summary(
            records,
            report_inputs,
            report_artifacts,
            annotation_export,
        )
        write_json(self.args.summary_json, summary, force=True)
        return summary

    def _validate_options(self) -> None:
        if self.args.repair_grounded_records and self.args.repair_output is None:
            raise ValueError("--repair-output is required with --repair-grounded-records.")

    def _load_review_inputs(self) -> tuple[object, list[object], set[str]]:
        args = self.args
        annotation_export = None
        if args.report_xlsx.is_file():
            annotation_export = export_reviewer_annotations(
                args.report_xlsx,
                args.reviewer_annotations,
            )
        reviewer_annotations = load_reviewer_annotations(args.reviewer_annotations)
        reset_review_sample_ids = load_sample_id_set(args.reset_review_sample_ids)
        return annotation_export, reviewer_annotations, reset_review_sample_ids

    def _load_filter_and_normalize_records(
        self,
    ) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
        args = self.args
        all_records = self.evaluator.load(args.teacher_trace.resolve())
        resolved_records, native_results = self._resolve_legacy_grounding(all_records)
        records = [
            record
            for record in resolved_records
            if (not args.scenario_id or record.get("scenario_id") == args.scenario_id)
            and (not args.sample_id or record.get("sample_id") == args.sample_id)
        ]
        if not records:
            raise ValueError("No teacher-trace records matched the requested filters.")
        return records, native_results

    @staticmethod
    def _build_report_inputs(
        records: list[dict[str, object]],
        native_results: dict[str, dict[str, object]],
    ) -> _ReportInputs:
        facts = build_teacher_report_facts(
            records,
            [native_results[str(record.get("sample_id") or "")] for record in records],
        )
        evaluations = materialize_teacher_report_value(facts["evaluations"])
        quality_sample_ids = {
            str(item.get("sample_id"))
            for item in evaluations
            if item.get("task1_quality_flag") == "pass"
        }
        return _ReportInputs(facts, evaluations, quality_sample_ids)

    def _write_report_artifacts(
        self,
        records: list[dict[str, object]],
        report_inputs: _ReportInputs,
        reviewer_annotations: list[object],
        reset_review_sample_ids: set[str],
    ) -> _ReportArtifacts:
        args = self.args
        write_jsonl(args.output_jsonl, report_inputs.evaluations, force=True)
        self.report_writer.write_schema(args.schema_json)
        filtered_run = bool(args.scenario_id or args.sample_id)
        compact_split_counts: Dict[str, int] = {}
        audit_split_counts: Dict[str, int] = {}
        if not filtered_run:
            write_split_records(
                args.compact_split_dir,
                [
                    {**record, "quality_flag": "pass"}
                    for record in records
                    if str(record.get("sample_id") or "") in report_inputs.quality_sample_ids
                ],
                force=True,
            )
            compact_split_counts = self._split_counts(args.compact_split_dir)
            audit_split_counts = self.report_writer.write_audit_splits(
                args.deliverable_dir,
                records,
                args.compact_split_dir,
                report_inputs.quality_sample_ids,
            )
        statistics = materialize_teacher_report_value(report_inputs.facts["statistics"])
        manual_records = (
            self.auditor.manual_review_queue(records, report_inputs.evaluations)
            if args.repair_grounded_records
            else self.auditor.manual_sample(records, report_inputs.evaluations)
        )
        rule_files = {
            name: (CONSTRAINT_LIBRARY / name).is_file()
            for name in REQUIRED_RULE_FILES
        }
        artifacts = {
            "teacher_trace_schema": display_path(args.schema_json),
            "teacher_trace_quality_report": display_path(args.report_xlsx),
            "teacher_trace_statistics": display_path(args.statistics_xlsx),
            "quality_evaluation_jsonl": display_path(args.output_jsonl),
            "quality_evaluation_summary": display_path(args.summary_json),
            "audit_split_counts": audit_split_counts,
            "compact_sft_split_dir": display_path(args.compact_split_dir),
            "compact_sft_split_counts": compact_split_counts,
            "pipeline_constraint_rule_library": rule_files,
            "filtered_run_did_not_rewrite_splits": filtered_run,
            "reviewer_annotations": display_path(args.reviewer_annotations),
            "reset_review_record_count": len(reset_review_sample_ids),
        }
        report_source = (
            args.repair_output
            if args.repair_grounded_records and args.repair_output is not None
            else args.teacher_trace
        )
        Task1StatisticsWorkbook.write(
            args.statistics_xlsx,
            report_inputs.facts,
            Path(display_path(report_source)),
        )
        self.report_writer.write_report(
            args.report_xlsx,
            report_inputs.facts,
            manual_records,
            Path(display_path(report_source)),
            artifacts,
            reviewer_annotations=reviewer_annotations,
            reset_review_sample_ids=sorted(reset_review_sample_ids),
        )
        workbook_verification = self.report_writer.verify_workbook(args.report_xlsx)
        statistics_workbook_verification = self.report_writer.verify_workbook(
            args.statistics_xlsx
        )
        return _ReportArtifacts(
            statistics,
            manual_records,
            artifacts,
            report_source,
            workbook_verification,
            statistics_workbook_verification,
        )

    def _build_summary(
        self,
        records: list[dict[str, object]],
        report_inputs: _ReportInputs,
        report_artifacts: _ReportArtifacts,
        annotation_export: object,
    ) -> Dict[str, object]:
        args = self.args
        return {
            "schema_version": "pipeclaw_task1_quality_v1",
            "teacher_trace": display_path(report_artifacts.report_source),
            "minimum_pass_score": args.minimum_score,
            "native_evaluation": self.evaluator.summarize([
                {
                    "quality_score": item["native_quality_score"],
                    "quality_flag": item["native_quality_flag"],
                    "profile": item["native_profile"],
                    "quality_issues": item["native_quality_issues"],
                }
                for item in report_inputs.evaluations
            ]),
            "task1_statistics": report_artifacts.statistics,
            "manual_spot_check": {
                "queue_mode": (
                    "all_remaining_needs_review"
                    if args.repair_grounded_records
                    else "deterministic_stratified_sample"
                ),
                "sample_rate_target": (
                    None if args.repair_grounded_records else args.manual_sample_rate
                ),
                "sample_count": len(report_artifacts.manual_records),
                "actual_sample_rate": round(len(report_artifacts.manual_records) / len(records), 6),
                "status": "pending_human_signoff",
            },
            "rule_library_complete": all(
                report_artifacts.artifacts["pipeline_constraint_rule_library"].values()
            ),
            "workbook_verification": report_artifacts.workbook_verification,
            "statistics_workbook_verification": report_artifacts.statistics_workbook_verification,
            "artifacts": report_artifacts.artifacts,
            "output_jsonl": display_path(args.output_jsonl),
            "repair_output": (
                display_path(args.repair_output)
                if args.repair_grounded_records
                else None
            ),
            "reviewer_annotation_export": annotation_export,
        }

    def _resolve_legacy_grounding(
        self, records: list[dict[str, object]]
    ) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
        provenance: dict[tuple[str, str, str], dict[str, object]] = {}
        resolved_records = []
        native_results = {}
        for source_record in records:
            record = dict(source_record)
            dataset_source = str(record.get("dataset_source") or "")
            context = []
            for source_turn in record.get("conversation_context") or []:
                turn = dict(source_turn)
                key = (
                    dataset_source,
                    str(turn.get("session_id") or ""),
                    str(turn.get("turn_id") or ""),
                )
                inherited = provenance.get(key)
                if inherited:
                    turn.update(inherited)
                context.append(turn)
            record["conversation_context"] = context
            self._refresh_csv_grounding(record)
            if self.args.repair_grounded_records:
                record = repair_grounded_record(record)
            native = self.evaluator.evaluate(record, trace_status=record.get("trace_status"))
            record_id = str(record.get("sample_id") or "")
            native_results[record_id] = native
            if self.args.repair_grounded_records:
                # Repaired records adopt the aliases of the re-run canonical
                # report rather than a second offline score calculation.
                apply_quality_aliases(record, native)
            evidence_summary = summarize_record_tool_evidence(record)
            grounding_verified = (
                native.get("quality_flag") == "pass"
                and evidence_summary.evidence_found
            )
            history_projection = build_history_turn(
                record,
                evidence_summary=evidence_summary,
            )
            tool_evidence_verified = bool(
                history_projection.get("tool_evidence_verified")
            )
            current_key = (
                dataset_source,
                str(record.get("session_id") or ""),
                str(record.get("turn_id") or ""),
            )
            provenance_entry: dict[str, object] = {
                "assistant_output": str(record.get("final_answer") or ""),
                "grounding_verified": grounding_verified,
                "tool_evidence_verified": tool_evidence_verified,
                "evidence_artifacts": list(evidence_summary.evidence_artifacts),
            }
            verified_evidence_summary = (
                dict(history_projection.get("verified_evidence_summary") or {})
                if tool_evidence_verified
                else {}
            )
            if verified_evidence_summary:
                provenance_entry["verified_evidence_summary"] = verified_evidence_summary
            provenance[current_key] = provenance_entry
            resolved_records.append(record)
        return resolved_records, native_results

    @staticmethod
    def _refresh_csv_grounding(record: dict[str, object]) -> None:
        """Rebuild deterministic CSV evidence from stored successful tool outputs."""
        csv_evidence = record_csv_evidence(
            record,
            scope_text=str(record.get("user_input") or ""),
        )
        if not csv_evidence:
            return
        evidence = dict(record.get("evidence") or {})
        evidence["csv_evidence"] = csv_evidence
        record["evidence"] = evidence

        issues = list(record.get("quality_issues") or [])
        if (
            "unsupported_numerical_claim" in issues
            and numeric_claims_are_grounded(
                str(record.get("final_answer") or ""),
                str(record.get("user_input") or ""),
                numeric_grounding_evidence(record),
            )
        ):
            record["quality_issues"] = [
                issue
                for issue in issues
                if issue != "unsupported_numerical_claim"
            ]

    @staticmethod
    def _split_counts(path: Path) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for split in ("train", "valid", "test"):
            source = path / f"teacher_trace_{split}.jsonl"
            counts[split] = (
                sum(
                    bool(line.strip())
                    for line in source.read_text(encoding="utf-8-sig").splitlines()
                )
                if source.is_file()
                else 0
            )
        return counts

def main() -> int:
    args = build_parser().parse_args()
    summary = TeacherTraceEvaluationRunner(args).run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
