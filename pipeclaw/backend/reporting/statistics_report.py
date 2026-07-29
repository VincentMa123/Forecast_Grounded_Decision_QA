from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Sequence

from evaluator.teacher_trace_audit import TeacherTraceQualityAuditor


class Task1StatisticsWorkbook:
    """Create the standalone Task 1.9 data-statistics workbook."""

    @staticmethod
    def write(
        path: Path,
        records: Sequence[Dict[str, Any]],
        evaluations: Sequence[Dict[str, Any]],
        statistics: Mapping[str, Any],
        source_path: Path,
    ) -> None:
        from openpyxl import Workbook
        from openpyxl.chart import BarChart, PieChart, Reference
        from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
        from openpyxl.styles import Font, PatternFill

        from .workbook_style import polish_workbook, style_chart

        path.parent.mkdir(parents=True, exist_ok=True)
        workbook = Workbook()
        overview = workbook.active
        overview.title = "Overview"
        source_sheet = workbook.create_sheet("Dataset Sources")
        scenario_sheet = workbook.create_sheet("Scenario Types")
        task_sheet = workbook.create_sheet("Task Types")
        constraint_sheet = workbook.create_sheet("Constraint Types")
        outcome_sheet = workbook.create_sheet("Constraint Outcomes")
        rule_sheet = workbook.create_sheet("Rule Outcomes")
        risk_sheet = workbook.create_sheet("Risk Levels")
        intervention_sheet = workbook.create_sheet("Intervention Labels")
        cross_sheet = workbook.create_sheet("Risk x Intervention")
        split_sheet = workbook.create_sheet("Split Statistics")
        evidence_sheet = workbook.create_sheet("Evidence Statistics")
        quality_sheet = workbook.create_sheet("Quality Statistics")
        notes = workbook.create_sheet("Method Notes")
        workbook.properties.title = "Task 1 Teacher Trace Data Statistics"
        workbook.properties.subject = "Task 1.9 data statistics tables"

        by_id = {str(item.get("sample_id")): item for item in evaluations}
        total = len(records)
        eligible = [record for record in records if Task1StatisticsWorkbook._sft_eligible(record, by_id)]
        evidence_counts = [TeacherTraceQualityAuditor.evidence_item_count(record.get("evidence") or {}) for record in records]
        answer_lengths = [len(str(record.get("final_answer") or "")) for record in records]
        tool_counts = [len(record.get("tool_calls") or []) for record in records]
        constraint_records = [record for record in records if record.get("constraint_check")]
        nonpass_constraints = sum(
            (record.get("constraint_check") or {}).get("overall_status") in {"warning", "fail"}
            for record in constraint_records
        )

        overview_rows = [
            ("Task 1 Teacher Trace Data Statistics", None),
            ("Generated at (UTC)", datetime.now(timezone.utc).isoformat()),
            ("Teacher trace source", source_path.as_posix()),
            ("Total samples", total),
            ("Dataset sources", len({str(record.get("dataset_source") or "unknown") for record in records})),
            ("Scenarios", len({str(record.get("scenario_id") or "") for record in records})),
            ("Sessions", len({str(record.get("session_id") or "") for record in records})),
            ("SFT-eligible samples", len(eligible)),
            ("SFT-eligible rate", len(eligible) / total if total else 0.0),
            ("Native quality pass rate", statistics["native_quality_pass_rate"]),
            ("Task 1 quality pass rate", statistics["task1_quality_pass_rate"]),
            ("Average evidence items", mean(evidence_counts) if evidence_counts else 0.0),
            ("Median evidence items", median(evidence_counts) if evidence_counts else 0.0),
            ("Average final-answer characters", mean(answer_lengths) if answer_lengths else 0.0),
            ("Average tool calls", mean(tool_counts) if tool_counts else 0.0),
            ("Records with constraint verification", len(constraint_records)),
            ("Records with warning/fail constraints", nonpass_constraints),
        ]
        for row in overview_rows:
            overview.append(row)
        overview["A1"].font = Font(name="Arial", size=16, bold=True, color="FFFFFF")
        overview["A1"].fill = PatternFill("solid", fgColor="17365D")
        overview.merge_cells("A1:B1")
        for row in range(2, overview.max_row + 1):
            if "rate" in str(overview.cell(row, 1).value).lower():
                overview.cell(row, 2).number_format = "0.0%"

        source_groups = Task1StatisticsWorkbook._groups(records, "dataset_source")
        source_statistics: Dict[str, Dict[str, float | int]] = {}
        for source, group in sorted(source_groups.items()):
            results = [by_id[str(record.get("sample_id"))] for record in group]
            evidence = [item.get("evidence_item_count", 0) for item in results]
            passed = sum(item.get("task1_quality_flag") == "pass" for item in results)
            source_statistics[source] = {
                "sample_count": len(group),
                "percentage": len(group) / total if total else 0.0,
                "scenario_count": len({str(item.get("scenario_id") or "") for item in group}),
                "session_count": len({str(item.get("session_id") or "") for item in group}),
                "openclaw": sum(item.get("scenario_type") == "openclaw" for item in group),
                "pipeformer": sum(item.get("scenario_type") == "pipeformer" for item in group),
                "train": sum(item.get("split") == "train" for item in group),
                "valid": sum(item.get("split") == "valid" for item in group),
                "test": sum(item.get("split") == "test" for item in group),
                "quality_pass": passed,
                "needs_review": len(group) - passed,
                "quality_pass_rate": passed / len(group),
                "average_evidence_items": mean(evidence) if evidence else 0.0,
                "median_evidence_items": median(evidence) if evidence else 0.0,
                "p95_evidence_items": Task1StatisticsWorkbook._percentile(evidence, 0.95),
                "average_answer_chars": mean(len(str(item.get("final_answer") or "")) for item in group),
                "average_tool_calls": mean(len(item.get("tool_calls") or []) for item in group),
            }
        sources = sorted(source_statistics)
        source_sheet.append(["metric", *sources])
        for metric in next(iter(source_statistics.values()), {}):
            source_sheet.append([metric, *(source_statistics[source][metric] for source in sources)])
            if metric in {"percentage", "quality_pass_rate"}:
                for cell in source_sheet[source_sheet.max_row][1:]:
                    cell.number_format = "0.0%"

        Task1StatisticsWorkbook._distribution_sheet(
            scenario_sheet,
            statistics["scenario_type_distribution"],
            "scenario_type",
        )
        Task1StatisticsWorkbook._distribution_sheet(
            task_sheet,
            statistics["task_type_distribution"],
            "pipeformer_task_type",
        )
        Task1StatisticsWorkbook._distribution_sheet(
            constraint_sheet,
            statistics["constraint_type_distribution"],
            "constraint_type",
        )
        Task1StatisticsWorkbook._distribution_sheet(
            risk_sheet,
            statistics["risk_level_distribution"],
            "risk_level",
        )
        Task1StatisticsWorkbook._distribution_sheet(
            intervention_sheet,
            statistics["human_intervention_distribution"],
            "intervention_label",
        )

        requested = Counter()
        category_outcomes: Dict[str, Counter[str]] = defaultdict(Counter)
        rule_outcomes: Dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            constraint = dict(record.get("constraint_check") or {})
            statuses = dict(constraint.get("category_status") or {})
            for category in constraint.get("requested_categories") or []:
                category = str(category)
                requested[category] += 1
                category_outcomes[category][str(statuses.get(category) or "missing")] += 1
            for rule, status in (constraint.get("rule_status") or {}).items():
                rule_outcomes[str(rule)][str(status or "missing")] += 1

        outcome_sheet.append([
            "constraint_category", "requested", "evaluated", "pass", "warning", "fail",
            "not_evaluated_or_missing", "coverage_rate", "pass_rate", "non_pass_rate",
        ])
        for category in sorted(requested):
            counts = category_outcomes[category]
            evaluated = sum(counts.get(value, 0) for value in ("pass", "warning", "fail"))
            nonpass = counts.get("warning", 0) + counts.get("fail", 0)
            outcome_sheet.append([
                category, requested[category], evaluated, counts.get("pass", 0),
                counts.get("warning", 0), counts.get("fail", 0), requested[category] - evaluated,
                evaluated / requested[category], counts.get("pass", 0) / evaluated if evaluated else None,
                nonpass / evaluated if evaluated else None,
            ])
            for column in (8, 9, 10):
                outcome_sheet.cell(outcome_sheet.max_row, column).number_format = "0.0%"

        rule_sheet.append([
            "rule_id", "evaluated", "pass", "warning", "fail", "not_evaluated_or_missing",
            "pass_rate", "non_pass_rate",
        ])
        for rule in sorted(rule_outcomes):
            counts = rule_outcomes[rule]
            evaluated = sum(counts.get(value, 0) for value in ("pass", "warning", "fail"))
            nonpass = counts.get("warning", 0) + counts.get("fail", 0)
            rule_sheet.append([
                rule, evaluated, counts.get("pass", 0), counts.get("warning", 0),
                counts.get("fail", 0), sum(counts.values()) - evaluated,
                counts.get("pass", 0) / evaluated if evaluated else None,
                nonpass / evaluated if evaluated else None,
            ])
            rule_sheet.cell(rule_sheet.max_row, 7).number_format = "0.0%"
            rule_sheet.cell(rule_sheet.max_row, 8).number_format = "0.0%"

        risk_values = sorted({str(record.get("risk_level") or "not_applicable") for record in records})
        intervention_values = sorted({str(record.get("manual_intervention_label") or "not_applicable") for record in records})
        cross_sheet.append(["risk_level", *intervention_values, "row_total"])
        for risk in risk_values:
            row = [risk]
            for intervention in intervention_values:
                row.append(sum(
                    str(record.get("risk_level") or "not_applicable") == risk
                    and str(record.get("manual_intervention_label") or "not_applicable") == intervention
                    for record in records
                ))
            cross_sheet.append([*row, sum(row[1:])])

        split_sheet.append([
            "split", "master_samples", "scenario_count", "session_count", "quality_pass",
            "needs_review", "sft_eligible", "sft_eligible_rate", "average_evidence_items",
            "average_answer_chars",
        ])
        for split in ("train", "valid", "test"):
            group = [record for record in records if record.get("split") == split]
            results = [by_id[str(record.get("sample_id"))] for record in group]
            passed = sum(item.get("task1_quality_flag") == "pass" for item in results)
            split_eligible = sum(Task1StatisticsWorkbook._sft_eligible(record, by_id) for record in group)
            split_sheet.append([
                split, len(group), len({str(item.get("scenario_id") or "") for item in group}),
                len({str(item.get("session_id") or "") for item in group}), passed, len(group) - passed,
                split_eligible, split_eligible / len(group) if group else 0.0,
                mean(item.get("evidence_item_count", 0) for item in results) if results else 0.0,
                mean(len(str(item.get("final_answer") or "")) for item in group) if group else 0.0,
            ])
            split_sheet.cell(split_sheet.max_row, 8).number_format = "0.0%"

        evidence_sheet.append([
            "dataset_source", "scenario_type", "sample_count", "zero_evidence_samples",
            "minimum", "p25", "median", "mean", "p75", "p90", "p95", "maximum",
        ])
        evidence_groups: Dict[tuple[str, str], List[int]] = defaultdict(list)
        for record in records:
            evidence_groups[(
                str(record.get("dataset_source") or "unknown"),
                str(record.get("scenario_type") or "unknown"),
            )].append(TeacherTraceQualityAuditor.evidence_item_count(record.get("evidence") or {}))
        for key, values in sorted(evidence_groups.items()):
            evidence_sheet.append([
                *key, len(values), sum(value == 0 for value in values), min(values),
                Task1StatisticsWorkbook._percentile(values, 0.25), median(values), mean(values),
                Task1StatisticsWorkbook._percentile(values, 0.75),
                Task1StatisticsWorkbook._percentile(values, 0.90),
                Task1StatisticsWorkbook._percentile(values, 0.95), max(values),
            ])

        check_names = ("schema", "numerical_consistency", "rule_consistency", "dispatch_consistency")
        quality_sheet.append([
            "quality_dimension", "pass", "fail_or_needs_review", "not_applicable",
            "applicable_count", "applicable_pass_rate",
        ])
        for name in check_names:
            counts = Counter(item["checks"][name]["status"] for item in evaluations)
            applicable = counts.get("pass", 0) + counts.get("fail", 0)
            quality_sheet.append([
                name, counts.get("pass", 0), counts.get("fail", 0),
                counts.get("not_applicable", 0), applicable,
                counts.get("pass", 0) / applicable if applicable else None,
            ])
            quality_sheet.cell(quality_sheet.max_row, 6).number_format = "0.0%"
        for name, distribution in (
            ("native_quality", statistics["native_quality_distribution"]),
            ("task1_quality", statistics["task1_quality_distribution"]),
        ):
            applicable = sum(distribution.values())
            quality_sheet.append([
                name, distribution.get("pass", 0), distribution.get("needs_review", 0),
                0, applicable, distribution.get("pass", 0) / applicable if applicable else None,
            ])
            quality_sheet.cell(quality_sheet.max_row, 6).number_format = "0.0%"

        notes.append(["topic", "definition"])
        notes.append(["Purpose", "Standalone Task 1.9 data-statistics tables for the master teacher-trace dataset."])
        notes.append(["SFT eligible", "Task 1 pass, no explicit SFT exclusion, and all preceding conversation turns pass quality."])
        notes.append(["Evidence items", "Recursive count of non-empty scalar/list items in the top-level evidence object."])
        notes.append(["Scenario types", "Scenario-family distribution across all teacher-trace records."])
        notes.append(["Task types", "PipeFormer task-subtype distribution across PipeFormer scenario records only; unspecified means no subtype was stored."])
        notes.append(["Constraint requested", "A category listed in constraint_check.requested_categories."])
        notes.append(["Constraint evaluated", "Requested category with pass, warning, or fail in category_status."])
        notes.append(["Percentiles", "Nearest-rank interpolation over per-record evidence-item counts."])
        notes.append(["Workbook formulas", "Fixed verification snapshot; no formulas or external links are used."])

        polish_workbook(
            workbook,
            ILLEGAL_CHARACTERS_RE,
            summary_sheets={"Overview"},
        )
        source_sheet.freeze_panes = "B2"
        for cell in source_sheet["A"][1:]:
            cell.font = Font(name="Arial", size=10, bold=True, color="1F1F1F")
            cell.fill = PatternFill("solid", fgColor="D9EAF7")
        if source_sheet.max_column > 1:
            chart = BarChart()
            chart.type = "bar"
            chart.style = 10
            chart.title = "Samples by Dataset Source"
            chart.y_axis.title = "dataset_source"
            chart.x_axis.title = "sample_count"
            chart.add_data(
                Reference(source_sheet, min_col=1, max_col=source_sheet.max_column, min_row=2, max_row=2),
                titles_from_data=True,
                from_rows=True,
            )
            chart.set_categories(Reference(source_sheet, min_col=2, max_col=source_sheet.max_column, min_row=1))
            style_chart(chart, ("4472C4",))
            source_sheet.add_chart(chart, "F2")
        Task1StatisticsWorkbook._add_bar_chart(scenario_sheet, BarChart, Reference, "Scenario-Type Distribution", 2, "F2")
        Task1StatisticsWorkbook._add_bar_chart(task_sheet, BarChart, Reference, "PipeFormer Task-Type Distribution", 2, "F2")
        Task1StatisticsWorkbook._add_bar_chart(constraint_sheet, BarChart, Reference, "Constraint-Type Distribution", 2, "F2")
        if risk_sheet.max_row > 1:
            chart = PieChart()
            chart.title = "Risk-Level Distribution"
            chart.add_data(Reference(risk_sheet, min_col=2, min_row=1, max_row=risk_sheet.max_row), titles_from_data=True)
            chart.set_categories(Reference(risk_sheet, min_col=1, min_row=2, max_row=risk_sheet.max_row))
            style_chart(
                chart,
                ("4472C4", "70AD47", "FFC000", "ED7D31", "C00000"),
            )
            risk_sheet.add_chart(chart, "F2")
        workbook.save(path)

    @staticmethod
    def _sft_eligible(record: Dict[str, Any], by_id: Mapping[str, Dict[str, Any]]) -> bool:
        result = by_id.get(str(record.get("sample_id"))) or {}
        return bool(
            result.get("task1_quality_flag") == "pass"
            and not record.get("sft_exclusion_reason")
            and all(
                item.get("quality_flag") == "pass"
                for item in record.get("conversation_context") or []
            )
        )

    @staticmethod
    def _groups(records: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record.get(key) or "unknown")].append(record)
        return groups

    @staticmethod
    def _distribution_sheet(sheet: Any, values: Mapping[str, int], label: str) -> None:
        total = sum(values.values()) or 1
        sheet.append([label, "count", "percentage"])
        for value, count in sorted(values.items()):
            sheet.append([value, count, count / total])
            sheet.cell(sheet.max_row, 3).number_format = "0.0%"

    @staticmethod
    def _percentile(values: Sequence[int], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    @staticmethod
    def _add_bar_chart(sheet: Any, chart_cls: Any, reference_cls: Any, title: str, value_column: int, anchor: str) -> None:
        if sheet.max_row < 2:
            return
        chart = chart_cls()
        chart.type = "bar"
        chart.style = 10
        chart.title = title
        chart.y_axis.title = sheet.cell(1, 1).value
        chart.x_axis.title = sheet.cell(1, value_column).value
        chart.add_data(reference_cls(sheet, min_col=value_column, min_row=1, max_row=sheet.max_row), titles_from_data=True)
        chart.set_categories(reference_cls(sheet, min_col=1, min_row=2, max_row=sheet.max_row))
        chart.height = 7
        chart.width = 12
        from .workbook_style import style_chart

        style_chart(chart, ("4472C4",))
        sheet.add_chart(chart, anchor)
