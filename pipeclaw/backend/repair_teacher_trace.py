from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from evaluator.deterministic_repairs import (
    CONDITIONAL_EVIDENCE_SAMPLE_ID,
    DETERMINISTIC_REPAIR_SAMPLE_IDS,
    apply_deterministic_repairs,
    apply_staged_answer_repairs,
    update_session_records,
)
from evaluator.csv_evidence import build_csv_evidence
from evaluator.reviewer_annotations import (
    export_reviewer_annotations,
    load_reviewer_annotations,
)
from evaluator.scorer import NativeEvaluationConfig, NativeTraceEvaluator
from evaluator.task1 import Task1QualityVerifier, Task1VerificationConfig
from evaluator.tool_evidence import (
    ToolEvidenceState,
    classify_tool_evidence,
)
from generate_teacher_trace import default_scenario_files, load_scenario_sources, write_split_records
from pipeline.io_utils import write_json
from pipeline.teacher_trace_store import TeacherTracePaths, TeacherTraceStore


BACKEND_ROOT = Path(__file__).resolve().parent
GENERATED_ROOT = BACKEND_ROOT / "generated_teacher_traces"
DELIVERABLE_ROOT = GENERATED_ROOT / "task1_deliverables"
DEFAULT_MASTER_JSON = GENERATED_ROOT / "teacher_trace.json"
DEFAULT_MASTER_JSONL = GENERATED_ROOT / "teacher_trace.jsonl"
DEFAULT_SESSIONS = GENERATED_ROOT / "teacher_trace_sessions.jsonl"
DEFAULT_SPLITS = GENERATED_ROOT / "splits"
DEFAULT_REPORT = DELIVERABLE_ROOT / "teacher_trace_quality_report.xlsx"
DEFAULT_ANNOTATIONS = DELIVERABLE_ROOT / "manual_quality_decisions.jsonl"
DEFAULT_RESET_IDS = DELIVERABLE_ROOT / "repaired_sample_ids.json"
DEFAULT_STAGING = GENERATED_ROOT / "repair_staging"


REGENERATION_TARGETS = (
    # ("pipeclaw_dataset_v2", "scenario_openclaw_013"), 
    # ("pipeclaw_dataset_v2", "scenario_openclaw_014"), 
    # ("pipeclaw_dataset_v2", "scenario_openclaw_015"), 
    # ("pipeclaw_dataset_v2", "scenario_openclaw_016"),
    # ("pipeclaw_dataset_v2", "scenario_openclaw_021"),
    # ("pipeclaw_dataset_v2", "scenario_openclaw_022"),
    # ("pipeclaw_dataset_v2", "scenario_openclaw_023"),
    # ("pipeclaw_dataset_v2", "scenario_openclaw_024"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_003"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_004"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_005"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_007"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_008"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_009"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_010"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_011"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_012"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_013"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_014"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_015"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_019"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_prediction_012"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_prediction_015"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_004"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_008"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_011"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_012"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_014"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_015"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_020"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_prediction_012"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_prediction_015"),
    # ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_prediction_016"),
)
CONDITIONAL_REGENERATION_TARGET = (
    "pipeclaw_dataset_v2",
    "scenario_openclaw_006",
)
AMBIGUOUS_RELEVANCE_TARGETS = {
    ("pipeclaw_dataset_v2", f"scenario_openclaw_{number:03d}")
    for number in range(21, 25)
}


@dataclass(frozen=True)
class AttemptPaths:
    root: Path
    records_json: Path
    records_jsonl: Path
    sessions_jsonl: Path
    splits: Path
    preflight: Path
    evaluation_jsonl: Path
    manifest: Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair teacher traces without regex-based tool routing."
    )
    parser.add_argument("--export-annotations", action="store_true")
    parser.add_argument("--apply-deterministic", action="store_true")
    parser.add_argument("--stage-regeneration", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenarios that already have complete generated record and session outputs.",
    )
    parser.add_argument("--merge-approved", type=Path)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help=(
            "Concurrent scenario subprocesses. Start with 2; use 1 for "
            "constrained GPU memory or provider capacity."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--master-json", type=Path, default=DEFAULT_MASTER_JSON)
    parser.add_argument("--master-jsonl", type=Path, default=DEFAULT_MASTER_JSONL)
    parser.add_argument("--sessions-jsonl", type=Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--report-xlsx", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reviewer-annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--reset-review-sample-ids", type=Path, default=DEFAULT_RESET_IDS)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not any((args.export_annotations, args.apply_deterministic, args.stage_regeneration, args.merge_approved)):
        raise ValueError("Choose at least one repair action.")
    if args.attempts < 1 or args.attempts > 3:
        raise ValueError("--attempts must be between 1 and 3.")
    results: Dict[str, Any] = {}
    if args.export_annotations:
        results["annotation_export"] = export_reviewer_annotations(
            args.report_xlsx, args.reviewer_annotations
        )
    results["assignment_validation"] = _validate_failed_record_assignments(args)
    if args.apply_deterministic:
        results["deterministic"] = _apply_deterministic(args)
    if args.stage_regeneration:
        results["regeneration"] = _stage_regeneration(args)
    if args.merge_approved:
        results["merge"] = _merge_approved(args, args.merge_approved)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _store(args: argparse.Namespace) -> TeacherTraceStore:
    return TeacherTraceStore(
        TeacherTracePaths(
            output_jsonl=args.master_jsonl,
            output_json=args.master_json,
            session_output_jsonl=args.sessions_jsonl,
        )
    )


def _apply_deterministic(args: argparse.Namespace) -> Dict[str, Any]:
    store = _store(args)
    before = store.load_master()
    before_ids = [str(item.get("sample_id") or "") for item in before]
    before_splits = {str(item.get("sample_id") or ""): item.get("split") for item in before}
    repaired, result = apply_deterministic_repairs(before)
    _assert_master_invariants(before_ids, before_splits, repaired)
    sessions = update_session_records(store.load_sessions(), repaired)
    store.write_master(repaired)
    store.write_sessions(sessions)
    write_split_records(args.split_dir, repaired, force=True)
    write_json(
        args.reset_review_sample_ids,
        {"sample_ids": result["repaired_sample_ids"]},
        force=True,
    )
    return result


def _generation_start_message(
    *,
    position: int,
    total: int,
    dataset_source: str,
    scenario_id: str,
    attempt: int,
    attempt_count: int,
    output_root: Path,
) -> str:
    return (
        f"[{position}/{total}] START {dataset_source}:{scenario_id} "
        f"attempt {attempt}/{attempt_count} "
        f"output={Path(output_root).as_posix()}"
    )


def _generation_finish_message(
    *,
    position: int,
    total: int,
    dataset_source: str,
    scenario_id: str,
    attempt: int,
    command_exit_code: int,
    automatic_pass: bool,
    elapsed_seconds: float,
    manifest_path: Path,
) -> str:
    status = "DONE" if automatic_pass else "FAILED"
    return (
        f"[{position}/{total}] {status} {dataset_source}:{scenario_id} "
        f"attempt {attempt} exit_code={command_exit_code} "
        f"automatic_pass={str(bool(automatic_pass)).lower()} "
        f"elapsed={elapsed_seconds:.1f}s "
        f"manifest={Path(manifest_path).as_posix()}"
    )


def _stage_scenario(
    args: argparse.Namespace,
    annotations: Dict[tuple[str, str], Dict[str, Any]],
    dataset_source: str,
    scenario_id: str,
    position: int,
    target_count: int,
) -> Dict[str, Any]:
    if args.resume:
        resumed = _generated_staged_scenario(
            args.staging_root,
            dataset_source,
            scenario_id,
        )
        if resumed is not None:
            print(
                "Skipping already generated staged scenario: "
                f"{dataset_source}:{scenario_id} "
                f"attempt={resumed['candidate_attempt']}",
                flush=True,
            )
            return resumed
    scenario_summary = {
        "dataset_source": dataset_source,
        "scenario_id": scenario_id,
        "status": "unresolved",
        "attempts": [],
    }
    for attempt in range(1, args.attempts + 1):
        paths = _attempt_paths(args.staging_root, dataset_source, scenario_id, attempt)
        _prepare_attempt_outputs(paths)
        command = _generation_command(
            args,
            dataset_source,
            scenario_id,
            paths,
        )
        print(
            _generation_start_message(
                position=position,
                total=target_count,
                dataset_source=dataset_source,
                scenario_id=scenario_id,
                attempt=attempt,
                attempt_count=args.attempts,
                output_root=paths.root,
            ),
            flush=True,
        )
        started_at = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=BACKEND_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        attempt_result: Dict[str, Any] = {
            "attempt": attempt,
            "command_exit_code": completed.returncode,
            "stdout_tail": completed.stdout[-4_000:],
            "stderr_tail": completed.stderr[-4_000:],
            "root": paths.root.as_posix(),
        }
        if completed.returncode == 0:
            attempt_result.update(
                _evaluate_attempt(dataset_source, scenario_id, paths)
            )
        else:
            attempt_result["automatic_pass"] = False
            attempt_result["automatic_failures"] = ["generation_command_failed"]
        attempt_result["manual_review_required"] = True
        attempt_result["manual_review_focus"] = (
            "Confirm the chosen tool is relevant and the answer is grounded; a successful but irrelevant PipeFormer forecast is unacceptable."
            if (dataset_source, scenario_id) in AMBIGUOUS_RELEVANCE_TARGETS
            else "Review every regenerated turn against the original reviewer notes, including turns that previously passed."
        )
        attempt_result["original_reviewer_notes"] = _scenario_reviewer_notes(
            dataset_source,
            scenario_id,
            annotations,
        )
        attempt_result["required_human_checks"] = [
            "all_turns_reviewed",
            "original_reviewer_notes_satisfied",
            "chosen_tool_is_relevant",
            "answer_is_grounded",
        ]
        paths.root.mkdir(parents=True, exist_ok=True)
        write_json(paths.manifest, attempt_result, force=True)
        print(
            _generation_finish_message(
                position=position,
                total=target_count,
                dataset_source=dataset_source,
                scenario_id=scenario_id,
                attempt=attempt,
                command_exit_code=completed.returncode,
                automatic_pass=attempt_result.get("automatic_pass") is True,
                elapsed_seconds=time.monotonic() - started_at,
                manifest_path=paths.manifest,
            ),
            flush=True,
        )
        scenario_summary["attempts"].append(attempt_result)
        if attempt_result.get("automatic_pass"):
            scenario_summary["status"] = "pending_human_review"
            scenario_summary["candidate_attempt"] = attempt
            break
    return scenario_summary


def _run_regeneration_targets(
    args: argparse.Namespace,
    annotations: Dict[tuple[str, str], Dict[str, Any]],
    targets: Sequence[tuple[str, str]],
) -> List[Dict[str, Any]]:
    target_count = len(targets)
    jobs = [
        (
            args,
            annotations,
            dataset_source,
            scenario_id,
            position,
            target_count,
        )
        for position, (dataset_source, scenario_id) in enumerate(targets, 1)
    ]
    if not jobs:
        return []
    workers = getattr(args, "workers", 1)
    if workers == 1:
        return [_stage_scenario(*job) for job in jobs]

    def run_job(job: tuple[Any, ...]) -> Dict[str, Any]:
        return _stage_scenario(*job)

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        return list(executor.map(run_job, jobs))


def _stage_regeneration(args: argparse.Namespace) -> Dict[str, Any]:
    args.staging_root.mkdir(parents=True, exist_ok=True)
    annotations = load_reviewer_annotations(args.reviewer_annotations)
    targets = _planned_regeneration_targets(args)
    summaries = _run_regeneration_targets(args, annotations, targets)
    summary_path = args.staging_root / "staging_summary.json"
    previous = (
        json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if summary_path.is_file()
        else {}
    )
    summary = _merge_staging_summary(previous, summaries)
    write_json(summary_path, summary, force=True)
    return summary


def _merge_staging_summary(
    previous: Dict[str, Any],
    updates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    scenarios = [
        dict(item)
        for item in previous.get("scenarios") or []
        if isinstance(item, dict)
    ]
    positions = {
        (
            str(item.get("dataset_source") or ""),
            str(item.get("scenario_id") or ""),
        ): index
        for index, item in enumerate(scenarios)
    }
    for update in updates:
        item = dict(update)
        key = (
            str(item.get("dataset_source") or ""),
            str(item.get("scenario_id") or ""),
        )
        if key in positions:
            scenarios[positions[key]] = item
        else:
            positions[key] = len(scenarios)
            scenarios.append(item)
    return {
        "schema_version": "teacher_trace_repair_staging_v1",
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }


def _generated_staged_scenario(
    staging_root: Path,
    dataset_source: str,
    scenario_id: str,
) -> Dict[str, Any] | None:
    scenario_root = (
        staging_root
        / dataset_source.replace("/", "_").replace("\\", "_")
        / scenario_id
    )
    for attempt_root in sorted(scenario_root.glob("attempt_*")):
        attempt = int(attempt_root.name.rsplit("_", 1)[-1])
        paths = _attempt_paths(staging_root, dataset_source, scenario_id, attempt)
        generated_outputs = (
            paths.records_json,
            paths.records_jsonl,
            paths.sessions_jsonl,
        )
        if not all(path.is_file() and path.stat().st_size > 0 for path in generated_outputs):
            continue
        manifest = (
            json.loads(paths.manifest.read_text(encoding="utf-8-sig"))
            if paths.manifest.is_file()
            else None
        )
        if manifest is None:
            status = "generated_unevaluated"
        elif manifest.get("automatic_pass") is True:
            status = "pending_human_review"
        else:
            status = "generated_needs_review"
        return {
            "dataset_source": dataset_source,
            "scenario_id": scenario_id,
            "status": status,
            "candidate_attempt": attempt,
            "attempts": [manifest] if manifest is not None else [],
            "resume_action": "skipped_generated_outputs",
        }
    return None


def _planned_regeneration_targets(args: argparse.Namespace) -> List[tuple[str, str]]:
    targets = list(REGENERATION_TARGETS)
    records = _store(args).load_master()
    conditional = next(
        (
            item
            for item in records
            if str(item.get("sample_id") or "") == CONDITIONAL_EVIDENCE_SAMPLE_ID
        ),
        None,
    )
    if conditional:
        evidence = build_csv_evidence(
            conditional.get("tool_calls") or [],
            conditional.get("tool_outputs") or [],
            str(conditional.get("final_answer") or ""),
            scope_text=str(conditional.get("user_input") or ""),
        )
        recoverable = (
            len(evidence.get("answer_rows") or []) == 12
            and len(evidence.get("derived_results") or []) >= 3
        )
        if not recoverable and CONDITIONAL_REGENERATION_TARGET not in targets:
            targets.append(CONDITIONAL_REGENERATION_TARGET)
    return targets


def _scenario_reviewer_notes(
    dataset_source: str,
    scenario_id: str,
    annotations: Dict[tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prefix = f"{dataset_source}:{scenario_id}_"
    notes = []
    for (_, sample_id), item in sorted(annotations.items()):
        if not sample_id.startswith(prefix):
            continue
        review = dict(item.get("review") or {})
        if not review.get("reviewer_notes"):
            continue
        notes.append({
            "sheet": item.get("sheet"),
            "sample_id": sample_id,
            "reviewer_notes": review.get("reviewer_notes"),
            "final_disposition": review.get("final_disposition"),
        })
    return notes


def _evaluate_attempt(
    dataset_source: str,
    scenario_id: str,
    paths: AttemptPaths,
) -> Dict[str, Any]:
    records = TeacherTraceStore.load(paths.records_jsonl)
    sessions = TeacherTraceStore.load(paths.sessions_jsonl)
    records, staged_repairs = apply_staged_answer_repairs(records)
    if staged_repairs["repaired_record_count"]:
        staging_store = TeacherTraceStore(
            TeacherTracePaths(
                output_jsonl=paths.records_jsonl,
                output_json=paths.records_json,
                session_output_jsonl=paths.sessions_jsonl,
            )
        )
        sessions = update_session_records(sessions, records)
        staging_store.write_master(records)
        staging_store.write_sessions(sessions)
        write_split_records(paths.splits, records, force=True)
    native_evaluator = NativeTraceEvaluator(NativeEvaluationConfig())
    task1 = Task1QualityVerifier(Task1VerificationConfig())
    evaluations = []
    failures = []
    for record in records:
        native = native_evaluator.evaluate(record, trace_status=record.get("trace_status"))
        task = task1.evaluate(record, native)
        evaluations.append({**task, "native": native})
        sample_id = str(record.get("sample_id") or "")
        if native.get("quality_flag") != "pass":
            failures.append(f"native_quality:{sample_id}")
        if task.get("task1_quality_flag") != "pass":
            failures.append(f"task1_quality:{sample_id}")
        failed_tools = _execution_failed_tool_names(record)
        if failed_tools:
            failures.append(f"failed_tools:{sample_id}:{','.join(failed_tools)}")
        failures.extend(_parsing_and_verification_failures(record))
    if not sessions or any(not bool(item.get("complete")) for item in sessions):
        failures.append("incomplete_session")
    expected_ids = _expected_sample_ids(dataset_source, scenario_id)
    actual_ids = {str(item.get("sample_id") or "") for item in records}
    if actual_ids != expected_ids:
        failures.append("incomplete_or_foreign_sample_id_set")
    with paths.evaluation_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for item in evaluations:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {
        "record_count": len(records),
        "session_count": len(sessions),
        "automatic_pass": not failures,
        "automatic_failures": list(dict.fromkeys(failures)),
        "deterministic_repairs": staged_repairs,
    }


def _execution_failed_tool_names(record: Dict[str, Any]) -> List[str]:
    """Return tools that failed to execute, not successful no-evidence calls."""
    return [
        str(item.get("name") or "unknown")
        for item in record.get("tool_outputs") or []
        if classify_tool_evidence(item).state
        is ToolEvidenceState.EXECUTION_FAILED
    ]


def _parsing_and_verification_failures(record: Dict[str, Any]) -> List[str]:
    sample_id = str(record.get("sample_id") or "")
    failures = []
    for item in record.get("tool_outputs") or []:
        output = dict(item.get("output") or {})
        resolution = dict(output.get("task_resolution") or {})
        if any(resolution.get(key) for key in (
            "unresolved_attention_targets",
            "unresolved_output_state_variables",
            "invalid_normalized_variables",
        )):
            failures.append(f"unresolved_parsing:{sample_id}")
        if item.get("name") == "run_pipeformer_forecast" and output.get("success") is True:
            verification = dict(output.get("verification") or output.get("constraint_check") or {})
            if verification.get("verification_complete") is not True:
                failures.append(f"incomplete_verification:{sample_id}")
            for evidence in dict(output.get("evidence") or {}).get("boundary_application_evidence") or []:
                if evidence.get("verified") is not True:
                    failures.append(f"unverified_boundary_application:{sample_id}")
    return failures


def _merge_approved(args: argparse.Namespace, approval_path: Path) -> Dict[str, Any]:
    approvals = _load_approvals(approval_path)
    store = _store(args)
    original = store.load_master()
    original_sessions = store.load_sessions()
    original_ids = [str(item.get("sample_id") or "") for item in original]
    original_splits = {str(item.get("sample_id") or ""): item.get("split") for item in original}
    combined = original
    combined_sessions = original_sessions
    reset_ids = set(_load_reset_ids(args.reset_review_sample_ids))
    merged = []
    for approval in approvals:
        dataset_source = str(approval["dataset_source"])
        scenario_id = str(approval["scenario_id"])
        attempt = int(approval["attempt"])
        if (dataset_source, scenario_id) not in set(REGENERATION_TARGETS) | {CONDITIONAL_REGENERATION_TARGET}:
            raise ValueError(f"Approval contains an unplanned scenario: {dataset_source}:{scenario_id}")
        paths = _attempt_paths(args.staging_root, dataset_source, scenario_id, attempt)
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8-sig"))
        if manifest.get("automatic_pass") is not True:
            raise ValueError(f"Attempt did not pass automatic checks: {paths.manifest}")
        generated = TeacherTraceStore.load(paths.records_jsonl)
        generated_sessions = TeacherTraceStore.load(paths.sessions_jsonl)
        for record in generated:
            prior_split = original_splits.get(str(record.get("sample_id") or ""))
            if prior_split != record.get("split"):
                raise ValueError(f"Split changed for {record.get('sample_id')}")
        combined, _ = store.replace_scenario(
            combined,
            generated,
            dataset_source=dataset_source,
            scenario_id=scenario_id,
            id_field="sample_id",
        )
        combined_sessions, _ = store.replace_scenario(
            combined_sessions,
            generated_sessions,
            dataset_source=dataset_source,
            scenario_id=scenario_id,
            id_field="session_record_id",
        )
        reset_ids.update(str(item.get("sample_id") or "") for item in generated)
        merged.append({"dataset_source": dataset_source, "scenario_id": scenario_id, "attempt": attempt})
    _assert_master_invariants(original_ids, original_splits, combined)
    store.write_master(combined)
    store.write_sessions(combined_sessions)
    write_split_records(args.split_dir, combined, force=True)
    write_json(args.reset_review_sample_ids, {"sample_ids": sorted(reset_ids)}, force=True)
    return {"merged_scenario_count": len(merged), "merged": merged}


def _generation_command(
    args: argparse.Namespace,
    dataset_source: str,
    scenario_id: str,
    paths: AttemptPaths,
) -> List[str]:
    paths.root.mkdir(parents=True, exist_ok=True)
    return [
        str(args.python_executable),
        str(BACKEND_ROOT / "generate_teacher_trace.py"),
        "--dataset-source", dataset_source,
        "--scenario-id", scenario_id,
        "--device", str(args.device),
        "--output-jsonl", str(paths.records_jsonl),
        "--output-json", str(paths.records_json),
        "--session-output-jsonl", str(paths.sessions_jsonl),
        "--split-output-dir", str(paths.splits),
        "--preflight-output", str(paths.preflight),
        "--force",
    ]


def _prepare_attempt_outputs(paths: AttemptPaths) -> None:
    """Start an intentional rerun without reusing files from an older run."""
    paths.root.mkdir(parents=True, exist_ok=True)
    managed_paths = (
        paths.records_json,
        paths.records_jsonl,
        paths.sessions_jsonl,
        paths.preflight,
        paths.evaluation_jsonl,
        paths.manifest,
        paths.splits,
    )
    resolved_root = paths.root.resolve()
    for path in managed_paths:
        try:
            relative = path.resolve().relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"Refusing to remove attempt output outside attempt root: {path}"
            ) from exc
        if relative == Path("."):
            raise ValueError(f"Refusing to remove the attempt root itself: {path}")

    for path in managed_paths[:-1]:
        if path.is_file():
            path.unlink()
    if paths.splits.is_dir():
        shutil.rmtree(paths.splits)


def _attempt_paths(
    staging_root: Path,
    dataset_source: str,
    scenario_id: str,
    attempt: int,
) -> AttemptPaths:
    safe_dataset = dataset_source.replace("/", "_").replace("\\", "_")
    root = staging_root / safe_dataset / scenario_id / f"attempt_{attempt:02d}"
    return AttemptPaths(
        root=root,
        records_json=root / "teacher_trace.json",
        records_jsonl=root / "teacher_trace.jsonl",
        sessions_jsonl=root / "teacher_trace_sessions.jsonl",
        splits=root / "splits",
        preflight=root / "scenario_preflight.json",
        evaluation_jsonl=root / "quality_evaluation.jsonl",
        manifest=root / "attempt_manifest.json",
    )


def _expected_sample_ids(dataset_source: str, scenario_id: str) -> set[str]:
    for source in load_scenario_sources(default_scenario_files()):
        if source["dataset_source"] != dataset_source:
            continue
        for scenario in source.get("scenarios") or []:
            if scenario.get("scenario_id") == scenario_id:
                return set(TeacherTraceStore.sample_ids(scenario))
    raise ValueError(f"Scenario source not found: {dataset_source}:{scenario_id}")


def _validate_failed_record_assignments(args: argparse.Namespace) -> Dict[str, Any]:
    annotations = load_reviewer_annotations(args.reviewer_annotations)
    failed_ids = {
        sample_id
        for (_, sample_id), item in annotations.items()
        if str(dict(item.get("review") or {}).get("final_disposition") or "").casefold() == "failed"
    }
    if not failed_ids:
        return {"failed_record_count": 0, "assignment_counts": {}}
    records = {str(item.get("sample_id") or ""): item for item in _store(args).load_master()}
    assignments: Dict[str, str] = {}
    for sample_id in failed_ids:
        if sample_id in DETERMINISTIC_REPAIR_SAMPLE_IDS:
            assignments[sample_id] = "deterministic_answer_repair"
        elif sample_id == CONDITIONAL_EVIDENCE_SAMPLE_ID:
            assignments[sample_id] = "conditional_evidence_recovery"
        else:
            record = records.get(sample_id) or {}
            pair = (str(record.get("dataset_source") or ""), str(record.get("scenario_id") or ""))
            if pair in REGENERATION_TARGETS:
                assignments[sample_id] = "scenario_regeneration"
    unassigned = sorted(failed_ids - set(assignments))
    if len(failed_ids) != 37:
        raise ValueError(f"Expected 37 unique failed reviewer records, found {len(failed_ids)}")
    assignment_counts: Dict[str, int] = {}
    for repair_path in assignments.values():
        assignment_counts[repair_path] = assignment_counts.get(repair_path, 0) + 1
    return {
        "failed_record_count": len(failed_ids),
        "assignment_counts": assignment_counts,
        "unassigned_record_count": len(unassigned),
        "unassigned_sample_ids": unassigned,
        "unassigned_warning": (
            "These reviewer-failed records are not selected by the current "
            "REGENERATION_TARGETS subset."
            if unassigned
            else None
        ),
    }


def _assert_master_invariants(
    expected_ids: Sequence[str],
    expected_splits: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
) -> None:
    actual_ids = [str(item.get("sample_id") or "") for item in records]
    if len(actual_ids) != 1_140:
        raise ValueError(f"Teacher-trace record count changed: {len(actual_ids)} != 1140")
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Teacher-trace sample-ID set changed or contains duplicates.")
    for record in records:
        sample_id = str(record.get("sample_id") or "")
        if expected_splits.get(sample_id) != record.get("split"):
            raise ValueError(f"Split assignment changed for {sample_id}")


def _load_approvals(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    approvals = value.get("approved") if isinstance(value, dict) else value
    if not isinstance(approvals, list):
        raise ValueError("Approval file must be a list or contain an 'approved' list.")
    required = {
        "dataset_source",
        "scenario_id",
        "attempt",
        "all_turns_reviewed",
        "original_reviewer_notes_satisfied",
        "chosen_tool_is_relevant",
        "answer_is_grounded",
    }
    for item in approvals:
        if not isinstance(item, dict) or not required <= set(item):
            raise ValueError(f"Invalid approval entry: {item}")
        if any(item.get(key) is not True for key in required - {"dataset_source", "scenario_id", "attempt"}):
            raise ValueError(f"Approval lacks required human sign-off: {item}")
    return approvals


def _load_reset_ids(path: Path) -> Iterable[str]:
    if not Path(path).is_file():
        return []
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return value.get("sample_ids") or []


if __name__ == "__main__":
    raise SystemExit(main())
