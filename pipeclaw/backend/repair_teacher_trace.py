from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from copy import deepcopy
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
from evaluator.decision_trace_state import (
    DecisionTraceState,
    bounded_recent_turns,
    serialize_verified_decision_state,
)
from evaluator.grounding_contract import (
    GroundingContractBuilder,
    finalize_applied_disturbance_disclosure,
)
from evaluator.reviewer_annotations import (
    export_reviewer_annotations,
    load_reviewer_annotations,
)
from evaluator.scorer import NativeEvaluationConfig, NativeTraceEvaluator
from evaluator.task1 import Task1QualityVerifier, Task1VerificationConfig
from evaluator.tool_evidence import (
    ToolEvidenceState,
    attach_tool_arguments,
    classify_tool_evidence,
)
from generate_teacher_trace import (
    _history_turn,
    default_scenario_files,
    load_scenario_sources,
    write_split_records,
)
from pipeline.forecast_registry_contract import authorize_forecast_registry
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
    ("pipeclaw_dataset_v2", "scenario_openclaw_013"),
    ("pipeclaw_dataset_v2", "scenario_openclaw_014"),
    ("pipeclaw_dataset_v2", "scenario_openclaw_015"),
    ("pipeclaw_dataset_v2", "scenario_openclaw_016"),
    ("pipeclaw_dataset_v2", "scenario_openclaw_021"),
    ("pipeclaw_dataset_v2", "scenario_openclaw_022"),
    ("pipeclaw_dataset_v2", "scenario_openclaw_023"),
    ("pipeclaw_dataset_v2", "scenario_openclaw_024"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_003"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_004"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_005"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_007"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_008"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_009"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_010"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_011"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_012"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_013"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_014"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_015"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_dispatch_019"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_prediction_012"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v4", "scenario_pipeformer_prediction_015"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_004"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_008"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_011"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_012"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_014"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_015"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_dispatch_020"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_prediction_012"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_prediction_015"),
    ("Pipeline_Full_Life_Cycle_Test_Dataset-v7", "scenario_pipeformer_prediction_016"),
)
AMBIGUOUS_RELEVANCE_TARGETS = {
    ("pipeclaw_dataset_v2", f"scenario_openclaw_{number:03d}")
    for number in range(21, 25)
}
PLANNED_TARGETS_SCHEMA_VERSION = "teacher_trace_regeneration_plan_v1"
PLANNED_TARGETS_FILENAME = "planned_targets.json"
CHILD_HEARTBEAT_SECONDS = 30.0
CHILD_OUTPUT_TAIL_CHARS = 4_000
_LIVE_LOG_PRINT_LOCK = threading.Lock()


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


def _parse_target(value: str) -> tuple[str, str]:
    dataset_source, separator, scenario_id = str(value).partition("::")
    if (
        separator != "::"
        or not dataset_source.strip()
        or not scenario_id.strip()
    ):
        raise argparse.ArgumentTypeError(
            "--target must be DATASET_SOURCE::SCENARIO_ID"
        )
    return dataset_source.strip(), scenario_id.strip()


def _forecast_argument_issues(arguments: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    disturbance = str(arguments.get("disturbance_variable") or "")
    if not disturbance:
        issues.append("arguments:missing_disturbance_variable")
    if arguments.get("forecast_horizon_minutes") is None:
        issues.append("arguments:missing_forecast_horizon")
    if (
        arguments.get("case_id") in (None, "")
        and arguments.get("current_operating_condition_number") is None
    ):
        issues.append("arguments:missing_operating_case")
    boundary = dict(arguments.get("boundary_conditions") or {})
    percentage_changes = dict(boundary.get("percentage_changes") or {})
    setpoints = dict(boundary.get("setpoints") or {})
    if disturbance.endswith(":ST"):
        if arguments.get("disturbance_setpoint") not in (0, 1):
            issues.append("arguments:binary_disturbance_setpoint_invalid")
        if arguments.get("disturbance_magnitude_percent") is not None:
            issues.append("arguments:binary_disturbance_percentage_forbidden")
    for variable, value in percentage_changes.items():
        if str(variable).endswith(":ST"):
            issues.append(f"arguments:binary_percentage_forbidden:{variable}")
    for variable, value in setpoints.items():
        if str(variable).endswith(":ST") and value not in (0, 1):
            issues.append(f"arguments:binary_setpoint_invalid:{variable}")
    if disturbance and disturbance in setpoints:
        issues.append("arguments:same_variable_disturbance_action")
    if disturbance and disturbance in percentage_changes:
        issues.append("arguments:same_variable_disturbance_action")
    return issues


def _trajectory_reasons(record: Dict[str, Any]) -> List[str]:
    calls = [
        dict(item)
        for item in record.get("tool_calls") or []
        if isinstance(item, dict)
    ]
    outputs_by_id = {
        str(item.get("tool_call_id") or ""): dict(item.get("output") or {})
        for item in record.get("tool_outputs") or []
        if isinstance(item, dict)
    }
    forecast_indices = [
        index
        for index, call in enumerate(calls)
        if call.get("name") == "run_pipeformer_forecast"
    ]
    reasons: List[str] = []
    if not forecast_indices:
        return reasons
    stored_applications = [
        dict(item)
        for item in dict(record.get("grounding_contract") or {}).get(
            "applied_disturbances"
        )
        or []
        if isinstance(item, dict)
    ]
    for forecast_index in forecast_indices:
        call = calls[forecast_index]
        call_id = str(call.get("tool_call_id") or "")
        arguments = dict(call.get("arguments") or {})
        preceding = [
            {
                **prior,
                "output": outputs_by_id.get(
                    str(prior.get("tool_call_id") or "")
                ),
            }
            for prior in calls[:forecast_index]
        ]
        authorization = authorize_forecast_registry(arguments, preceding)
        for issue in authorization.get("issues") or []:
            code = str(dict(issue).get("code") or "unknown")
            reasons.append(f"registry:{code}")
        reasons.extend(_forecast_argument_issues(arguments))
        output = outputs_by_id.get(call_id)
        if (
            not isinstance(output, dict)
            or output.get("success") is not True
            or output.get("error")
            or output.get("exit_code") not in (None, 0)
        ):
            reasons.append("trajectory:forecast_execution_failed")
            continue
        resolution = dict(output.get("task_resolution") or {})
        if any(
            resolution.get(key)
            for key in (
                "unresolved_attention_targets",
                "unresolved_output_state_variables",
                "invalid_normalized_variables",
            )
        ):
            reasons.append("trajectory:invalid_forecast_resolution")
        verification = dict(
            output.get("verification")
            or output.get("constraint_check")
            or {}
        )
        if (
            verification.get("verification_complete") is not True
            or verification.get("not_evaluated_rules")
        ):
            reasons.append("trajectory:incomplete_verification")
        applications = list(
            dict(output.get("evidence") or {}).get(
                "boundary_application_evidence"
            )
            or []
        )
        stored_application_verified = any(
            str(item.get("variable") or "").casefold()
            == str(arguments.get("disturbance_variable") or "").casefold()
            and item.get("verified") is True
            for item in stored_applications
        )
        if not applications and not stored_application_verified:
            reasons.append("trajectory:application_evidence_missing")
        elif any(
            not isinstance(item, dict) or item.get("verified") is not True
            for item in applications
        ):
            reasons.append("trajectory:unverified_application")
    return list(dict.fromkeys(reasons))


def _discover_regeneration_targets(
    records: Iterable[Dict[str, Any]],
    *,
    profile: str,
) -> List[Dict[str, Any]]:
    if profile not in {"registry-contract", "reviewer"}:
        raise ValueError(f"Unsupported target profile: {profile}")
    if profile == "reviewer":
        return [
            {
                "dataset_source": dataset_source,
                "scenario_id": scenario_id,
                "reasons": ["legacy:reviewer_target"],
            }
            for dataset_source, scenario_id in sorted(REGENERATION_TARGETS)
        ]
    grouped: Dict[tuple[str, str], List[str]] = {}
    scenario_has_forecast: Dict[tuple[str, str], bool] = {}
    for record in records:
        pair = (
            str(record.get("dataset_source") or ""),
            str(
                record.get("source_scenario_id")
                or record.get("scenario_id")
                or ""
            ),
        )
        if not all(pair):
            continue
        scenario_has_forecast[pair] = (
            scenario_has_forecast.get(pair, False)
            or any(
                dict(call).get("name") == "run_pipeformer_forecast"
                for call in record.get("tool_calls") or []
                if isinstance(call, dict)
            )
        )
        reasons = _trajectory_reasons(dict(record))
        if not reasons:
            continue
        bucket = grouped.setdefault(pair, [])
        for reason in reasons:
            if reason not in bucket:
                bucket.append(reason)
    for pair, has_forecast in scenario_has_forecast.items():
        if (
            "pipeformer" in pair[1].casefold()
            and not has_forecast
        ):
            grouped.setdefault(pair, []).append("trajectory:forecast_missing")
    return [
        {
            "dataset_source": pair[0],
            "scenario_id": pair[1],
            "reasons": sorted(reasons),
        }
        for pair, reasons in sorted(grouped.items())
    ]


def _master_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_planned_targets(
    staging_root: Path,
    master_json: Path,
    profile: str,
    targets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "schema_version": PLANNED_TARGETS_SCHEMA_VERSION,
        "target_profile": profile,
        "master_path": Path(master_json).resolve().as_posix(),
        "master_sha256": _master_sha256(master_json),
        "target_count": len(targets),
        "targets": targets,
    }
    staging_root = Path(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    path = staging_root / PLANNED_TARGETS_FILENAME
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8-sig"))
        if existing != payload:
            raise ValueError(
                "The existing planned_targets.json is immutable and does not "
                "match this discovery. Use a new --staging-root."
            )
        return existing
    write_json(path, payload, force=True)
    return payload


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
    parser.add_argument(
        "--migrate-memory-and-disclosures",
        action="store_true",
        help=(
            "Rebuild verified_decision_state_v1, canonical disclosure blocks, "
            "sessions, and SFT splits without calling an LLM."
        ),
    )
    parser.add_argument(
        "--list-regeneration-targets",
        action="store_true",
        help="Discover and print trajectory defects without generating.",
    )
    parser.add_argument("--stage-regeneration", action="store_true")
    parser.add_argument(
        "--target-profile",
        choices=("registry-contract", "reviewer"),
        default="registry-contract",
    )
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target,
        default=[],
        metavar="DATASET_SOURCE::SCENARIO_ID",
        help=(
            "Restrict the discovered profile. Repeat for one-at-a-time or "
            "small-batch regeneration."
        ),
    )
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
    if not any((
        args.export_annotations,
        args.apply_deterministic,
        args.migrate_memory_and_disclosures,
        args.list_regeneration_targets,
        args.stage_regeneration,
        args.merge_approved,
    )):
        raise ValueError("Choose at least one repair action.")
    if args.attempts < 1 or args.attempts > 3:
        raise ValueError("--attempts must be between 1 and 3.")
    results: Dict[str, Any] = {}
    if args.export_annotations:
        results["annotation_export"] = export_reviewer_annotations(
            args.report_xlsx, args.reviewer_annotations
        )
    if args.apply_deterministic:
        results["assignment_validation"] = _validate_failed_record_assignments(args)
    if args.apply_deterministic:
        results["deterministic"] = _apply_deterministic(args)
    if args.migrate_memory_and_disclosures:
        results["migration"] = _migrate_memory_and_disclosures(args)
    if args.list_regeneration_targets:
        targets = _selected_discovered_targets(args)
        results["regeneration_targets"] = {
            "target_profile": args.target_profile,
            "target_count": len(targets),
            "targets": targets,
        }
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


def _selected_discovered_targets(
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    discovered = _discover_regeneration_targets(
        _store(args).load_master(),
        profile=args.target_profile,
    )
    requested = set(args.target or [])
    if not requested:
        return discovered
    by_pair = {
        (item["dataset_source"], item["scenario_id"]): item
        for item in discovered
    }
    missing = sorted(requested - set(by_pair))
    if missing:
        raise ValueError(
            "Requested --target values are not failures in the selected "
            "profile: "
            + ", ".join(f"{dataset}::{scenario}" for dataset, scenario in missing)
        )
    return [by_pair[pair] for pair in sorted(requested)]


def _filter_planned_targets(
    targets: Sequence[Dict[str, Any]],
    requested_targets: Sequence[tuple[str, str]],
) -> List[Dict[str, Any]]:
    requested = set(requested_targets)
    if not requested:
        return [dict(item) for item in targets]
    by_pair = {
        (
            str(item.get("dataset_source") or ""),
            str(item.get("scenario_id") or ""),
        ): dict(item)
        for item in targets
    }
    missing = sorted(requested - set(by_pair))
    if missing:
        raise ValueError(
            "Requested --target is absent from frozen target manifest: "
            + ", ".join(
                f"{dataset}::{scenario}" for dataset, scenario in missing
            )
        )
    return [by_pair[pair] for pair in sorted(requested)]


def _migrate_records(
    records: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    histories: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    migrated: List[Dict[str, Any]] = []
    disclosure_changes = 0
    native_evaluator = NativeTraceEvaluator(NativeEvaluationConfig())
    for source_record in records:
        record = dict(source_record)
        pair = (
            str(record.get("dataset_source") or ""),
            str(
                record.get("source_scenario_id")
                or record.get("scenario_id")
                or ""
            ),
        )
        history = histories.setdefault(pair, [])
        state = DecisionTraceState.from_history(history)
        try:
            state_before = serialize_verified_decision_state(
                state,
                max_chars=int(
                    os.getenv("VERIFIED_STATE_MAX_CHARS", "16000")
                ),
            )
        except ValueError as exc:
            raise ValueError(
                f"{record.get('sample_id')}: {exc}"
            ) from exc
        recent_turns = bounded_recent_turns(
            history,
            max_turns=2,
            max_chars=int(os.getenv("RECENT_TURNS_MAX_CHARS", "4000")),
        )
        tool_results = attach_tool_arguments(
            record.get("tool_outputs") or [],
            record.get("tool_calls") or [],
        )
        rebuilt_contract = GroundingContractBuilder().build(
            str(record.get("user_input") or ""),
            tool_results,
            require_decision_policy=True,
            prior_candidate_results=state.candidates,
            prior_decision_policy=state.decision_policy,
            prior_decision_policy_source_question=(
                state.decision_policy_source_question
            ),
            prior_applied_disturbances=state.applied_disturbances,
        )
        stored_contract = record.get("grounding_contract")
        contract = (
            deepcopy(dict(stored_contract))
            if isinstance(stored_contract, dict) and stored_contract
            else rebuilt_contract
        )
        verified_applied = [
            deepcopy(dict(item))
            for item in [
                *(rebuilt_contract.get("applied_disturbances") or []),
                *state.applied_disturbances,
            ]
            if isinstance(item, dict) and item.get("verified") is True
        ]
        if verified_applied:
            contract["applied_disturbances"] = verified_applied
        else:
            contract.pop("applied_disturbances", None)
        before_answer = str(record.get("final_answer") or "")
        after_answer = finalize_applied_disturbance_disclosure(
            before_answer,
            contract,
        )
        if after_answer != before_answer:
            disclosure_changes += 1
            record["final_answer"] = after_answer
            record["repair_provenance"] = {
                "method": "canonical_disclosure_migration_v1",
                "external_llm_calls": 0,
                "reason": (
                    "Exact machine-generated application disclosure rebuilt "
                    "from successful stored forecast evidence."
                ),
            }
        record["grounding_contract"] = contract
        record["answer_mode"] = contract.get("answer_mode")
        record["decision_summary"] = dict(
            contract.get("decision_summary") or {}
        )
        record["state_before"] = state_before
        record["recent_turns"] = recent_turns
        record["context_injection"] = {
            "verified_state_schema": state_before["schema_version"],
            "recent_dialogue_turns": len(recent_turns),
            "history_policy": (
                "verified_decision_state_v1 plus at most two bounded recent "
                "dialogue turns; conversation_context is audit-only"
            ),
        }
        native = native_evaluator.evaluate(
            record,
            trace_status=record.get("trace_status"),
        )
        record["quality_flag"] = native["quality_flag"]
        record["quality_score"] = native["quality_score"]
        record["quality_profile"] = native["profile"]
        record["quality_failed_checks"] = native["failed_checks"]
        record["quality_issues"] = native["quality_issues"]
        migrated.append(record)
        history.append(_history_turn(record))
    return migrated, {
        "record_count": len(migrated),
        "canonical_disclosure_change_count": disclosure_changes,
        "state_schema": "verified_decision_state_v1",
        "external_llm_calls": 0,
    }


def _migrate_memory_and_disclosures(args: argparse.Namespace) -> Dict[str, Any]:
    store = _store(args)
    original = store.load_master()
    original_ids = [str(item.get("sample_id") or "") for item in original]
    original_splits = {
        str(item.get("sample_id") or ""): item.get("split")
        for item in original
    }
    migrated, result = _migrate_records(original)
    _assert_master_invariants(original_ids, original_splits, migrated)
    sessions = update_session_records(store.load_sessions(), migrated)
    store.write_master(migrated)
    store.write_sessions(sessions)
    result["sft_record_count"] = write_split_records(
        args.split_dir,
        migrated,
        force=True,
    )
    return result


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
    discovery_reasons: Sequence[str] = (),
) -> str:
    reason_text = (
        f" reasons={','.join(discovery_reasons)}"
        if discovery_reasons
        else ""
    )
    return (
        f"[{position}/{total}] START {dataset_source}:{scenario_id} "
        f"attempt {attempt}/{attempt_count} "
        f"output={Path(output_root).as_posix()}{reason_text}"
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


def _print_live_log(message: str) -> None:
    with _LIVE_LOG_PRINT_LOCK:
        print(message, flush=True)


def _run_child_process_with_live_logs(
    command: Sequence[str],
    *,
    cwd: Path,
    position: int,
    total: int,
    dataset_source: str,
    scenario_id: str,
    attempt: int,
    heartbeat_seconds: float = CHILD_HEARTBEAT_SECONDS,
    poll_seconds: float = 0.25,
    tail_chars: int = CHILD_OUTPUT_TAIL_CHARS,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    events: queue.Queue[tuple[str, str | None]] = queue.Queue()
    tails = {"stdout": "", "stderr": ""}

    def read_stream(stream_name: str, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                events.put((stream_name, line))
        finally:
            stream.close()
            events.put((stream_name, None))

    readers = [
        threading.Thread(
            target=read_stream,
            args=("stdout", process.stdout),
            daemon=True,
        ),
        threading.Thread(
            target=read_stream,
            args=("stderr", process.stderr),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    started_at = time.monotonic()
    last_activity_at = started_at
    last_heartbeat_at = started_at
    completed_streams: set[str] = set()
    while (
        process.poll() is None
        or len(completed_streams) < len(readers)
        or not events.empty()
    ):
        try:
            stream_name, line = events.get(timeout=poll_seconds)
        except queue.Empty:
            now = time.monotonic()
            if (
                process.poll() is None
                and heartbeat_seconds > 0
                and now - last_activity_at >= heartbeat_seconds
                and now - last_heartbeat_at >= heartbeat_seconds
            ):
                _print_live_log(
                    f"[{position}/{total}] RUNNING "
                    f"{dataset_source}:{scenario_id} attempt={attempt} "
                    f"elapsed={now - started_at:.1f}s "
                    f"quiet_for={now - last_activity_at:.1f}s"
                )
                last_heartbeat_at = now
            continue
        if line is None:
            completed_streams.add(stream_name)
            continue

        last_activity_at = time.monotonic()
        tails[stream_name] = (tails[stream_name] + line)[-tail_chars:]
        rendered = line.rstrip("\r\n")
        if rendered:
            _print_live_log(
                f"[{position}/{total}] CHILD "
                f"{dataset_source}:{scenario_id} attempt={attempt} "
                f"{stream_name}: {rendered}"
            )

    for reader in readers:
        reader.join()
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=process.wait(),
        stdout=tails["stdout"],
        stderr=tails["stderr"],
    )


def _stage_scenario(
    args: argparse.Namespace,
    annotations: Dict[tuple[str, str], Dict[str, Any]],
    dataset_source: str,
    scenario_id: str,
    position: int,
    target_count: int,
    discovery_reasons: Sequence[str] = (),
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
        "discovery_reasons": list(discovery_reasons),
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
        _print_live_log(
            _generation_start_message(
                position=position,
                total=target_count,
                dataset_source=dataset_source,
                scenario_id=scenario_id,
                attempt=attempt,
                attempt_count=args.attempts,
                output_root=paths.root,
                discovery_reasons=discovery_reasons,
            )
        )
        started_at = time.monotonic()
        completed = _run_child_process_with_live_logs(
            command,
            cwd=BACKEND_ROOT,
            position=position,
            total=target_count,
            dataset_source=dataset_source,
            scenario_id=scenario_id,
            attempt=attempt,
        )
        attempt_result: Dict[str, Any] = {
            "attempt": attempt,
            "command_exit_code": completed.returncode,
            "stdout_tail": completed.stdout[-4_000:],
            "stderr_tail": completed.stderr[-4_000:],
            "root": paths.root.as_posix(),
            "discovery_reasons": list(discovery_reasons),
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
        _print_live_log(
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
            )
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
    targets: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    target_count = len(targets)
    jobs = [
        (
            args,
            annotations,
            str(target["dataset_source"]),
            str(target["scenario_id"]),
            position,
            target_count,
            list(target.get("reasons") or []),
        )
        for position, target in enumerate(targets, 1)
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
    annotations = load_reviewer_annotations(args.reviewer_annotations)
    planned_path = args.staging_root / PLANNED_TARGETS_FILENAME
    if args.resume and planned_path.is_file():
        planned = json.loads(planned_path.read_text(encoding="utf-8-sig"))
        if (
            planned.get("schema_version") != PLANNED_TARGETS_SCHEMA_VERSION
            or planned.get("target_profile") != args.target_profile
            or planned.get("master_sha256") != _master_sha256(args.master_json)
        ):
            raise ValueError(
                "Frozen target manifest does not match the current profile "
                "and master hash; use a new --staging-root."
            )
    else:
        discovered = _discover_regeneration_targets(
            _store(args).load_master(),
            profile=args.target_profile,
        )
        planned = _freeze_planned_targets(
            args.staging_root,
            args.master_json,
            args.target_profile,
            discovered,
        )
    targets = _filter_planned_targets(
        list(planned.get("targets") or []),
        args.target or [],
    )
    summaries = _run_regeneration_targets(args, annotations, targets)
    summary_path = args.staging_root / "staging_summary.json"
    previous = (
        json.loads(summary_path.read_text(encoding="utf-8-sig"))
        if summary_path.is_file()
        else {}
    )
    summary = _merge_staging_summary(previous, summaries)
    summary["planned_targets_path"] = (
        args.staging_root / PLANNED_TARGETS_FILENAME
    ).as_posix()
    summary["target_profile"] = args.target_profile
    summary["master_sha256"] = planned["master_sha256"]
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
    """Backward-compatible pair projection of dynamic profile discovery."""
    return [
        (str(item["dataset_source"]), str(item["scenario_id"]))
        for item in _selected_discovered_targets(args)
    ]


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
    planned_path = args.staging_root / PLANNED_TARGETS_FILENAME
    if not planned_path.is_file():
        raise ValueError(
            "Missing frozen planned_targets.json. Run --stage-regeneration "
            "with the desired target profile before merging."
        )
    planned = json.loads(planned_path.read_text(encoding="utf-8-sig"))
    if planned.get("schema_version") != PLANNED_TARGETS_SCHEMA_VERSION:
        raise ValueError(f"Unsupported planned-target manifest: {planned_path}")
    current_master_hash = _master_sha256(args.master_json)
    if current_master_hash != planned.get("master_sha256"):
        raise ValueError(
            "Master teacher trace changed after regeneration targets were "
            "frozen; refusing to merge approvals against stale inputs."
        )
    planned_pairs = {
        (
            str(item.get("dataset_source") or ""),
            str(item.get("scenario_id") or ""),
        )
        for item in planned.get("targets") or []
        if isinstance(item, dict)
    }
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
        if (dataset_source, scenario_id) not in planned_pairs:
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
