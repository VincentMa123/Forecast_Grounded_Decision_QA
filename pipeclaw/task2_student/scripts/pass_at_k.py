"""pass@k rollout harness for the python-writing (write_file+run_command) scenarios.

GRPO Phase 0: measures whether the current checkpoint's failures (syntax slips,
wrong paths, thrash) are a *capability* problem or a *reliability* problem by
sampling K episodes per python scenario at several temperatures and scoring each
episode with the deterministic backend evaluator.

Zero-model-weight mode: --emit-grpo-prompts only renders the prompt dataset used
by the GRPO stage and never loads a checkpoint.

Run from the repository root in the task2-ms-swift conda env.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeclaw.task2_student.release_artifacts import (
    atomic_jsonl_writer,
    atomic_write_text,
)
from pipeclaw.task2_student.rollout.models import RolloutConfig
from pipeclaw.task2_student.rollout.prompting import PromptCaseBuilder
from pipeclaw.task2_student.rollout.scenarios import (
    evaluation_workspace_key,
    workspace_for,
)
from pipeclaw.task2_student.rollout.suite import read_jsonl, tool_schema_index

from tqdm.auto import tqdm


RUN_TOOL = "run_command"
TOPOLOGY_TOOL = "analyze_pipeline_topology"
# Policy-ladder rejects priced identically to sandbox_violation; keep in one
# shared lane or each dodge reopens with its own price discovery.
_PRECONDITION_REJECTS = (
    "sandbox_violation",
    "forecast_registry_precondition_failed",
    "decision_metric_used_as_output_state_variable",
    "duplicate_equivalent_forecast",
    "decision_policy_source_not_in_current_user_request",
)
# Any of these means the episode died before presenting a real final answer.
_ERROR_CODES = (
    "python_syntax_error",
    "sandbox_violation",
    "tool_arguments_schema_invalid",
    "duplicate_failed_tool_call",
    "invalid_arguments",
    "unknown_tool",
    "tool_not_allowed",
    *_PRECONDITION_REJECTS,
)
_ABORT_STATUSES = frozenset(
    {"max_turns_exceeded", "max_completion_length", "empty_response", "generation_error"}
)


def select_python_scenarios(
    records: Sequence[Mapping[str, Any]],
    *,
    include_topology: bool = False,
) -> list[dict[str, Any]]:
    """Keep records whose teacher trace executes a python script (or topology tool)."""
    wanted = {RUN_TOOL, TOPOLOGY_TOOL} if include_topology else {RUN_TOOL}
    return [
        dict(record)
        for record in records
        if wanted
        & {
            str(item.get("name"))
            for item in record.get("tool_calls") or []
            if isinstance(item, Mapping)
        }
    ]


def first_user_message(record: Mapping[str, Any]) -> str:
    """Return the user request exactly as the teacher trace recorded it."""
    user_input = record.get("user_input")
    if isinstance(user_input, str) and user_input.strip():
        return user_input
    for message in reversed(record.get("messages") or []):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return str(message.get("content") or "")
    raise ValueError(f"record {record.get('sample_id')} has no user message")


def training_system_prompt(record: Mapping[str, Any]) -> str:
    """Rebuild the SFT system prompt from the CURRENT policy (prepare_dataset.py:542).

    NOTE: prompt_policy.py drifted after the released train.jsonl was frozen
    (7,322 -> 8,249 chars), so this does NOT byte-match released SFT prompts;
    use frozen_system_prompt() with the released trace_level data when the
    model must see the exact SFT prompt distribution.
    """
    from pipeclaw.task2_student.scripts.prepare_dataset import _trace_system_content

    return _trace_system_content(dict(record))


def frozen_system_prompts(schema_source: Path) -> dict[str, str]:
    """Map example_id -> the system prompt frozen in the released SFT data.

    The released records ARE the SFT prompt distribution; copying their system
    message beats regenerating from the drifted live prompt_policy source.
    """
    frozen: dict[str, str] = {}
    for record in read_jsonl(schema_source):
        for message in record.get("messages") or []:
            if message.get("role") == "system":
                frozen[str(record.get("sample_id") or record.get("example_id") or "")] = str(
                    message["content"]
                )
    return frozen


def episode_stats(rollout: Mapping[str, Any]) -> dict[str, Any]:
    """Rule-based per-episode stats used by the gate and the GRPO reward."""
    errors = {code: 0 for code in _ERROR_CODES}
    first_run: Mapping[str, Any] | None = None
    for output in rollout.get("tool_outputs") or []:
        payload = output.get("output")
        if not isinstance(payload, Mapping):
            continue
        code = payload.get("error_code")
        if code in errors:
            errors[code] += 1
        if output.get("name") == RUN_TOOL and first_run is None:
            first_run = payload
    first_run_compile = bool(
        first_run
        and first_run.get("exit_code") is not None
        and first_run.get("error_code") != "python_syntax_error"
    )
    calls = rollout.get("tool_calls") or []
    failed_signatures: dict[str, int] = {}
    success_signatures: dict[str, int] = {}
    for call, output in zip(calls, rollout.get("tool_outputs") or []):
        payload = output.get("output") if isinstance(output, Mapping) else None
        succeeded = call.get("execution_success") is not False and (
            not isinstance(payload, Mapping) or payload.get("success", True) is not False
        )
        signature = json.dumps(
            [call.get("name"), call.get("arguments")], sort_keys=True, default=str
        )
        pool = success_signatures if succeeded else failed_signatures
        pool[signature] = pool.get(signature, 0) + 1
    thrash = sum(count - 1 for count in failed_signatures.values() if count > 1)
    duplicate_success = sum(count - 1 for count in success_signatures.values() if count > 1)
    return {
        "turns": rollout.get("turns", 0),
        "trace_status": rollout.get("trace_status", ""),
        "first_run_compile": first_run_compile,
        "first_run_exit0": first_run is not None and first_run.get("exit_code") == 0,
        "error_counts": errors,
        "thrash_count": thrash,
        "duplicate_success_count": duplicate_success,
        # error_codes cover only sandbox/audit rejections; plain exit!=0 failures
        # arrive with error_code=None, so track failed calls directly.
        "failed_call_count": sum(failed_signatures.values()),
        "malformed_json": bool(rollout.get("json_errors")),
        "timeout_hit": rollout.get("trace_status") in _ABORT_STATUSES,
    }


def _schemas_for(
    record: Mapping[str, Any],
    schemas_by_key: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    """Match a record's schemas via its identifiers; None when the index misses."""
    for key in (
        str(record.get("sample_id") or ""),
        str(record.get("example_id") or ""),
        str(record.get("source_id") or ""),
    ):
        matched = schemas_by_key.get(key)
        if matched:
            return list(matched)
    return None


def _record_key(record: Mapping[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("example_id") or "")


def _frozen_prompt(frozen: Mapping[str, str], record: Mapping[str, Any]) -> str:
    prompt = frozen.get(_record_key(record))
    if prompt is None:
        raise ValueError(
            f"record {_record_key(record)} misses the frozen system prompt; "
            "refusing the drifted live fallback"
        )
    return prompt


def emit_grpo_prompts(
    selected: Sequence[Mapping[str, Any]],
    path: Path,
    schemas_by_key: Mapping[str, Any] | None = None,
    frozen: Mapping[str, str] | None = None,
) -> int:
    """Write the prompt-only rows the GRPO stage consumes (no model needed).

    Stacked dataset vintages collide on (scenario_id, session_id, turn_id):
    keep only the newest vintage prefix per key — double-counted scenarios get
    double GRPO sampling weight. Toolless records carry no executable behavior
    to ground an episode on, so they are skipped as well.
    """

    def _vintage(sample_id: str) -> tuple[int, str]:
        # released ids use hyphen markers: Pipeline_Full_Life_Cycle_Test_Dataset-v4:/-v7:
        for marker, rank in (("-v7:", 7), ("-v4:", 6)):
            if marker in sample_id:
                return rank, sample_id
        return 5, sample_id

    by_key = {
        (str(r.get("scenario_id") or ""), str(r.get("session_id") or ""), int(r.get("turn_id") or 1)): r
        for r in sorted(
            selected,
            key=lambda r: _vintage(str(r.get("sample_id") or "")),
        )
        if r.get("tool_calls") or r.get("tool_outputs")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_jsonl_writer(path, default=str) as write:
        for record in by_key.values():
            tools = record.get("tools")
            if tools is None and schemas_by_key is not None:
                matched = _schemas_for(record, schemas_by_key)
                if matched:
                    tools = json.dumps(matched, ensure_ascii=False)
                else:
                    tools = None
            if tools is None:
                raise ValueError(
                    f"record {_record_key(record)} has no tools; pass --tool-schema-source"
                )
            system = (
                _frozen_prompt(frozen, record)
                if frozen is not None
                else training_system_prompt(record)
            )
            write(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": first_user_message(record)},
                    ],
                    "tools": tools,
                    "reference": record,
                    "scenario_id": record.get("scenario_id") or "",
                    "scenario_type": record.get("scenario_type") or "openclaw",
                    "sample_id": _record_key(record),
                }
            )
    return len(by_key)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def rate(values: Sequence[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    # GRPO's reward group is (prompt row, temperature), so the share metric is
    # keyed the same way. (Using scenario_id lumps unrelated prompt groups into
    # one flat group, losing the count.)
    scenario_groups: dict[tuple[str, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        scenario_groups.setdefault(
            (str(row.get("sample_id")), row.get("temperature")), []
        ).append(row)
    zero_variance_share = (
        sum(
            1
            for env_rows in scenario_groups.values()
            if len(env_rows) > 1
            and statistics.pstdev([float(row.get("reward", 0.0)) for row in env_rows])
            == 0
        )
        / len(scenario_groups)
        if scenario_groups
        else 0.0
    )
    return {
        "episodes": len(rows),
        "scenarios": len({str(row.get("scenario_id")) for row in rows}),
        "first_try_exit0_rate": rate([r["first_run_exit0"] for r in rows]),
        "recovered_after_error_rate": rate(
            [
                r["passed"]
                and (
                    r["thrash_count"] > 0
                    or any(r["error_counts"].values())
                    or r.get("failed_call_count", 0) > 0
                )
                for r in rows
            ]
        ),
        "thrash_rate": rate([r["thrash_count"] > 0 for r in rows]),
        "mean_overall_score": statistics.fmean(
            [float(r.get("overall_score") or 0.0) for r in rows]
        )
        if rows
        else 0.0,
        "passed_rate": rate([r["passed"] for r in rows]),
        "mean_reward": statistics.fmean([float(r.get("reward", 0.0)) for r in rows])
        if rows
        else 0.0,
        "zero_reward_prompt_share": zero_variance_share,
        "syntax_error_share": rate(
            [r["error_counts"]["python_syntax_error"] > 0 for r in rows]
        ),
    }


def composite_reward(stats: Mapping[str, Any], report_fields: Mapping[str, Any]) -> float:
    """Dense rule reward; mirrors the GRPO plugin so the gate measures what RL sees."""
    reward = 0.0
    reward += 0.55 * float(report_fields.get("overall_score") or 0.0) / 100.0
    reward += 0.20 * (1.0 if report_fields.get("hard_gate_passed") else 0.0)
    reward += 0.10 * (1.0 if stats["first_run_compile"] else 0.0)
    reward += 0.05 * (1.0 if stats["first_run_exit0"] else 0.0)
    reward += 0.05 * (1.0 if report_fields.get("passed") else 0.0)
    reward -= 0.05 * (1.0 if stats["timeout_hit"] else 0.0)
    reward -= min(0.15, 0.05 * int(stats["thrash_count"]))
    reward -= 0.10 * (1.0 if stats["malformed_json"] else 0.0)
    # Repeated identical *successful* calls are free in rollout training but
    # rejected by the live audit layer; make them cost reward here too.
    reward -= min(0.06, 0.02 * int(stats.get("duplicate_success_count", 0)))
    # Rotated-arg retry loops dodge the identical-signature thrash signature;
    # count every failed call past the first (the first failure is free: an
    # honest one-shot error is information, not a loop).
    reward -= min(0.10, 0.05 * max(0, int(stats.get("failed_call_count", 0)) - 1))
    # A sandbox-policy attempt is the cheapest possible reject: the tool never
    # validates arguments, the failure costs only the free allowance. Its
    # precondition siblings (registry-less forecast, metric-guarded forecast,
    # duplicate forecast, decision-policy quote rejects) inherit the same
    # ladder — the bare attempt and the metric-padded dodge must BOTH lose to
    # stop-and-declare (AGENTS rule 3).
    reward -= 0.10 * float(
        any(stats.get("error_counts", {}).get(code) for code in _PRECONDITION_REJECTS)
    )
    # A recovery bonus (+0.05 for pass-with-any-failure) enabled the "1 junk
    # fail then pass outscores clean" exploit; ranking is already safe via the
    # penalty ladders above, so no bonus is emitted.
    return round(reward, 6)


def run_episodes(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(Path(args.source))
    if getattr(args, "all_scenarios", False):
        sources = [
            dict(r)
            for r in records
            if r.get("tool_calls") or r.get("tool_outputs")
        ]
    elif getattr(args, "pipeformer", False):
        sources = [
            dict(r)
            for r in records
            if r.get("scenario_type") == "pipeformer" and int(r.get("turn_id") or 1) == 1
        ]
    else:
        sources = select_python_scenarios(
            records, include_topology=bool(getattr(args, "include_topology", False))
        )
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise ValueError("no scenarios selected from --source")

    schemas_by_key = (
        tool_schema_index(read_jsonl(Path(args.tool_schema_source)))
        if args.tool_schema_source
        else {}
    )
    frozen_prompts = (
        frozen_system_prompts(Path(args.tool_schema_source))
        if args.tool_schema_source
        else {}
    )

    if args.emit_grpo_prompts:
        written = emit_grpo_prompts(
            sources, Path(args.emit_grpo_prompts), schemas_by_key, frozen_prompts
        )
        return {"emitted_grpo_prompts": written, "scenarios": len(sources)}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from pipeclaw.backend.evaluator import EvaluationProfile, evaluate

    builder = PromptCaseBuilder()
    all_schemas = _union_schemas(sources, schemas_by_key)
    runner = _build_runner(args, all_schemas)
    rows: list[dict[str, Any]] = []
    rollouts_path = output_dir / "episodes.jsonl"
    trajectories_path = output_dir / "trajectories.jsonl"
    progress = tqdm(
        total=len(sources) * len(args.temps) * args.episodes,
        desc="pass_at_k",
        unit="episode",
    )
    with atomic_jsonl_writer(rollouts_path, default=str) as write_rollout, atomic_jsonl_writer(
        trajectories_path, default=str
    ) as write_trajectory:
        for source in sources:
            case_schemas = _schemas_for(source, schemas_by_key) or builder.build(
                source,
                workspace_root=workspace_for(output_dir, "schema-probe"),
            ).tools
            frozen_prompt = (
                _frozen_prompt(frozen_prompts, source)
                if args.execution_mode == "raw-student"
                and args.system_prompt_mode == "training"
                else None
            )
            for temp in args.temps:
                for k in range(args.episodes):
                    env_key = (
                        f"{evaluation_workspace_key(source)}__k{k}__t{temp:.2f}"
                    )
                    case = builder.build(
                        source,
                        workspace_root=workspace_for(output_dir, env_key),
                        tool_schemas=case_schemas,
                    )
                    if frozen_prompt is not None:
                        case.messages[0] = {
                            "role": "system",
                            "content": frozen_prompt,
                        }
                    result = runner.run(
                        case,
                        RolloutConfig(
                            max_turns=args.max_turns,
                            max_new_tokens=args.max_new_tokens,
                            temperature=temp,
                        ),
                    )
                    rollout = result.to_dict()
                    write_trajectory(
                        {
                            "scenario_id": source.get("scenario_id") or "",
                            "sample_id": source.get("sample_id")
                            or source.get("example_id"),
                            "temperature": temp,
                            "episode": k,
                            "execution_mode": args.execution_mode,
                            "rollout": rollout,
                        }
                    )
                    stats = episode_stats(rollout)
                    report_fields = evaluate(
                        rollout,
                        profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
                        reference=source,
                    ).to_dict()
                    row = {
                        "scenario_id": source.get("scenario_id") or "",
                        "sample_id": source.get("sample_id")
                        or source.get("example_id"),
                        "temperature": temp,
                        "episode": k,
                        "execution_mode": args.execution_mode,
                        **stats,
                        "overall_score": report_fields.get("overall_score"),
                        "hard_gate_passed": report_fields.get("hard_gate_passed"),
                        "critical_failures": report_fields.get("critical_failures")
                        or (),
                        "failed_checks": report_fields.get("failed_checks") or (),
                        "passed": bool(report_fields.get("passed")),
                        "reward": composite_reward(stats, report_fields),
                        "final_answer": rollout.get("final_answer", ""),
                    }
                    write_rollout(row)
                    rows.append(row)
                    progress.set_postfix(
                        scenario=str(row["scenario_id"])[-40:],
                        status=row["trace_status"],
                        passed=row["passed"],
                        reward=row["reward"],
                        refresh=False,
                    )
                    progress.update()
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
    progress.close()
    summary = _aggregate(rows)
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
    )
    return summary


def _union_schemas(
    sources: Sequence[Mapping[str, Any]],
    schemas_by_key: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Union every selected scenario's schemas, deduplicated by function name.

    The dispatcher must allow-list what any case in the run may call; an empty
    schema set allow-lists nothing (tools.py:372,386,395).
    """
    seen: set[str] = set()
    schemas: list[dict[str, Any]] = []
    for record in sources:
        for schema in _schemas_for(record, schemas_by_key) or []:
            name = str((schema.get("function") or {}).get("name") or "")
            if name and name not in seen:
                seen.add(name)
                schemas.append(dict(schema))
    return schemas


def _build_runner(args: argparse.Namespace, schemas: Sequence[Mapping[str, Any]]):
    if getattr(args, "execution_mode", "raw-student") == "production-agent":
        from pipeclaw.task2_student.rollout.production_agent import (
            ProductionAgentRunner,
        )

        return ProductionAgentRunner()

    from pipeclaw.task2_student.rollout.runner import RolloutRunner
    from pipeclaw.task2_student.rollout.scenarios import (
        ScenarioPolicy,
        build_openclaw_dispatcher,
    )
    from pipeclaw.task2_student.rollout.swift_generator import (
        SwiftGenerator,
        discover_base_model,
    )

    model = args.model or discover_base_model(Path(args.adapters))
    generator = SwiftGenerator.from_args(
        model=model,
        adapters=args.adapters,
        device=getattr(args, "device", None),
        quant_bits=getattr(args, "quant_bits", None),
        no_quantization=bool(getattr(args, "no_quantization", False)),
        enable_thinking=bool(getattr(args, "enable_thinking", False)),
        model_type=getattr(args, "model_type", None),
    )
    dispatcher = build_openclaw_dispatcher(list(schemas), Path(args.repo_root))
    runner = RolloutRunner(generator, dispatcher, policy=ScenarioPolicy())
    return runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="teacher_trace_*.jsonl")
    parser.add_argument("--tool-schema-source", help="trace_level/*.jsonl for schemas")
    parser.add_argument("--adapters", help="LoRA adapter checkpoint directory")
    parser.add_argument("--model", help="Base model id/path when no --adapters")
    parser.add_argument("--output-dir", default="pipeclaw/task2_student/outputs/evaluation/pass_at_k")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--temps", type=float, nargs="+", default=[0.7, 1.0])
    parser.add_argument(
        "--max-turns",
        type=int,
        help="default: 8 for raw-student, 30 for production-agent",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--execution-mode",
        choices=["raw-student", "production-agent"],
        default="raw-student",
        help=(
            "raw-student loads the model directly; production-agent uses the "
            "deployed OpenAI-compatible model through AgentOrchestrator"
        ),
    )
    parser.add_argument("--system-prompt-mode", choices=["training", "production"], default="training")
    parser.add_argument("--emit-grpo-prompts", help="write the GRPO prompt dataset and exit")
    parser.add_argument(
        "--pipeformer",
        action="store_true",
        help="select scenario_type=pipeformer records instead of python (run_command) scenarios",
    )
    parser.add_argument(
        "--include-topology",
        action="store_true",
        help="also include records whose teacher trace uses analyze_pipeline_topology",
    )
    parser.add_argument("--device")
    parser.add_argument("--quant-bits", type=int)
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--model-type",
        help="explicit ms-swift model_type when auto-matching is ambiguous (e.g., qwen3_5 for Qwen3.8-27B)",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="evaluate every scenario family (openclaw + topology + pipeformer turn-1), not only python scenarios",
    )
    parser.add_argument("--repo-root", default=".")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_turns is None:
        args.max_turns = 30 if args.execution_mode == "production-agent" else 8
    if (
        args.execution_mode == "raw-student"
        and not args.emit_grpo_prompts
        and not args.adapters
        and not args.model
    ):
        parser.error("--adapters or --model is required unless --emit-grpo-prompts")
    if not args.tool_schema_source:
        # emit-mode needs it too: frozen prompts keyed from the same source;
        # without it the emit dies at the first record in one cold raise.
        parser.error("--tool-schema-source is required")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    summary = run_episodes(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
