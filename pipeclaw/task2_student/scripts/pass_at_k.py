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
_ERROR_CODES = (
    "python_syntax_error",
    "sandbox_violation",
    "tool_arguments_schema_invalid",
    "duplicate_failed_tool_call",
)


def select_python_scenarios(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep records whose teacher trace executes a python script."""
    selected = []
    for record in records:
        names = {
            str(item.get("name"))
            for item in record.get("tool_calls") or []
            if isinstance(item, Mapping)
        }
        if RUN_TOOL in names:
            selected.append(dict(record))
    return selected


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
                frozen[str(record.get("example_id") or "")] = str(message["content"])
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
    for call, output in zip(calls, rollout.get("tool_outputs") or []):
        payload = output.get("output") if isinstance(output, Mapping) else None
        succeeded = call.get("execution_success") is not False and (
            not isinstance(payload, Mapping) or payload.get("success", True) is not False
        )
        if succeeded:
            continue
        signature = json.dumps(
            [call.get("name"), call.get("arguments")], sort_keys=True, default=str
        )
        failed_signatures[signature] = failed_signatures.get(signature, 0) + 1
    thrash = sum(count - 1 for count in failed_signatures.values() if count > 1)
    return {
        "turns": rollout.get("turns", 0),
        "trace_status": rollout.get("trace_status", ""),
        "first_run_compile": first_run_compile,
        "first_run_exit0": first_run is not None and first_run.get("exit_code") == 0,
        "error_counts": errors,
        "thrash_count": thrash,
        "malformed_json": bool(rollout.get("json_errors")),
        "timeout_hit": rollout.get("trace_status") == "max_turns_exceeded",
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


def emit_grpo_prompts(
    selected: Sequence[Mapping[str, Any]],
    path: Path,
    schemas_by_key: Mapping[str, Any] | None = None,
    frozen: Mapping[str, str] | None = None,
) -> int:
    """Write the prompt-only rows the GRPO stage consumes (no model needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_jsonl_writer(path, default=str) as write:
        for record in selected:
            tools = record.get("tools")
            if tools is None and schemas_by_key is not None:
                matched = _schemas_for(record, schemas_by_key)
                if matched:
                    tools = json.dumps(matched, ensure_ascii=False)
                else:
                    tools = None
            else:
                tools = record.get("tools")
            if tools is None:
                raise ValueError(
                    f"record {_record_key(record)} has no tools; pass --tool-schema-source"
                )
            system = (frozen or {}).get(_record_key(record)) or training_system_prompt(
                record
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
                    "sample_id": _record_key(record),
                }
            )
    return len(selected)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def rate(values: Sequence[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    scenario_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        scenario_groups.setdefault(str(row["scenario_id"]), []).append(row)
    zero_variance_share = 0.0
    if scenario_groups:
        flat = 0
        for env_rows in scenario_groups.values():
            scores = [float(row.get("reward", 0.0)) for row in env_rows]
            if len(scores) > 1 and statistics.pstdev(scores) == 0:
                flat += 1
        zero_variance_share = flat / len(scenario_groups)
    return {
        "episodes": len(rows),
        "scenarios": len(scenario_groups),
        "first_try_exit0_rate": rate([r["first_run_exit0"] for r in rows]),
        "recovered_after_error_rate": rate(
            [
                r["passed"]
                and (
                    r["thrash_count"] > 0 or any(r["error_counts"].values())
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
    return round(reward, 6)


def run_episodes(args: argparse.Namespace) -> dict[str, Any]:
    sources = select_python_scenarios(read_jsonl(Path(args.source)))
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise ValueError("no python scenarios selected from --source")

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
    progress = tqdm(
        total=len(sources) * len(args.temps) * args.episodes,
        desc="pass_at_k",
        unit="episode",
    )
    with atomic_jsonl_writer(rollouts_path, default=str) as write_rollout:
        for source in sources:
            case_schemas = _schemas_for(source, schemas_by_key) or builder.build(
                source,
                workspace_root=workspace_for(output_dir, "schema-probe"),
            ).tools
            frozen_prompt = (
                frozen_prompts.get(_record_key(source)) or training_system_prompt(source)
                if args.system_prompt_mode == "training"
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
                        **stats,
                        "overall_score": report_fields.get("overall_score"),
                        "hard_gate_passed": report_fields.get("hard_gate_passed"),
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
    )
    dispatcher = build_openclaw_dispatcher(list(schemas), Path(args.repo_root))
    runner = RolloutRunner(generator, dispatcher, policy=ScenarioPolicy())
    return runner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="teacher_trace_*.jsonl")
    parser.add_argument("--tool-schema-source", help="trace_level/*.jsonl for schemas")
    parser.add_argument("--adapters", help="LoRA adapter checkpoint directory")
    parser.add_argument("--model", help="Base model id/path when no --adapters")
    parser.add_argument("--output-dir", default="pipeclaw/task2_student/outputs/evaluation/pass_at_k")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--temps", type=float, nargs="+", default=[0.7, 1.0])
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system-prompt-mode", choices=["training", "production"], default="training")
    parser.add_argument("--emit-grpo-prompts", help="write the GRPO prompt dataset and exit")
    parser.add_argument("--device")
    parser.add_argument("--quant-bits", type=int)
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    if not args.emit_grpo_prompts:
        if not args.adapters and not args.model:
            parser.error("--adapters or --model is required unless --emit-grpo-prompts")
        if not args.tool_schema_source:
            parser.error("--tool-schema-source is required unless --emit-grpo-prompts")
    summary = run_episodes(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
