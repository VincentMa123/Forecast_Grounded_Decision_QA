from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeclaw.student_distillation.release_artifacts import (
    atomic_jsonl_writer,
    atomic_write_text,
)
from pipeclaw.student_distillation.reward import composite_reward, episode_stats
from pipeclaw.student_distillation.rollout.models import RolloutConfig
from pipeclaw.student_distillation.rollout.prompting import build_prompt_case
from pipeclaw.student_distillation.rollout.scenarios import (
    evaluation_workspace_key,
    workspace_for,
)
from pipeclaw.student_distillation.rollout.suite import (
    build_runner_selector,
    lookup_tool_schemas,
    read_jsonl,
    release_cuda_cache,
    tool_schema_index,
)

from tqdm.auto import tqdm


RUN_TOOL = "run_command"
TOPOLOGY_TOOL = "analyze_pipeline_topology"


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


def frozen_system_prompts(schema_source: Path) -> dict[str, str]:
    """Map example_id -> the system prompt frozen in the released SFT data.

    The released records ARE the SFT prompt distribution; copying their system
    message beats regenerating from the drifted live prompt_policy source.
    """
    return {
        _record_key(record): str(message["content"])
        for record in read_jsonl(schema_source)
        for message in record.get("messages") or []
        if message.get("role") == "system"
    }


def _record_key(record: Mapping[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("example_id") or "")


def _frozen_prompt(frozen: Mapping[str, str], record: Mapping[str, Any]) -> str:
    key = _record_key(record)
    prompt = frozen.get(key)
    if prompt is None:
        raise ValueError(
            f"record {key} misses the frozen system prompt; "
            "refusing the drifted live fallback"
        )
    return prompt


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def rate(values: Sequence[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    def mean(field: str) -> float:
        return statistics.fmean(float(row.get(field) or 0.0) for row in rows) if rows else 0.0

    # Group repeated evaluations of one prompt at one temperature.
    groups: dict[tuple[str, Any], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("sample_id")), row.get("temperature")), []).append(row)
    zero_variance_share = (
        sum(
            1
            for group in groups.values()
            if len(group) > 1
            and statistics.pstdev([float(row.get("reward", 0.0)) for row in group]) == 0
        )
        / len(groups)
        if groups
        else 0.0
    )
    return {
        "episodes": len(rows),
        "scenarios": len({str(row.get("scenario_id")) for row in rows}),
        "first_try_exit0_rate": rate([r["first_run_exit0"] for r in rows]),
        "recovered_after_error_rate": rate(
            [
                row["passed"] and (
                    row["thrash_count"] > 0
                    or any(row["error_counts"].values())
                    or row.get("failed_call_count", 0) > 0
                ) for row in rows
            ]
        ),
        "thrash_rate": rate([row["thrash_count"] > 0 for row in rows]),
        "mean_overall_score": mean("overall_score"),
        "passed_rate": rate([row["passed"] for row in rows]),
        "mean_reward": mean("reward"),
        "zero_variance_share": zero_variance_share,
        "syntax_error_share": rate(
            [row["error_counts"]["python_syntax_error"] > 0 for row in rows]
        ),
    }


def run_episodes(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(Path(args.source))
    if getattr(args, "all_scenarios", False):
        sources = [dict(r) for r in records if r.get("tool_calls") or r.get("tool_outputs")]
    elif getattr(args, "pipeformer", False):
        sources = [
            dict(r)
            for r in records
            if r.get("scenario_type") == "pipeformer"
            and int(r.get("turn_id") or 1) == 1
        ]
    else:
        sources = select_python_scenarios(records, include_topology=bool(
            getattr(args, "include_topology", False)))
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from pipeclaw.backend.evaluator import EvaluationProfile, evaluate

    source_cases: list[tuple[Mapping[str, Any], list[dict[str, Any]]]] = []
    for source in sources:
        case_schemas = lookup_tool_schemas(source, schemas_by_key, missing_policy="none")
        if not case_schemas:
            case_schemas = build_prompt_case(
                source,
                workspace_root=workspace_for(output_dir, "schema-probe"),
            ).tools
        source_cases.append((source, case_schemas))

    select_runner = build_runner_selector(
        args,
        ((source.get("scenario_type"), schemas) for source, schemas in source_cases),
        copy_schemas=True,
        model_type=getattr(args, "model_type", None),
    )
    rows: list[dict[str, Any]] = []
    progress = tqdm(total=len(sources) * len(args.temps) * args.episodes,
                    desc="pass_at_k", unit="episode")
    with (
        atomic_jsonl_writer(output_dir / "episodes.jsonl", default=str) as write_rollout,
        atomic_jsonl_writer(output_dir / "trajectories.jsonl", default=str) as write_trajectory,
    ):
        for source, case_schemas in source_cases:
            frozen_prompt = (
                _frozen_prompt(frozen_prompts, source)
                if args.execution_mode == "raw-student"
                and args.system_prompt_mode == "training"
                else None
            )
            for temp in args.temps:
                for k in range(args.episodes):
                    env_key = f"{evaluation_workspace_key(source)}__k{k}__t{temp!r}"
                    case = build_prompt_case(
                        source,
                        workspace_root=workspace_for(output_dir, env_key),
                        tool_schemas=case_schemas,
                    )
                    if frozen_prompt is not None:
                        case.messages[0] = {"role": "system", "content": frozen_prompt}
                    result = select_runner(source.get("scenario_type")).run(case, RolloutConfig(
                        max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
                        temperature=temp))
                    rollout = result.to_dict()
                    identity = {
                        "scenario_id": source.get("scenario_id") or "",
                        "sample_id": source.get("sample_id") or source.get("example_id"),
                        "temperature": temp,
                        "episode": k,
                        "execution_mode": args.execution_mode,
                    }
                    write_trajectory({**identity, "rollout": rollout})
                    stats = episode_stats(rollout)
                    report_fields = evaluate(
                        rollout,
                        profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
                        reference=source,
                    ).to_dict()
                    row = {
                        **identity,
                        **stats,
                        "overall_score": report_fields.get("overall_score"),
                        "hard_gate_passed": report_fields.get("hard_gate_passed"),
                        "critical_failures": report_fields.get("critical_failures") or (),
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
                    release_cuda_cache()
    progress.close()
    summary = _aggregate(rows)
    atomic_write_text(output_dir / "summary.json",
                      json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="teacher_trace_*.jsonl")
    parser.add_argument("--tool-schema-source", help="trace_level/*.jsonl for schemas")
    parser.add_argument("--adapters", help="Optional LoRA adapter checkpoint directory")
    parser.add_argument("--model", help="Base model id/path when no --adapters")
    parser.add_argument(
        "--execution-mode",
        choices=("raw-student", "production-agent"),
        default="raw-student",
        help="Use the direct student loop or the deployed AgentOrchestrator",
    )
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--device", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--model-type",
        help="Explicit ms-swift model_type when auto-matching is ambiguous "
        "(for example, qwen3_5 for Qwen3.8-27B)",
    )
    parser.add_argument("--repo-root", default=".", help="Repository root used to import PipeClaw tools")
    parser.add_argument("--quant-bits", type=int)
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--output-dir", default="pipeclaw/student_distillation/outputs/evaluation/pass_at_k")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--temps", type=float, nargs="+", default=[0.7, 1.0])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--system-prompt-mode", choices=["training", "production"], default="training")
    parser.add_argument(
        "--pipeformer",
        action="store_true",
        help="select scenario_type=pipeformer records instead of python "
        "(run_command) scenarios",
    )
    parser.add_argument(
        "--include-topology",
        action="store_true",
        help="also include records whose teacher trace uses analyze_pipeline_topology",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="evaluate every record with tool activity (including OpenClaw, "
        "topology, and PipeFormer traces), not only Python/run_command scenarios",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_turns is None:
        args.max_turns = 30 if args.execution_mode == "production-agent" else 8
    if (
        args.execution_mode == "raw-student"
        and not args.adapters
        and not args.model
    ):
        parser.error("--adapters or --model is required")
    if not args.tool_schema_source:
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
