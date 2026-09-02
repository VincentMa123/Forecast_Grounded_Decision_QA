from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from pipeclaw.backend.evaluator import (
    EvaluationProfile,
    EvaluationReport,
    evaluate,
    summarize,
)
from pipeclaw.student_distillation.release_artifacts import (
    atomic_jsonl_writer,
    atomic_write_text,
    read_jsonl_domain,
    stable_json,
)

from .models import PromptCase, RolloutConfig
from .prompting import build_prompt_case, parse_tool_schemas
from .runner import RolloutRunner
from .scenarios import (
    ScenarioPolicy,
    build_dispatcher,
    evaluation_workspace_key,
    is_openclaw_scenario,
    scenario_key,
    workspace_for,
)
from .tools import ToolDispatcher

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - MS-SWIFT normally installs tqdm

    def tqdm(iterable=None, *args, **kwargs):
        del args, kwargs
        return iterable if iterable is not None else []


# The canonical schema-v3 fields copied from the report onto every rollout
# record.  ``schema_version`` is renamed so the rollout keeps one unambiguous
# evaluation-schema field alongside its trajectory fields.
_REPORT_ROOT_FIELDS = (
    "overall_score",
    "hard_gate_passed",
    "passed",
    "metrics",
    "diagnostics",
    "failed_checks",
    "critical_failures",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line, reporting the offending line on failure."""

    return read_jsonl_domain(path, skip_blank_lines=True)


def sample_keys(record: Mapping[str, Any]) -> set[str]:
    """Return every identifier a record may be joined on across files."""

    return {
        str(record[key])
        for key in ("sample_id", "example_id", "source_sample_id", "source_id")
        if record.get(key) is not None
    }


def tool_schema_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Index OpenAI tool schemas by every identifier of their record."""

    index: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        normalised = parse_tool_schemas(record.get("tools"))
        candidate_authoritative = _authoritative_schema_set(normalised)
        for key in sample_keys(record):
            existing = index.get(key)
            existing_authoritative = _authoritative_schema_set(existing)
            if existing is None or (
                candidate_authoritative and not existing_authoritative
            ) or (
                candidate_authoritative == existing_authoritative
                and stable_json(normalised) < stable_json(existing)
            ):
                index[key] = normalised
    return index


def generic_schemas(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fallback schemas for dry-runs when a projection file is unavailable."""

    tool_calls = source.get("tool_calls")
    items = tool_calls if isinstance(tool_calls, Sequence) and not isinstance(
        tool_calls, (str, bytes)
    ) else ()
    names = {
        str(item["name"])
        for item in items
        if isinstance(item, Mapping) and item.get("name")
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Tool schema unavailable in the source projection.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            },
        }
        for name in sorted(names)
    ]


_MISSING_SCHEMA_POLICIES = {"error", "generic", "none"}


def lookup_tool_schemas(
    source: Mapping[str, Any],
    schemas_by_key: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    missing_policy: str = "error",
) -> list[dict[str, Any]] | None:
    """Select deterministic authoritative schemas with an explicit miss policy."""

    if missing_policy not in _MISSING_SCHEMA_POLICIES:
        raise ValueError(
            f"missing_policy must be one of {sorted(_MISSING_SCHEMA_POLICIES)}"
        )
    keys = sorted(sample_keys(source))
    matched = next(
        (
            candidate
            for key in keys
            if _authoritative_schema_set(candidate := schemas_by_key.get(key))
        ),
        None,
    )
    if matched is not None:
        return [dict(schema) for schema in matched or []]
    if missing_policy == "generic":
        return generic_schemas(source)
    if missing_policy == "none":
        return None
    identifier = keys[0] if keys else "record without a sample identifier"
    raise ValueError(f"no authoritative tool schema for {identifier}")


def _authoritative_schema_set(
    schemas: Sequence[Mapping[str, Any]] | None,
) -> bool:
    """Return whether schemas carry an explicit OpenAI function contract."""

    return bool(schemas) and all(
        isinstance(schema, Mapping)
        and schema.get("type") == "function"
        and isinstance((function := schema.get("function")), Mapping)
        and isinstance((name := function.get("name")), str)
        and bool(name.strip())
        for schema in schemas
    )


def build_cases(
    source_records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    schemas_by_key: Mapping[str, Sequence[Mapping[str, Any]]],
    scenario_type: str | None = None,
    limit: int | None = None,
    allow_generic_schemas: bool = False,
) -> list[PromptCase]:
    """Build prompt-only cases with per-scenario isolated workspaces.

    Generic schemas are intentionally opt-in for dry-run inspection only.
    """

    requested = str(scenario_type or "")
    requested_key = (
        "openclaw" if is_openclaw_scenario(requested) else requested.casefold()
    )
    cases: list[PromptCase] = []
    for source in source_records:
        source_type = source.get("scenario_type")
        source_key = (
            "openclaw"
            if is_openclaw_scenario(source_type)
            else str(source_type or "").casefold()
        )
        if requested and source_key != requested_key:
            continue
        schemas = lookup_tool_schemas(
            source,
            schemas_by_key,
            missing_policy="generic" if allow_generic_schemas else "error",
        )
        cases.append(
            build_prompt_case(
                source,
                workspace_root=workspace_for(
                    output_dir, evaluation_workspace_key(source)
                ),
                tool_schemas=schemas,
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def schemas_by_scenario_family(
    cases: Sequence[PromptCase],
) -> dict[str, list[Mapping[str, Any]]]:
    """Union each family's tool schemas, deduplicated by function name."""

    families: dict[str, list[Mapping[str, Any]]] = {}
    seen: dict[str, set[str]] = {}
    for case in cases:
        family = scenario_key(case.scenario_type)
        schemas = families.setdefault(family, [])
        names = seen.setdefault(family, set())
        for schema in case.tools:
            function = schema.get("function") if isinstance(schema, Mapping) else None
            name = function.get("name") if isinstance(function, Mapping) else None
            if name and str(name) not in names:
                schemas.append(schema)
                names.add(str(name))
    return families


def attach_report(rollout: dict[str, Any], report: EvaluationReport) -> dict[str, Any]:
    """Copy the canonical schema-v3 root fields onto one rollout record.

    The report is flattened rather than nested so a rollout carries exactly one
    score, one metric mapping, and one set of diagnostics.
    """

    payload = report.to_dict()
    rollout.update(
        {"evaluation_schema_version": payload["schema_version"]}
        | {field: payload[field] for field in _REPORT_ROOT_FIELDS}
    )
    return rollout


def _set_postfix(progress: Any, **fields: Any) -> None:
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(refresh=False, **fields)


def release_cuda_cache() -> None:
    """Release unreachable per-case tensors and unused CUDA allocator blocks."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _summary(
    reports: Sequence[EvaluationReport],
    *,
    mode: str,
    record_count: int,
) -> dict[str, Any]:
    """Build one schema-v3 summary, including for dry runs with no reports."""

    return dict(summarize(reports), mode=mode, record_count=record_count)


def _by_scenario_type(
    cases: Sequence[PromptCase],
    reports: Sequence[EvaluationReport | None],
    *,
    mode: str,
) -> dict[str, dict[str, Any]]:
    """Score each scenario type separately so one family cannot mask another.

    A metric that is inapplicable to PipeFormer or OpenClaw changes that
    family's denominator, so the combined score alone can hide a regression.
    """

    indexes_by_type: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        indexes_by_type.setdefault(str(case.scenario_type or "unknown"), []).append(index)
    grouped: dict[str, dict[str, Any]] = {}
    for scenario_type in sorted(indexes_by_type):
        indexes = indexes_by_type[scenario_type]
        summary = _summary(
            [
                reports[index]
                for index in indexes
                if index < len(reports) and reports[index] is not None
            ],
            mode=mode,
            record_count=len(indexes),
        )
        summary.pop("by_scenario_type", None)
        grouped[scenario_type] = summary
    return grouped


def evaluate_dataset(args: Any) -> dict[str, Any]:
    """Run every case in one dataset and write ``rollouts.jsonl``/``summary.json``."""

    source_records = read_jsonl(Path(args.source))
    dry_run = bool(getattr(args, "dry_run", False))
    schema_source = getattr(args, "tool_schema_source", None)
    if not dry_run and not schema_source:
        raise ValueError("--tool-schema-source is required for non-dry evaluation")
    schema_records = (
        read_jsonl(Path(schema_source)) if schema_source else source_records
    )
    schemas_by_key = tool_schema_index(schema_records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(
        source_records,
        output_dir=output_dir,
        schemas_by_key=schemas_by_key,
        scenario_type=getattr(args, "scenario_type", None),
        limit=getattr(args, "limit", None),
        allow_generic_schemas=dry_run,
    )
    if not dry_run and not cases:
        raise ValueError("no evaluation records matched the requested scenario")

    rollouts_path = output_dir / "rollouts.jsonl"
    reports: list[EvaluationReport | None] = []
    latencies: list[float] = []
    record_count = len(cases)
    mode = "dry_run" if dry_run else "autonomous"
    config = None if dry_run else RolloutConfig(
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        capture_raw_responses=bool(getattr(args, "save_raw_responses", False)),
        capture_raw_tool_outputs=bool(getattr(args, "save_raw_tool_outputs", False)),
    )
    runner = None if dry_run else _build_runner(args, cases)
    with atomic_jsonl_writer(rollouts_path, default=str) as write_rollout:
        progress = tqdm(
            cases,
            total=len(cases),
            desc="Preparing evaluation" if dry_run else "Evaluating",
            unit="case",
        )
        for case in progress:
            report = None
            if dry_run:
                write_rollout(
                    {
                        "sample_id": case.sample_id,
                        "scenario_id": case.scenario_id,
                        "scenario_type": case.scenario_type,
                        "prompt_messages": case.messages,
                        "tools": case.tools,
                        "teacher_future_hidden": True,
                        "status": "dry_run",
                    }
                )
                status = "dry_run"
            else:
                source = case.source_record
                started = perf_counter()
                rollout = runner(case).run(case, config).to_dict()
                latency = perf_counter() - started
                rollout["latency_seconds"] = round(latency, 6)
                latencies.append(latency)
                report = evaluate(
                    rollout,
                    profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
                    reference=source,
                )
                attach_report(rollout, report)
                write_rollout(rollout)
                status = rollout.get("trace_status", "unknown")
                release_cuda_cache()
            reports.append(report)
            _set_postfix(progress, scenario=case.scenario_type, status=status)

    summary = _summary(
        [report for report in reports if report is not None],
        mode=mode,
        record_count=record_count,
    )
    summary["by_scenario_type"] = _by_scenario_type(cases, reports, mode=mode)
    summary["execution_mode"] = getattr(args, "execution_mode", "raw-student")
    summary["runtime"] = {
        "count": len(latencies),
        "average_latency_seconds": (
            round(sum(latencies) / len(latencies), 6) if latencies else None
        ),
        "minimum_latency_seconds": round(min(latencies), 6) if latencies else None,
        "maximum_latency_seconds": round(max(latencies), 6) if latencies else None,
    }
    atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
    )
    return summary


def _build_runner(args: Any, cases: Sequence[PromptCase]):
    """Load the model once and return a per-case runner selector.

    MS-SWIFT, PEFT, and torch are imported here rather than at module import so
    the dry-run path and the test suite never touch model weights or CUDA.
    """

    if getattr(args, "execution_mode", "raw-student") == "production-agent":
        from .production_agent import ProductionAgentRunner

        runner = ProductionAgentRunner()
        return lambda _case: runner

    adapters = getattr(args, "adapters", None)
    model = getattr(args, "model", None)
    if not adapters and not model:
        raise ValueError("--model or --adapters is required unless --dry-run is used")
    from .swift_generator import SwiftGenerator, discover_base_model

    if not model:
        model = discover_base_model(Path(adapters))
    generator = SwiftGenerator.from_args(
        model=model,
        adapters=adapters,
        device=getattr(args, "device", None),
        quant_bits=getattr(args, "quant_bits", None),
        no_quantization=bool(getattr(args, "no_quantization", False)),
        enable_thinking=bool(getattr(args, "enable_thinking", False)),
    )
    repo_root = Path(getattr(args, "repo_root", "."))
    policy = ScenarioPolicy()
    runners = {
        family: RolloutRunner(
            generator, build_dispatcher(family, schemas, repo_root), policy=policy
        )
        for family, schemas in schemas_by_scenario_family(cases).items()
    }

    def select(case: PromptCase) -> RolloutRunner:
        return runners[scenario_key(case.scenario_type)]

    return select
