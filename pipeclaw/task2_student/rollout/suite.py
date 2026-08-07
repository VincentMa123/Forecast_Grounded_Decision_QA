"""Dataset-level rollout orchestration and schema-v2 scoring.

This module is the single place where rollout *execution* meets rollout
*evaluation*: it runs each case through :class:`RolloutRunner` and then scores
the resulting trajectory with ``pipeclaw.backend.evaluator``.  ``models``,
``prompting``, ``runner``, and ``tools`` stay free of evaluator imports so the
execution core remains testable without teacher oracles or score weights.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeclaw.backend.evaluator import (
    EvaluationProfile,
    EvaluationReport,
    evaluate,
    summarize,
)

from .models import PromptCase, RolloutConfig
from .prompting import PromptCaseBuilder
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


# The canonical schema-v2 fields copied from the report onto every rollout
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

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, Mapping):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(dict(record))
    return records


def _progress_cases(
    cases: Sequence[tuple[dict[str, Any], PromptCase]],
    *,
    description: str,
):
    """Wrap case iterations with a terminal progress bar."""

    return tqdm(cases, total=len(cases), desc=description, unit="case")


def sample_keys(record: Mapping[str, Any]) -> set[str]:
    """Return every identifier a record may be joined on across files."""

    keys = set()
    for key in ("sample_id", "example_id", "source_sample_id", "source_id"):
        value = record.get(key)
        if value is not None:
            keys.add(str(value))
    return keys


def tool_schema_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Index OpenAI tool schemas by every identifier of their record."""

    index: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        schemas = record.get("tools")
        if isinstance(schemas, str):
            try:
                schemas = json.loads(schemas)
            except json.JSONDecodeError:
                schemas = []
        if isinstance(schemas, Mapping):
            schemas = [schemas]
        if not isinstance(schemas, Sequence) or isinstance(schemas, (str, bytes)):
            schemas = []
        normalised = [dict(schema) for schema in schemas if isinstance(schema, Mapping)]
        for key in sample_keys(record):
            index[key] = normalised
    return index


def generic_schemas(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Fallback schemas for dry-runs when a projection file is unavailable."""

    tool_calls = source.get("tool_calls")
    items = (
        tool_calls
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes))
        else []
    )
    names: set[str] = set()
    for item in items:
        if isinstance(item, Mapping) and item.get("name"):
            names.add(str(item["name"]))
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


def _scenario_matches(source: Mapping[str, Any], requested: str) -> bool:
    """Compare a record's scenario type with the requested one, alias-aware."""

    if not requested:
        return True
    requested_key = (
        "openclaw" if is_openclaw_scenario(requested) else str(requested).casefold()
    )
    source_type = source.get("scenario_type")
    source_key = (
        "openclaw"
        if is_openclaw_scenario(source_type)
        else str(source_type or "").casefold()
    )
    return source_key == requested_key


def build_cases(
    source_records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    schemas_by_key: Mapping[str, Sequence[Mapping[str, Any]]],
    scenario_type: str | None = None,
    limit: int | None = None,
) -> list[tuple[dict[str, Any], PromptCase]]:
    """Build prompt-only cases with per-scenario isolated workspaces."""

    builder = PromptCaseBuilder()
    requested = str(scenario_type or "")
    cases: list[tuple[dict[str, Any], PromptCase]] = []
    for source in source_records:
        if not _scenario_matches(source, requested):
            continue
        keys = sample_keys(source)
        schemas = next(
            (schemas_by_key[key] for key in keys if key in schemas_by_key),
            generic_schemas(source),
        )
        case = builder.build(
            source,
            workspace_root=workspace_for(
                output_dir, evaluation_workspace_key(source)
            ),
            tool_schemas=schemas,
        )
        cases.append((dict(source), case))
        if limit is not None and len(cases) >= limit:
            break
    return cases


def schemas_by_scenario_family(
    cases: Sequence[tuple[Mapping[str, Any], PromptCase]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Union each family's tool schemas, deduplicated by function name."""

    families: dict[str, list[Mapping[str, Any]]] = {}
    seen: dict[str, set[str]] = {}
    for _, case in cases:
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
    """Copy the canonical schema-v2 root fields onto one rollout record.

    The report is flattened rather than nested so a rollout carries exactly one
    score, one metric mapping, and one set of diagnostics.
    """

    payload = report.to_dict()
    rollout["evaluation_schema_version"] = payload["schema_version"]
    for field in _REPORT_ROOT_FIELDS:
        rollout[field] = payload[field]
    return rollout


def _dry_run_item(case: PromptCase) -> dict[str, Any]:
    return {
        "sample_id": case.sample_id,
        "scenario_id": case.scenario_id,
        "scenario_type": case.scenario_type,
        "prompt_messages": case.messages,
        "tools": case.tools,
        "teacher_future_hidden": True,
        "status": "dry_run",
    }


def _set_postfix(progress: Any, **fields: Any) -> None:
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(refresh=False, **fields)


def _release_cuda_cache() -> None:
    """Release unreachable per-case tensors and unused CUDA allocator blocks."""

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _summary(
    reports: Sequence[EvaluationReport],
    *,
    mode: str,
    record_count: int,
) -> dict[str, Any]:
    """Build one schema-v2 summary, including for dry runs with no reports."""

    summary = summarize(reports)
    summary["mode"] = mode
    summary["record_count"] = record_count
    return summary


def _scenario_summary(
    reports: Sequence[EvaluationReport],
    *,
    mode: str,
    record_count: int,
) -> dict[str, Any]:
    """Build one per-family summary without a recursive scenario breakdown."""

    summary = _summary(reports, mode=mode, record_count=record_count)
    summary.pop("by_scenario_type", None)
    return summary


def _by_scenario_type(
    cases: Sequence[tuple[Mapping[str, Any], PromptCase]],
    reports: Sequence[EvaluationReport | None],
    *,
    mode: str,
) -> dict[str, dict[str, Any]]:
    """Score each scenario type separately so one family cannot mask another.

    A metric that is inapplicable to PipeFormer or OpenClaw changes that
    family's denominator, so the combined score alone can hide a regression.
    """

    grouped: dict[str, dict[str, Any]] = {}
    for scenario_type in sorted(
        {str(case.scenario_type or "unknown") for _, case in cases}
    ):
        indexes = [
            index
            for index, (_, case) in enumerate(cases)
            if str(case.scenario_type or "unknown") == scenario_type
        ]
        grouped[scenario_type] = _scenario_summary(
            [
                reports[index]
                for index in indexes
                if index < len(reports) and reports[index] is not None
            ],
            mode=mode,
            record_count=len(indexes),
        )
    return grouped


def evaluate_dataset(args: Any) -> dict[str, Any]:
    """Run every case in one dataset and write ``rollouts.jsonl``/``summary.json``."""

    source_records = read_jsonl(Path(args.source))
    schema_records = (
        read_jsonl(Path(args.tool_schema_source))
        if getattr(args, "tool_schema_source", None)
        else source_records
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
    )

    rollouts_path = output_dir / "rollouts.jsonl"
    results: list[dict[str, Any]] = []
    reports: list[EvaluationReport | None] = []
    mode = "dry_run" if args.dry_run else "autonomous"
    with rollouts_path.open("w", encoding="utf-8") as handle:
        if args.dry_run:
            progress = _progress_cases(cases, description="Preparing evaluation")
            for _, case in progress:
                item = _dry_run_item(case)
                handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
                results.append(item)
                reports.append(None)
                _set_postfix(progress, scenario=case.scenario_type, status="dry_run")
        else:
            runner = _build_runner(args, cases)
            progress = _progress_cases(cases, description="Evaluating")
            for source, case in progress:
                config = RolloutConfig(
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    capture_raw_responses=bool(
                        getattr(args, "save_raw_responses", False)
                    ),
                    capture_raw_tool_outputs=bool(
                        getattr(args, "save_raw_tool_outputs", False)
                    ),
                )
                rollout = runner(case).run(case, config).to_dict()
                report = evaluate(
                    rollout,
                    profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
                    reference=source,
                )
                attach_report(rollout, report)
                handle.write(json.dumps(rollout, ensure_ascii=False, default=str) + "\n")
                results.append(rollout)
                reports.append(report)
                _set_postfix(
                    progress,
                    scenario=case.scenario_type,
                    status=rollout.get("trace_status", "unknown"),
                )
                _release_cuda_cache()

    summary = _summary(
        [report for report in reports if report is not None],
        mode=mode,
        record_count=len(results),
    )
    summary["by_scenario_type"] = _by_scenario_type(cases, reports, mode=mode)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    return summary


def _build_runner(args: Any, cases: Sequence[tuple[Mapping[str, Any], PromptCase]]):
    """Load the model once and return a per-case runner selector.

    MS-SWIFT, PEFT, and torch are imported here rather than at module import so
    the dry-run path and the test suite never touch model weights or CUDA.
    """

    if not getattr(args, "adapters", None):
        raise ValueError("--adapters is required unless --dry-run is used")
    from .swift_generator import SwiftGenerator, discover_base_model

    model = getattr(args, "model", None) or discover_base_model(Path(args.adapters))
    generator = SwiftGenerator.from_args(
        model=model,
        adapters=args.adapters,
        device=getattr(args, "device", None),
        quant_bits=getattr(args, "quant_bits", None),
        no_quantization=bool(getattr(args, "no_quantization", False)),
    )
    repo_root = Path(getattr(args, "repo_root", "."))
    policy = ScenarioPolicy()
    dispatchers: dict[str, ToolDispatcher] = {
        family: build_dispatcher(family, schemas, repo_root)
        for family, schemas in schemas_by_scenario_family(cases).items()
    }
    runners = {
        family: RolloutRunner(generator, dispatcher, policy=policy)
        for family, dispatcher in dispatchers.items()
    }

    def select(case: PromptCase) -> RolloutRunner:
        return runners[scenario_key(case.scenario_type)]

    return select
