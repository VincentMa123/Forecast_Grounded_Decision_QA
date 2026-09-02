from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pipeclaw.backend.agent.prompt_policy import static_forecast_policy
from pipeclaw.student_distillation.path_contract import (
    canonicalize_recorded_tool_arguments,
    redact_host_paths,
)
from pipeclaw.student_distillation.release_artifacts import (
    atomic_write_text as _atomic_write_text,
    sha256_bytes as _sha256_bytes,
    sha256_file as _sha256_file,
    stable_json,
    utc_now as _utc_now,
)
from pipeclaw.student_distillation.rollout.prompting import (
    trace_system_content as _trace_system_content,
)
from pipeclaw.student_distillation.scripts.validate_dataset import (
    CORRECTION_SPLITS,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    EXPECTED_SOURCE_COUNTS,
    PROJECTIONS,
    REPO_ROOT,
    SPLITS,
    DatasetValidationError,
    _normalize_tool_schemas,
    load_registered_tool_schemas,
    projection_writes_python,
    read_jsonl,
    validate_projection_records,
    validate_source_records,
)

TASK_PROMPTS = {
    "condition_parsing": "Parse the current pipeline request into the verified "
    "structured task. Return JSON only; do not add hidden reasoning.",
    "tool_planning": "Use the available tools and bounded verified context to "
    "execute the teacher's verified tool plan. Preserve exact canonical identifiers.",
    "constraint_judgment": "Convert the verified forecast checks into typed "
    "constraint, risk, and intervention judgments. Return JSON only.",
    "evidence_extraction": "Extract only the supplied verified evidence needed for the decision. Return JSON only.",
    "answer_generation": "Answer the current request using only the supplied "
    "verified evidence and decision state. Preserve canonical application disclosures.",
}
JUDGMENT_FIELDS = (
    "category_status",
    "overall_status",
    "verification_complete",
    "risk_level",
    "failure_count",
    "warning_count",
    "failed_rule_ids",
    "warning_rule_ids",
    "triggered_flags",
    "human_intervention_label",
    "dispatch_recommendation",
)
CONVERTER_VERSION = "1.1.0"


def project_answer_only(source: dict[str, Any], split: str) -> dict[str, Any]:
    """Project one compact Task 1 record into the answer-only baseline."""
    return _project_public(source, split, "answer_only")[0]


def project_trace_level(
    source: dict[str, Any], split: str, tool_schemas: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Project one compact Task 1 record into MS-SWIFT agent format."""
    return _project_public(source, split, "trace_level", tool_schemas)[0]


def project_constraint_multitask(
    source: dict[str, Any], split: str, tool_schemas: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Create nonempty structured auxiliary examples for the five Task 2 skills."""
    return _project_public(source, split, "constraint_multitask", tool_schemas)


def _project_public(
    source: dict[str, Any],
    split: str,
    projection: str,
    tool_schemas: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    validate_source_records([source], split=split, expected_count=1)
    source = redact_host_paths(source)
    schemas, names = (
        ([], set())
        if projection == "answer_only"
        else _normalize_tool_schemas(tool_schemas)
    )
    records = _project_records(source, split, projection, schemas, names)
    validate_projection_records(
        records,
        projection=projection,
        split=split,
        registered_tool_names=names,
    )
    return records


def _project_records(
    source: dict[str, Any],
    split: str,
    projection: str,
    tool_schemas: Sequence[dict[str, Any]],
    registered_names: set[str],
) -> list[dict[str, Any]]:
    """Project a source already validated and redacted exactly once."""
    if projection == "constraint_multitask":
        return _constraint_examples(source, split, tool_schemas, registered_names)
    record = _identity_fields(
        source,
        split=split,
        projection=projection,
        example_id=str(source["sample_id"]),
    )
    if projection == "answer_only":
        record["messages"] = [
            {"role": "system", "content": static_forecast_policy()},
            {"role": "user", "content": source["user_input"]},
            {"role": "assistant", "content": source["final_answer"], "loss": True},
        ]
    else:
        record["tools"] = stable_json(tool_schemas)
        record["messages"] = [
            {"role": "system", "content": _trace_system_content(source)},
            {"role": "user", "content": source["user_input"]},
            *_paired_tool_messages(source, registered_names),
            {"role": "assistant", "content": source["final_answer"], "loss": True},
        ]
    return [record]


def _constraint_examples(
    source: dict[str, Any],
    split: str,
    tool_schemas: Sequence[dict[str, Any]],
    registered_names: set[str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if source.get("parsed_task"):
        examples.append(_assistant_task_example(
            source, split=split, task_type="condition_parsing",
            input_payload={"user_input": source["user_input"]},
            target=stable_json(source["parsed_task"]),
        ))
    if source.get("tool_calls"):
        planning = _task_identity(source, split, "tool_planning")
        planning["tools"] = stable_json(tool_schemas)
        planning["messages"] = [
            {"role": "system", "content": _system_content(source, TASK_PROMPTS["tool_planning"])},
            {"role": "user", "content": source["user_input"]},
            *_paired_tool_messages(source, registered_names),
        ]
        examples.append(planning)
    judgments = extract_constraint_judgments(source)
    if judgments:
        examples.append(_assistant_task_example(
            source, split=split, task_type="constraint_judgment",
            input_payload={
                "user_input": source["user_input"],
                "verified_forecast_results": _forecast_outputs(source),
            },
            target=stable_json({"judgments": judgments}),
        ))
    if source.get("evidence"):
        examples.append(_assistant_task_example(
            source, split=split, task_type="evidence_extraction",
            input_payload={
                "user_input": source["user_input"],
                "verified_context": _verified_context(source, include_evidence=False),
            },
            target=stable_json(source["evidence"]),
        ))
    examples.append(_assistant_task_example(
        source, split=split, task_type="answer_generation",
        input_payload={
            "user_input": source["user_input"],
            "parsed_task": source["parsed_task"],
            "verified_context": _verified_context(source),
        },
        target=source["final_answer"],
    ))
    return examples


def extract_constraint_judgments(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract typed judgment labels from successful PipeFormer verification."""

    judgments: list[dict[str, Any]] = []
    for output in source.get("tool_outputs") or []:
        if output.get("name") != "run_pipeformer_forecast":
            continue
        payload = output.get("output")
        if not isinstance(payload, dict) or payload.get("success") is not True:
            continue
        verification = payload.get("verification")
        if not isinstance(verification, dict):
            continue
        judgment = {
            field: verification[field]
            for field in JUDGMENT_FIELDS
            if field in verification
        }
        if judgment:
            judgments.append(
                {
                    "tool_call_id": output.get("tool_call_id"),
                    **judgment,
                }
            )
    return judgments


def generate_datasets(
    *,
    source_root: Path,
    output_root: Path,
    manifest_path: Path,
    expected_counts: dict[str, int],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Generate and validate all Task 2 projections plus their manifest."""
    source_root, output_root, manifest_path = (
        source_root.resolve(), output_root.resolve(), manifest_path.resolve()
    )
    tool_schemas = load_registered_tool_schemas(REPO_ROOT)
    registered_names = {str(schema["function"]["name"]) for schema in tool_schemas}
    source_records: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    source_ids: dict[str, set[str]] = {}
    for split in SPLITS:
        if split not in expected_counts:
            raise DatasetValidationError(f"missing expected count for {split}")
        path = source_root / f"teacher_trace_{split}.jsonl"
        records = read_jsonl(path)
        validate_source_records(records, split=split, expected_count=expected_counts[split])
        source_records[split] = [redact_host_paths(record) for record in records]
        source_hashes[split] = _sha256_file(path)
        source_ids[split] = {str(record["sample_id"]) for record in records}
    _validate_disjoint_source_splits(source_ids)

    manifest: dict[str, Any] = {
        "schema_version": "task2_ms_swift_manifest_v1",
        "converter_version": CONVERTER_VERSION,
        "created_at_utc": created_at or _utc_now(),
        "tool_schemas": {
            "count": len(tool_schemas),
            "names": sorted(registered_names),
            "sha256": _sha256_bytes(stable_json(tool_schemas).encode("utf-8")),
        },
        "sources": {
            split: {
                "file": f"teacher_trace_{split}.jsonl",
                "record_count": len(source_records[split]),
                "sha256": source_hashes[split],
            }
            for split in SPLITS
        },
        "projections": {projection: {} for projection in PROJECTIONS},
        "corrective_datasets": {"python_script": {}},
    }
    trace_records: dict[str, list[dict[str, Any]]] = {}
    for projection in PROJECTIONS:
        for split in SPLITS:
            derived_records = [
                derived
                for source in source_records[split]
                for derived in _project_records(
                    source, split, projection, tool_schemas, registered_names
                )
            ]
            validate_projection_records(
                derived_records,
                projection=projection,
                split=split,
                registered_tool_names=registered_names,
            )
            output_path = output_root / projection / f"{split}.jsonl"
            _atomic_write_text(
                output_path,
                "".join(f"{stable_json(record)}\n" for record in derived_records),
            )
            reloaded = read_jsonl(output_path)
            validate_projection_records(
                reloaded,
                projection=projection,
                split=split,
                registered_tool_names=registered_names,
            )
            if reloaded != derived_records:
                raise DatasetValidationError(
                    f"{projection}/{split}: written records changed during serialization"
                )
            details: dict[str, Any] = {
                "file": output_path.relative_to(output_root).as_posix(),
                "record_count": len(derived_records),
                "sha256": _sha256_file(output_path),
            }
            if projection == "constraint_multitask":
                details["task_counts"] = dict(
                    sorted(Counter(str(record["task_type"]) for record in derived_records).items())
                )
            elif projection == "trace_level":
                trace_records[split] = derived_records
            manifest["projections"][projection][split] = details

    for split in CORRECTION_SPLITS:
        records = select_python_correction_records(trace_records[split])
        validate_projection_records(
            records,
            projection="trace_level",
            split=split,
            registered_tool_names=registered_names,
        )
        output_path = output_root / "python_correction" / f"{split}.jsonl"
        _atomic_write_text(
            output_path,
            "".join(f"{stable_json(record)}\n" for record in records),
        )
        manifest["corrective_datasets"]["python_script"][split] = {
            "file": output_path.relative_to(output_root).as_posix(),
            "record_count": len(records),
            "python_record_count": sum(projection_writes_python(record) for record in records),
            "sha256": _sha256_file(output_path),
        }
    for split in SPLITS:
        if _sha256_file(source_root / f"teacher_trace_{split}.jsonl") != source_hashes[split]:
            raise DatasetValidationError(f"{split}: authoritative source changed during generation")
    _atomic_write_text(manifest_path, stable_json(manifest) + "\n")
    return manifest


def _identity_fields(source: dict[str, Any], *, split: str, projection: str, example_id: str) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "source_sample_id": source["sample_id"],
        "scenario_id": source["scenario_id"],
        "session_id": source["session_id"],
        "turn_id": source["turn_id"],
        "scenario_type": source["scenario_type"],
        "split": split,
        "projection": projection,
    }


def select_python_correction_records(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep every Python-writing trace plus two deterministic replay traces each."""
    code_ids = {
        str(record.get("example_id") or "")
        for record in records
        if projection_writes_python(record)
    }
    replay = sorted(
        (r for r in records if str(r.get("example_id") or "") not in code_ids),
        key=lambda r: _sha256_bytes(str(r.get("example_id") or "").encode()),
    )[: 2 * len(code_ids)]
    selected = code_ids | {str(r.get("example_id") or "") for r in replay}
    return [r for r in records if str(r.get("example_id") or "") in selected]


def _task_identity(source: dict[str, Any], split: str, task_type: str) -> dict[str, Any]:
    record = _identity_fields(
        source,
        split=split,
        projection="constraint_multitask",
        example_id=f"{source['sample_id']}::{task_type}",
    )
    record["task_type"] = task_type
    return record


def _assistant_task_example(
    source: dict[str, Any],
    *,
    split: str,
    task_type: str,
    input_payload: dict[str, Any],
    target: str,
) -> dict[str, Any]:
    record = _task_identity(source, split, task_type)
    record["messages"] = [
        {"role": "system", "content": _system_content(source, TASK_PROMPTS[task_type])},
        {"role": "user", "content": stable_json(input_payload)},
        {"role": "assistant", "content": target, "loss": True},
    ]
    return record


def _system_content(source: dict[str, Any], prompt: str) -> str:
    context = {
        "state_before": source.get("state_before") or {},
        "recent_turns": source.get("recent_turns") or [],
    }
    return (
        f"{static_forecast_policy()}\n\n"
        f"{prompt}\nBounded verified context:\n{stable_json(context)}"
    )


def _verified_context(
    source: dict[str, Any], *, include_evidence: bool = True
) -> dict[str, Any]:
    defaults = {
        "state_before": {},
        "recent_turns": [],
        "tool_outputs": [],
        "decision_summary": {},
    }
    context = {key: source.get(key) or default for key, default in defaults.items()}
    if include_evidence:
        context["evidence"] = source.get("evidence") or {}
    return context


def _forecast_outputs(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        output
        for output in source.get("tool_outputs") or []
        if output.get("name") == "run_pipeformer_forecast"
        and isinstance(output.get("output"), dict)
        and output["output"].get("success") is True
    ]


def _paired_tool_messages(
    source: dict[str, Any], registered_names: set[str]
) -> list[dict[str, Any]]:
    outputs_by_id = {
        str(output["tool_call_id"]): output
        for output in source.get("tool_outputs") or []
    }
    messages: list[dict[str, Any]] = []
    for call in source.get("tool_calls") or []:
        call_id = str(call["tool_call_id"])
        name = str(call["name"])
        if name not in registered_names:
            raise DatasetValidationError(f"{source['sample_id']}: tool {name!r} is not registered")
        output = outputs_by_id.get(call_id)
        if output is None:
            raise DatasetValidationError(f"{source['sample_id']}: tool call {call_id} lacks an output")
        if output.get("name") != name:
            raise DatasetValidationError(f"{source['sample_id']}: tool name mismatch for {call_id}")
        payload = output.get("output")
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise DatasetValidationError(f"{source['sample_id']}: tool output {call_id} must be successful")
        arguments = canonicalize_recorded_tool_arguments(
            name, call.get("arguments") or {}
        )
        messages.extend((
            {
                "role": "tool_call",
                "content": stable_json({"name": name, "arguments": arguments}),
                "loss": True,
            },
            {
                "role": "tool_response",
                "content": stable_json(payload),
                "loss": False,
            },
        ))
    return messages


def _validate_disjoint_source_splits(source_ids_by_split: dict[str, set[str]]) -> None:
    for index, left_split in enumerate(SPLITS):
        for right_split in SPLITS[index + 1 :]:
            overlap = source_ids_by_split.get(
                left_split, set()
            ) & source_ids_by_split.get(right_split, set())
            if overlap:
                raise DatasetValidationError(
                    f"source split leakage between {left_split} and "
                    f"{right_split}: {sorted(overlap)[:3]}"
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert frozen Task 1 splits into MS-SWIFT datasets."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--created-at",
        help="Optional fixed UTC timestamp for reproducible manifest tests.",
    )
    args = parser.parse_args(argv)
    manifest = generate_datasets(
        source_root=args.source_root,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
        expected_counts=EXPECTED_SOURCE_COUNTS,
        created_at=args.created_at,
    )
    print(stable_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
