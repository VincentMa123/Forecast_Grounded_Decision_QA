from __future__ import annotations

import argparse
import ast
import json
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Sequence

from pipeclaw.backend.grounding.evidence.tool import (
    command_python_scripts,
    normalized_tool_path,
)
from pipeclaw.student_distillation.path_contract import (
    is_host_absolute_path,
    redact_host_paths,
)
from pipeclaw.student_distillation.release_artifacts import (
    read_jsonl_domain,
    required_text,
    sha256_bytes as _sha256_bytes,
    sha256_file as _sha256_file,
)


SOURCE_REQUIRED_FIELDS = frozenset(
    {
        "sample_id",
        "scenario_id",
        "session_id",
        "turn_id",
        "scenario_type",
        "state_before",
        "recent_turns",
        "user_input",
        "parsed_task",
        "tool_calls",
        "tool_outputs",
        "evidence",
        "decision_summary",
        "final_answer",
    }
)
CONSTRAINT_TASK_TYPES = frozenset(
    {
        "condition_parsing",
        "tool_planning",
        "constraint_judgment",
        "evidence_extraction",
        "answer_generation",
    }
)


class DatasetValidationError(ValueError):
    """Raised when a source or derived dataset violates a safety invariant."""


_required_text = partial(required_text, error_factory=DatasetValidationError)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file and reject blank, invalid, or non-object rows."""

    return read_jsonl_domain(path, error_factory=DatasetValidationError)


def validate_source_records(
    records: Sequence[dict[str, Any]],
    *,
    split: str,
    expected_count: int,
    tool_schemas: Iterable[dict[str, Any]] | None = None,
) -> None:
    """Validate one immutable compact Task 1 split before projection."""

    if len(records) != expected_count:
        raise DatasetValidationError(
            f"{split}: expected {expected_count} records, found {len(records)}"
        )
    seen_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        missing = SOURCE_REQUIRED_FIELDS - record.keys()
        if missing:
            raise DatasetValidationError(
                f"{split}:{index}: missing source fields {sorted(missing)}"
            )
        sample_id = _required_text(record, "sample_id", f"{split}:{index}")
        if sample_id in seen_ids:
            raise DatasetValidationError(f"{split}: duplicate sample_id {sample_id}")
        seen_ids.add(sample_id)
        for field in ("scenario_id", "session_id", "scenario_type", "user_input"):
            _required_text(record, field, sample_id)
        if not isinstance(record.get("turn_id"), int):
            raise DatasetValidationError(f"{sample_id}: turn_id must be an integer")
        _required_text(record, "final_answer", sample_id)
        _validate_source_tool_pairs(
            record,
            sample_id,
            tool_schemas=tool_schemas,
        )
        _validate_v8_dispatch_lifecycle(record, sample_id)


def validate_projection_records(
    records: Sequence[dict[str, Any]],
    *,
    projection: str,
    split: str,
    registered_tool_names: Iterable[str],
) -> None:
    """Validate one derived projection split before it is released."""

    allowed_tool_names = set(registered_tool_names)
    seen_example_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        location = f"{projection}/{split}:{index}"
        example_id = _required_text(record, "example_id", location)
        if example_id in seen_example_ids:
            raise DatasetValidationError(
                f"{projection}/{split}: duplicate example_id {example_id}"
            )
        seen_example_ids.add(example_id)
        _required_text(record, "source_sample_id", example_id)
        for field in ("scenario_id", "session_id", "scenario_type"):
            _required_text(record, field, example_id)
        if not isinstance(record.get("turn_id"), int):
            raise DatasetValidationError(f"{example_id}: turn_id must be an integer")
        if record.get("projection") != projection:
            raise DatasetValidationError(
                f"{example_id}: projection {record.get('projection')!r} "
                f"does not match {projection!r}"
            )
        if record.get("split") != split:
            raise DatasetValidationError(
                f"{example_id}: split {record.get('split')!r} does not match {split!r}"
            )
        _validate_messages(
            record,
            example_id=example_id,
            projection=projection,
            registered_tool_names=allowed_tool_names,
        )


def validate_release(
    *,
    source_root: Path,
    output_root: Path,
    manifest_path: Path,
    expected_counts: dict[str, int],
    registered_tool_names: Iterable[str],
    tool_schemas: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a complete generated release against sources and manifest."""

    split_names = ("train", "valid", "test")
    projection_names = (
        "answer_only",
        "trace_level",
        "constraint_multitask",
    )
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    manifest_path = manifest_path.resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"{manifest_path}: invalid or unreadable manifest"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "task2_ms_swift_manifest_v1"
    ):
        raise DatasetValidationError("unsupported dataset manifest schema")

    source_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    all_source_ids: dict[str, str] = {}
    for split in split_names:
        source_path = source_root / f"teacher_trace_{split}.jsonl"
        records = read_jsonl(source_path)
        validate_source_records(
            records,
            split=split,
            expected_count=expected_counts[split],
            tool_schemas=tool_schemas,
        )
        source_by_split[split] = {
            str(record["sample_id"]): record for record in records
        }
        for sample_id in source_by_split[split]:
            prior_split = all_source_ids.get(sample_id)
            if prior_split is not None:
                raise DatasetValidationError(
                    f"source split leakage for {sample_id}: {prior_split} and {split}"
                )
            all_source_ids[sample_id] = split
        source_details = (manifest.get("sources") or {}).get(split) or {}
        if source_details.get("record_count") != len(records):
            raise DatasetValidationError(
                f"{split}: source manifest record count mismatch"
            )
        if source_details.get("sha256") != _sha256_file(source_path):
            raise DatasetValidationError(f"{split}: source checksum mismatch")

    registered_names = set(registered_tool_names)
    tool_manifest = manifest.get("tool_schemas")
    if not isinstance(tool_manifest, dict):
        raise DatasetValidationError("tool schema manifest is missing")
    if tool_manifest.get("count") != len(registered_names):
        raise DatasetValidationError("tool schema count mismatch")
    if tool_manifest.get("names") != sorted(registered_names):
        raise DatasetValidationError("tool schema names mismatch")
    observed_tool_schema_hashes: set[str] = set()
    validated_files = 0
    for projection in projection_names:
        projection_manifest = (manifest.get("projections") or {}).get(projection)
        if not isinstance(projection_manifest, dict):
            raise DatasetValidationError(
                f"{projection}: projection manifest is missing"
            )
        for split in split_names:
            details = projection_manifest.get(split)
            if not isinstance(details, dict):
                raise DatasetValidationError(
                    f"{projection}/{split}: manifest entry is missing"
                )
            expected_relative = f"{projection}/{split}.jsonl"
            if details.get("file") != expected_relative:
                raise DatasetValidationError(
                    f"{projection}/{split}: unsafe or unexpected output path"
                )
            output_path = output_root / projection / f"{split}.jsonl"
            records = read_jsonl(output_path)
            validate_projection_records(
                records,
                projection=projection,
                split=split,
                registered_tool_names=registered_names,
            )
            if projection == "trace_level":
                observed_tool_schema_hashes.update(
                    _sha256_bytes(str(record["tools"]).encode("utf-8"))
                    for record in records
                )
            if details.get("record_count") != len(records):
                raise DatasetValidationError(
                    f"{projection}/{split}: manifest record count mismatch"
                )
            if details.get("sha256") != _sha256_file(output_path):
                raise DatasetValidationError(
                    f"{projection}/{split}: output checksum mismatch"
                )
            _validate_derived_identities(
                records,
                sources=source_by_split[split],
                projection=projection,
                split=split,
            )
            if projection == "constraint_multitask":
                task_counts: dict[str, int] = {}
                for record in records:
                    task_type = str(record["task_type"])
                    task_counts[task_type] = task_counts.get(task_type, 0) + 1
                if details.get("task_counts") != dict(sorted(task_counts.items())):
                    raise DatasetValidationError(
                        f"{projection}/{split}: task counts mismatch"
                    )
            validated_files += 1
    if observed_tool_schema_hashes != {tool_manifest.get("sha256")}:
        raise DatasetValidationError("tool schema checksum mismatch")
    correction_manifest = (manifest.get("corrective_datasets") or {}).get(
        "python_script"
    )
    if correction_manifest is not None and not isinstance(correction_manifest, dict):
        raise DatasetValidationError("python correction manifest is invalid")
    for split in ("train", "valid") if correction_manifest is not None else ():
        details = correction_manifest.get(split)
        if not isinstance(details, dict):
            raise DatasetValidationError(
                f"python_correction/{split}: manifest entry is missing"
            )
        expected_relative = f"python_correction/{split}.jsonl"
        if details.get("file") != expected_relative:
            raise DatasetValidationError(
                f"python_correction/{split}: unsafe or unexpected output path"
            )
        path = output_root / expected_relative
        records = read_jsonl(path)
        validate_projection_records(
            records,
            projection="trace_level",
            split=split,
            registered_tool_names=registered_names,
        )
        if details.get("record_count") != len(records):
            raise DatasetValidationError(
                f"python_correction/{split}: record count mismatch"
            )
        if details.get("python_record_count") != sum(
            projection_writes_python(record) for record in records
        ):
            raise DatasetValidationError(
                f"python_correction/{split}: Python record count mismatch"
            )
        if details.get("sha256") != _sha256_file(path):
            raise DatasetValidationError(
                f"python_correction/{split}: checksum mismatch"
            )
        source_ids = source_by_split[split]
        if any(record.get("source_sample_id") not in source_ids for record in records):
            raise DatasetValidationError(
                f"python_correction/{split}: unknown source record"
            )
        validated_files += 1
    return {
        "validated_projection_files": validated_files,
        "validated_source_records": sum(
            len(records) for records in source_by_split.values()
        ),
    }


def _required_tool_arguments(
    tool_schemas: Iterable[dict[str, Any]] | None,
) -> dict[str, set[str]]:
    required_by_name: dict[str, set[str]] = {}
    for schema in tool_schemas or ():
        function = schema.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        required = parameters.get("required")
        if isinstance(required, list):
            required_by_name[name] = {
                str(field) for field in required if isinstance(field, str)
            }
    return required_by_name


def _validate_source_tool_pairs(
    record: dict[str, Any],
    sample_id: str,
    *,
    tool_schemas: Iterable[dict[str, Any]] | None = None,
) -> None:
    calls = record.get("tool_calls")
    outputs = record.get("tool_outputs")
    if not isinstance(calls, list) or not isinstance(outputs, list):
        raise DatasetValidationError(
            f"{sample_id}: tool_calls and tool_outputs must be lists"
        )
    calls_by_id: dict[str, dict[str, Any]] = {}
    required_by_name = _required_tool_arguments(tool_schemas)
    for call in calls:
        if not isinstance(call, dict):
            raise DatasetValidationError(f"{sample_id}: tool call must be an object")
        call_id = _required_text(call, "tool_call_id", sample_id)
        if call_id in calls_by_id:
            raise DatasetValidationError(
                f"{sample_id}: duplicate tool_call_id {call_id}"
            )
        _required_text(call, "name", sample_id)
        if not isinstance(call.get("arguments"), dict):
            raise DatasetValidationError(
                f"{sample_id}: tool call {call_id} arguments must be an object"
            )
        tool_name = str(call.get("name") or "")
        missing_arguments = required_by_name.get(tool_name, set()) - set(
            call["arguments"]
        )
        if missing_arguments:
            raise DatasetValidationError(
                f"{sample_id}: tool {tool_name} missing required arguments "
                f"{sorted(missing_arguments)}"
            )
        calls_by_id[call_id] = call

    _validate_python_tool_sequence(calls, sample_id)

    outputs_by_id: dict[str, dict[str, Any]] = {}
    for output in outputs:
        if not isinstance(output, dict):
            raise DatasetValidationError(f"{sample_id}: tool output must be an object")
        call_id = _required_text(output, "tool_call_id", sample_id)
        if call_id in outputs_by_id:
            raise DatasetValidationError(
                f"{sample_id}: duplicate tool output for {call_id}"
            )
        outputs_by_id[call_id] = output

    if calls_by_id.keys() != outputs_by_id.keys():
        raise DatasetValidationError(
            f"{sample_id}: every tool call must have exactly one matching output"
        )
    for call_id, call in calls_by_id.items():
        output = outputs_by_id[call_id]
        if output.get("name") != call.get("name"):
            raise DatasetValidationError(
                f"{sample_id}: tool name mismatch for {call_id}"
            )
        payload = output.get("output")
        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise DatasetValidationError(
                f"{sample_id}: tool output {call_id} must be successful"
            )


def _validate_v8_dispatch_lifecycle(record: dict[str, Any], sample_id: str) -> None:
    is_v8 = record.get(
        "dataset_source"
    ) == "Pipeline_Full_Life_Cycle_Test_Dataset-v8" or sample_id.startswith(
        "Pipeline_Full_Life_Cycle_Test_Dataset-v8:"
    )
    if not is_v8 or not str(record.get("scenario_id") or "").startswith(
        "scenario_pipeformer_dispatch_"
    ):
        return

    calls = record.get("tool_calls") or []
    names = [call.get("name") for call in calls]
    registry_count = next(
        (
            index
            for index, name in enumerate(names)
            if name != "search_pipeformer_registry"
        ),
        len(names),
    )
    expected_names = ["search_pipeformer_registry"] * registry_count + [
        "run_pipeformer_forecast",
        "run_pipeformer_forecast",
        "run_pipeformer_forecast",
        "set_decision_policy",
    ]
    candidate_ids = [
        call.get("arguments", {}).get("candidate_id")
        for call in calls
        if call.get("name") == "run_pipeformer_forecast"
    ]
    forecast_arguments = [
        dict(call.get("arguments") or {})
        for call in calls
        if call.get("name") == "run_pipeformer_forecast"
    ]
    context_keys = (
        "case_id",
        "disturbance_variable",
        "disturbance_direction",
        "disturbance_magnitude_percent",
        "forecast_horizon_minutes",
    )
    contexts = {
        tuple(arguments.get(key) for key in context_keys)
        for arguments in forecast_arguments
    }
    actions = {
        json.dumps(
            {
                key: dict((arguments.get("boundary_conditions") or {}).get(key) or {})
                for key in ("setpoints", "percentage_changes")
            },
            sort_keys=True,
        )
        for arguments in forecast_arguments
    }
    registry_arguments = [dict(call.get("arguments") or {}) for call in calls[:2]]
    disturbance_variable = (
        forecast_arguments[0].get("disturbance_variable")
        if forecast_arguments
        else None
    )
    answer = str(record.get("final_answer") or "")
    if (
        registry_count != 2
        or names != expected_names
        or candidate_ids != ["candidate_1", "candidate_2", "candidate_3"]
        or registry_arguments[0].get("query") != disturbance_variable
        or registry_arguments[1].get("query") != ""
        or registry_arguments[1].get("role") != "input"
        or registry_arguments[1].get("controllable") is not True
        or len(contexts) != 1
        or len(actions) != 3
        or "selected_candidate_id" not in answer
        or any(candidate_id not in answer for candidate_id in candidate_ids)
    ):
        raise DatasetValidationError(
            f"{sample_id}: v8 dispatch lifecycle must be two registry lookups, "
            "candidate_1/2/3 forecasts, one decision policy, and a ranked answer"
        )


def _validate_messages(
    record: dict[str, Any],
    *,
    example_id: str,
    projection: str,
    registered_tool_names: set[str],
) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise DatasetValidationError(f"{example_id}: messages must be a nonempty list")
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise DatasetValidationError(
                f"{example_id}: message {message_index} must be an object"
            )
        if message.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool_call",
            "tool_response",
        }:
            raise DatasetValidationError(
                f"{example_id}: unsupported message role {message.get('role')!r}"
            )
        if not isinstance(message.get("content"), str):
            raise DatasetValidationError(
                f"{example_id}: message {message_index} content must be a string"
            )
        content = message["content"]
        if (
            message.get("role") in {"system", "tool_call"}
            and "windows-first" in content.casefold()
        ):
            raise DatasetValidationError(
                f"{example_id}: generated contract contains host-specific path policy"
            )
        if message.get("role") == "tool_call":
            try:
                parsed_call = json.loads(content)
            except json.JSONDecodeError:
                parsed_call = None
            if isinstance(parsed_call, dict):
                arguments = parsed_call.get("arguments")
                if isinstance(arguments, dict) and is_host_absolute_path(
                    arguments.get("cwd")
                ):
                    raise DatasetValidationError(
                        f"{example_id}: tool_call cwd must be workspace-relative or omitted"
                    )

    task_type = record.get("task_type")
    if projection == "constraint_multitask":
        if task_type not in CONSTRAINT_TASK_TYPES:
            raise DatasetValidationError(
                f"{example_id}: unsupported constraint task_type {task_type!r}"
            )
    elif task_type is not None:
        raise DatasetValidationError(
            f"{example_id}: task_type is only valid for constraint_multitask"
        )

    requires_tools = projection == "trace_level" or task_type == "tool_planning"
    if requires_tools:
        schema_names = _validate_tools(record, example_id)
        if not schema_names <= registered_tool_names:
            unknown = sorted(schema_names - registered_tool_names)
            raise DatasetValidationError(
                f"{example_id}: unregistered tool schema {unknown}"
            )

    # Recovery records are student-failure-shaped rows: they deliberately include
    # the failed tool_response so the model learns the post-error transition.
    # Allowed ONLY under the explicit opt-in; ordinary teacher traces never qualify.
    allow_failures = record.get("quality_flag") == "recovery"
    call_count = _validate_tool_messages(
        messages,
        example_id=example_id,
        registered_tool_names=registered_tool_names,
        allow_failed_responses=allow_failures,
    )
    if projection == "trace_level" and messages[-1].get("role") != "assistant":
        raise DatasetValidationError(
            f"{example_id}: trace must end with a final assistant answer"
        )
    if projection == "trace_level" and call_count == 0 and "tools" not in record:
        raise DatasetValidationError(f"{example_id}: trace tools are missing")
    if task_type == "tool_planning" and call_count == 0:
        raise DatasetValidationError(
            f"{example_id}: tool_planning must contain at least one tool_call"
        )

    assistant_messages = [
        message for message in messages if message.get("role") == "assistant"
    ]
    if projection != "constraint_multitask" or task_type != "tool_planning":
        if not assistant_messages or not assistant_messages[-1]["content"].strip():
            raise DatasetValidationError(
                f"{example_id}: final assistant response is required"
            )
        if assistant_messages[-1].get("loss") is not True:
            raise DatasetValidationError(f"{example_id}: assistant must receive loss")
    if task_type in {
        "condition_parsing",
        "constraint_judgment",
        "evidence_extraction",
    }:
        structured_target = _parse_json(
            assistant_messages[-1]["content"],
            f"{example_id}: structured assistant target",
        )
        if not isinstance(structured_target, dict):
            raise DatasetValidationError(
                f"{example_id}: structured assistant target must be an object"
            )
    for message in [
        *assistant_messages,
        *(item for item in messages if item.get("role") == "tool_call"),
    ]:
        if "<think>" in message["content"].casefold():
            raise DatasetValidationError(
                f"{example_id}: hidden chain-of-thought targets are forbidden"
            )


def _validate_tools(record: dict[str, Any], example_id: str) -> set[str]:
    raw_tools = record.get("tools")
    if not isinstance(raw_tools, str):
        raise DatasetValidationError(f"{example_id}: tools must be a JSON string")
    if "windows-first" in raw_tools.casefold():
        raise DatasetValidationError(
            f"{example_id}: tool schema contains host-specific path policy"
        )
    tools = _parse_json(raw_tools, f"{example_id}: tools")
    if not isinstance(tools, list) or not tools:
        raise DatasetValidationError(f"{example_id}: tools must contain schemas")
    names: set[str] = set()
    for schema in tools:
        try:
            name = schema["function"]["name"]
        except (KeyError, TypeError) as exc:
            raise DatasetValidationError(
                f"{example_id}: invalid function tool schema"
            ) from exc
        if not isinstance(name, str) or not name:
            raise DatasetValidationError(f"{example_id}: tool schema name is required")
        if name in names:
            raise DatasetValidationError(f"{example_id}: duplicate tool schema {name}")
        names.add(name)
    return names


def _validate_tool_messages(
    messages: Sequence[dict[str, Any]],
    *,
    example_id: str,
    registered_tool_names: set[str],
    allow_failed_responses: bool = False,
) -> int:
    call_count = 0
    parsed_calls: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "tool_response":
            raise DatasetValidationError(
                f"{example_id}: tool_response must follow a tool_call"
            )
        if role != "tool_call":
            index += 1
            continue
        call_count += 1
        call = _parse_json(message["content"], f"{example_id}: tool_call")
        if not isinstance(call, dict):
            raise DatasetValidationError(
                f"{example_id}: tool_call content must be an object"
            )
        name = call.get("name")
        if name not in registered_tool_names:
            raise DatasetValidationError(f"{example_id}: unregistered tool {name!r}")
        if not isinstance(call.get("arguments"), dict):
            raise DatasetValidationError(
                f"{example_id}: tool_call arguments must be an object"
            )
        parsed_calls.append(call)
        if message.get("loss") is not True:
            raise DatasetValidationError(f"{example_id}: tool_call must receive loss")
        if (
            index + 1 >= len(messages)
            or messages[index + 1].get("role") != "tool_response"
        ):
            raise DatasetValidationError(
                f"{example_id}: tool_call must have a matching tool_response"
            )
        response = messages[index + 1]
        if response.get("loss") is not False:
            raise DatasetValidationError(
                f"{example_id}: tool_response must not receive loss"
            )
        payload = _parse_json(response["content"], f"{example_id}: tool_response")
        if not isinstance(payload, dict) or payload.get("success") is not True:
            if not allow_failed_responses:
                raise DatasetValidationError(
                    f"{example_id}: tool_response must be successful"
                )
        index += 2
    _validate_python_tool_sequence(parsed_calls, example_id)
    return call_count


def _validate_python_tool_sequence(
    calls: Sequence[dict[str, Any]],
    location: str,
) -> None:
    written_scripts: set[str] = set()
    for call in calls:
        name = str(call.get("name") or "")
        arguments = call.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        if name == "write_file":
            path = normalized_tool_path(arguments.get("path"))
            if not path.endswith(".py"):
                continue
            content = arguments.get("content")
            try:
                if (
                    not isinstance(content, str)
                    or "truncated for sft" in content.casefold()
                ):
                    raise SyntaxError("truncated or missing Python content")
                ast.parse(content)
            except (SyntaxError, ValueError, TypeError) as exc:
                raise DatasetValidationError(
                    f"{location}: write_file for {path!r} must contain complete valid Python"
                ) from exc
            written_scripts.add(path)
        elif name == "run_command":
            for script in command_python_scripts(arguments):
                if script not in written_scripts:
                    raise DatasetValidationError(
                        f"{location}: Python execution for {script!r} requires a "
                        "preceding write_file with the complete script"
                    )


def projection_writes_python(record: dict[str, Any]) -> bool:
    for message in record.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "tool_call":
            continue
        try:
            call = json.loads(str(message.get("content") or ""))
        except json.JSONDecodeError:
            continue
        arguments = call.get("arguments") if isinstance(call, dict) else None
        if (
            call.get("name") == "write_file"
            and isinstance(arguments, dict)
            and normalized_tool_path(arguments.get("path")).endswith(".py")
        ):
            return True
    return False


def _parse_json(value: str, location: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"{location}: invalid JSON") from exc


def _validate_derived_identities(
    records: Sequence[dict[str, Any]],
    *,
    sources: dict[str, dict[str, Any]],
    projection: str,
    split: str,
) -> None:
    seen_source_task_pairs: set[tuple[str, str]] = set()
    seen_source_ids: set[str] = set()
    for record in records:
        source_sample_id = str(record["source_sample_id"])
        source = sources.get(source_sample_id)
        if source is None:
            raise DatasetValidationError(
                f"{projection}/{split}: unknown source_sample_id {source_sample_id}"
            )
        source = redact_host_paths(source)
        for field in ("scenario_id", "session_id", "turn_id", "scenario_type"):
            if record.get(field) != source.get(field):
                raise DatasetValidationError(
                    f"{record['example_id']}: source identity field {field} changed"
                )
        messages = record["messages"]
        if projection in {"answer_only", "trace_level"}:
            user_messages = [
                message for message in messages if message.get("role") == "user"
            ]
            if not user_messages or user_messages[0].get("content") != source.get(
                "user_input"
            ):
                raise DatasetValidationError(
                    f"{record['example_id']}: user input changed"
                )
            if messages[-1].get("content") != source.get("final_answer"):
                raise DatasetValidationError(
                    f"{record['example_id']}: final answer changed"
                )
        if projection == "constraint_multitask":
            task_type = str(record["task_type"])
            key = (source_sample_id, task_type)
            if key in seen_source_task_pairs:
                raise DatasetValidationError(
                    f"{projection}/{split}: duplicate source task {key}"
                )
            seen_source_task_pairs.add(key)
            if task_type == "answer_generation":
                if messages[-1].get("content") != source.get("final_answer"):
                    raise DatasetValidationError(
                        f"{record['example_id']}: final answer changed"
                    )
            elif task_type == "condition_parsing":
                if _parse_json(
                    messages[-1]["content"],
                    f"{record['example_id']}: condition target",
                ) != source.get("parsed_task"):
                    raise DatasetValidationError(
                        f"{record['example_id']}: parsed task changed"
                    )
            elif task_type == "evidence_extraction":
                if _parse_json(
                    messages[-1]["content"],
                    f"{record['example_id']}: evidence target",
                ) != source.get("evidence"):
                    raise DatasetValidationError(
                        f"{record['example_id']}: evidence target changed"
                    )
        else:
            if source_sample_id in seen_source_ids:
                raise DatasetValidationError(
                    f"{projection}/{split}: duplicate source_sample_id "
                    f"{source_sample_id}"
                )
            seen_source_ids.add(source_sample_id)
    if projection in {"answer_only", "trace_level"}:
        if seen_source_ids != sources.keys():
            raise DatasetValidationError(
                f"{projection}/{split}: source coverage mismatch"
            )
    else:
        answer_sources = {
            source_id
            for source_id, task_type in seen_source_task_pairs
            if task_type == "answer_generation"
        }
        if answer_sources != sources.keys():
            raise DatasetValidationError(
                f"{projection}/{split}: answer_generation coverage mismatch"
            )


def main(argv: Sequence[str] | None = None) -> int:
    from pipeclaw.student_distillation.scripts.prepare_dataset import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_SOURCE_ROOT,
        EXPECTED_SOURCE_COUNTS,
        REPO_ROOT,
        load_registered_tool_schemas,
        stable_json,
    )

    parser = argparse.ArgumentParser(
        description="Validate generated Task 2 MS-SWIFT datasets."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    args = parser.parse_args(argv)
    schemas = load_registered_tool_schemas(REPO_ROOT)
    result = validate_release(
        source_root=args.source_root,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
        expected_counts=EXPECTED_SOURCE_COUNTS,
        registered_tool_names={str(schema["function"]["name"]) for schema in schemas},
        tool_schemas=schemas,
    )
    print(stable_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
