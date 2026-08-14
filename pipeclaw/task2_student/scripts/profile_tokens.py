"""Exact, fail-closed token profiling for Task 2 MS-SWIFT datasets."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from pipeclaw.task2_student.release_artifacts import (
    JsonlArtifactError,
    atomic_write_text,
    read_jsonl as _read_jsonl_artifact,
    sha256_bytes,
    sha256_file as _sha256_file,
    stable_json as _stable_json,
    utc_now as _utc_now,
)


DEFAULT_THRESHOLDS = (1024, 2048, 4096, 8192, 16384)
PROFILE_PROJECTIONS = (
    "answer_only",
    "trace_level",
    "constraint_multitask",
)
PROFILE_PROJECTION_CHOICES = (*PROFILE_PROJECTIONS, "python_correction")
PROFILE_SPLITS = ("train", "valid")
FIELD_ROLES = {
    "system": "system_prompt",
    "user": "user_messages",
    "tool_call": "tool_calls",
    "tool_response": "tool_responses",
    "assistant": "assistant_targets",
}
FIELD_NAMES = (
    "system_prompt",
    "user_messages",
    "tool_schemas",
    "tool_calls",
    "tool_responses",
    "assistant_targets",
)
DEFAULT_LOSS_SCALE = "default+ignore_empty_think"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "pipeclaw" / "task2_student" / "data"
DEFAULT_MANIFEST_PATH = (
    DEFAULT_DATA_ROOT / "manifests" / "task2_dataset_manifest.json"
)
DEFAULT_SUMMARY_PATH = (
    DEFAULT_DATA_ROOT / "token_profiles" / "qwen35_08b_token_profile.json"
)
DEFAULT_RECORDS_PATH = (
    DEFAULT_DATA_ROOT / "token_profiles" / "qwen35_08b_token_records.jsonl"
)
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


class TokenProfileError(ValueError):
    """Raised when a token profile cannot be produced safely."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return _read_jsonl_artifact(path)
    except JsonlArtifactError as exc:
        raise TokenProfileError(str(exc)) from exc


@dataclass(frozen=True)
class EncodedRecord:
    """Token counts returned by a complete, untruncated template encoding."""

    total_tokens: int
    supervised_tokens: int


@dataclass(frozen=True)
class ProfileInput:
    """One manifest-verified projection file selected for profiling."""

    projection: str
    split: str
    path: Path
    records: list[dict[str, Any]]
    sha256: str


class RecordEncoder(Protocol):
    """Boundary implemented by the exact MS-SWIFT template adapter."""

    def encode_record(self, record: dict[str, Any]) -> EncodedRecord:
        """Encode one complete record without a maximum length."""


class TextTokenCounter(Protocol):
    """Boundary for counting raw field content with the selected processor."""

    def count_text(self, text: str) -> int:
        """Return the number of tokenizer tokens in one text value."""


class SwiftTemplateEncoder:
    """Processor-only MS-SWIFT adapter for exact training-template encoding."""

    def __init__(
        self,
        model_id: str,
        *,
        loss_scale: str = DEFAULT_LOSS_SCALE,
    ) -> None:
        swift_module = load_swift_api()
        processor = swift_module.get_processor(model_id)
        template = swift_module.get_template(
            processor,
            loss_scale=loss_scale,
        )
        self._initialize(
            model_id=model_id,
            processor=processor,
            template=template,
            loss_scale=loss_scale,
            swift_version=str(getattr(swift_module, "__version__", "unknown")),
        )

    @classmethod
    def from_components(
        cls,
        *,
        model_id: str,
        processor: Any,
        template: Any,
        swift_version: str,
        loss_scale: str = DEFAULT_LOSS_SCALE,
    ) -> SwiftTemplateEncoder:
        """Construct an adapter around already-loaded processor components."""

        instance = cls.__new__(cls)
        instance._initialize(
            model_id=model_id,
            processor=processor,
            template=template,
            loss_scale=loss_scale,
            swift_version=swift_version,
        )
        return instance

    def _initialize(
        self,
        *,
        model_id: str,
        processor: Any,
        template: Any,
        loss_scale: str,
        swift_version: str,
    ) -> None:
        if not model_id.strip():
            raise TokenProfileError("model_id must be nonempty text")
        if not hasattr(template, "set_mode") or not hasattr(template, "encode"):
            raise TokenProfileError("MS-SWIFT template lacks the required API")
        template.set_mode("train")
        self.model_id = model_id
        self.loss_scale = loss_scale
        self.processor = processor
        self.template = template
        self.swift_version = swift_version

    def encode_record(self, record: dict[str, Any]) -> EncodedRecord:
        encoded = self.template.encode(record)
        if not isinstance(encoded, dict):
            raise TokenProfileError("MS-SWIFT template returned a non-object encoding")
        if encoded.get("truncated") is True or encoded.get("is_truncated") is True:
            raise TokenProfileError("MS-SWIFT reported a truncated encoding")
        input_ids = _flat_vector(encoded.get("input_ids"), "input_ids")
        labels = _flat_vector(encoded.get("labels"), "labels")
        if len(input_ids) != len(labels):
            raise TokenProfileError("input_ids and labels length mismatch")
        loss_scale = encoded.get("loss_scale")
        scales: list[Any] | None = None
        if loss_scale is not None:
            scales = _flat_vector(loss_scale, "loss_scale")
            if len(scales) != len(labels):
                raise TokenProfileError("labels and loss_scale length mismatch")
        supervised = sum(
            label != -100 and (scales is None or float(scales[index]) > 0)
            for index, label in enumerate(labels)
        )
        return EncodedRecord(
            total_tokens=len(input_ids),
            supervised_tokens=supervised,
        )

    def count_text(self, text: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            raise TokenProfileError("Qwen3.5 processor lacks tokenizer.encode")
        token_ids = encode(text, add_special_tokens=False)
        return len(_flat_vector(token_ids, "text token ids"))

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "ms_swift_version": self.swift_version,
            "processor_class": type(self.processor).__name__,
            "template_class": type(self.template).__name__,
            "loss_scale": self.loss_scale,
            "template_mode": "train",
            "model_weights_loaded": False,
        }


def load_swift_api(
    *,
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType:
    """Load the processor/template API with a focused dependency error."""

    try:
        swift_module = import_module("swift")
    except (ImportError, ModuleNotFoundError) as exc:
        raise TokenProfileError(
            "MS-SWIFT is required in the dedicated Task 2 environment"
        ) from exc
    if not callable(getattr(swift_module, "get_processor", None)) or not callable(
        getattr(swift_module, "get_template", None)
    ):
        raise TokenProfileError(
            "installed MS-SWIFT lacks get_processor/get_template"
        )
    return swift_module


def nearest_rank(values: Sequence[int], percentile: float) -> int:
    """Return a deterministic nearest-rank percentile."""

    if not values:
        raise TokenProfileError("at least one token length is required")
    if not 0 < percentile <= 1:
        raise TokenProfileError("percentile must be greater than 0 and at most 1")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return int(ordered[rank - 1])


def summarize_lengths(
    values: Sequence[int],
    *,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Summarize one nonempty collection of exact record lengths."""

    if not values:
        raise TokenProfileError("at least one token length is required")
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise TokenProfileError("token lengths must be positive integers")
    if any(not isinstance(limit, int) or limit <= 0 for limit in thresholds):
        raise TokenProfileError("coverage thresholds must be positive integers")

    ordered = sorted(values)
    count = len(ordered)
    coverage: dict[str, dict[str, int | float]] = {}
    for limit in thresholds:
        fitting = sum(value <= limit for value in ordered)
        coverage[str(limit)] = {
            "count": fitting,
            "percent": round(100.0 * fitting / count, 6),
        }
    return {
        "count": count,
        "minimum": ordered[0],
        "median": nearest_rank(ordered, 0.50),
        "p95": nearest_rank(ordered, 0.95),
        "p99": nearest_rank(ordered, 0.99),
        "maximum": ordered[-1],
        "coverage": coverage,
    }


def profile_records(
    records: Iterable[dict[str, Any]],
    *,
    encoder: RecordEncoder,
    projection: str,
    split: str,
) -> list[dict[str, Any]]:
    """Encode records exactly and retain the identity needed for auditing."""

    if split == "test":
        raise TokenProfileError(
            "the test split cannot be used for sequence-length selection"
        )
    if split not in {"train", "valid"}:
        raise TokenProfileError(f"unsupported profile split {split!r}")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        example_id = _required_text(record, "example_id", f"{projection}/{split}:{index}")
        source_sample_id = _required_text(record, "source_sample_id", example_id)
        scenario_type = _required_text(record, "scenario_type", example_id)
        measured = encoder.encode_record(record)
        if measured.total_tokens <= 0:
            raise TokenProfileError(f"{example_id}: encoding produced no tokens")
        if measured.supervised_tokens <= 0:
            raise TokenProfileError(f"{example_id}: encoding has no supervised tokens")
        if measured.supervised_tokens > measured.total_tokens:
            raise TokenProfileError(
                f"{example_id}: supervised tokens exceed total tokens"
            )
        rows.append(
            {
                "example_id": example_id,
                "source_sample_id": source_sample_id,
                "projection": projection,
                "split": split,
                "scenario_type": scenario_type,
                "task_type": record.get("task_type"),
                "total_tokens": measured.total_tokens,
                "supervised_tokens": measured.supervised_tokens,
            }
        )
    if not rows:
        raise TokenProfileError(f"{projection}/{split}: no records to profile")
    return rows


def summarize_profile_rows(
    rows: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Build overall and categorical summaries from per-record measurements."""

    if not rows:
        raise TokenProfileError("at least one profile row is required")

    def summarize_group(group_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        summary = summarize_lengths(
            [int(row["total_tokens"]) for row in group_rows],
            thresholds=thresholds,
        )
        summary["supervised_tokens"] = summarize_lengths(
            [int(row["supervised_tokens"]) for row in group_rows],
            thresholds=thresholds,
        )
        return summary

    def group_by(field: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            raw_value = row.get(field)
            value = "none" if raw_value is None else str(raw_value)
            groups.setdefault(value, []).append(row)
        return {
            value: summarize_group(group_rows)
            for value, group_rows in sorted(groups.items())
        }

    longest_fields = (
        "example_id",
        "projection",
        "split",
        "scenario_type",
        "task_type",
        "total_tokens",
        "supervised_tokens",
    )
    longest = sorted(
        rows,
        key=lambda row: (-int(row["total_tokens"]), str(row["example_id"])),
    )[:20]
    return {
        "overall": summarize_group(rows),
        "by_projection": group_by("projection"),
        "by_split": group_by("split"),
        "by_scenario_type": group_by("scenario_type"),
        "by_task_type": group_by("task_type"),
        "longest_examples": [
            {field: row.get(field) for field in longest_fields}
            for row in longest
        ],
    }


def profile_inputs(
    inputs: Sequence[ProfileInput],
    encoder: SwiftTemplateEncoder,
) -> list[dict[str, Any]]:
    """Profile all manifest-verified files and attach raw field token counts."""

    if not inputs:
        raise TokenProfileError("at least one profile input is required")
    rows: list[dict[str, Any]] = []
    for profile_input in inputs:
        file_rows = profile_records(
            profile_input.records,
            encoder=encoder,
            projection=profile_input.projection,
            split=profile_input.split,
        )
        for row, record in zip(file_rows, profile_input.records, strict=True):
            row["field_content_tokens"] = measure_field_content(record, encoder)
        rows.extend(file_rows)
    return rows


def build_profile_report(
    *,
    rows: Sequence[dict[str, Any]],
    inputs: Sequence[ProfileInput],
    encoder: SwiftTemplateEncoder,
    data_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    created_at: str | None = None,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Build the reviewable Phase 5 summary from exact per-record rows."""

    if not rows:
        raise TokenProfileError("at least one profile row is required")
    field_totals = {field: 0 for field in FIELD_NAMES}
    for row in rows:
        fields = row.get("field_content_tokens")
        if not isinstance(fields, dict):
            raise TokenProfileError(
                f"{row.get('example_id')}: field token counts are missing"
            )
        for field in FIELD_NAMES:
            value = fields.get(field)
            if not isinstance(value, int) or value < 0:
                raise TokenProfileError(
                    f"{row.get('example_id')}: invalid field count for {field}"
                )
            field_totals[field] += value

    field_order = {field: index for index, field in enumerate(FIELD_NAMES)}
    summary = summarize_profile_rows(rows, thresholds=thresholds)
    summary["field_content_totals"] = field_totals
    summary["field_content_ranking"] = [
        {"field": field, "tokens": tokens}
        for field, tokens in sorted(
            field_totals.items(),
            key=lambda item: (-item[1], field_order[item[0]]),
        )
    ]
    return {
        "schema_version": "task2_token_profile_v1",
        "created_at_utc": created_at or _utc_now(),
        "encoder": encoder.metadata(),
        "dataset_manifest": {
            "file": _relative_report_path(manifest_path, data_root),
            "sha256": manifest_sha256,
        },
        "projections": sorted({profile_input.projection for profile_input in inputs}),
        "splits": sorted({profile_input.split for profile_input in inputs}),
        "thresholds": list(thresholds),
        "input_files": [
            {
                "projection": profile_input.projection,
                "split": profile_input.split,
                "file": _relative_report_path(profile_input.path, data_root),
                "record_count": len(profile_input.records),
                "sha256": profile_input.sha256,
            }
            for profile_input in inputs
        ],
        "field_attribution": (
            "Exact record totals use the MS-SWIFT training template. Field totals "
            "count raw content with the same Qwen3.5 processor and exclude chat "
            "template/agent-format overhead."
        ),
        "summary": summary,
    }


def write_profile_reports(
    *,
    summary_path: Path,
    records_path: Path,
    report: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> None:
    """Atomically write the summary JSON and per-record JSONL report."""

    if summary_path.resolve() == records_path.resolve():
        raise TokenProfileError("summary and records destinations must differ")
    summary_text = _stable_json(report) + "\n"
    records_text = _records_text(rows)
    atomic_write_text(summary_path, summary_text)
    atomic_write_text(records_path, records_text)


def _commit_profile_reports(
    *,
    summary_path: Path,
    records_path: Path,
    staged_summary_path: Path,
    staged_records_path: Path,
) -> None:
    """Replace records first and summary last; summary is the commit marker."""

    summary_path = summary_path.resolve()
    records_path = records_path.resolve()
    atomic_write_text(
        records_path,
        staged_records_path.read_text(encoding="utf-8"),
    )
    atomic_write_text(
        summary_path,
        staged_summary_path.read_text(encoding="utf-8"),
    )


def validate_profile_provenance(
    profile: Mapping[str, Any] | Path | str,
    *,
    data_root: Path,
    manifest_path: Path,
    projections: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    records_path: Path,
) -> None:
    """Fail closed unless a profile binds to the current selected release."""

    profile_data = (
        _read_profile_json(Path(profile))
        if isinstance(profile, (Path, str))
        else profile
    )
    if not isinstance(profile_data, Mapping):
        raise TokenProfileError("profile must be a JSON object")
    if profile_data.get("schema_version") != "task2_token_profile_v1":
        raise TokenProfileError("unsupported token profile schema")

    profile_projections = _selection(
        profile_data.get("projections"), "profile projections"
    )
    profile_splits = _selection(profile_data.get("splits"), "profile splits")
    selected_projections = (
        _selection(projections, "projection selection")
        if projections is not None
        else profile_projections
    )
    selected_splits = (
        _selection(splits, "split selection")
        if splits is not None
        else profile_splits
    )
    if set(selected_projections) != set(profile_projections):
        raise TokenProfileError("profile projection selection does not match the requested selection")
    if set(selected_splits) != set(profile_splits):
        raise TokenProfileError("profile split selection does not match the requested selection")

    manifest_path = manifest_path.resolve()
    data_root = data_root.resolve()
    recorded_manifest = profile_data.get("dataset_manifest")
    if not isinstance(recorded_manifest, Mapping):
        raise TokenProfileError("profile dataset manifest provenance is missing")
    expected_manifest_file = _relative_report_path(manifest_path, data_root)
    if recorded_manifest.get("file") != expected_manifest_file:
        raise TokenProfileError(
            "profile manifest path does not match the selected dataset manifest"
        )
    try:
        current_manifest_sha256 = _sha256_file(manifest_path)
    except OSError as exc:
        raise TokenProfileError(
            f"selected dataset manifest is unreadable: {manifest_path}"
        ) from exc
    if recorded_manifest.get("sha256") != current_manifest_sha256:
        raise TokenProfileError(
            "profile manifest checksum mismatch: "
            f"recorded {recorded_manifest.get('sha256')!r}, "
            f"current {current_manifest_sha256!r}"
        )

    inputs = load_profile_inputs(
        data_root=data_root,
        manifest_path=manifest_path,
        projections=selected_projections,
        splits=selected_splits,
    )
    recorded_inputs = profile_data.get("input_files")
    if not isinstance(recorded_inputs, list) or not all(
        isinstance(item, Mapping) for item in recorded_inputs
    ):
        raise TokenProfileError("profile input provenance is missing")
    expected_inputs = [
        {
            "projection": item.projection,
            "split": item.split,
            "file": _relative_report_path(item.path, data_root),
            "record_count": len(item.records),
            "sha256": item.sha256,
        }
        for item in inputs
    ]
    actual_inputs = sorted(
        recorded_inputs,
        key=lambda item: (
            str(item.get("projection")),
            str(item.get("split")),
        ),
    )
    if actual_inputs != sorted(
        expected_inputs,
        key=lambda item: (item["projection"], item["split"]),
    ):
        raise TokenProfileError("profile input provenance mismatch")
    recorded_records_sha256 = profile_data.get("records_sha256")
    if not isinstance(recorded_records_sha256, str):
        raise TokenProfileError("profile records checksum is missing")
    if records_path is None:
        raise TokenProfileError("profile records path is required")
    records_path = records_path.resolve()
    _read_profile_rows(records_path)
    try:
        current_records_sha256 = _sha256_file(records_path)
    except OSError as exc:
        raise TokenProfileError(
            f"profile records are unreadable: {records_path}"
        ) from exc
    if recorded_records_sha256 != current_records_sha256:
        raise TokenProfileError(
            "profile records checksum mismatch: "
            f"recorded {recorded_records_sha256!r}, "
            f"current {current_records_sha256!r}"
        )


def measure_field_content(
    record: dict[str, Any],
    encoder: TextTokenCounter,
) -> dict[str, int]:
    """Count raw content by semantic field with the selected tokenizer."""

    measured = {
        "system_prompt": 0,
        "user_messages": 0,
        "tool_schemas": 0,
        "tool_calls": 0,
        "tool_responses": 0,
        "assistant_targets": 0,
    }
    if "tools" in record:
        tools = record["tools"]
        if not isinstance(tools, str):
            raise TokenProfileError("tools content must be text")
        measured["tool_schemas"] = _count_text(encoder, tools, "tools")

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise TokenProfileError("messages must be a nonempty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TokenProfileError(f"message {index} must be an object")
        role = message.get("role")
        field = FIELD_ROLES.get(str(role))
        if field is None:
            raise TokenProfileError(f"message {index} has unsupported role {role!r}")
        content = message.get("content")
        if not isinstance(content, str):
            raise TokenProfileError(f"message {index} content must be text")
        measured[field] += _count_text(
            encoder,
            content,
            f"message {index} content",
        )
    return measured


def load_profile_inputs(
    *,
    data_root: Path,
    manifest_path: Path,
    projections: Sequence[str] = PROFILE_PROJECTIONS,
    splits: Sequence[str] = PROFILE_SPLITS,
) -> list[ProfileInput]:
    """Load only train/valid files whose counts and hashes match the manifest."""

    if any(split == "test" for split in splits):
        raise TokenProfileError(
            "the test split cannot be used for sequence-length selection"
        )
    unsupported_splits = set(splits) - set(PROFILE_SPLITS)
    if unsupported_splits:
        raise TokenProfileError(
            f"unsupported profile splits {sorted(unsupported_splits)}"
        )
    unsupported_projections = set(projections) - set(PROFILE_PROJECTION_CHOICES)
    if unsupported_projections:
        raise TokenProfileError(
            f"unsupported projections {sorted(unsupported_projections)}"
        )
    if not projections or not splits:
        raise TokenProfileError("at least one projection and split are required")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenProfileError(f"{manifest_path}: invalid or unreadable manifest") from exc
    if not isinstance(manifest, dict) or manifest.get(
        "schema_version"
    ) != "task2_ms_swift_manifest_v1":
        raise TokenProfileError("unsupported dataset manifest schema")

    data_root = data_root.resolve()
    inputs: list[ProfileInput] = []
    for projection in projections:
        projection_manifest = (
            (manifest.get("corrective_datasets") or {}).get("python_script")
            if projection == "python_correction"
            else (manifest.get("projections") or {}).get(projection)
        )
        if not isinstance(projection_manifest, dict):
            raise TokenProfileError(f"{projection}: manifest entry is missing")
        for split in splits:
            details = projection_manifest.get(split)
            if not isinstance(details, dict):
                raise TokenProfileError(
                    f"{projection}/{split}: manifest entry is missing"
                )
            expected_relative = f"{projection}/{split}.jsonl"
            if details.get("file") != expected_relative:
                raise TokenProfileError(
                    f"{projection}/{split}: unexpected dataset path"
                )
            path = data_root / projection / f"{split}.jsonl"
            records = _read_jsonl(path)
            if details.get("record_count") != len(records):
                raise TokenProfileError(
                    f"{projection}/{split}: record count does not match manifest"
                )
            digest = _sha256_file(path)
            if details.get("sha256") != digest:
                raise TokenProfileError(
                    f"{projection}/{split}: checksum does not match manifest"
                )
            inputs.append(
                ProfileInput(
                    projection=projection,
                    split=split,
                    path=path,
                    records=records,
                    sha256=digest,
                )
            )
    return inputs


def _count_text(encoder: TextTokenCounter, text: str, location: str) -> int:
    count = encoder.count_text(text)
    if not isinstance(count, int) or count < 0:
        raise TokenProfileError(
            f"{location}: tokenizer returned an invalid token count"
        )
    return count


def _read_profile_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenProfileError(f"profile JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise TokenProfileError(f"profile JSON must contain an object: {path}")
    return value


def _read_profile_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = _read_jsonl(path)
    except (OSError, ValueError) as exc:
        raise TokenProfileError(f"profile records are unreadable: {path}") from exc
    if not rows:
        raise TokenProfileError(f"profile records are empty: {path}")
    return rows


def _validate_staged_profile(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    inputs: Sequence[ProfileInput],
) -> None:
    if report.get("schema_version") != "task2_token_profile_v1":
        raise TokenProfileError("staged profile has an unsupported schema")
    summary = report.get("summary")
    overall = summary.get("overall") if isinstance(summary, Mapping) else None
    count = overall.get("count") if isinstance(overall, Mapping) else None
    expected_count = sum(len(item.records) for item in inputs)
    if not isinstance(count, int) or count != len(rows) or count != expected_count:
        raise TokenProfileError(
            "staged profile record count mismatch: "
            f"summary={count!r}, records={len(rows)}, inputs={expected_count}"
        )
    expected_pairs = {(item.projection, item.split) for item in inputs}
    row_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row.get("projection")), str(row.get("split")))
        row_counts[key] = row_counts.get(key, 0) + 1
    if set(row_counts) != expected_pairs:
        raise TokenProfileError("staged profile projection/split selection mismatch")
    for item in inputs:
        if row_counts.get((item.projection, item.split)) != len(item.records):
            raise TokenProfileError(
                f"staged profile record count mismatch for {item.projection}/{item.split}"
            )


def _selection(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TokenProfileError(f"{label} must be a nonempty list")
    values = tuple(str(item) for item in value)
    if not values or any(not item for item in values):
        raise TokenProfileError(f"{label} must be a nonempty list")
    if len(set(values)) != len(values):
        raise TokenProfileError(f"{label} contains duplicates")
    return values


def _relative_report_path(path: Path, data_root: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError as exc:
        raise TokenProfileError(
            f"{path}: report input is outside data root {data_root}"
        ) from exc


def _flat_vector(value: Any, location: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TokenProfileError(f"{location} must be a token sequence")
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if any(isinstance(item, (list, tuple)) for item in value):
        raise TokenProfileError(f"{location} must describe exactly one record")
    if not value:
        raise TokenProfileError(f"{location} must not be empty")
    return value


def _records_text(rows: Sequence[dict[str, Any]]) -> str:
    return "".join(_stable_json(row) + "\n" for row in rows)


def _required_text(record: dict[str, Any], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TokenProfileError(f"{location}: {field} must be nonempty text")
    return value


def main(
    argv: Sequence[str] | None = None,
    *,
    encoder_factory: Callable[..., SwiftTemplateEncoder] = SwiftTemplateEncoder,
) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile exact MS-SWIFT/Qwen3.5 train and validation token lengths "
            "without loading model weights or truncating records."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    parser.add_argument(
        "--records-output",
        type=Path,
        default=DEFAULT_RECORDS_PATH,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--loss-scale", default=DEFAULT_LOSS_SCALE)
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=PROFILE_PROJECTION_CHOICES,
        default=list(PROFILE_PROJECTIONS),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=PROFILE_SPLITS,
        default=list(PROFILE_SPLITS),
        help="Only train/valid are accepted; test cannot select configuration.",
    )
    parser.add_argument(
        "--created-at",
        default=None,
        help="Optional fixed UTC timestamp for reproducible report tests.",
    )
    parser.add_argument(
        "--validate-profile",
        action="store_true",
        help=(
            "Validate the existing summary provenance and exit before loading "
            "MS-SWIFT or a tokenizer."
        ),
    )
    args = parser.parse_args(argv)
    if args.summary_output.resolve() == args.records_output.resolve():
        raise TokenProfileError("summary and records destinations must differ")

    if args.validate_profile:
        validate_profile_provenance(
            args.summary_output,
            data_root=args.data_root,
            manifest_path=args.manifest_path,
            projections=args.projections,
            splits=args.splits,
            records_path=args.records_output,
        )
        print(_stable_json({"validated_profile": args.summary_output.resolve().as_posix()}))
        return 0

    inputs = load_profile_inputs(
        data_root=args.data_root,
        manifest_path=args.manifest_path,
        projections=args.projections,
        splits=args.splits,
    )
    encoder = encoder_factory(
        args.model_id,
        loss_scale=args.loss_scale,
    )
    rows = profile_inputs(inputs, encoder)
    report = build_profile_report(
        rows=rows,
        inputs=inputs,
        encoder=encoder,
        data_root=args.data_root,
        manifest_path=args.manifest_path.resolve(),
        manifest_sha256=_sha256_file(args.manifest_path),
        created_at=args.created_at,
    )
    report = {
        **report,
        "records_sha256": sha256_bytes(_records_text(rows).encode("utf-8")),
    }
    with tempfile.TemporaryDirectory(prefix="task2-token-profile-") as scratch:
        staged_summary_path = Path(scratch) / "candidate_profile.json"
        staged_records_path = Path(scratch) / "candidate_records.jsonl"
        write_profile_reports(
            summary_path=staged_summary_path,
            records_path=staged_records_path,
            report=report,
            rows=rows,
        )
        staged_report = _read_profile_json(staged_summary_path)
        staged_rows = _read_profile_rows(staged_records_path)
        validate_profile_provenance(
            staged_report,
            data_root=args.data_root,
            manifest_path=args.manifest_path,
            projections=args.projections,
            splits=args.splits,
            records_path=staged_records_path,
        )
        _validate_staged_profile(staged_report, staged_rows, inputs)
        _commit_profile_reports(
            summary_path=args.summary_output,
            records_path=args.records_output,
            staged_summary_path=staged_summary_path,
            staged_records_path=staged_records_path,
        )
    print(
        _stable_json(
            {
                "profiled_records": len(rows),
                "summary_output": args.summary_output.resolve().as_posix(),
                "records_output": args.records_output.resolve().as_posix(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TokenProfileError as exc:
        raise SystemExit(f"Token profile validation failed: {exc}") from exc
