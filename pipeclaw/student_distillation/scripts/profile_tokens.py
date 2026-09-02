"""Measure released dataset tokens with the exact MS-SWIFT training template.

The complete workflow stays together because encoding, summarization, provenance
validation, and report publication form one sequential release command.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from pipeclaw.student_distillation.release_artifacts import (
    atomic_write_text as _atomic_write_text,
    read_jsonl_domain,
    required_text,
    sha256_bytes,
    sha256_file as _sha256_file,
    stable_json as _stable_json,
    utc_now as _utc_now,
)
from pipeclaw.student_distillation.scripts.validate_dataset import (
    CORRECTION_SPLITS as PROFILE_SPLITS,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_ROOT as DEFAULT_DATA_ROOT,
    PROJECTIONS as PROFILE_PROJECTIONS,
    REPO_ROOT,
)


DEFAULT_THRESHOLDS = (1024, 2048, 4096, 8192, 16384)
PROFILE_PROJECTION_CHOICES = (*PROFILE_PROJECTIONS, "python_correction")
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
DEFAULT_SUMMARY_PATH = (
    DEFAULT_DATA_ROOT / "token_profiles" / "qwen35_08b_token_profile.json"
)
DEFAULT_RECORDS_PATH = (
    DEFAULT_DATA_ROOT / "token_profiles" / "qwen35_08b_token_records.jsonl"
)
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


class TokenProfileError(ValueError):
    """Raised when a token profile cannot be produced safely."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise TokenProfileError(message)


_required_text = partial(required_text, error_factory=TokenProfileError)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_domain(path, error_factory=TokenProfileError)


# === Token measurement =====================================================
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
    def encode_record(self, record: dict[str, Any]) -> EncodedRecord: ...


class TextTokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...


class SwiftTemplateEncoder:
    """Processor-only MS-SWIFT adapter for exact training-template encoding."""

    def __init__(self, model_id: str, *, loss_scale: str = DEFAULT_LOSS_SCALE) -> None:
        swift_module = load_swift_api()
        processor = swift_module.get_processor(model_id)
        template = swift_module.get_template(processor, loss_scale=loss_scale)
        self._initialize(
            model_id, processor, template, loss_scale,
            str(getattr(swift_module, "__version__", "unknown")),
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
        instance._initialize(model_id, processor, template, loss_scale, swift_version)
        return instance

    def _initialize(
        self,
        model_id: str,
        processor: Any,
        template: Any,
        loss_scale: str,
        swift_version: str,
    ) -> None:
        _require(model_id.strip(), "model_id must be nonempty text")
        _require(
            hasattr(template, "set_mode") and hasattr(template, "encode"),
            "MS-SWIFT template lacks the required API",
        )
        template.set_mode("train")
        self.model_id, self.processor, self.template = model_id, processor, template
        self.loss_scale, self.swift_version = loss_scale, swift_version

    def encode_record(self, record: dict[str, Any]) -> EncodedRecord:
        encoded = self.template.encode(record)
        _require(isinstance(encoded, dict), "MS-SWIFT template returned a non-object encoding")
        _require(
            encoded.get("truncated") is not True and encoded.get("is_truncated") is not True,
            "MS-SWIFT reported a truncated encoding",
        )
        input_ids = _flat_vector(encoded.get("input_ids"), "input_ids")
        labels = _flat_vector(encoded.get("labels"), "labels")
        _require(len(input_ids) == len(labels), "input_ids and labels length mismatch")
        loss_scale = encoded.get("loss_scale")
        scales: list[Any] | None = None
        if loss_scale is not None:
            scales = _flat_vector(loss_scale, "loss_scale")
            _require(len(scales) == len(labels), "labels and loss_scale length mismatch")
        return EncodedRecord(
            len(input_ids),
            sum(
                label != -100 and (scales is None or float(scales[index]) > 0)
                for index, label in enumerate(labels)
            ),
        )

    def count_text(self, text: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        encode = getattr(tokenizer, "encode", None)
        _require(callable(encode), "Qwen3.5 processor lacks tokenizer.encode")
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
    _require(
        callable(getattr(swift_module, "get_processor", None))
        and callable(getattr(swift_module, "get_template", None)),
        "installed MS-SWIFT lacks get_processor/get_template",
    )
    return swift_module


# === Profile summaries =====================================================
def nearest_rank(values: Sequence[int], percentile: float) -> int:
    """Return a deterministic nearest-rank percentile."""

    _require(values, "at least one token length is required")
    _require(0 < percentile <= 1, "percentile must be greater than 0 and at most 1")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return int(ordered[rank - 1])


def summarize_lengths(
    values: Sequence[int],
    *,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Summarize one nonempty collection of exact record lengths."""

    _require(values, "at least one token length is required")
    _require(
        all(isinstance(value, int) and value > 0 for value in values),
        "token lengths must be positive integers",
    )
    _require(
        all(isinstance(limit, int) and limit > 0 for limit in thresholds),
        "coverage thresholds must be positive integers",
    )

    ordered, count = sorted(values), len(values)

    def coverage(limit: int) -> dict[str, int | float]:
        fitting = sum(value <= limit for value in ordered)
        return {"count": fitting, "percent": round(100.0 * fitting / count, 6)}

    return {
        "count": count,
        "minimum": ordered[0],
        "median": nearest_rank(ordered, 0.50),
        "p95": nearest_rank(ordered, 0.95),
        "p99": nearest_rank(ordered, 0.99),
        "maximum": ordered[-1],
        "coverage": {str(limit): coverage(limit) for limit in thresholds},
    }


def profile_records(
    records: Iterable[dict[str, Any]],
    *,
    encoder: RecordEncoder,
    projection: str,
    split: str,
) -> list[dict[str, Any]]:
    """Encode records exactly and retain the identity needed for auditing."""

    _require(split != "test", "the test split cannot be used for sequence-length selection")
    _require(split in {"train", "valid"}, f"unsupported profile split {split!r}")

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        example_id = _required_text(record, "example_id", f"{projection}/{split}:{index}")
        source_sample_id = _required_text(record, "source_sample_id", example_id)
        scenario_type = _required_text(record, "scenario_type", example_id)
        measured = encoder.encode_record(record)
        _require(measured.total_tokens > 0, f"{example_id}: encoding produced no tokens")
        _require(measured.supervised_tokens > 0, f"{example_id}: encoding has no supervised tokens")
        _require(
            measured.supervised_tokens <= measured.total_tokens,
            f"{example_id}: supervised tokens exceed total tokens",
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
    _require(rows, f"{projection}/{split}: no records to profile")
    return rows


def summarize_profile_rows(
    rows: Sequence[dict[str, Any]],
    *,
    thresholds: Sequence[int] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Build overall and categorical summaries from per-record measurements."""

    _require(rows, "at least one profile row is required")

    def summarize_group(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        summary = summarize_lengths(
            [int(row["total_tokens"]) for row in group], thresholds=thresholds
        )
        summary["supervised_tokens"] = summarize_lengths(
            [int(row["supervised_tokens"]) for row in group], thresholds=thresholds
        )
        return summary

    def grouped(field: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            value = "none" if row.get(field) is None else str(row[field])
            groups.setdefault(value, []).append(row)
        return {
            value: summarize_group(group) for value, group in sorted(groups.items())
        }

    longest_fields = (
        "example_id", "projection", "split", "scenario_type", "task_type",
        "total_tokens", "supervised_tokens",
    )
    longest = sorted(rows, key=lambda row: (
        -int(row["total_tokens"]), str(row["example_id"])))[:20]
    return {
        "overall": summarize_group(rows),
        **{f"by_{field}": grouped(field) for field in (
            "projection", "split", "scenario_type", "task_type")},
        "longest_examples": [
            {field: row.get(field) for field in longest_fields} for row in longest
        ],
    }


def profile_inputs(
    inputs: Sequence[ProfileInput],
    encoder: SwiftTemplateEncoder,
) -> list[dict[str, Any]]:
    """Profile all manifest-verified files and attach raw field token counts."""

    _require(inputs, "at least one profile input is required")
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


# === Release provenance ====================================================
def _input_provenance(
    inputs: Sequence[ProfileInput], data_root: Path
) -> list[dict[str, Any]]:
    return [
        {
            "projection": item.projection,
            "split": item.split,
            "file": _relative_report_path(item.path, data_root),
            "record_count": len(item.records),
            "sha256": item.sha256,
        }
        for item in inputs
    ]


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

    _require(rows, "at least one profile row is required")
    field_totals = dict.fromkeys(FIELD_NAMES, 0)
    for row in rows:
        fields = row.get("field_content_tokens")
        _require(
            isinstance(fields, dict),
            f"{row.get('example_id')}: field token counts are missing",
        )
        for field in FIELD_NAMES:
            value = fields.get(field)
            _require(
                isinstance(value, int) and value >= 0,
                f"{row.get('example_id')}: invalid field count for {field}",
            )
            field_totals[field] += value

    summary = summarize_profile_rows(rows, thresholds=thresholds)
    summary["field_content_totals"] = field_totals
    summary["field_content_ranking"] = [
        {"field": field, "tokens": tokens}
        for field, tokens in sorted(field_totals.items(), key=lambda item: (
            -item[1], FIELD_NAMES.index(item[0])))
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
        "input_files": _input_provenance(inputs, data_root),
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

    _require(
        summary_path.resolve() != records_path.resolve(),
        "summary and records destinations must differ",
    )
    summary_text = _stable_json(report) + "\n"
    records_text = _records_text(rows)
    _atomic_write_text(summary_path, summary_text)
    _atomic_write_text(records_path, records_text)


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
    _atomic_write_text(
        records_path,
        staged_records_path.read_text(encoding="utf-8"),
    )
    _atomic_write_text(
        summary_path,
        staged_summary_path.read_text(encoding="utf-8"),
    )


def _profile_digest(path: Path, subject: str) -> str:
    try:
        return _sha256_file(path)
    except OSError as exc:
        raise TokenProfileError(f"{subject} is unreadable: {path}") from exc


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
    _require(isinstance(profile_data, Mapping), "profile must be a JSON object")
    _require(
        profile_data.get("schema_version") == "task2_token_profile_v1",
        "unsupported token profile schema",
    )

    profile_projections = _selection(
        profile_data.get("projections"), "profile projections"
    )
    profile_splits = _selection(profile_data.get("splits"), "profile splits")
    selections = []
    for requested, profiled, name in (
        (projections, profile_projections, "projection"),
        (splits, profile_splits, "split"),
    ):
        selected = (
            _selection(requested, f"{name} selection")
            if requested is not None
            else profiled
        )
        _require(
            set(selected) == set(profiled),
            f"profile {name} selection does not match the requested selection",
        )
        selections.append(selected)
    selected_projections, selected_splits = selections

    manifest_path = manifest_path.resolve()
    data_root = data_root.resolve()
    recorded_manifest = profile_data.get("dataset_manifest")
    _require(
        isinstance(recorded_manifest, Mapping),
        "profile dataset manifest provenance is missing",
    )
    expected_manifest_file = _relative_report_path(manifest_path, data_root)
    _require(
        recorded_manifest.get("file") == expected_manifest_file,
        "profile manifest path does not match the selected dataset manifest",
    )
    current_manifest_sha256 = _profile_digest(
        manifest_path, "selected dataset manifest"
    )
    _require(
        recorded_manifest.get("sha256") == current_manifest_sha256,
        "profile manifest checksum mismatch: "
        f"recorded {recorded_manifest.get('sha256')!r}, current {current_manifest_sha256!r}",
    )

    inputs = load_profile_inputs(
        data_root=data_root,
        manifest_path=manifest_path,
        projections=selected_projections,
        splits=selected_splits,
    )
    recorded_inputs = profile_data.get("input_files")
    _require(
        isinstance(recorded_inputs, list)
        and all(isinstance(item, Mapping) for item in recorded_inputs),
        "profile input provenance is missing",
    )
    expected_inputs = _input_provenance(inputs, data_root)
    input_key = lambda item: (str(item.get("projection")), str(item.get("split")))
    _require(
        sorted(recorded_inputs, key=input_key) == sorted(expected_inputs, key=input_key),
        "profile input provenance mismatch",
    )
    recorded_records_sha256 = profile_data.get("records_sha256")
    _require(isinstance(recorded_records_sha256, str), "profile records checksum is missing")
    _require(records_path is not None, "profile records path is required")
    records_path = records_path.resolve()
    _read_profile_rows(records_path)
    current_records_sha256 = _profile_digest(records_path, "profile records")
    _require(
        recorded_records_sha256 == current_records_sha256,
        "profile records checksum mismatch: "
        f"recorded {recorded_records_sha256!r}, current {current_records_sha256!r}",
    )


def measure_field_content(
    record: dict[str, Any],
    encoder: TextTokenCounter,
) -> dict[str, int]:
    """Count raw content by semantic field with the selected tokenizer."""

    measured = dict.fromkeys(FIELD_NAMES, 0)
    if "tools" in record:
        tools = record["tools"]
        _require(isinstance(tools, str), "tools content must be text")
        measured["tool_schemas"] = _count_text(encoder, tools, "tools")

    messages = record.get("messages")
    _require(isinstance(messages, list) and messages, "messages must be a nonempty list")
    for index, message in enumerate(messages):
        _require(isinstance(message, dict), f"message {index} must be an object")
        role = message.get("role")
        field = FIELD_ROLES.get(str(role))
        _require(field is not None, f"message {index} has unsupported role {role!r}")
        content = message.get("content")
        _require(isinstance(content, str), f"message {index} content must be text")
        measured[field] += _count_text(encoder, content, f"message {index} content")
    return measured


def load_profile_inputs(
    *,
    data_root: Path,
    manifest_path: Path,
    projections: Sequence[str] = PROFILE_PROJECTIONS,
    splits: Sequence[str] = PROFILE_SPLITS,
) -> list[ProfileInput]:
    """Load only train/valid files whose counts and hashes match the manifest."""

    _require(
        all(split != "test" for split in splits),
        "the test split cannot be used for sequence-length selection",
    )
    for values, allowed, label in (
        (splits, PROFILE_SPLITS, "profile splits"),
        (projections, PROFILE_PROJECTION_CHOICES, "projections"),
    ):
        unsupported = set(values) - set(allowed)
        _require(not unsupported, f"unsupported {label} {sorted(unsupported)}")
    _require(projections and splits, "at least one projection and split are required")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenProfileError(f"{manifest_path}: invalid or unreadable manifest") from exc
    _require(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == "task2_ms_swift_manifest_v1",
        "unsupported dataset manifest schema",
    )

    data_root = data_root.resolve()
    projection_entries = manifest.get("projections") or {}
    correction_entries = manifest.get("corrective_datasets") or {}
    inputs: list[ProfileInput] = []
    for projection in projections:
        projection_manifest = (
            correction_entries.get("python_script")
            if projection == "python_correction"
            else projection_entries.get(projection)
        )
        _require(
            isinstance(projection_manifest, dict),
            f"{projection}: manifest entry is missing",
        )
        for split in splits:
            details = projection_manifest.get(split)
            _require(
                isinstance(details, dict),
                f"{projection}/{split}: manifest entry is missing",
            )
            relative = f"{projection}/{split}.jsonl"
            _require(
                details.get("file") == relative,
                f"{projection}/{split}: unexpected dataset path",
            )
            path = data_root / projection / f"{split}.jsonl"
            records = _read_jsonl(path)
            _require(
                details.get("record_count") == len(records),
                f"{projection}/{split}: record count does not match manifest",
            )
            digest = _sha256_file(path)
            _require(
                details.get("sha256") == digest,
                f"{projection}/{split}: checksum does not match manifest",
            )
            inputs.append(ProfileInput(projection, split, path, records, digest))
    return inputs


def _count_text(encoder: TextTokenCounter, text: str, location: str) -> int:
    count = encoder.count_text(text)
    _require(
        isinstance(count, int) and count >= 0,
        f"{location}: tokenizer returned an invalid token count",
    )
    return count


def _read_profile_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenProfileError(f"profile JSON is unreadable: {path}") from exc
    _require(isinstance(value, dict), f"profile JSON must contain an object: {path}")
    return value


def _read_profile_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = _read_jsonl(path)
    except (OSError, ValueError) as exc:
        raise TokenProfileError(f"profile records are unreadable: {path}") from exc
    _require(rows, f"profile records are empty: {path}")
    return rows


def _validate_staged_profile(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    inputs: Sequence[ProfileInput],
) -> None:
    _require(
        report.get("schema_version") == "task2_token_profile_v1",
        "staged profile has an unsupported schema",
    )
    summary = report.get("summary")
    overall = summary.get("overall") if isinstance(summary, Mapping) else None
    count = overall.get("count") if isinstance(overall, Mapping) else None
    expected_count = sum(len(item.records) for item in inputs)
    _require(
        isinstance(count, int) and count == len(rows) == expected_count,
        (
            "staged profile record count mismatch: "
            f"summary={count!r}, records={len(rows)}, inputs={expected_count}"
        ),
    )
    expected_pairs = {(item.projection, item.split) for item in inputs}
    row_counts = Counter(
        (str(row.get("projection")), str(row.get("split"))) for row in rows
    )
    _require(
        set(row_counts) == expected_pairs,
        "staged profile projection/split selection mismatch",
    )
    for item in inputs:
        _require(
            row_counts.get((item.projection, item.split)) == len(item.records),
            f"staged profile record count mismatch for {item.projection}/{item.split}",
        )


def _selection(value: Any, label: str) -> tuple[str, ...]:
    _require(
        not isinstance(value, (str, bytes)) and isinstance(value, Sequence),
        f"{label} must be a nonempty list",
    )
    values = tuple(str(item) for item in value)
    _require(values and all(values), f"{label} must be a nonempty list")
    _require(len(set(values)) == len(values), f"{label} contains duplicates")
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
    _require(isinstance(value, list), f"{location} must be a token sequence")
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    _require(
        not any(isinstance(item, (list, tuple)) for item in value),
        f"{location} must describe exactly one record",
    )
    _require(value, f"{location} must not be empty")
    return value


def _records_text(rows: Sequence[dict[str, Any]]) -> str:
    return "".join(_stable_json(row) + "\n" for row in rows)


# === Command-line entrypoint ===============================================
def main(
    argv: Sequence[str] | None = None,
    *,
    encoder_factory: Callable[..., SwiftTemplateEncoder] = SwiftTemplateEncoder,
) -> int:
    parser = argparse.ArgumentParser(description=
        "Profile exact MS-SWIFT/Qwen3.5 train and validation token lengths "
        "without loading model weights or truncating records.")
    for flag, options in (
        ("--data-root", {"type": Path, "default": DEFAULT_DATA_ROOT}),
        ("--manifest-path", {"type": Path, "default": DEFAULT_MANIFEST_PATH}),
        ("--summary-output", {"type": Path, "default": DEFAULT_SUMMARY_PATH}),
        ("--records-output", {"type": Path, "default": DEFAULT_RECORDS_PATH}),
        ("--model-id", {"default": DEFAULT_MODEL_ID}),
        ("--loss-scale", {"default": DEFAULT_LOSS_SCALE}),
        ("--projections", {"nargs": "+", "choices": PROFILE_PROJECTION_CHOICES,
                           "default": list(PROFILE_PROJECTIONS)}),
        ("--splits", {
            "nargs": "+", "choices": PROFILE_SPLITS, "default": list(PROFILE_SPLITS),
            "help": "Only train/valid are accepted; test cannot select configuration.",
        }),
        ("--created-at", {"default": None,
                          "help": "Optional fixed UTC timestamp for reproducible report tests."}),
        ("--validate-profile", {
            "action": "store_true",
            "help": "Validate the existing summary provenance and exit before loading "
                    "MS-SWIFT or a tokenizer.",
        }),
    ):
        parser.add_argument(flag, **options)
    args = parser.parse_args(argv)
    _require(args.summary_output.resolve() != args.records_output.resolve(),
             "summary and records destinations must differ")

    selection = {
        "data_root": args.data_root,
        "manifest_path": args.manifest_path,
        "projections": args.projections,
        "splits": args.splits,
    }
    if args.validate_profile:
        validate_profile_provenance(args.summary_output,
                                    records_path=args.records_output, **selection)
        print(_stable_json({"validated_profile": args.summary_output.resolve().as_posix()}))
        return 0

    inputs = load_profile_inputs(**selection)
    encoder = encoder_factory(args.model_id, loss_scale=args.loss_scale)
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
    report = {**report, "records_sha256": sha256_bytes(
        _records_text(rows).encode("utf-8"))}
    with tempfile.TemporaryDirectory(prefix="task2-token-profile-") as scratch:
        staged_summary_path = Path(scratch) / "candidate_profile.json"
        staged_records_path = Path(scratch) / "candidate_records.jsonl"
        write_profile_reports(summary_path=staged_summary_path,
                              records_path=staged_records_path, report=report, rows=rows)
        staged_report = _read_profile_json(staged_summary_path)
        staged_rows = _read_profile_rows(staged_records_path)
        validate_profile_provenance(staged_report,
                                    records_path=staged_records_path, **selection)
        _validate_staged_profile(staged_report, staged_rows, inputs)
        _commit_profile_reports(summary_path=args.summary_output,
                                records_path=args.records_output,
                                staged_summary_path=staged_summary_path,
                                staged_records_path=staged_records_path)
    print(_stable_json({
        "profiled_records": len(rows),
        "summary_output": args.summary_output.resolve().as_posix(),
        "records_output": args.records_output.resolve().as_posix(),
    }))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TokenProfileError as exc:
        raise SystemExit(f"Token profile validation failed: {exc}") from exc
