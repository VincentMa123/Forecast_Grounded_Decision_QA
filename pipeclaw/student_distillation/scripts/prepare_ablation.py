from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
from typing import Any

from pipeclaw.student_distillation.release_artifacts import (
    atomic_jsonl_writer,
    atomic_write_text,
    sha256_file,
    stable_json,
    utc_now,
)
from pipeclaw.student_distillation.scripts.prepare_dataset import project_trace_level
from pipeclaw.student_distillation.scripts.validate_dataset import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_ROOT,
    SPLITS,
    DatasetValidationError,
    load_registered_tool_schemas,
    read_jsonl,
)


CONDITIONS = {
    "A": (False, False),
    "B": (False, True),
    "C": (True, False),
    "D": (True, True),
}


def project_ablation(trace: dict[str, Any], condition: str) -> dict[str, Any]:
    """Toggle tool definitions and call supervision on a validated full trace."""
    include_tools, supervise_calls = CONDITIONS[condition]
    projected = dict(trace)
    if not include_tools:
        projected.pop("tools")
    projected["messages"] = [
        {**message, "loss": False}
        if message["role"] == "tool_call" and not supervise_calls
        else dict(message)
        for message in trace["messages"]
    ]
    return projected


def generate_ablation(
    source_root: Path, output_root: Path, *, skip_invalid: bool = False
) -> dict[str, Any]:
    """Build a complete release in a staging folder before publishing it."""
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Choose a new ablation output directory: {output_root}")
    schemas = load_registered_tool_schemas()
    manifest = {
        "schema_version": "task2_trace_ablation_v1",
        "created_at_utc": utc_now(),
        "source_root": str(source_root),
        "conditions": {
            condition: {"include_tool_definitions": definitions, "supervise_successful_calls": calls}
            for condition, (definitions, calls) in CONDITIONS.items()
        },
        "sources": {},
        "files": {},
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ablation_", dir=output_root.parent)).resolve()
    source_ids: set[str] = set()
    try:
        for split in SPLITS:
            path = source_root / f"teacher_trace_{split}.jsonl"
            source_hash = sha256_file(path)
            rows = read_jsonl(path)
            traces, exclusions = [], []
            for row in rows:
                sample_id = row["sample_id"]
                if sample_id in source_ids:
                    raise DatasetValidationError(f"Repeated source record: {sample_id}")
                source_ids.add(sample_id)
                if row.get("split") != split:
                    raise DatasetValidationError(f"{sample_id}: split differs from {path.name}")
                try:
                    traces.append(project_trace_level(row, split, schemas))
                except DatasetValidationError as error:
                    if not skip_invalid:
                        raise
                    exclusions.append({"source_sample_id": sample_id, "reason": str(error)})
            if not traces:
                raise DatasetValidationError(f"No eligible records for {split}")
            if sha256_file(path) != source_hash:
                raise DatasetValidationError(f"Source changed during projection: {path}")
            manifest["sources"][split] = {
                "file": path.name, "sha256": source_hash, "record_count": len(rows),
                "eligible_count": len(traces), "exclusions": exclusions,
            }
            for condition in CONDITIONS:
                relative = f"{condition}/{split}.jsonl"
                output = stage / relative
                with atomic_jsonl_writer(output) as write:
                    for trace in traces:
                        write(project_ablation(trace, condition))
                manifest["files"][relative] = {
                    "record_count": len(traces), "sha256": sha256_file(output),
                }
        atomic_write_text(stage / "manifest.json", stable_json(manifest) + "\n")
        stage.rename(output_root)
    finally:
        if stage.exists() and stage.resolve().parent == output_root.parent:
            shutil.rmtree(stage)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "ablation")
    parser.add_argument("--skip-invalid", action="store_true", help="Log rejected source rows and exclude them from every condition.")
    args = parser.parse_args()
    manifest = generate_ablation(args.source_root, args.output_root, skip_invalid=args.skip_invalid)
    for split, source in manifest["sources"].items():
        print(f"{split}: {source['eligible_count']} eligible; {len(source['exclusions'])} excluded")


if __name__ == "__main__":
    main()
