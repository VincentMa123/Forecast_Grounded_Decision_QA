from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.scorer import (
    DEFAULT_MINIMUM_SCORE,
    evaluate_native_record,
    load_records,
    summarize_evaluations,
)


BACKEND_ROOT = Path(__file__).resolve().parent
DEFAULT_TRACE = BACKEND_ROOT / "generated_teacher_traces" / "teacher_trace.json"
DEFAULT_OUTPUT = BACKEND_ROOT / "generated_teacher_traces" / "quality_evaluation.jsonl"
DEFAULT_SUMMARY = BACKEND_ROOT / "generated_teacher_traces" / "quality_evaluation_summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate compact teacher-trace records using the native PipeClaw workflow."
    )
    parser.add_argument("--teacher-trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--scenario-id")
    parser.add_argument("--sample-id")
    parser.add_argument("--minimum-score", type=float, default=DEFAULT_MINIMUM_SCORE)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    records = [
        record
        for record in load_records(args.teacher_trace.resolve())
        if (not args.scenario_id or record.get("scenario_id") == args.scenario_id)
        and (not args.sample_id or record.get("sample_id") == args.sample_id)
    ]
    if not records:
        raise ValueError("No teacher-trace records matched the requested filters.")
    evaluations = []
    for record in records:
        result = evaluate_native_record(
            record,
            trace_status=record.get("trace_status"),
            minimum_score=args.minimum_score,
        )
        evaluations.append(
            {
                "sample_id": record.get("sample_id"),
                "scenario_id": record.get("scenario_id"),
                **result,
            }
        )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for result in evaluations:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "native_pipeclaw_quality_v1",
        "teacher_trace": str(args.teacher_trace.resolve()),
        "minimum_pass_score": args.minimum_score,
        **summarize_evaluations(evaluations),
        "output_jsonl": str(args.output_jsonl.resolve()),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
