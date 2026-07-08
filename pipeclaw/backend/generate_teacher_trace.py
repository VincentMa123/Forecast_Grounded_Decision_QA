"""
Build a PDF-format PipeFormer teacher trace for the public mock data.

This file is intentionally only the CLI coordinator. The pipeline stages live in
backend/pipeline so parsing, checkpoint inference, rule checks, evidence
extraction, and trace formatting can evolve independently.

Example:
    python generate_teacher_trace.py --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.io_utils import write_json, write_jsonl
from pipeline.pipeformer_inference import find_default_checkpoint_dir, find_default_forecast_csv
from pipeline.scenario_loader import find_scenario, load_scenarios
from pipeline.teacher_trace_pipeline import build_trace_record


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    root = repo_root_from_here()
    backend_root = Path(__file__).resolve().parent
    pipeformer_root = root / "pipeFormer"
    parser = argparse.ArgumentParser(description="Build PDF-format mock PipeFormer teacher traces for PipeClaw.")
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=backend_root / "pipeclaw_data" / "mock_pipeformer_tiny_scenarios.json",
        help="Scenario JSON file to read.",
    )
    parser.add_argument(
        "--scenario-id",
        default="mock_pipeformer_prediction_tiny_001",
        help="Scenario id to build into a trace.",
    )
    parser.add_argument(
        "--pipeformer-root",
        type=Path,
        default=pipeformer_root,
        help="Local pipeFormer repository root.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=find_default_checkpoint_dir(root),
        help="PipeFormer checkpoint directory to use for inference.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for checkpoint inference, for example cpu or cuda.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional PipeFormer data directory override. Defaults to checkpoint training_config.json.",
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=None,
        help="Optional PipeFormer static directory override. Defaults to checkpoint training_config.json.",
    )
    parser.add_argument(
        "--forecast-csv",
        type=Path,
        default=find_default_forecast_csv(root),
        help="Existing PipeFormer sample prediction CSV, used only with --use-sample-csv.",
    )
    parser.add_argument(
        "--use-sample-csv",
        action="store_true",
        help="Use the existing sample prediction CSV instead of running checkpoint inference.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=root / "pipeFormer" / "data" / "mock_tiny" / "static" / "mock_tiny" / "index_variable_mapping.csv",
        help="PipeFormer variable mapping CSV.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=backend_root / "generated_teacher_traces" / "mock_pipeformer_teacher_trace_format.jsonl",
        help="PDF section 1.7 teacher-trace JSONL output.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=backend_root / "generated_teacher_traces" / "mock_pipeformer_teacher_trace_format.pretty.json",
        help="Pretty JSON copy of the PDF section 1.7 record.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite outputs if they already exist.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenario_file)
    scenario = find_scenario(scenarios, args.scenario_id)
    record = build_trace_record(
        scenario=scenario,
        scenario_path=args.scenario_file.resolve(),
        forecast_csv=args.forecast_csv.resolve(),
        mapping_path=args.mapping_csv.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        pipeformer_root=args.pipeformer_root.resolve(),
        data_dir=args.data_dir.resolve() if args.data_dir else None,
        static_dir=args.static_dir.resolve() if args.static_dir else None,
        device=args.device,
        use_sample_csv=args.use_sample_csv,
    )

    write_jsonl(args.output_jsonl, [record], force=args.force)
    write_json(args.output_json, record, force=args.force)

    print(json.dumps({
        "status": "ok",
        "scenario_id": record["scenario_id"],
        "forecast_mode": record["prediction_summary"]["forecast_mode"],
        "overall_status": record["constraint_check"]["overall_status"],
        "risk_level": record["risk_level"],
        "output_jsonl": args.output_jsonl.as_posix(),
        "output_json": args.output_json.as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())