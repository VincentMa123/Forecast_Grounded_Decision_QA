"""Run prompt-only autonomous rollouts and score them against teacher oracles.

Examples::

    python -m pipeclaw.task2_student.scripts.evaluate_autonomous \
        --source pipeclaw/task2_student/data/trace_level/test.jsonl \
        --adapters pipeclaw/task2_student/outputs/qwen35-9b \
        --output-dir pipeclaw/task2_student/outputs/evaluation/autonomous

Use ``--dry-run`` to inspect the exact PromptBuilder messages and tool schemas
without loading a model or executing tools.

This module owns argument parsing only.  Rollout execution lives in
``pipeclaw.task2_student.rollout`` and scoring lives in
``pipeclaw.backend.evaluator``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipeclaw.task2_student.rollout.suite import evaluate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Teacher-source JSONL used for prompts and oracle metrics",
    )
    parser.add_argument(
        "--tool-schema-source",
        help="Projection JSONL containing OpenAI tool schemas",
    )
    parser.add_argument("--adapters", help="Optional LoRA adapter directory")
    parser.add_argument(
        "--model",
        help=(
            "Base model name/path; inferred from adapter metadata when --adapters "
            "is used, otherwise required"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for rollouts.jsonl and summary.json",
    )
    parser.add_argument(
        "--repo-root", default=".", help="Repository root used to import PipeClaw tools"
    )
    parser.add_argument(
        "--scenario-type",
        help=(
            "Evaluate one scenario type: pipeformer, openclaw (or the pipeclaw "
            "alias); omit for both"
        ),
    )
    parser.add_argument("--limit", type=int, help="Limit the number of cases")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable the model's reasoning mode; disabled by default for benchmark parity",
    )
    parser.add_argument("--device", help="CUDA_VISIBLE_DEVICES value")
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument(
        "--quant-bits",
        type=int,
        choices=(4, 8),
        help=(
            "Override the adapter checkpoint's base-model quantization with "
            "bitsandbytes 4-bit or 8-bit loading"
        ),
    )
    quantization.add_argument(
        "--no-quantization",
        action="store_true",
        help="Ignore saved quantization metadata and load the base model unquantized",
    )
    parser.add_argument(
        "--save-raw-responses",
        action="store_true",
        help="Save serialized generator responses before parsing them",
    )
    parser.add_argument(
        "--save-raw-tool-outputs",
        action="store_true",
        help=(
            "Save complete tool payloads separately; model-facing and default "
            "rollout outputs stay compact"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and schemas without loading a model",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = evaluate_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
