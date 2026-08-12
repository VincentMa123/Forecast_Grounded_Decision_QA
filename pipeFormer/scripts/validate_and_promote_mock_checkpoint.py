#!/usr/bin/env python3
"""Promote a candidate mock checkpoint only after causal acceptance checks pass."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


POLICY_THRESHOLDS = {
    "strict": {
        "intervention_coverage": 1.0,
        "nonzero_response_rate": 0.95,
        "expected_direction_rate": 0.90,
        "unchanged_baseline_max_delta": 1e-8,
    },
    "synthetic-hybrid": {
        "intervention_coverage": 1.0,
        "nonzero_response_rate": 0.95,
        "expected_direction_rate": 0.55,
        "unchanged_baseline_max_delta": 1e-8,
        "tokenization_pass_rate": 0.90,
        "data_generation_pass_rate": 1.0,
        "attention_routing_pass_rate": 1.0,
        "logit_response_pass_rate": 1.0,
    },
}


def acceptance_result(report: dict[str, Any], *, policy: str = "strict") -> dict[str, Any]:
    if policy not in POLICY_THRESHOLDS:
        raise ValueError(f"Unknown checkpoint acceptance policy: {policy}")
    thresholds = dict(POLICY_THRESHOLDS[policy])
    required = {
        "intervention_coverage",
        "nonzero_response_rate",
        "expected_direction_rate",
        "unchanged_baseline_max_delta",
        "candidate_eval_loss",
        "current_eval_loss",
    }
    checks: dict[str, bool] = {
        "required_metrics_present": required.issubset(report),
    }
    if checks["required_metrics_present"]:
        checks.update({
            "intervention_coverage": float(report["intervention_coverage"]) >= thresholds["intervention_coverage"],
            "nonzero_response_rate": float(report["nonzero_response_rate"]) >= thresholds["nonzero_response_rate"],
            "expected_direction_rate": float(report["expected_direction_rate"]) >= thresholds["expected_direction_rate"],
            "unchanged_baseline": float(report["unchanged_baseline_max_delta"]) <= thresholds["unchanged_baseline_max_delta"],
            "validation_loss": float(report["candidate_eval_loss"]) <= float(report["current_eval_loss"]),
        })
    if policy == "synthetic-hybrid":
        stages = dict((report.get("stage_diagnostics") or {}).get("stages") or {})
        for stage in ("data_generation", "attention_routing", "logit_response"):
            checks[f"{stage}_stage"] = (
                float((stages.get(stage) or {}).get("pass_rate", 0.0))
                >= thresholds[f"{stage}_pass_rate"]
            )
        checks["tokenization_stage"] = (
            float((stages.get("tokenization") or {}).get("pass_rate", 0.0))
            >= thresholds["tokenization_pass_rate"]
        )

    accepted = bool(checks) and all(checks.values())
    limitations = []
    if policy == "synthetic-hybrid" and accepted:
        stages = dict((report.get("stage_diagnostics") or {}).get("stages") or {})
        if float(report["expected_direction_rate"]) < POLICY_THRESHOLDS["strict"]["expected_direction_rate"]:
            limitations.append("expected_direction_below_strict_threshold")
        if float((stages.get("tokenization") or {}).get("pass_rate", 0.0)) < 1.0:
            limitations.append("incomplete_control_token_sensitivity")
        if float((stages.get("argmax_decoding") or {}).get("pass_rate", 0.0)) < 1.0:
            limitations.append("raw_argmax_decoding_failed")
        limitations.extend(("hybrid_projection_required", "synthetic_not_physically_validated"))
    return {
        "accepted": accepted,
        "accepted_with_limitations": accepted and bool(limitations),
        "policy": policy,
        "thresholds": thresholds,
        "checks": checks,
        "limitations": limitations,
    }


def acceptance_passes(report: dict[str, Any]) -> bool:
    return bool(acceptance_result(report, policy="strict")["accepted"])


def write_active_manifest(
    manifest_path: Path,
    checkpoint_dir: Path,
    report: dict[str, Any],
    *,
    policy: str = "strict",
) -> None:
    manifest_path = Path(manifest_path).resolve()
    checkpoint_dir = Path(checkpoint_dir).resolve(strict=True)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Candidate checkpoint is not a directory: {checkpoint_dir}")
    reported_checkpoint = report.get("checkpoint_dir")
    if not isinstance(reported_checkpoint, str) or not reported_checkpoint.strip():
        raise ValueError("Validation report must contain checkpoint_dir.")
    if Path(reported_checkpoint).resolve(strict=True) != checkpoint_dir:
        raise ValueError("Validation report checkpoint_dir does not match the candidate checkpoint.")
    outputs_root = manifest_path.parent
    decision = acceptance_result(report, policy=policy)
    if not decision["accepted"]:
        failed = sorted(key for key, passed in decision["checks"].items() if not passed)
        raise ValueError(
            f"Candidate checkpoint did not satisfy the {policy} acceptance criteria: {failed}."
        )
    try:
        relative_checkpoint = checkpoint_dir.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("Candidate checkpoint must be under the PipeFormer outputs directory.") from exc
    payload = {
        "accepted": True,
        "accepted_with_limitations": decision["accepted_with_limitations"],
        "acceptance_policy": policy,
        "acceptance_thresholds": decision["thresholds"],
        "acceptance_checks": decision["checks"],
        "limitations": decision["limitations"],
        "checkpoint_dir": relative_checkpoint.as_posix(),
        "validation_report": report,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(json.dumps(payload, indent=2) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, manifest_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--manifest", default="outputs/mock_decoder_active.json")
    parser.add_argument(
        "--policy",
        choices=tuple(POLICY_THRESHOLDS),
        default="strict",
        help="Use strict scientific validation or explicit synthetic-hybrid acceptance.",
    )
    args = parser.parse_args()
    report = json.loads(Path(args.validation_report).read_text(encoding="utf-8-sig"))
    write_active_manifest(
        Path(args.manifest),
        Path(args.checkpoint_dir),
        report,
        policy=args.policy,
    )
    print(json.dumps({
        "status": "promoted",
        "checkpoint_dir": args.checkpoint_dir,
        "acceptance_policy": args.policy,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
