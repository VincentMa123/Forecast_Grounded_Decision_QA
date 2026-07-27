#!/usr/bin/env python3
"""Evaluate a candidate checkpoint on unseen control interventions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable


DIAGNOSTIC_STAGE_ORDER = (
    "data_generation",
    "tokenization",
    "attention_routing",
    "logit_response",
    "argmax_decoding",
)


def attention_reaches_control(
    attention_indices: Any,
    control_index: int,
    target_index: int,
    layer_count: int,
) -> bool:
    """Return whether sparse decoder attention can carry a control to a target."""
    frontier = {int(target_index)}
    if int(control_index) in frontier:
        return True
    for _ in range(max(int(layer_count), 0)):
        next_frontier = set(frontier)
        for variable_index in frontier:
            if variable_index < 0 or variable_index >= len(attention_indices):
                continue
            row = attention_indices[variable_index]
            values = row.tolist() if hasattr(row, "tolist") else row
            next_frontier.update(int(value) for value in values if int(value) >= 0)
        if int(control_index) in next_frontier:
            return True
        if next_frontier == frontier:
            break
        frontier = next_frontier
    return False


def first_failed_stage(stages: dict[str, dict[str, Any]]) -> str | None:
    """Return the earliest explicitly failed causal pipeline stage."""
    return next(
        (
            stage
            for stage in DIAGNOSTIC_STAGE_ORDER
            if stages.get(stage, {}).get("passed") is False
        ),
        None,
    )


def summarize_stage_diagnostics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate compact per-control stage results without retaining model tensors."""
    stages: dict[str, dict[str, Any]] = {}
    for stage in DIAGNOSTIC_STAGE_ORDER:
        results = [
            bool(item.get("stages", {}).get(stage, {}).get("passed"))
            for item in observations
            if stage in item.get("stages", {})
        ]
        passed_count = sum(results)
        stages[stage] = {
            "passed": bool(results) and passed_count == len(results),
            "pass_count": passed_count,
            "evaluated_count": len(results),
            "pass_rate": passed_count / len(results) if results else 0.0,
        }
    return {
        "control_count": len(observations),
        "stages": stages,
        "first_failed_stage": first_failed_stage(stages),
        "controls": observations,
    }


def summarize_sensitivity(observations: list[dict[str, Any]], threshold: float = 1e-4) -> dict[str, Any]:
    direction_matches = [
        bool(match)
        for observation in observations
        for match in observation.get("direction_matches", [])
    ]
    return {
        "evaluated_intervention_count": len(observations),
        "nonzero_response_rate": (
            sum(float(item.get("max_abs_delta", 0.0)) > threshold for item in observations) / len(observations)
            if observations else 0.0
        ),
        "expected_direction_rate": (
            sum(direction_matches) / len(direction_matches) if direction_matches else 0.0
        ),
    }


def logit_margin_diagnostics(
    baseline_logits: Any,
    disturbed_logits: Any,
    target_names: list[str],
) -> list[dict[str, Any]]:
    """Describe how far each affected output remains from an argmax flip."""
    import numpy as np

    def as_array(values: Any) -> Any:
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values, dtype=np.float64)

    baseline = as_array(baseline_logits)
    disturbed = as_array(disturbed_logits)
    if baseline.shape != disturbed.shape or baseline.ndim != 2:
        raise ValueError("baseline and disturbed logits must have matching [target, vocab] shapes")
    if baseline.shape[0] != len(target_names):
        raise ValueError("target_names length must match the target-logit dimension")
    results: list[dict[str, Any]] = []
    for target_name, base_row, disturbed_row in zip(target_names, baseline, disturbed):
        base_order = np.argsort(base_row)[::-1]
        disturbed_order = np.argsort(disturbed_row)[::-1]
        base_top1 = int(base_order[0])
        base_top2 = int(base_order[1]) if len(base_order) > 1 else base_top1
        disturbed_top1 = int(disturbed_order[0])
        disturbed_top2 = int(disturbed_order[1]) if len(disturbed_order) > 1 else disturbed_top1
        alternatives = np.delete(disturbed_row, base_top1)
        best_alternative = float(np.max(alternatives)) if alternatives.size else float(disturbed_row[base_top1])
        margin = float(disturbed_row[base_top1] - best_alternative)
        results.append(
            {
                "target_variable": target_name,
                "baseline_top1_token": base_top1,
                "baseline_top1_logit": float(base_row[base_top1]),
                "baseline_top2_token": base_top2,
                "baseline_top2_logit": float(base_row[base_top2]),
                "disturbed_top1_token": disturbed_top1,
                "disturbed_top1_logit": float(disturbed_row[disturbed_top1]),
                "disturbed_top2_token": disturbed_top2,
                "disturbed_top2_logit": float(disturbed_row[disturbed_top2]),
                "baseline_winner_logit_delta": float(disturbed_row[base_top1] - base_row[base_top1]),
                "baseline_winner_margin_after_disturbance": margin,
                "argmax_changed": disturbed_top1 != base_top1,
            }
        )
    return results


def summarize_logit_margins(entries: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(item["baseline_winner_margin_after_disturbance"]) for item in entries]
    return {
        "minimum_baseline_winner_margin_after_disturbance": min(margins) if margins else None,
        "baseline_winner_overtaken_count": sum(bool(item.get("argmax_changed")) for item in entries),
    }


def _read_registry(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    entries = json.loads(path.read_text(encoding="utf-8-sig"))["variables"]
    return entries, {str(item["variable"]): item for item in entries}


def _read_graph(path: Path) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            left, right = str(row.get("node") or "").strip(), str(row.get("connected_node") or "").strip()
            if left and right:
                graph.setdefault(left, set()).add(right)
                graph.setdefault(right, set()).add(left)
    return graph


def _distances(graph: dict[str, set[str]], start: str) -> dict[str, int]:
    found = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, ()):
            if neighbor not in found:
                found[neighbor] = found[node] + 1
                queue.append(neighbor)
    return found


def _nearest_outputs(
    control: dict[str, Any],
    quantity: str,
    outputs: list[dict[str, Any]],
    graph: dict[str, set[str]],
) -> list[str]:
    matches = [item for item in outputs if item.get("physical_quantity") == quantity]
    if not matches:
        return []
    distances = _distances(graph, str(control.get("equipment_id")))
    ranked = [(distances.get(str(item.get("equipment_id")), math.inf), str(item["variable"])) for item in matches]
    minimum = min(distance for distance, _ in ranked)
    return [variable for distance, variable in ranked if distance == minimum][:3]


def _row_means(forecast: dict[str, Any]) -> dict[str, float]:
    rows = forecast.get("predict_rows") or []
    totals: dict[str, float] = {}
    for row in rows:
        values = row.get("values") if isinstance(row, dict) else getattr(row, "values", None)
        for variable, value in dict(values or {}).items():
            totals[variable] = totals.get(variable, 0.0) + float(value)
    count = max(len(rows), 1)
    return {variable: total / count for variable, total in totals.items()}


def _loss(checkpoint_dir: Path) -> float:
    candidates = [checkpoint_dir / "trainer_state.json", checkpoint_dir.parent / "trainer_state.json"]
    for path in candidates:
        if path.is_file():
            state = json.loads(path.read_text(encoding="utf-8-sig"))
            value = state.get("best_metric")
            if value is not None:
                return float(value)
            history = [item.get("eval_loss") for item in state.get("log_history", []) if item.get("eval_loss") is not None]
            if history:
                return float(min(history))
    raise ValueError(f"No validation loss found near checkpoint {checkpoint_dir}.")


def _coverage(data_dir: Path, controls: list[dict[str, Any]]) -> float:
    train_root = data_dir / "dataset" / "train"
    expected = 2 * len(controls)
    present = sum((train_root / f"case_{1001 + index:04d}" / "Boundary.csv").is_file() for index in range(expected))
    return present / expected if expected else 0.0


def _control_family(control: dict[str, Any]) -> str:
    quantity = str(control.get("physical_quantity") or "")
    if quantity in {"supply_flow_setpoint", "source_pressure_setpoint"}:
        return "supply"
    if quantity == "demand_flow_setpoint":
        return "demand"
    if quantity == "valve_flow_ratio":
        return "valve"
    if quantity == "downstream_pressure_setpoint":
        return "regulator"
    if quantity in {"outlet_pressure_setpoint", "rotational_speed_setpoint"}:
        return "compressor"
    if quantity == "equipment_status":
        return "equipment_status"
    return quantity or "unknown"


def status_intervention_direction(baseline_value: float) -> str:
    """Toggle a binary status away from its actual loaded baseline."""
    return "up" if float(baseline_value) < 0.5 else "down"


def _case_control_baseline(data_dir: Path, variable: str) -> float:
    boundary_path = data_dir / "dataset" / "train" / "case_001" / "Boundary.csv"
    with boundary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if not row or variable not in row:
        raise ValueError(f"Missing baseline value for {variable} in {boundary_path}")
    return float(row[variable])


def _representative_controls(
    controls: list[dict[str, Any]],
    *,
    all_controls: bool,
) -> list[dict[str, Any]]:
    if all_controls:
        return controls
    selected: dict[str, dict[str, Any]] = {}
    for control in controls:
        selected.setdefault(_control_family(control), control)
    return list(selected.values())


def diagnose_causal_stages(
    *,
    checkpoint_dir: Path,
    repo_root: Path,
    controls: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    graph: dict[str, set[str]],
    all_controls: bool = False,
    device: str = "cpu",
    disturbance_timing_mode: str = "current_step",
    threshold: float = 1e-4,
) -> dict[str, Any]:
    """Locate where synthetic control sensitivity first disappears."""
    import numpy as np
    import torch

    backend_root = repo_root / "pipeclaw" / "backend"
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    from pipeline.pipeformer_inference import (
        add_pipeformer_import_paths,
        apply_condition_to_matrix,
        attach_hybrid_token_statistics,
        build_hybrid_token_features,
        load_attention_indices,
        load_case_matrix,
        load_pipeformer_model,
        load_prediction_mask,
        load_tokenizer,
        load_training_config,
        load_variable_mapping,
    )

    pipeformer_root = repo_root / "pipeFormer"
    data_dir = pipeformer_root / "data" / "mock_lifecycle"
    static_dir = data_dir / "static" / "mock_lifecycle"
    add_pipeformer_import_paths(pipeformer_root)
    training_config = load_training_config(checkpoint_dir, pipeformer_root)
    variable_mapping = load_variable_mapping(static_dir / "index_variable_mapping.csv")
    variable_names = [
        name for name, _ in sorted(variable_mapping.items(), key=lambda item: item[1]["index"])
    ]
    name_to_index = {name: index for index, name in enumerate(variable_names)}
    base_matrix, _ = load_case_matrix(data_dir / "dataset" / "train" / "case_001", variable_names)
    sequence_length = int(training_config.get("sequence_length", 5))
    base_window = np.asarray(base_matrix[:sequence_length], dtype=np.float32)
    tokenizer = load_tokenizer(static_dir, vocab_size=training_config.get("tokenizer_vocab_size"))
    model, model_config, _, _ = load_pipeformer_model(checkpoint_dir, pipeformer_root, device)
    attach_hybrid_token_statistics(model, model_config, tokenizer)
    projection_type = str(model_config.get("input_projection_type", "token_embedding"))
    attention_indices = load_attention_indices(static_dir)
    prediction_mask = load_prediction_mask(static_dir, variable_names)
    layer_count = int(model_config.get("n_layers", 1))
    attention_tensor = torch.as_tensor(attention_indices, dtype=torch.long, device=device).unsqueeze(0)
    mask_tensor = torch.as_tensor(prediction_mask, dtype=torch.float32, device=device).unsqueeze(0)

    def forward(window: Any) -> Any:
        tokens = np.asarray(tokenizer.transform_to_tokens(window), dtype=np.int64)
        hybrid_inputs: dict[str, Any] = {}
        if projection_type.lower() == "hybrid":
            token_medians, token_offsets = build_hybrid_token_features(
                tokenizer, window, tokens
            )
            hybrid_inputs = {
                "input_token_medians": torch.as_tensor(
                    token_medians, dtype=torch.float32, device=device
                ).unsqueeze(0),
                "input_token_offsets": torch.as_tensor(
                    token_offsets, dtype=torch.float32, device=device
                ).unsqueeze(0),
            }
        with torch.no_grad():
            result = model(
                input_ids=torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0),
                input_tokens=torch.as_tensor(tokens, dtype=torch.long, device=device).unsqueeze(0),
                prediction_mask=mask_tensor,
                attention_indices=attention_tensor,
                **hybrid_inputs,
            )
        logits = result.get("token_logits") if isinstance(result, dict) else None
        if logits is None:
            raise RuntimeError("PipeFormer checkpoint did not return token_logits for stage diagnosis.")
        return tokens, logits

    selected_controls = _representative_controls(controls, all_controls=all_controls)
    observations: list[dict[str, Any]] = []
    for control in selected_controls:
        variable = str(control["variable"])
        global_control_index = next(
            index for index, item in enumerate(controls) if item["variable"] == variable
        )
        is_status = control.get("physical_quantity") == "equipment_status"
        direction = (
            status_intervention_direction(base_window[0, name_to_index[variable]])
            if is_status
            else ("up" if global_control_index % 2 == 0 else "down")
        )
        sign = 1.0 if direction == "up" else -1.0
        task = {
            "disturbance_variable": variable,
            "disturbance_direction": direction,
            "disturbance_magnitude_percent": 100.0 if is_status else 12.0,
            "boundary_conditions": (
                {"setpoints": {variable: 1.0 if direction == "up" else 0.0}}
                if is_status
                else {"percentage_changes": {variable: sign * 12.0}}
            ),
        }
        disturbed_window = apply_condition_to_matrix(
            base_window,
            task,
            variable_mapping,
            timing_mode=disturbance_timing_mode,
        )
        control_index = name_to_index[variable]
        numeric_input_delta = float(
            np.max(np.abs(disturbed_window[:, control_index] - base_window[:, control_index]))
        )
        target_names: list[str] = []
        for effect in control.get("effect_targets") or []:
            for target in _nearest_outputs(
                control,
                str(effect.get("physical_quantity")),
                outputs,
                graph,
            ):
                if target in name_to_index and target not in target_names:
                    target_names.append(target)
        target_indices = [name_to_index[name] for name in target_names]

        heldout_case = data_dir / "dataset" / "test" / f"case_{2001 + global_control_index:04d}"
        generated_target_delta = 0.0
        if heldout_case.is_dir() and target_indices:
            heldout_matrix, _ = load_case_matrix(heldout_case, variable_names)
            pre = heldout_matrix[20:30, target_indices].mean(axis=0)
            post = heldout_matrix[35:45, target_indices].mean(axis=0)
            generated_target_delta = float(np.max(np.abs(post - pre)))

        base_tokens, base_logits = forward(base_window)
        disturbed_tokens, disturbed_logits = forward(disturbed_window)
        changed_control_tokens = int(
            np.count_nonzero(base_tokens[:, control_index] != disturbed_tokens[:, control_index])
        )
        reachable_targets = [
            name
            for name, target_index in zip(target_names, target_indices)
            if attention_reaches_control(
                attention_indices,
                control_index,
                target_index,
                layer_count,
            )
        ]

        if target_indices:
            base_target_logits = base_logits[0, -1, target_indices, :]
            disturbed_target_logits = disturbed_logits[0, -1, target_indices, :]
            margin_entries = logit_margin_diagnostics(
                base_target_logits, disturbed_target_logits, target_names
            )
            margin_summary = summarize_logit_margins(margin_entries)
            logit_delta = float(
                torch.max(torch.abs(disturbed_target_logits - base_target_logits)).item()
            )
            base_all_tokens = torch.argmax(base_logits[0, -1, :, :], dim=-1)
            disturbed_all_tokens = torch.argmax(disturbed_logits[0, -1, :, :], dim=-1)
            base_target_tokens = base_all_tokens[target_indices]
            disturbed_target_tokens = disturbed_all_tokens[target_indices]
            changed_target_count = int(
                torch.count_nonzero(base_target_tokens != disturbed_target_tokens).item()
            )
            base_values = tokenizer.tokens_to_values(base_all_tokens.unsqueeze(0))
            disturbed_values = tokenizer.tokens_to_values(disturbed_all_tokens.unsqueeze(0))
            base_values = torch.as_tensor(base_values, dtype=torch.float32).squeeze(0)[target_indices]
            disturbed_values = torch.as_tensor(disturbed_values, dtype=torch.float32).squeeze(0)[target_indices]
            decoded_value_delta = float(torch.max(torch.abs(disturbed_values - base_values)).item())
        else:
            logit_delta = 0.0
            changed_target_count = 0
            decoded_value_delta = 0.0
            margin_entries = []
            margin_summary = summarize_logit_margins([])

        stages = {
            "data_generation": {
                "passed": numeric_input_delta > 0.0 and generated_target_delta > threshold,
                "numeric_input_max_abs_delta": numeric_input_delta,
                "generated_target_max_abs_delta": generated_target_delta,
            },
            "tokenization": {
                "passed": changed_control_tokens > 0,
                "changed_control_token_count": changed_control_tokens,
            },
            "attention_routing": {
                "passed": bool(target_names) and len(reachable_targets) == len(target_names),
                "declared_target_count": len(target_names),
                "reachable_target_count": len(reachable_targets),
            },
            "logit_response": {
                "passed": logit_delta > 1e-7,
                "max_abs_delta": logit_delta,
                **margin_summary,
                "affected_output_logit_margins": margin_entries,
            },
            "argmax_decoding": {
                "passed": changed_target_count > 0 and decoded_value_delta > threshold,
                "changed_target_count": changed_target_count,
                "decoded_value_max_abs_delta": decoded_value_delta,
            },
        }
        observations.append(
            {
                "variable": variable,
                "family": _control_family(control),
                "direction": direction,
                "declared_targets": target_names,
                "first_failed_stage": first_failed_stage(stages),
                "stages": stages,
            }
        )
    return summarize_stage_diagnostics(observations)


def evaluate_checkpoint(
    *,
    checkpoint_dir: Path,
    current_checkpoint_dir: Path,
    repo_root: Path,
    forecast_runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    magnitude_percent: float = 12.0,
    threshold: float = 1e-4,
    diagnose_stages: bool = False,
    all_controls: bool = False,
    device: str = "cpu",
    disturbance_timing_mode: str = "current_step",
) -> dict[str, Any]:
    pipeformer_root = repo_root / "pipeFormer"
    static_dir = pipeformer_root / "data" / "mock_lifecycle" / "static" / "mock_lifecycle"
    data_dir = pipeformer_root / "data" / "mock_lifecycle"
    entries, _ = _read_registry(static_dir / "variable_registry.json")
    controls = [item for item in entries if item.get("role") == "input" and item.get("controllable") is True]
    outputs = [item for item in entries if item.get("role") == "output"]
    graph = _read_graph(static_dir / "save_connect_all_nodes.csv")

    if forecast_runner is None:
        backend_root = repo_root / "pipeclaw" / "backend"
        sys.path.insert(0, str(backend_root))
        from pipeline.pipeformer_inference import run_checkpoint_inference

        def forecast_runner(task: dict[str, Any]) -> dict[str, Any]:
            return run_checkpoint_inference(
                parsed_task=task,
                checkpoint_dir=checkpoint_dir,
                pipeformer_root=pipeformer_root,
                data_dir=data_dir,
                static_dir=static_dir,
                mapping_path=static_dir / "index_variable_mapping.csv",
                device=device,
                disturbance_timing_mode=disturbance_timing_mode,
            )

    common = {
        "case_id": "mock_test_001",
        "current_operating_condition_number": 1,
        "forecast_horizon_minutes": 5,
    }
    first = str(controls[0]["variable"])
    baseline_task = {
        **common,
        "disturbance_variable": first,
        "disturbance_direction": "up",
        "disturbance_magnitude_percent": 0.0,
        "boundary_conditions": {"percentage_changes": {first: 0.0}},
    }
    baseline = _row_means(forecast_runner(dict(baseline_task)))
    repeated_baseline = _row_means(forecast_runner(dict(baseline_task)))
    unchanged_delta = max((abs(repeated_baseline.get(name, value) - value) for name, value in baseline.items()), default=0.0)

    observations = []
    for index, control in enumerate(controls):
        variable = str(control["variable"])
        is_status = control.get("physical_quantity") == "equipment_status"
        direction = (
            status_intervention_direction(_case_control_baseline(data_dir, variable))
            if is_status
            else ("up" if index % 2 == 0 else "down")
        )
        sign = 1.0 if direction == "up" else -1.0
        boundary = {"setpoints": {variable: 1.0 if direction == "up" else 0.0}} if is_status else {
            "percentage_changes": {variable: sign * magnitude_percent}
        }
        task = {
            **common,
            "disturbance_variable": variable,
            "disturbance_direction": direction,
            "disturbance_magnitude_percent": 100.0 if is_status else magnitude_percent,
            "boundary_conditions": boundary,
        }
        candidate = _row_means(forecast_runner(task))
        output_deltas = {
            str(item["variable"]): candidate.get(str(item["variable"]), baseline.get(str(item["variable"]), 0.0))
            - baseline.get(str(item["variable"]), 0.0)
            for item in outputs
        }
        matches = []
        for effect in control.get("effect_targets") or []:
            target_variables = _nearest_outputs(control, str(effect.get("physical_quantity")), outputs, graph)
            if not target_variables:
                continue
            delta = sum(output_deltas[name] for name in target_variables) / len(target_variables)
            declared_sign = 1.0 if effect.get("direction") == "positive" else -1.0
            matches.append(delta * declared_sign * sign > 0.0)
        observations.append(
            {
                "variable": variable,
                "direction": direction,
                "max_abs_delta": max((abs(value) for value in output_deltas.values()), default=0.0),
                "direction_matches": matches,
            }
        )

    report = {
        "checkpoint_dir": checkpoint_dir.resolve().as_posix(),
        "current_checkpoint_dir": current_checkpoint_dir.resolve().as_posix(),
        "intervention_coverage": _coverage(data_dir, controls),
        "unchanged_baseline_max_delta": unchanged_delta,
        "candidate_eval_loss": _loss(checkpoint_dir),
        "current_eval_loss": _loss(current_checkpoint_dir),
        "response_threshold": threshold,
        **summarize_sensitivity(observations, threshold),
        "interventions": observations,
    }
    if diagnose_stages:
        report["stage_diagnostics"] = diagnose_causal_stages(
            checkpoint_dir=checkpoint_dir,
            repo_root=repo_root,
            controls=controls,
            outputs=outputs,
            graph=graph,
            all_controls=all_controls,
            device=device,
            disturbance_timing_mode=disturbance_timing_mode,
            threshold=threshold,
        )
        report["first_failed_stage"] = report["stage_diagnostics"]["first_failed_stage"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--current-checkpoint-dir", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output", default="outputs/mock_decoder_candidate/sensitivity_report.json")
    parser.add_argument("--diagnose-stages", action="store_true")
    parser.add_argument("--all-controls", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--disturbance-timing-mode",
        choices=("legacy_observation_window", "current_step"),
        default="current_step",
    )
    args = parser.parse_args()
    report = evaluate_checkpoint(
        checkpoint_dir=Path(args.checkpoint_dir),
        current_checkpoint_dir=Path(args.current_checkpoint_dir),
        repo_root=Path(args.repo_root),
        diagnose_stages=args.diagnose_stages,
        all_controls=args.all_controls,
        device=args.device,
        disturbance_timing_mode=args.disturbance_timing_mode,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "interventions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
