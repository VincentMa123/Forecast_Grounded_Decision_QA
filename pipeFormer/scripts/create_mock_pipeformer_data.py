#!/usr/bin/env python3
"""Create a coherent synthetic PipeFormer fixture for lifecycle task controls.

The generated signals are causal but synthetic. They validate data, model, and
teacher-trace integration; they are not calibrated pipeline simulations.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BOUNDARY_DIMS = 538
VARIABLE_RE = re.compile(r"\b[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?\b")
CONTROL_SPECS = {
    ("T", "SNQ"): ("supply_flow_setpoint", "p.u.", 0.4, 1.6),
    ("E", "SNQ"): ("demand_flow_setpoint", "p.u.", 0.4, 1.6),
    ("B", "FR"): ("valve_flow_ratio", "p.u.", 0.0, 1.2),
    ("R", "SPD"): ("downstream_pressure_setpoint", "p.u.", 0.4, 1.6),
    ("C", "SP_"): ("rotational_speed_setpoint", "p.u.", 0.4, 1.6),
    ("C", "SP_out"): ("outlet_pressure_setpoint", "p.u.", 0.4, 1.6),
    ("T", "SP"): ("source_pressure_setpoint", "p.u.", 0.4, 1.6),
    ("C", "ST"): ("equipment_status", "binary", 0.0, 1.0),
    ("R", "ST"): ("equipment_status", "binary", 0.0, 1.0),
}


def extract_inventory(
    dataset_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any], dict[str, list[str]]]:
    all_scenarios: list[dict[str, Any]] = []
    source_summary: dict[str, Any] = {}
    variable_sources: dict[str, list[str]] = defaultdict(list)
    for dataset_path in dataset_paths:
        scenarios = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
        if not isinstance(scenarios, list):
            raise TypeError(f"Lifecycle dataset must contain a JSON list: {dataset_path}")
        texts = []
        scenario_types: dict[str, int] = defaultdict(int)
        session_count = 0
        turn_count = 0
        for scenario in scenarios:
            scenario_types[str(scenario.get("scenario_type") or "unknown")] += 1
            texts.append(str(scenario.get("scenario_description") or ""))
            sessions = scenario.get("sessions") or []
            session_count += len(sessions)
            for session in sessions:
                dialogue = session.get("dialogue") or []
                turn_count += len(dialogue)
                for turn in dialogue:
                    texts.append(str(turn.get("user_input") or ""))
        variables = sorted({match.group(0) for match in VARIABLE_RE.finditer("\n".join(texts)) if ":" in match.group(0)})
        for variable in variables:
            variable_sources[variable].append(dataset_path.name)
        all_scenarios.extend(scenarios)
        source_summary[dataset_path.name] = {
            "scenario_count": len(scenarios),
            "scenario_types": dict(sorted(scenario_types.items())),
            "session_count": session_count,
            "turn_count": turn_count,
            "control_variable_count": len(variables),
        }
    return all_scenarios, sorted(variable_sources), source_summary, dict(variable_sources)


def control_registry(variable: str, sources: list[str]) -> dict[str, Any]:
    equipment, tag = variable.split(":", 1)
    equipment_prefix = equipment.split("_", 1)[0]
    equipment_type = {
        "T": "gas_source",
        "E": "demand_boundary",
        "B": "ball_valve",
        "R": "pressure_regulator",
        "C": "compressor",
    }.get(equipment_prefix, "boundary_control")
    quantity, unit, lower, upper = CONTROL_SPECS.get(
        (equipment_prefix, tag),
        ("control_setpoint", "p.u.", 0.0, 2.0),
    )
    return {
        "variable": variable,
        "equipment_id": equipment,
        "equipment_type": equipment_type,
        "physical_quantity": quantity,
        "role": "input",
        "unit": unit,
        "controllable": True,
        "lower_limit": lower,
        "upper_limit": upper,
        "warning_lower_limit": lower,
        "warning_upper_limit": upper,
        "source": sources[0] if len(sources) == 1 else "multiple_lifecycle_sources",
        "sources": sources,
    }


def state_registry(control_names: list[str]) -> list[dict[str, Any]]:
    specs: list[tuple[str, str, str, str, float, float, float, float]] = []
    for device in ("N_001", "N_002", "N_003"):
        specs.append((f"{device}_v000", device, "node", "pressure", -3.0, 3.0, -2.5, 2.5))
    for device in ("P_001", "P_002"):
        specs.extend(
            [
                (f"{device}_v000", device, "pipeline_segment", "pressure", -3.0, 3.0, -2.5, 2.5),
                (f"{device}_v001", device, "pipeline_segment", "flow", -3.0, 3.0, -2.2, 2.2),
            ]
        )
    devices_by_prefix: dict[str, list[str]] = defaultdict(list)
    for variable in control_names:
        device = variable.split(":", 1)[0]
        prefix = device.split("_", 1)[0]
        if device not in devices_by_prefix[prefix]:
            devices_by_prefix[prefix].append(device)
    for device in sorted(devices_by_prefix["B"]):
        specs.extend(
            [
                (f"{device}_v000", device, "ball_valve", "valve_opening", -3.0, 3.0, -2.5, 2.5),
                (f"{device}_v001", device, "ball_valve", "flow", -3.0, 3.0, -2.2, 2.2),
            ]
        )
    for device in ("H_001", "H_002"):
        specs.append((f"{device}_v000", device, "pipeline_segment", "linepack", -3.0, 3.0, -2.5, 2.5))
    for device in sorted(devices_by_prefix["C"]):
        specs.extend(
            [
                (f"{device}_v000", device, "compressor", "compressor_load", -2.0, 2.0, -1.2, 1.2),
                (f"{device}_v001", device, "compressor", "compression_ratio", -2.0, 2.0, -1.2, 1.2),
                (f"{device}_v002", device, "compressor", "rotational_speed", -2.0, 2.0, -1.2, 1.2),
            ]
        )
        power_device = device.replace("C_", "TE_", 1)
        specs.append((f"{power_device}_v000", power_device, "compressor_power", "power", -3.0, 3.0, -2.5, 2.5))
    for device in sorted(devices_by_prefix["R"]):
        specs.append((f"{device}_v000", device, "pressure_regulator", "regulator_range", -3.0, 3.0, -2.5, 2.5))
    return [
        {
            "variable": variable,
            "equipment_id": equipment,
            "equipment_type": equipment_type,
            "physical_quantity": quantity,
            "role": "output",
            "unit": "p.u.",
            "controllable": False,
            "lower_limit": lower,
            "upper_limit": upper,
            "warning_lower_limit": warning_lower,
            "warning_upper_limit": warning_upper,
            "source": "synthetic_derived_state",
        }
        for variable, equipment, equipment_type, quantity, lower, upper, warning_lower, warning_upper in specs
    ]


def make_dirs(root: Path, force: bool) -> dict[str, Path]:
    if root.exists() and force:
        shutil.rmtree(root)
    paths = {
        "train": root / "dataset" / "train",
        "test": root / "dataset" / "test",
        "static_full": root / "static" / "full",
        "static_active": root / "static" / "mock_lifecycle",
        "process": root / "process_eq_argu",
        "relative": root / "relative_PPT",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_mapping(path: Path, names: list[str], global_indices: list[int] | None = None) -> None:
    payload: dict[str, Any] = {"index": np.arange(len(names), dtype=np.int32), "variable_name": names}
    if global_indices is not None:
        payload["global_index"] = np.asarray(global_indices, dtype=np.int32)
    pd.DataFrame(payload).to_csv(path, index=False)


def full_mapping(control_names: list[str], state_names: list[str]) -> list[str]:
    fillers = [f"T_999:BC{i:03d}" for i in range(BOUNDARY_DIMS - len(control_names))]
    equipment = list(state_names)
    for prefix, count in (("B", 2058), ("C", 161), ("H", 192), ("N", 1716), ("P", 1610), ("R", 50), ("TE", 387)):
        existing = sum(name.startswith(f"{prefix}_") for name in equipment)
        equipment.extend(f"{prefix}_999_v{i:04d}" for i in range(count - existing))
    return control_names + fillers + equipment


def equipment_id(variable: str) -> str:
    base = variable.split(":", 1)[0]
    parts = base.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else base


def graph_edges(registry: list[dict[str, Any]]) -> list[tuple[str, str]]:
    entities = sorted({str(item["equipment_id"]) for item in registry})
    groups = defaultdict(list)
    for entity in entities:
        groups[entity.split("_", 1)[0]].append(entity)
    backbone = groups["T"] + groups["N"] + groups["P"] + groups["H"] + groups["B"] + groups["C"] + groups["R"] + groups["E"] + groups["TE"]
    return list(dict.fromkeys((backbone[index], backbone[index + 1]) for index in range(len(backbone) - 1)))


def write_static(paths: dict[str, Path], root: Path, registry: list[dict[str, Any]]) -> None:
    controls = [item["variable"] for item in registry if item["role"] == "input"]
    states = [item["variable"] for item in registry if item["role"] == "output"]
    full_names = full_mapping(controls, states)
    active_names = controls + states
    lookup = {name: index for index, name in enumerate(full_names)}
    write_mapping(paths["static_full"] / "index_variable_mapping.csv", full_names)
    write_mapping(paths["static_active"] / "index_variable_mapping.csv", active_names, [lookup[name] for name in active_names])
    pd.DataFrame({"variable_name": active_names, "predict": [0] * len(controls) + [1] * len(states)}).to_csv(
        paths["static_active"] / "prediction_mask.csv", index=False
    )
    registry_payload = {
        "schema_version": "mock_lifecycle_registry_v1",
        "synthetic": True,
        "physical_validation_status": "not_validated",
        "variables": registry,
    }
    for directory in (root, paths["static_active"]):
        (directory / "variable_registry.json").write_text(json.dumps(registry_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(registry).to_csv(root / "variable_registry.csv", index=False)

    edges = graph_edges(registry)
    graph = defaultdict(set)
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)
    graph_data = {
        "graph": dict(graph),
        "equipment_dir": "equipment_arguments",
        "connected_nodes_file": "relative_PPT/connected_nodes.csv",
        "target_equipment_types": {"T", "E", "C", "R", "B", "N", "P", "H"},
    }
    with (paths["static_active"] / "pipeline_graph_cache.pkl").open("wb") as handle:
        pickle.dump(graph_data, handle)
    pd.DataFrame([{"node": left, "connected_node": right} for left, right in edges]).to_csv(
        paths["static_active"] / "save_connect_all_nodes.csv", index=False
    )
    hyper = {
        "graph": {"is_subgraph": True, "center_node": "P_001", "total_nodes": len(graph), "total_edges": len(edges), "mock_data": True},
        "variables": {"total_variables": len(active_names), "boundary_variables": len(controls), "equipment_variables": len(states)},
        "tokenizer": {"quantile_step": 0.1, "constant_freq_threshold": 0.995, "constant_variable_threshold": 0.995},
    }
    (paths["static_active"] / "graph_hyperparameters.json").write_text(json.dumps(hyper, indent=2), encoding="utf-8")
    (paths["static_full"] / "graph_hyperparameters.json").write_text(
        json.dumps({"variables": {"total_variables": len(full_names), "boundary_variables": BOUNDARY_DIMS}}, indent=2), encoding="utf-8"
    )
    write_attention(paths["static_active"], active_names, graph)


def write_attention(static_dir: Path, names: list[str], graph: dict[str, set[str]], max_neighbors: int = 12) -> None:
    by_entity: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(names):
        by_entity[equipment_id(name)].append(index)
    indices = np.zeros((len(names), max_neighbors), dtype=np.int32)
    labels = np.empty((len(names), max_neighbors), dtype=object)
    for index, name in enumerate(names):
        entity = equipment_id(name)
        candidates = [index] + [value for value in by_entity[entity] if value != index]
        for neighbor in graph.get(entity, set()):
            candidates.extend(by_entity.get(neighbor, []))
        deduped = list(dict.fromkeys(candidates)) or [index]
        padded = (deduped + [index] * max_neighbors)[:max_neighbors]
        indices[index] = padded
        labels[index] = [names[value] for value in padded]
    with (static_dir / "attention_indices.pkl").open("wb") as handle:
        pickle.dump({"attention_indices": indices, "variable_names": labels}, handle)
    pd.DataFrame(indices).to_csv(static_dir / "attention_indices.csv", index_label="variable_index")
    pd.DataFrame(labels).to_csv(static_dir / "attention_variable_names.csv", index_label="variable_index")


def baseline(variable: str, case_number: int) -> float:
    tag = variable.split(":", 1)[1]
    prefix = variable.split("_", 1)[0]
    if tag == "ST":
        return 0.0 if (case_number + sum(map(ord, variable))) % 7 == 0 else 1.0
    if tag == "FR":
        return 0.78 + 0.05 * ((case_number % 5) - 2)
    if tag == "SNQ" and prefix == "T":
        return 1.0 + 0.06 * ((case_number % 7) - 3)
    if tag == "SNQ" and prefix == "E":
        return 0.95 + 0.05 * (((case_number * 3) % 7) - 3)
    return 1.0 + 0.05 * ((case_number % 5) - 2)


def control_series(control_names: list[str], case_number: int) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    times = pd.date_range("2025-01-01 00:00:00", "2025-01-01 23:30:00", freq="30min")
    phase = case_number * 0.17
    x = np.linspace(0.0, 2.0 * np.pi, len(times), dtype=np.float32)
    result = {}
    for index, variable in enumerate(control_names):
        tag = variable.split(":", 1)[1]
        if tag == "ST":
            values = np.full(len(times), baseline(variable, case_number), dtype=np.float32)
        else:
            values = baseline(variable, case_number) + 0.08 * np.sin(x + phase + index * 0.23)
        result[variable] = values.astype(np.float32)
    return times, result


def minute_controls(boundary_times: pd.DatetimeIndex, controls: dict[str, np.ndarray]) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    minute_times = pd.date_range("2025-01-01 00:01:00", "2025-01-01 23:59:00", freq="1min")
    result = {}
    for name, values in controls.items():
        series = pd.Series(values, index=boundary_times).reindex(minute_times, method="ffill")
        result[name] = series.fillna(float(values[0])).to_numpy(dtype=np.float32)
    return minute_times, result


def mean_controls(controls: dict[str, np.ndarray], prefixes: tuple[str, ...]) -> np.ndarray:
    selected = [values for name, values in controls.items() if name.startswith(prefixes)]
    return np.mean(selected, axis=0) if selected else np.zeros(len(next(iter(controls.values()))), dtype=np.float32)


def derive_states(controls: dict[str, np.ndarray], case_number: int, state_names: list[str]) -> dict[str, np.ndarray]:
    supply = mean_controls(controls, ("T_",))
    demand = mean_controls(controls, ("E_",))
    valves = mean_controls(controls, ("B_",))
    regulators = mean_controls(controls, ("R_",))
    compressor = mean_controls(controls, ("C_",))
    gap = supply - demand
    x = np.linspace(0.0, 2.0 * np.pi, len(supply), dtype=np.float32)
    wave = 0.06 * np.sin(x + case_number * 0.11)
    linepack = np.empty_like(gap)
    linepack[0] = 0.9 + 0.08 * gap[0]
    for index in range(1, len(linepack)):
        linepack[index] = 0.997 * linepack[index - 1] + 0.003 * (0.9 + 0.7 * gap[index])

    result: dict[str, np.ndarray] = {}
    for offset, name in enumerate(state_names):
        jitter = wave + 0.008 * np.sin(3.0 * x + offset * 0.3)
        if name.startswith(("N_", "P_")) and name.endswith("_v000"):
            values = 0.65 + 0.65 * gap + 0.18 * (compressor - 1.0) + 0.10 * (regulators - 1.0) - offset * 0.01 + jitter
        elif name.startswith("P_") and name.endswith("_v001"):
            values = 0.75 + 0.55 * demand + 0.12 * gap + jitter
        elif name.startswith("B_") and name.endswith("_v000"):
            values = 0.2 + 0.9 * valves + jitter
        elif name.startswith("B_") and name.endswith("_v001"):
            values = 0.72 + 0.55 * supply + 0.10 * (valves - 0.8) + jitter
        elif name.startswith("H_"):
            values = linepack - offset * 0.006 + jitter
        elif name.startswith("C_") and name.endswith("_v000"):
            values = 0.62 + 0.35 * demand - 0.20 * supply + 0.30 * compressor + jitter
        elif name.startswith("C_") and name.endswith("_v001"):
            values = 0.55 + 0.28 * demand - 0.15 * supply + 0.26 * compressor + jitter + 0.14 * np.sin(2.0 * x + offset * 0.2)
        elif name.startswith("C_") and name.endswith("_v002"):
            values = 0.25 + 0.78 * compressor + jitter
        elif name.startswith("TE_"):
            values = 0.35 + 0.48 * compressor + 0.25 * demand + jitter
        elif name.startswith("R_"):
            values = 0.15 + 0.85 * regulators + jitter
        else:
            values = 0.8 + jitter
        result[name] = np.asarray(values, dtype=np.float32)
    return result


def write_case(case_dir: Path, case_number: int, controls: list[str], states: list[str]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    boundary_times, control_values = control_series(controls, case_number)
    boundary_payload: dict[str, Any] = {"TIME": boundary_times.strftime("%Y-%m-%d %H:%M:%S")}
    boundary_payload.update(control_values)
    for index in range(BOUNDARY_DIMS - len(controls)):
        boundary_payload[f"T_999:BC{index:03d}"] = np.zeros(len(boundary_times), dtype=np.float32)
    pd.DataFrame(boundary_payload).to_csv(case_dir / "Boundary.csv", index=False)

    minute_times, minute_control_values = minute_controls(boundary_times, control_values)
    state_values = derive_states(minute_control_values, case_number, states)
    by_file: dict[str, dict[str, Any]] = defaultdict(lambda: {"TIME": minute_times.strftime("%Y-%m-%d %H:%M:%S")})
    for name, values in state_values.items():
        prefix = name.split("_", 1)[0]
        filename = "T&E" if prefix == "TE" else prefix
        by_file[filename][name] = values
    for filename, payload in by_file.items():
        pd.DataFrame(payload).to_csv(case_dir / f"{filename}.csv", index=False)


def write_samples(paths: dict[str, Path], controls: list[str], states: list[str]) -> None:
    for case_number in range(1, 41):
        write_case(paths["train"] / f"case_{case_number:03d}", case_number, controls, states)
    for case_number in range(101, 105):
        write_case(paths["test"] / f"case_{case_number:03d}", case_number, controls, states)


def write_placeholders(paths: dict[str, Path]) -> None:
    pd.DataFrame([[0.0] * 9], index=["P_001"], columns=[f"pipe_feature_{index}" for index in range(9)]).to_csv(
        paths["process"] / "pipe_features.csv"
    )
    pd.DataFrame([[0.0]], index=["C_001"], columns=["pca_feature_1"]).to_csv(paths["process"] / "compressor_features.csv")
    pd.DataFrame([{"num_pipes": 1, "num_compressors": 1, "pipe_feature_dim": 9, "compressor_feature_dim": 1}]).to_csv(
        paths["process"] / "metadata.csv", index=False
    )


def write_configs(project_root: Path, data_root: Path, active_dir: Path, variable_count: int, boundary_count: int) -> None:
    model_dir = project_root / "configs" / "models" / "decoder"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "model_name": "FluidDecoder", "model_version": "mock-lifecycle-integration", "input_dim": variable_count,
        "output_dim": variable_count, "sequence_length": 5, "boundary_dims": boundary_count,
        "equipment_dims": variable_count - boundary_count, "d_model": 48, "n_heads": 4, "n_layers": 2,
        "d_ff": 96, "attention_dropout": 0.1, "dropout_rate": 0.1, "tokenizer_vocab_size": 4096,
        "time_position_encoding": "learnable", "variable_position_encoding": "learnable", "max_time_positions": 10,
        "max_variable_positions": variable_count, "projection_hidden_dim": 48, "input_projection_type": "token_embedding",
        "use_topology_attention": True, "use_layer_norm": True, "activation": "gelu",
    }
    (model_dir / "mock_decoder.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    train_config = {
        "data_dir": data_root.relative_to(project_root).as_posix(), "static_dir": active_dir.relative_to(project_root).as_posix(),
        "cache_dir": (active_dir / "cache").relative_to(project_root).as_posix(), "train_batch_size": 4, "eval_batch_size": 4,
        "sequence_length": 5, "max_sequences_per_sample": 4, "num_train_epochs": 1, "learning_rate": 0.0001,
        "weight_decay": 0.00001, "model_config_path": "configs/models/decoder/mock_decoder.json",
        "output_dir": "./outputs/mock_decoder", "eval_steps": 10, "save_steps": 10, "save_total_limit": 1,
        "normalization_method": "standard", "device": "cpu", "dataloader_num_workers": 0, "mixed_precision": False,
        "use_swanlab": False, "debug_mode": True, "log_level": "info", "seed": 42,
    }
    (project_root / "configs" / "mock_decoder.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")


def write_readme(data_root: Path) -> None:
    (data_root / "README.md").write_text(
        """# Mock lifecycle PipeFormer fixture

This dataset covers the union of explicit control-variable vocabularies in the
v4 and v7 lifecycle datasets and adds equipment-specific derived forecast
states for the engineering constraint library. All signals are synthetic and
the checkpoint is not physically validated.

Regenerate and train from `pipeFormer/`:

```powershell
python scripts/create_mock_pipeformer_data.py --force
python build_cache.py --data-dir data/mock_lifecycle --static-dir data/mock_lifecycle/static/mock_lifecycle --skip-tokens --force
python data/compute_tokenizer_stats.py --data_dir data/mock_lifecycle --static-dir data/mock_lifecycle/static/mock_lifecycle --force
python build_cache.py --data-dir data/mock_lifecycle --static-dir data/mock_lifecycle/static/mock_lifecycle --force
python data/compute_normalization_stats.py --static_dir data/mock_lifecycle/static/mock_lifecycle --method standard --force
python train.py --config configs/mock_decoder.json
```
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a v4+v7-compatible synthetic PipeFormer fixture.")
    parser.add_argument("--dataset", action="append", default=None, help="Lifecycle task dataset; repeat for multiple sources.")
    parser.add_argument("--output-dir", default="data/mock_lifecycle")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_values = args.dataset or [
        "../pipeclaw/backend/pipeclaw_data/Pipeline_Full_Life_Cycle_Test_Dataset-v4.json",
        "../pipeclaw/backend/pipeclaw_data/Pipeline_Full_Life_Cycle_Test_Dataset-v7.json",
    ]
    dataset_paths = [(project_root / value).resolve() for value in dataset_values]
    data_root = (project_root / args.output_dir).resolve()
    scenarios, controls, source_summary, variable_sources = extract_inventory(dataset_paths)
    registry = [control_registry(name, variable_sources[name]) for name in controls] + state_registry(controls)
    paths = make_dirs(data_root, args.force)
    write_static(paths, data_root, registry)
    write_samples(paths, controls, [item["variable"] for item in registry if item["role"] == "output"])
    write_placeholders(paths)
    write_configs(project_root, data_root, paths["static_active"], len(registry), len(controls))
    write_readme(data_root)
    manifest = {
        "schema_version": "mock_lifecycle_manifest_v1", "synthetic": True, "physical_validation_status": "not_validated",
        "source_datasets": [path.name for path in dataset_paths], "source_summary": source_summary,
        "source_record_count": len(scenarios),
        "control_variable_count": len(controls), "forecast_state_variable_count": len(registry) - len(controls),
        "control_variables": controls,
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
