#!/usr/bin/env python3
"""
Create a tiny synthetic PipeFormer-compatible dataset for local smoke tests.

The generated data is intentionally fake. It is only meant to validate the
preprocessing/cache/training wiring when the original industrial topology and
sequence files are unavailable.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


BOUNDARY_DIMS = 538
EQUIPMENT_DIMS = {
    "B": 2058,
    "C": 161,
    "H": 192,
    "N": 1716,
    "P": 1610,
    "R": 50,
    "T&E": 387,
}
EQUIPMENT_ORDER = ["B", "C", "H", "N", "P", "R", "T&E"]


def make_dirs(root: Path, force: bool) -> dict[str, Path]:
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    paths = {
        "dataset_train": root / "dataset" / "train",
        "dataset_test": root / "dataset" / "test",
        "static_full": root / "static" / "full",
        "static_tiny": root / "static" / "mock_tiny",
        "relative": root / "relative_PPT",
        "process_eq_argu": root / "process_eq_argu",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def full_variable_names() -> list[str]:
    names: list[str] = [f"T_001:BC{i:03d}" for i in range(BOUNDARY_DIMS)]
    for equipment_type in EQUIPMENT_ORDER:
        count = EQUIPMENT_DIMS[equipment_type]
        prefix = "TE" if equipment_type == "T&E" else equipment_type
        for i in range(count):
            device_no = (i % 3) + 1
            feature_no = i // 3
            names.append(f"{prefix}_{device_no:03d}_v{feature_no:03d}")
    return names


def write_mapping(path: Path, names: list[str], global_indices: list[int] | None = None) -> None:
    data = {"index": np.arange(len(names), dtype=np.int32), "variable_name": names}
    if global_indices is not None:
        data["global_index"] = np.asarray(global_indices, dtype=np.int32)
    pd.DataFrame(data).to_csv(path, index=False)


def write_full_static_mapping(static_full: Path, names: list[str]) -> None:
    write_mapping(static_full / "index_variable_mapping.csv", names)
    boundary_count = BOUNDARY_DIMS
    hyper = {
        "graph": {
            "is_subgraph": False,
            "center_node": None,
            "total_nodes": 8,
            "total_edges": 8,
            "node_type_counts": {"B": 1, "C": 1, "E": 1, "N": 2, "P": 1, "R": 1, "T": 1},
        },
        "variables": {
            "total_variables": len(names),
            "boundary_variables": boundary_count,
            "equipment_variables": len(names) - boundary_count,
        },
        "tokenizer": {},
    }
    (static_full / "graph_hyperparameters.json").write_text(
        json.dumps(hyper, indent=2), encoding="utf-8"
    )


def choose_tiny_variables(names: list[str]) -> tuple[list[str], list[int]]:
    desired = [
        "T_001:BC000",
        "T_001:BC001",
        "T_001:BC002",
        "T_001:BC003",
        "B_001_v000",
        "B_001_v001",
        "C_001_v000",
        "C_001_v001",
        "C_001_v002",
        "P_001_v000",
        "P_001_v001",
        "R_001_v000",
        "TE_001_v000",
        "N_001_v000",
        "N_002_v000",
    ]
    lookup = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in desired if name not in lookup]
    if missing:
        raise RuntimeError(f"Internal mock variable list is invalid: {missing}")
    return desired, [lookup[name] for name in desired]


def write_graph(static_dir: Path, root: Path) -> None:
    graph = defaultdict(set)
    edges = [
        ("T_001", "N_001"),
        ("N_001", "P_001"),
        ("P_001", "N_002"),
        ("N_002", "B_001"),
        ("B_001", "C_001"),
        ("C_001", "R_001"),
        ("R_001", "E_001"),
        ("E_001", "TE_001"),
    ]
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)

    graph_data = {
        "graph": dict(graph),
        "equipment_dir": str(root / "equipment_arguments"),
        "connected_nodes_file": str(root / "relative_PPT" / "connected_nodes.csv"),
        "target_equipment_types": {"T", "E", "C", "R", "B"},
    }
    with (static_dir / "pipeline_graph_cache.pkl").open("wb") as handle:
        pickle.dump(graph_data, handle)

    rows = [{"node": min(a, b), "connected_node": max(a, b)} for a, b in edges]
    pd.DataFrame(rows).to_csv(static_dir / "save_connect_all_nodes.csv", index=False)


def write_tiny_static(static_tiny: Path, root: Path, tiny_names: list[str], tiny_global: list[int]) -> None:
    write_mapping(static_tiny / "index_variable_mapping.csv", tiny_names, tiny_global)
    write_graph(static_tiny, root)

    mask = pd.DataFrame(
        {
            "variable_name": tiny_names,
            "predict": [0 if ":" in name else 1 for name in tiny_names],
        }
    )
    mask.to_csv(static_tiny / "prediction_mask.csv", index=False)

    boundary_count = sum(1 for name in tiny_names if ":" in name)
    hyper = {
        "graph": {
            "is_subgraph": True,
            "center_node": "P_001",
            "neighbor_count": len(tiny_names) - 1,
            "total_nodes": 8,
            "total_edges": 8,
            "node_type_counts": {"B": 1, "C": 1, "E": 1, "N": 2, "P": 1, "R": 1, "T": 1},
            "mock_data": True,
        },
        "variables": {
            "total_variables": len(tiny_names),
            "boundary_variables": boundary_count,
            "equipment_variables": len(tiny_names) - boundary_count,
        },
        "tokenizer": {
            "quantile_step": 0.1,
            "constant_freq_threshold": 0.995,
            "constant_variable_threshold": 0.995,
        },
    }
    (static_tiny / "graph_hyperparameters.json").write_text(
        json.dumps(hyper, indent=2), encoding="utf-8"
    )


def daily_times() -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    boundary = pd.date_range("2025-01-01 00:00:00", "2025-01-01 23:30:00", freq="30min")
    equipment = pd.date_range("2025-01-01 00:01:00", "2025-01-01 23:59:00", freq="1min")
    return boundary, equipment


def smooth_signal(length: int, sample_idx: int, scale: float, phase: float = 0.0) -> np.ndarray:
    x = np.linspace(0.0, 2.0 * np.pi, length, dtype=np.float32)
    return scale * (np.sin(x + phase) + 0.2 * np.cos(3.0 * x + sample_idx))


def write_sample(sample_dir: Path, sample_idx: int, tiny_names: list[str]) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    boundary_times, equipment_times = daily_times()

    boundary_cols = [f"T_001:BC{i:03d}" for i in range(BOUNDARY_DIMS)]
    boundary_data = {"TIME": boundary_times.strftime("%Y-%m-%d %H:%M:%S")}
    base = 4.0 + 0.05 * sample_idx
    for i, col in enumerate(boundary_cols):
        boundary_data[col] = base + smooth_signal(len(boundary_times), sample_idx, 0.05, i * 0.03) + i * 0.0005
    pd.DataFrame(boundary_data).to_csv(sample_dir / "Boundary.csv", index=False)

    by_type: dict[str, list[str]] = {key: [] for key in EQUIPMENT_ORDER}
    for name in tiny_names:
        if ":" in name:
            continue
        prefix = name.split("_", 1)[0]
        if prefix == "TE":
            by_type["T&E"].append(name)
        elif prefix in by_type:
            by_type[prefix].append(name)

    for equipment_type, columns in by_type.items():
        if not columns:
            continue
        data = {"TIME": equipment_times.strftime("%Y-%m-%d %H:%M:%S")}
        for i, col in enumerate(columns):
            trend = np.linspace(0, 0.1 + sample_idx * 0.02, len(equipment_times), dtype=np.float32)
            data[col] = 1.0 + i * 0.25 + trend + smooth_signal(len(equipment_times), sample_idx, 0.08, i)
        pd.DataFrame(data).to_csv(sample_dir / f"{equipment_type}.csv", index=False)


def write_samples(dataset_train: Path, dataset_test: Path, tiny_names: list[str]) -> None:
    sample_names = ["case_001", "case_002"]
    chinese_smoke_name = "第001个算例"
    for idx, name in enumerate(sample_names, start=1):
        write_sample(dataset_train / name, idx, tiny_names)
    # Some PipeFormer helper code looks for this exact sample name when building
    # fallback variable maps, so keep a tiny duplicate around for compatibility.
    write_sample(dataset_train / chinese_smoke_name, 1, tiny_names)
    write_sample(dataset_test / "case_101", 101, tiny_names)


def write_static_feature_placeholders(process_eq_argu: Path) -> None:
    pd.DataFrame(
        [[0.0] * 9],
        index=["P_001"],
        columns=[
            "pipe_length_km",
            "outer_diameter_mm",
            "wall_thickness_mm",
            "heat_transfer_coefficient",
            "wall_roughness_mm",
            "outlet_elevation_m",
            "inlet_elevation_m",
            "outlet_soil_temp_c",
            "inlet_soil_temp_c",
        ],
    ).to_csv(process_eq_argu / "pipe_features.csv")
    pd.DataFrame([[0.0]], index=["C_001"], columns=["pca_feature_1"]).to_csv(
        process_eq_argu / "compressor_features.csv"
    )
    pd.DataFrame(
        [
            {
                "num_pipes": 1,
                "num_compressors": 1,
                "pipe_feature_dim": 9,
                "compressor_feature_dim": 1,
                "pipe_feature_columns": "['pipe_length_km', 'outer_diameter_mm', 'wall_thickness_mm', 'heat_transfer_coefficient', 'wall_roughness_mm', 'outlet_elevation_m', 'inlet_elevation_m', 'outlet_soil_temp_c', 'inlet_soil_temp_c']",
            }
        ]
    ).to_csv(process_eq_argu / "metadata.csv", index=False)


def write_configs(project_root: Path, data_root: Path, static_tiny: Path, tiny_dim: int, boundary_dim: int) -> None:
    model_dir = project_root / "configs" / "models" / "decoder"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "model_name": "FluidDecoder",
        "model_version": "mock-smoke-test",
        "input_dim": tiny_dim,
        "output_dim": tiny_dim,
        "sequence_length": 3,
        "boundary_dims": boundary_dim,
        "equipment_dims": tiny_dim - boundary_dim,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 64,
        "attention_dropout": 0.1,
        "dropout_rate": 0.1,
        "tokenizer_vocab_size": 4096,
        "time_position_encoding": "learnable",
        "variable_position_encoding": "learnable",
        "max_time_positions": 10,
        "max_variable_positions": tiny_dim,
        "projection_hidden_dim": 32,
        "input_projection_type": "token_embedding",
        "use_topology_attention": True,
        "use_layer_norm": True,
        "activation": "gelu",
    }
    (model_dir / "mock_tiny_decoder.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )

    train_config = {
        "data_dir": str(data_root.relative_to(project_root)).replace("\\", "/"),
        "static_dir": str(static_tiny.relative_to(project_root)).replace("\\", "/"),
        "cache_dir": str((static_tiny / "cache").relative_to(project_root)).replace("\\", "/"),
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "sequence_length": 3,
        "max_sequences_per_sample": 16,
        "num_train_epochs": 1,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "model_config_path": "configs/models/decoder/mock_tiny_decoder.json",
        "output_dir": "./outputs/mock_tiny_decoder",
        "eval_steps": 5,
        "save_steps": 5,
        "save_total_limit": 1,
        "normalization_method": "standard",
        "device": "cpu",
        "dataloader_num_workers": 0,
        "mixed_precision": False,
        "use_swanlab": False,
        "debug_mode": True,
        "log_level": "info",
        "seed": 42,
    }
    (project_root / "configs" / "mock_tiny_decoder.json").write_text(
        json.dumps(train_config, indent=2), encoding="utf-8"
    )


def _equipment_name(var_name: str) -> str:
    if ":" in var_name:
        return var_name.split(":", 1)[0]
    parts = var_name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else var_name


def write_attention_indices(static_tiny: Path, tiny_names: list[str], max_neighbors: int = 8) -> None:
    """Write a compact, valid topology attention artifact without importing PipeFormer."""
    graph_neighbors = {
        "T_001": ["N_001"],
        "N_001": ["T_001", "P_001"],
        "P_001": ["N_001", "N_002"],
        "N_002": ["P_001", "B_001"],
        "B_001": ["N_002", "C_001"],
        "C_001": ["B_001", "R_001"],
        "R_001": ["C_001", "E_001"],
        "E_001": ["R_001", "TE_001"],
        "TE_001": ["E_001"],
    }
    by_equipment: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(tiny_names):
        by_equipment[_equipment_name(name)].append(idx)

    indices = np.zeros((len(tiny_names), max_neighbors), dtype=np.int32)
    variable_names = np.empty((len(tiny_names), max_neighbors), dtype=object)

    for idx, name in enumerate(tiny_names):
        equipment = _equipment_name(name)
        candidates: list[int] = [idx]
        candidates.extend(i for i in by_equipment.get(equipment, []) if i != idx)
        for neighbor_equipment in graph_neighbors.get(equipment, []):
            candidates.extend(by_equipment.get(neighbor_equipment, []))

        deduped: list[int] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        if not deduped:
            deduped = [idx]

        padded = (deduped + [idx] * max_neighbors)[:max_neighbors]
        indices[idx, :] = padded
        variable_names[idx, :] = [tiny_names[i] for i in padded]

    with (static_tiny / "attention_indices.pkl").open("wb") as handle:
        pickle.dump({"attention_indices": indices, "variable_names": variable_names}, handle)
    pd.DataFrame(indices).to_csv(static_tiny / "attention_indices.csv", index_label="variable_index")
    pd.DataFrame(variable_names).to_csv(static_tiny / "attention_variable_names.csv", index_label="variable_index")


def write_readme(data_root: Path, rel_root: str, rel_static: str) -> None:
    text = f"""# Mock Tiny PipeFormer Data

This folder contains synthetic data for PipeFormer smoke tests only. It is not
real pipeline telemetry and should not be used for scientific results.

Generated contents:

- `dataset/train` and `dataset/test`: tiny case folders with `Boundary.csv` and
  equipment CSVs.
- `static/mock_tiny`: active variable mapping, prediction mask, graph cache, and
  topology attention files.
- `static/full`: full 6,712-variable mapping used by `DataProcessor.combine_all_data`.
- `process_eq_argu`: placeholder static feature files.

Regenerate from the project root:

```powershell
python scripts/create_mock_pipeformer_data.py --force
```

Smoke-test commands:

```powershell
python build_cache.py --data-dir {rel_root} --static-dir {rel_static} --skip-tokens --force
python data/compute_tokenizer_stats.py --data_dir {rel_root} --static-dir {rel_static} --force
python build_cache.py --data-dir {rel_root} --static-dir {rel_static} --force
python data/compute_normalization_stats.py --static_dir {rel_static} --method standard --force
python train.py --config configs/mock_tiny_decoder.json
```
"""
    (data_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny mock data for PipeFormer smoke tests.")
    parser.add_argument(
        "--output-dir",
        default="data/mock_tiny",
        help="Output data root relative to the PipeFormer project root.",
    )
    parser.add_argument("--force", action="store_true", help="Delete and recreate the output directory.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_root = (project_root / args.output_dir).resolve()
    paths = make_dirs(data_root, args.force)

    names = full_variable_names()
    tiny_names, tiny_global = choose_tiny_variables(names)
    boundary_count = sum(1 for name in tiny_names if ":" in name)

    write_full_static_mapping(paths["static_full"], names)
    write_tiny_static(paths["static_tiny"], data_root, tiny_names, tiny_global)
    write_samples(paths["dataset_train"], paths["dataset_test"], tiny_names)
    write_static_feature_placeholders(paths["process_eq_argu"])
    write_configs(project_root, data_root, paths["static_tiny"], len(tiny_names), boundary_count)
    write_attention_indices(paths["static_tiny"], tiny_names)

    rel_root = data_root.relative_to(project_root).as_posix()
    rel_static = paths["static_tiny"].relative_to(project_root).as_posix()
    write_readme(data_root, rel_root, rel_static)

    print("Mock PipeFormer data created.")
    print(f"  data_dir: {rel_root}")
    print(f"  static_dir: {rel_static}")
    print(f"  active variables: {len(tiny_names)} ({boundary_count} boundary, {len(tiny_names) - boundary_count} equipment)")
    print("Next smoke-test commands:")
    print(f"  python build_cache.py --data-dir {rel_root} --static-dir {rel_static} --skip-tokens --force")
    print(f"  python data/compute_tokenizer_stats.py --data_dir {rel_root} --static-dir {rel_static} --force")
    print(f"  python build_cache.py --data-dir {rel_root} --static-dir {rel_static} --force")
    print(f"  python data/compute_normalization_stats.py --static_dir {rel_static} --method standard --force")
    print("  python train.py --config configs/mock_tiny_decoder.json")


if __name__ == "__main__":
    main()
