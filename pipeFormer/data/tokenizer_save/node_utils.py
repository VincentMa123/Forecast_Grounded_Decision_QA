"""Node helper functions used by the tokenizer."""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def split_node_and_attr(variable_name: str) -> Tuple[str, str]:
    if ":" in variable_name:
        node, attr = variable_name.split(":", 1)
        return node.strip(), attr.strip()

    parts = variable_name.split("_")
    if len(parts) >= 2 and parts[0].isalpha() and parts[1].isdigit():
        node = f"{parts[0]}_{parts[1]}"
        attr = "_".join(parts[2:]) if len(parts) > 2 else ""
    else:
        node = parts[0]
        attr = "_".join(parts[1:]) if len(parts) > 1 else ""
    return node.strip(), attr.strip()


def group_variables_by_node(variable_names: List[str]) -> Dict[str, List[str]]:
    node_map: Dict[str, List[str]] = {}
    for name in variable_names:
        node, _ = split_node_and_attr(name)
        node_map.setdefault(node, []).append(name)
    return node_map


def load_variable_names(data_dir: Path, expected_dims: int) -> List[str]:
    mapping_path = data_dir / "static" / "full" / "index_variable_mapping.csv"
    if not mapping_path.exists():
        mapping_path = data_dir / "index_variable_mapping.csv"
    if mapping_path.exists():
        try:
            df = pd.read_csv(mapping_path)
            names = df.get("variable_name", pd.Series(dtype=str)).astype(str).tolist()
            if len(names) == expected_dims:
                return names
        except Exception as exc:  # pragma: no cover - best effort diagnostics
            # logging handled by caller
            raise RuntimeError(f"Failed to read {mapping_path}: {exc}")
    return [f"feature_{idx:04d}" for idx in range(expected_dims)]


def load_node_connections(data_dir: Path) -> Dict[str, List[str]]:
    csv_path = data_dir / "save_connect_all_nodes.csv"
    connections: Dict[str, set] = {}
    if not csv_path.exists():
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        raise RuntimeError(f"Failed to read connectivity CSV {csv_path}: {exc}")

    for row in df.itertuples(index=False):
        node = str(getattr(row, "node", "")).strip()
        connected = str(getattr(row, "connected_node", "")).strip()
        if not node or not connected:
            continue
        connections.setdefault(node, set()).add(connected)
        connections.setdefault(connected, set()).add(node)

    return {node: sorted(list(neighbours)) for node, neighbours in connections.items()}


def get_variable_group(variable_index: int, boundary_dims: int, equipment_dims: int) -> str:
    if variable_index < 0:
        return "special"
    boundary_limit = max(boundary_dims, 0)
    total_limit = boundary_limit + max(equipment_dims, 0)
    if variable_index < boundary_limit:
        return "boundary"
    if variable_index < total_limit:
        return "equipment"
    return "unknown"


def compute_group_from_index(
    idx: Union[int, float],
    boundary_dims: int,
    equipment_dims: int,
) -> str:
    if isinstance(idx, float) and np.isnan(idx):
        return "unknown"
    try:
        int_idx = int(idx)
    except (TypeError, ValueError):
        return "unknown"
    if int_idx < 0:
        return "special"
    return get_variable_group(int_idx, boundary_dims, equipment_dims)


__all__ = [
    "compute_group_from_index",
    "get_variable_group",
    "group_variables_by_node",
    "load_node_connections",
    "load_variable_names",
    "split_node_and_attr",
]
