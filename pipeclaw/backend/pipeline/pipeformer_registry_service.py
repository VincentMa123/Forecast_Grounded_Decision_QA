from __future__ import annotations

import csv
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

from .variable_registry import VariableRegistry


REGISTRY_RESULT_FIELDS = ("variable", "role", "controllable")


class PipeFormerRegistrySearchService:
    """Search and bound the PipeFormer registry result at the application boundary."""

    def __init__(self, repo_or_backend_root: Path) -> None:
        root = Path(repo_or_backend_root).resolve()
        self.repo_root = root if (root / "pipeFormer").is_dir() else root.parents[1]

    def _registry_path(self) -> Path:
        override = os.getenv("PIPEFORMER_VARIABLE_REGISTRY")
        if override:
            return Path(override).expanduser().resolve()
        return (
            self.repo_root
            / "pipeFormer"
            / "data"
            / "mock_lifecycle"
            / "static"
            / "mock_lifecycle"
            / "variable_registry.json"
        )

    def _topology_path(self) -> Path:
        return self._registry_path().parent / "save_connect_all_nodes.csv"

    def _topology_distances(self, targets: List[str]) -> Dict[str, int]:
        path = self._topology_path()
        if not targets or not path.is_file():
            return {}
        graph: Dict[str, set[str]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                left = str(row.get("node") or "").strip()
                right = str(row.get("connected_node") or "").strip()
                if not left or not right:
                    continue
                graph.setdefault(left, set()).add(right)
                graph.setdefault(right, set()).add(left)
        distances: Dict[str, int] = {}
        queue = deque()
        for target in targets:
            normalized = str(target).strip()
            if normalized in graph and normalized not in distances:
                distances[normalized] = 0
                queue.append(normalized)
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        return distances

    def search(self, **filters: Any) -> Dict[str, Any]:
        try:
            registry = VariableRegistry.read(self._registry_path())
            attention_targets = [str(value) for value in filters.pop("attention_targets", [])]
            offset = max(0, int(filters.pop("offset", 0)))
            limit = max(1, min(int(filters.pop("limit", 12)), 50))
            variables = registry.search(
                **filters,
                limit=max(1, len(registry.by_name)),
            )
            distances = self._topology_distances(attention_targets)
            if attention_targets:
                ranked = [
                    (
                        distances.get(str(item.get("equipment_id"))),
                        str(item.get("variable")),
                        item,
                    )
                    for item in variables
                ]
                ranked.sort(
                    key=lambda item: (
                        item[0] is None,
                        item[0] if item[0] is not None else 10**9,
                        item[1],
                    )
                )
                variables = [item for _, _, item in ranked]
            page = variables[offset : offset + limit]
            result = {
                "success": True,
                "matched_variable_count": len(page),
                "matched_total_count": len(variables),
                "offset": offset,
            }
            if offset + len(page) < len(variables):
                result["next_offset"] = offset + len(page)
            result["variables"] = [
                {
                    key: item[key]
                    for key in REGISTRY_RESULT_FIELDS
                    if key in item
                }
                for item in page
            ]
            return result
        except Exception as exc:
            return {"error": str(exc), "variables": []}
