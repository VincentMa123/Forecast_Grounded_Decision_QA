from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def default_scenario_files() -> List[Path]:
    root = BACKEND_ROOT / "pipeclaw_data"
    return [
        root / "pipeclaw_dataset_v2.json",
        root / "Pipeline_Full_Life_Cycle_Test_Dataset-v4.json",
        root / "Pipeline_Full_Life_Cycle_Test_Dataset-v7.json",
    ]


def load_scenario_sources(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen_names = set()
    for path in paths:
        resolved = Path(path).resolve()
        source_name = resolved.stem
        if source_name in seen_names:
            raise ValueError(
                f"Duplicate dataset source name {source_name!r}; rename one source "
                "file to keep record ids unique."
            )
        seen_names.add(source_name)
        scenarios = json.loads(resolved.read_text(encoding="utf-8-sig"))
        if not isinstance(scenarios, list):
            raise TypeError(f"Scenario file must contain a JSON list: {resolved}")
        scenarios = [
            {**scenario, "dataset_source": source_name, "source_file": resolved.name}
            for scenario in scenarios
        ]
        sources.append(
            {
                "dataset_source": source_name,
                "source_file": resolved.name,
                "path": resolved,
                "scenarios": scenarios,
            }
        )
    return sources


def flatten_source_scenarios(sources: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        scenario
        for source in sources
        for scenario in source.get("scenarios") or []
    ]


def combined_preflight_sources(
    all_sources: Sequence[Dict[str, Any]],
    selected_sources: Sequence[Dict[str, Any]],
    existing_records: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[str]]:
    required_pairs = {
        (str(record.get("dataset_source") or ""), str(record.get("scenario_id") or ""))
        for record in existing_records
        if record.get("dataset_source") and record.get("scenario_id")
    }
    required_pairs.update(
        (
            str(source.get("dataset_source") or ""),
            str(scenario.get("scenario_id") or ""),
        )
        for source in selected_sources
        for scenario in source.get("scenarios") or []
    )
    available_pairs = {
        (str(source.get("dataset_source") or ""), str(scenario.get("scenario_id") or ""))
        for source in all_sources
        for scenario in source.get("scenarios") or []
    }
    selected = []
    for source in all_sources:
        source_name = str(source.get("dataset_source") or "")
        scenarios = [
            scenario
            for scenario in source.get("scenarios") or []
            if (source_name, str(scenario.get("scenario_id") or "")) in required_pairs
        ]
        if scenarios:
            selected.append({**source, "scenarios": scenarios})
    missing = sorted(
        f"{source}:{scenario}" for source, scenario in required_pairs - available_pairs
    )
    return selected, missing


__all__ = [
    "combined_preflight_sources",
    "default_scenario_files",
    "flatten_source_scenarios",
    "load_scenario_sources",
]
