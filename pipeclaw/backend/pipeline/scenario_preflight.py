from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .registry.variable_registry import (
    registry_path_for_mapping,
    validate_variable_registry,
)


VARIABLE_RE = re.compile(r"\b[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?\b")
DATA_FILE_RE = re.compile(
    r"\b(?P<filename>\d{8}_(?P<kind>node|pipeline|consumer)\.csv)\b", re.IGNORECASE
)
TOPOLOGY_FOLLOWUP_RE = re.compile(
    r"相邻站点|相邻供气点|上游|下游|可达|最短路径|共享网关|反向追"
    r"|\b(?:adjacent|upstream|downstream|reachable|shortest path|shared gateway)\b",
    re.IGNORECASE,
)
SAME_DAY_TOPOLOGY_TARGETS = {
    "通州南分输站",
    "湘潭分输站",
    "阳曲压气站",
    "乌鲁木齐压气站",
}
DATA_FILE_SUBDIRECTORIES = {
    "node": "node_flow",
    "pipeline": "pipeline_flow",
    "consumer": "consumer_flow",
}
DEFAULT_PIPELINE_DATA_ROOT = Path(__file__).resolve().parents[1] / "pipeline_data"


def scenario_texts(scenario: Dict[str, Any]) -> Iterable[str]:
    yield str(scenario.get("scenario_description") or "")
    for session in scenario.get("sessions") or []:
        for turn in session.get("dialogue") or []:
            yield str(turn.get("user_input") or "")


def scenario_variables(scenario: Dict[str, Any]) -> List[str]:
    return sorted(
        {
            match.group(0)
            for text in scenario_texts(scenario)
            for match in VARIABLE_RE.finditer(text)
        }
    )


def mapping_variables(mapping_csv: Path) -> List[str]:
    with Path(mapping_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "variable_name" not in (reader.fieldnames or []):
            raise ValueError(f"Mapping must contain variable_name: {mapping_csv}")
        return [
            str(row["variable_name"]).strip()
            for row in reader
            if str(row.get("variable_name") or "").strip()
        ]


def scenario_data_files(scenario: Dict[str, Any]) -> List[str]:
    files = set()
    pending = [scenario]
    texts = []
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str):
            texts.append(value)
    for text in texts:
        for match in DATA_FILE_RE.finditer(text):
            kind = match.group("kind").casefold()
            files.add(
                f"pipeline_data/{DATA_FILE_SUBDIRECTORIES[kind]}/{match.group('filename')}"
            )
    return sorted(files)


def consumer_station_mappings(pipeline_data_root: Path) -> Dict[str, str]:
    path = Path(pipeline_data_root) / "consumer_station.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("供气点") or "").strip(): str(row.get("匹配站名") or "").strip()
            for row in csv.DictReader(handle)
            if str(row.get("供气点") or "").strip()
        }


def scenario_topology_requirements(
    scenario: Dict[str, Any],
    station_mappings: Dict[str, str],
    pipeline_data_root: Path,
) -> Dict[str, Any]:
    texts = list(scenario_texts(scenario))
    combined = "\n".join(texts)
    if not TOPOLOGY_FOLLOWUP_RE.search(combined):
        return {"required_data_files": [], "targets": [], "missing_target_mappings": []}
    consumer_files = sorted(
        {
            match.group("filename")
            for match in DATA_FILE_RE.finditer(combined)
            if match.group("kind").casefold() == "consumer"
        }
    )
    supply_points = set()
    for filename in consumer_files:
        path = (
            Path(pipeline_data_root) / DATA_FILE_SUBDIRECTORIES["consumer"] / filename
        )
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            supply_points.update(
                str(row.get("供气点") or "").strip()
                for row in csv.DictReader(handle)
                if str(row.get("供气点") or "").strip()
            )
    targets = sorted(point for point in supply_points if point in combined)
    missing_target_mappings = sorted(
        point for point in targets if not str(station_mappings.get(point) or "").strip()
    )
    topology_dates = {
        filename[:8]
        for filename in consumer_files
        if any(target in SAME_DAY_TOPOLOGY_TARGETS for target in targets)
    }
    required = sorted(
        {
            f"pipeline_data/{DATA_FILE_SUBDIRECTORIES[kind]}/{day}_{kind}.csv"
            for day in topology_dates
            for kind in ("node", "pipeline")
        }
    )
    return {
        "required_data_files": required,
        "targets": [
            {"requested": point, "canonical": station_mappings.get(point)}
            for point in targets
        ],
        "missing_target_mappings": missing_target_mappings,
    }


def validate_scenarios(
    scenarios: Sequence[Dict[str, Any]],
    mapping_csv: Path,
    supported_variables: Optional[Sequence[str]] = None,
    *,
    pipeline_data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    supported = set(supported_variables or mapping_variables(mapping_csv))
    data_root = Path(pipeline_data_root or DEFAULT_PIPELINE_DATA_ROOT).resolve()
    station_mappings = consumer_station_mappings(data_root)
    reports = []
    all_required = set()
    all_unsupported = set()
    all_required_data_files = set()
    all_missing_data_files = set()
    all_missing_target_mappings = set()
    for scenario in scenarios:
        required = scenario_variables(scenario)
        unsupported = sorted(set(required) - supported)
        topology = scenario_topology_requirements(scenario, station_mappings, data_root)
        required_data_files = sorted(
            set(scenario_data_files(scenario)) | set(topology["required_data_files"])
        )
        missing_data_files = sorted(
            logical_path
            for logical_path in required_data_files
            if not (
                data_root / Path(logical_path).relative_to("pipeline_data")
            ).is_file()
        )
        all_required.update(required)
        all_unsupported.update(unsupported)
        all_required_data_files.update(required_data_files)
        all_missing_data_files.update(missing_data_files)
        all_missing_target_mappings.update(topology["missing_target_mappings"])
        reports.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "required_variables": required,
                "unsupported_variables": unsupported,
                "required_data_files": required_data_files,
                "missing_data_files": missing_data_files,
                "topology_requirements": topology,
                "missing_target_mappings": topology["missing_target_mappings"],
                "supported": (
                    not unsupported
                    and not missing_data_files
                    and not topology["missing_target_mappings"]
                ),
            }
        )
    return {
        "mapping_csv": Path(mapping_csv).name,
        "scenario_count": len(scenarios),
        "required_variable_count": len(all_required),
        "unsupported_variable_count": len(all_unsupported),
        "unsupported_variables": sorted(all_unsupported),
        "required_data_file_count": len(all_required_data_files),
        "required_data_files": sorted(all_required_data_files),
        "missing_data_file_count": len(all_missing_data_files),
        "missing_data_files": sorted(all_missing_data_files),
        "missing_target_mapping_count": len(all_missing_target_mappings),
        "missing_target_mappings": sorted(all_missing_target_mappings),
        "supported": (
            not all_unsupported
            and not all_missing_data_files
            and not all_missing_target_mappings
        ),
        "scenarios": reports,
    }


def validate_scenario_sources(
    sources: Sequence[Dict[str, Any]],
    mapping_csv: Path,
    *,
    pipeline_data_root: Optional[Path] = None,
) -> Dict[str, Any]:
    mapped_variables = mapping_variables(mapping_csv)
    registry_report = validate_variable_registry(
        registry_path_for_mapping(mapping_csv),
        mapped_variables,
    )
    source_reports = []
    all_unsupported = set()
    all_required = set()
    all_required_data_files = set()
    all_missing_data_files = set()
    all_missing_target_mappings = set()
    scenario_occurrences: Dict[str, List[Dict[str, Any]]] = {}
    session_occurrences: Dict[str, List[str]] = {}
    sample_occurrences: Dict[str, List[str]] = {}
    scenario_count = 0
    for source in sources:
        source_name = str(source["dataset_source"])
        scenarios = list(source.get("scenarios") or [])
        report = validate_scenarios(
            scenarios,
            mapping_csv,
            mapped_variables,
            pipeline_data_root=pipeline_data_root,
        )
        report["dataset_source"] = source_name
        source_reports.append(report)
        scenario_count += len(scenarios)
        all_required.update(
            variable
            for item in report["scenarios"]
            for variable in item["required_variables"]
        )
        all_unsupported.update(report["unsupported_variables"])
        all_required_data_files.update(report["required_data_files"])
        all_missing_data_files.update(report["missing_data_files"])
        all_missing_target_mappings.update(report["missing_target_mappings"])
        for scenario in scenarios:
            scenario_id = str(scenario.get("scenario_id") or "")
            scenario_occurrences.setdefault(scenario_id, []).append(
                {
                    "dataset_source": source_name,
                    "scenario_type": scenario.get("scenario_type"),
                    "content_fingerprint": _content_fingerprint(scenario),
                }
            )
            for session in scenario.get("sessions") or []:
                session_id = str(session.get("session_id") or "")
                session_key = f"{scenario_id}::{session_id}"
                session_occurrences.setdefault(session_key, []).append(source_name)
                for fallback_turn, turn in enumerate(
                    session.get("dialogue") or [], start=1
                ):
                    turn_id = int(turn.get("turn_id") or fallback_turn)
                    sample_key = f"{scenario_id}::{session_id}::turn_{turn_id:03d}"
                    sample_occurrences.setdefault(sample_key, []).append(source_name)

    scenario_collisions = []
    for scenario_id, occurrences in sorted(scenario_occurrences.items()):
        if len(occurrences) < 2:
            continue
        fingerprints = {item["content_fingerprint"] for item in occurrences}
        scenario_collisions.append(
            {
                "scenario_id": scenario_id,
                "dataset_sources": [item["dataset_source"] for item in occurrences],
                "scenario_types": sorted(
                    {
                        str(item.get("scenario_type") or "unknown")
                        for item in occurrences
                    }
                ),
                "records_identical": len(fingerprints) == 1,
            }
        )
    return {
        "mapping_csv": Path(mapping_csv).name,
        "source_count": len(sources),
        "scenario_record_count": scenario_count,
        "canonical_scenario_count": len(scenario_occurrences),
        "required_variable_count": len(all_required),
        "unsupported_variable_count": len(all_unsupported),
        "unsupported_variables": sorted(all_unsupported),
        "required_data_file_count": len(all_required_data_files),
        "required_data_files": sorted(all_required_data_files),
        "missing_data_file_count": len(all_missing_data_files),
        "missing_data_files": sorted(all_missing_data_files),
        "supported": (
            not all_unsupported
            and not all_missing_data_files
            and not all_missing_target_mappings
            and registry_report["supported"]
        ),
        "missing_target_mapping_count": len(all_missing_target_mappings),
        "missing_target_mappings": sorted(all_missing_target_mappings),
        "variable_registry": registry_report,
        "id_collisions": {
            "scenario_id": scenario_collisions,
            "session_id_count": sum(
                len(values) > 1 for values in session_occurrences.values()
            ),
            "sample_id_count": sum(
                len(values) > 1 for values in sample_occurrences.values()
            ),
            "namespacing_required": bool(scenario_collisions),
        },
        "sources": source_reports,
    }


def _content_fingerprint(value: Dict[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
