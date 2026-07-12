from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


VARIABLE_RE = re.compile(r"\b[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?\b")


def scenario_texts(scenario: Dict[str, Any]) -> Iterable[str]:
    yield str(scenario.get("scenario_description") or "")
    for session in scenario.get("sessions") or []:
        for turn in session.get("dialogue") or []:
            yield str(turn.get("user_input") or "")


def scenario_variables(scenario: Dict[str, Any]) -> List[str]:
    return sorted({match.group(0) for text in scenario_texts(scenario) for match in VARIABLE_RE.finditer(text)})


def mapping_variables(mapping_csv: Path) -> List[str]:
    with Path(mapping_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "variable_name" not in (reader.fieldnames or []):
            raise ValueError(f"Mapping must contain variable_name: {mapping_csv}")
        return [str(row["variable_name"]).strip() for row in reader if str(row.get("variable_name") or "").strip()]


def validate_scenarios(scenarios: Sequence[Dict[str, Any]], mapping_csv: Path) -> Dict[str, Any]:
    supported = set(mapping_variables(mapping_csv))
    reports = []
    all_required = set()
    all_unsupported = set()
    for scenario in scenarios:
        required = scenario_variables(scenario)
        unsupported = sorted(set(required) - supported)
        all_required.update(required)
        all_unsupported.update(unsupported)
        reports.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "required_variables": required,
                "unsupported_variables": unsupported,
                "supported": not unsupported,
            }
        )
    return {
        "mapping_csv": Path(mapping_csv).name,
        "scenario_count": len(scenarios),
        "required_variable_count": len(all_required),
        "unsupported_variable_count": len(all_unsupported),
        "unsupported_variables": sorted(all_unsupported),
        "supported": not all_unsupported,
        "scenarios": reports,
    }


def require_supported_scenarios(scenarios: Sequence[Dict[str, Any]], mapping_csv: Path) -> Dict[str, Any]:
    report = validate_scenarios(scenarios, mapping_csv)
    if not report["supported"]:
        joined = ", ".join(report["unsupported_variables"])
        raise ValueError(f"Scenario preflight failed; variables absent from {mapping_csv}: {joined}")
    return report


def validate_scenario_sources(
    sources: Sequence[Dict[str, Any]], mapping_csv: Path
) -> Dict[str, Any]:
    source_reports = []
    all_unsupported = set()
    all_required = set()
    scenario_occurrences: Dict[str, List[Dict[str, Any]]] = {}
    session_occurrences: Dict[str, List[str]] = {}
    sample_occurrences: Dict[str, List[str]] = {}
    scenario_count = 0
    for source in sources:
        source_name = str(source["dataset_source"])
        scenarios = list(source.get("scenarios") or [])
        report = validate_scenarios(scenarios, mapping_csv)
        report["dataset_source"] = source_name
        source_reports.append(report)
        scenario_count += len(scenarios)
        all_required.update(
            variable
            for item in report["scenarios"]
            for variable in item["required_variables"]
        )
        all_unsupported.update(report["unsupported_variables"])
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
                for fallback_turn, turn in enumerate(session.get("dialogue") or [], start=1):
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
                "scenario_types": sorted({str(item.get("scenario_type") or "unknown") for item in occurrences}),
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
        "supported": not all_unsupported,
        "id_collisions": {
            "scenario_id": scenario_collisions,
            "session_id_count": sum(len(values) > 1 for values in session_occurrences.values()),
            "sample_id_count": sum(len(values) > 1 for values in sample_occurrences.values()),
            "namespacing_required": bool(scenario_collisions),
        },
        "sources": source_reports,
    }


def require_supported_scenario_sources(
    sources: Sequence[Dict[str, Any]], mapping_csv: Path
) -> Dict[str, Any]:
    report = validate_scenario_sources(sources, mapping_csv)
    if not report["supported"]:
        joined = ", ".join(report["unsupported_variables"])
        raise ValueError(f"Scenario preflight failed; variables absent from {mapping_csv}: {joined}")
    return report


def _content_fingerprint(value: Dict[str, Any]) -> str:
    import hashlib
    import json

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
