from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_VARIABLE_FIELDS = (
    "variable",
    "equipment_type",
    "physical_quantity",
    "role",
    "unit",
    "controllable",
    "lower_limit",
    "upper_limit",
)


def registry_path_for_mapping(mapping_csv: Path) -> Path:
    return Path(mapping_csv).resolve().parent / "variable_registry.json"


def validate_variable_registry(
    path: Path,
    required_variables: Iterable[str],
) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return _missing_registry_report(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return _invalid_registry_report(path, f"Could not read registry JSON: {exc}")
    if not isinstance(document, dict) or not isinstance(document.get("variables"), list):
        return _invalid_registry_report(path, "Registry must contain a variables array.")
    return _registry_report(path, document, required_variables)


def load_variable_registry(
    path: Path,
    required_variables: Iterable[str],
) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        report = _missing_registry_report(path)
        raise ValueError("Variable registry validation failed: " + "; ".join(report["errors"]))
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Variable registry validation failed: could not read {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("variables"), list):
        raise ValueError("Variable registry validation failed: registry must contain a variables array.")
    report = _registry_report(path, document, required_variables)
    if not report["supported"]:
        raise ValueError("Variable registry validation failed: " + "; ".join(report["errors"]))
    return document


def _registry_report(
    path: Path,
    document: Dict[str, Any],
    required_variables: Iterable[str],
) -> Dict[str, Any]:
    entries = [item for item in document["variables"] if isinstance(item, dict)]
    names = [str(item.get("variable") or "").strip() for item in entries]
    counts = Counter(name for name in names if name)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    registry_variables = set(counts)
    required = set(required_variables)
    missing_variables = sorted(required - registry_variables)
    incomplete_variables = {
        str(item.get("variable") or f"entry_{index}"): [
            field for field in REQUIRED_VARIABLE_FIELDS if field not in item
        ]
        for index, item in enumerate(entries)
        if any(field not in item for field in REQUIRED_VARIABLE_FIELDS)
    }
    errors: List[str] = []
    if len(entries) != len(document["variables"]):
        errors.append("Registry variables must all be JSON objects.")
    if duplicates:
        errors.append("Duplicate registry variables: " + ", ".join(duplicates))
    if missing_variables:
        errors.append("Mapped variables missing from registry: " + ", ".join(missing_variables))
    if incomplete_variables:
        errors.append(
            "Registry entries missing required fields: "
            + ", ".join(
                f"{name}({','.join(fields)})"
                for name, fields in sorted(incomplete_variables.items())
            )
        )
    return {
        "registry_json": path.name,
        "registry_schema_version": document.get("schema_version"),
        "variable_count": len(entries),
        "missing_variables": missing_variables,
        "extra_variables": sorted(registry_variables - required),
        "duplicate_variables": duplicates,
        "incomplete_variables": incomplete_variables,
        "errors": errors,
        "supported": not errors,
    }


def _missing_registry_report(path: Path) -> Dict[str, Any]:
    return _invalid_registry_report(path, f"Variable registry does not exist: {path}")


def _invalid_registry_report(path: Path, error: str) -> Dict[str, Any]:
    return {
        "registry_json": path.name,
        "registry_schema_version": None,
        "variable_count": 0,
        "missing_variables": [],
        "extra_variables": [],
        "duplicate_variables": [],
        "incomplete_variables": {},
        "errors": [error],
        "supported": False,
    }
