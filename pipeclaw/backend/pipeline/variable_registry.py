from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
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


@dataclass(frozen=True)
class VariableRegistry:
    """Validated variable metadata loaded from one registry document."""

    path: Path
    document: Dict[str, Any]

    @classmethod
    def read(cls, path: Path) -> "VariableRegistry":
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Variable registry validation failed: variable registry does not exist: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Variable registry validation failed: could not read {path}: {exc}") from exc
        if not isinstance(document, dict) or not isinstance(document.get("variables"), list):
            raise ValueError("Variable registry validation failed: registry must contain a variables array.")
        return cls(path=path, document=document)

    def validate(self, required_variables: Iterable[str]) -> Dict[str, Any]:
        return _registry_report(self.path, self.document, required_variables)

    def require(self, required_variables: Iterable[str]) -> Dict[str, Any]:
        report = self.validate(required_variables)
        if not report["supported"]:
            raise ValueError("Variable registry validation failed: " + "; ".join(report["errors"]))
        return self.document

    @property
    def by_name(self) -> Dict[str, Dict[str, Any]]:
        return {
            str(item["variable"]): item
            for item in self.document.get("variables") or []
            if isinstance(item, dict) and item.get("variable")
        }

    def resolve_name(self, requested: str) -> Dict[str, Any]:
        """Resolve one requested name without guessing equipment identity."""
        raw = str(requested)
        stripped = raw.strip()
        if not stripped:
            raise ValueError("Variable name cannot be empty.")

        names = list(self.by_name)
        if stripped in self.by_name:
            method = "exact" if raw == stripped else "whitespace"
            return _variable_resolution(raw, stripped, method)

        case_matches = [name for name in names if name.casefold() == stripped.casefold()]
        if case_matches:
            return _unique_resolution(raw, case_matches, "case_insensitive")

        alias_matches = [
            name
            for name, item in self.by_name.items()
            for alias in item.get("aliases") or []
            if str(alias).strip().casefold() == stripped.casefold()
        ]
        if alias_matches:
            return _unique_resolution(raw, alias_matches, "declared_alias")

        terminal_variant = stripped[:-1] if stripped.endswith("_") else stripped + "_"
        terminal_matches = [
            name for name in names if name.casefold() == terminal_variant.casefold()
        ]
        if terminal_matches:
            return _unique_resolution(raw, terminal_matches, "terminal_underscore_alias")

        raise ValueError(f"Variable {raw!r} is not present in registry {self.path}.")

    def require_controllable_inputs(self, variables: Iterable[str]) -> List[Dict[str, Any]]:
        """Resolve and validate boundary adjustments against registry semantics."""
        validated = []
        for requested in variables:
            resolution = self.resolve_name(str(requested))
            item = self.by_name[resolution["resolved_variable"]]
            if item.get("role") != "input" or item.get("controllable") is not True:
                raise ValueError(
                    f"Variable {requested!r} is not a controllable input in registry {self.path}."
                )
            validated.append(item)
        return validated

    def search(
        self,
        *,
        query: str = "",
        role: str | None = None,
        controllable: bool | None = None,
        equipment_ids: Iterable[str] = (),
        equipment_types: Iterable[str] = (),
        physical_quantities: Iterable[str] = (),
        offset: int = 0,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        """Return a deterministic compact registry projection for tool use."""
        page_offset = max(0, int(offset))
        page_limit = max(1, int(limit))
        query_terms = [term for term in str(query).casefold().split() if term]
        equipment_id_filter = {str(value).casefold() for value in equipment_ids}
        equipment_type_filter = {str(value).casefold() for value in equipment_types}
        quantity_filter = {str(value).casefold() for value in physical_quantities}
        ranked = []
        for item in self.by_name.values():
            if role is not None and str(item.get("role")).casefold() != role.casefold():
                continue
            if controllable is not None and item.get("controllable") is not controllable:
                continue
            if equipment_id_filter and str(item.get("equipment_id", "")).casefold() not in equipment_id_filter:
                continue
            if equipment_type_filter and str(item.get("equipment_type", "")).casefold() not in equipment_type_filter:
                continue
            if quantity_filter and str(item.get("physical_quantity", "")).casefold() not in quantity_filter:
                continue
            effect_text = " ".join(
                str(effect.get("physical_quantity") or "")
                for effect in item.get("effect_targets") or []
                if isinstance(effect, dict)
            )
            search_text = " ".join(
                [
                    str(item.get("variable") or ""),
                    str(item.get("equipment_id") or ""),
                    str(item.get("equipment_type") or ""),
                    str(item.get("physical_quantity") or ""),
                    " ".join(str(alias) for alias in item.get("aliases") or []),
                    effect_text,
                ]
            ).casefold()
            variable = str(item.get("variable") or "").casefold()
            canonical_id_in_query = bool(variable and variable in str(query).casefold())
            if query_terms and not canonical_id_in_query and not all(term in search_text for term in query_terms):
                continue
            exact_score = 0 if canonical_id_in_query else 1
            ranked.append((exact_score, str(item.get("variable")), item))
        ranked.sort(key=lambda value: (value[0], value[1]))
        page = ranked[page_offset : page_offset + page_limit]
        return [_compact_registry_entry(item) for _, _, item in page]


def _variable_resolution(requested: str, resolved: str, method: str) -> Dict[str, Any]:
    return {
        "requested_variable": requested,
        "resolved_variable": resolved,
        "normalization_applied": requested != resolved,
        "method": method,
    }


def _compact_registry_entry(item: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "variable",
        "equipment_id",
        "equipment_type",
        "physical_quantity",
        "role",
        "unit",
        "controllable",
        "lower_limit",
        "upper_limit",
        "effect_targets",
    )
    return {key: copy.deepcopy(item[key]) for key in keys if key in item}


def _unique_resolution(requested: str, matches: Iterable[str], method: str) -> Dict[str, Any]:
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise ValueError(
            f"Variable {requested!r} is ambiguous in the registry: {', '.join(unique)}"
        )
    return _variable_resolution(requested, unique[0], method)


def normalize_task_variables(
    parsed_task: Dict[str, Any],
    registry: VariableRegistry,
) -> Dict[str, Any]:
    """Normalize explicit task variable references using one registry."""
    normalized = copy.deepcopy(parsed_task)
    resolutions: List[Dict[str, Any]] = []
    seen = set()

    def resolve(value: Any) -> str:
        resolution = registry.resolve_name(str(value))
        identity = (
            resolution["requested_variable"],
            resolution["resolved_variable"],
            resolution["method"],
        )
        if resolution["normalization_applied"] and identity not in seen:
            seen.add(identity)
            resolutions.append(resolution)
        return str(resolution["resolved_variable"])

    if normalized.get("disturbance_variable"):
        normalized["disturbance_variable"] = resolve(normalized["disturbance_variable"])

    boundary = dict(normalized.get("boundary_conditions") or {})
    if boundary.get("disturbance_variable"):
        boundary["disturbance_variable"] = resolve(boundary["disturbance_variable"])
    for field in ("setpoints", "percentage_changes"):
        values = boundary.get(field)
        if not isinstance(values, dict):
            continue
        resolved_values: Dict[str, Any] = {}
        for requested_name, value in values.items():
            resolved_name = resolve(requested_name)
            if resolved_name in resolved_values and resolved_values[resolved_name] != value:
                raise ValueError(
                    f"Multiple boundary values resolve to {resolved_name!r}."
                )
            resolved_values[resolved_name] = value
        boundary[field] = resolved_values
    normalized["boundary_conditions"] = boundary
    normalized["variable_normalizations"] = resolutions
    return normalized


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
        registry = VariableRegistry.read(path)
    except ValueError as exc:
        return _invalid_registry_report(path, f"Could not read registry JSON: {exc}")
    return registry.validate(required_variables)


def load_variable_registry(
    path: Path,
    required_variables: Iterable[str],
) -> Dict[str, Any]:
    path = Path(path)
    return VariableRegistry.read(path).require(required_variables)


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
