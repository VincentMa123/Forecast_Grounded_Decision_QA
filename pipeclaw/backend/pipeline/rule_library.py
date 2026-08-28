from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .schemas import ConstraintSpec


RULE_LIBRARY_ROOT = Path(__file__).resolve().parent / "constraint_library"
GENERIC_EVALUATORS = {
    "predicted_range",
    "max_abs_prediction",
    "mean_abs_delta_vs_observed",
    "max_abs_step_change",
    "max_step_decline",
    "max_decline_from_start",
    "boundary_disturbance_percent",
}


_documents: Dict[str, Dict[str, Any]] = {}


def _load_document(filename: str) -> Dict[str, Any]:
    if filename not in _documents:
        path = RULE_LIBRARY_ROOT / filename
        if not path.is_file():
            raise FileNotFoundError(f"Constraint rule file not found: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"Constraint rule file must contain a JSON object: {path}")
        _documents[filename] = document
    return _documents[filename]


def load_pipeline_constraints() -> Dict[str, Any]:
    document = _load_document("pipeline_constraints.json")
    required = {"library_name", "rule_files", "category_order"}
    missing = sorted(required - set(document))
    if missing:
        raise ValueError(f"pipeline_constraints.json is missing required fields: {missing}")
    return document


def load_rule_document(category: str) -> Dict[str, Any]:
    manifest = load_pipeline_constraints()
    filename = manifest["rule_files"].get(category)
    if not filename:
        raise KeyError(f"No rule file is configured for category {category!r}")
    document = _load_document(str(filename))
    if document.get("category") == category:
        return document
    nested_rules = document.get(f"{category}_rules")
    if isinstance(nested_rules, list):
        return {"category": category, "rules": nested_rules}
    raise ValueError(f"{filename} does not define rules for category {category!r}")


def load_constraint_specs(category: str) -> Tuple[ConstraintSpec, ...]:
    rules = load_rule_document(category).get("rules", [])
    specs = []
    seen = set()
    for rule in rules:
        rule_id = str(rule.get("rule_id") or "").strip()
        if not rule_id or rule_id in seen:
            raise ValueError(
                f"Rule ids must be non-empty and unique in category {category!r}: {rule_id!r}"
            )
        seen.add(rule_id)
        evaluator = str(rule.get("evaluator") or "").strip()
        if evaluator not in GENERIC_EVALUATORS:
            continue
        selector = dict(rule.get("selector") or {})
        limits = dict(rule.get("limits") or {})
        flags = dict(rule.get("flags") or {})
        specs.append(
            ConstraintSpec(
                name=rule_id,
                category=category,
                description=str(rule.get("description") or ""),
                priority=int(rule.get("priority", 999)),
                metric=evaluator,
                physical_quantities=tuple(selector.get("physical_quantities") or ()),
                equipment_types=tuple(selector.get("equipment_types") or ()),
                roles=tuple(selector.get("roles") or ()),
                use_registry_limits=bool(rule.get("use_registry_limits", False)),
                warning_low=limits.get("warning_low"),
                warning_high=limits.get("warning_high"),
                fail_low=limits.get("fail_low"),
                fail_high=limits.get("fail_high"),
                warning_threshold=limits.get("warning_threshold"),
                fail_threshold=limits.get("fail_threshold"),
                pass_flag=flags.get("pass"),
                warning_flag=flags.get("warning"),
                fail_flag=flags.get("fail"),
            )
        )
    return tuple(specs)


def load_rule_definition(category: str, rule_id: str) -> Dict[str, Any]:
    for rule in load_rule_document(category).get("rules", []):
        if rule.get("rule_id") == rule_id:
            return dict(rule)
    raise KeyError(f"Rule {rule_id!r} is not defined for category {category!r}")
