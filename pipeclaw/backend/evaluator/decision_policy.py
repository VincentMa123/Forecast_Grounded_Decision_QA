from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MetricDefinition:
    metric: str
    path: Tuple[str, ...]
    direction: str
    label_zh: str
    label_en: str
    unit_path: Tuple[str, ...] = ()


METRIC_CATALOG: Dict[str, MetricDefinition] = {
    "warning.count": MetricDefinition(
        "warning.count",
        ("warning_count",),
        "minimize",
        "告警数",
        "warnings",
    ),
    "risk.rank": MetricDefinition(
        "risk.rank",
        ("risk_rank",),
        "minimize",
        "风险等级",
        "risk rank",
    ),
    "compressor.maximum_load": MetricDefinition(
        "compressor.maximum_load",
        ("compressor_metrics", "maximum_load", "value"),
        "minimize",
        "压缩机最大负荷",
        "maximum compressor load",
    ),
    "pressure.minimum_operating_window_margin": MetricDefinition(
        "pressure.minimum_operating_window_margin",
        ("pressure_metrics", "minimum_operating_window_margin", "value"),
        "maximize",
        "最小压力窗裕度",
        "minimum pressure-window margin",
    ),
    "flow.max_abs_supply_demand_gap": MetricDefinition(
        "flow.max_abs_supply_demand_gap",
        ("flow_metrics", "supply_demand_balance", "value"),
        "minimize",
        "最大供需差",
        "maximum absolute supply-demand gap",
    ),
    "flow.maximum_segment_flow_change": MetricDefinition(
        "flow.maximum_segment_flow_change",
        ("flow_metrics", "maximum_segment_flow_change", "value"),
        "minimize",
        "最大管段流量变化",
        "maximum segment-flow change",
    ),
    "linepack.maximum_decline_from_start": MetricDefinition(
        "linepack.maximum_decline_from_start",
        ("linepack_metrics", "maximum_decline_from_start", "value"),
        "minimize",
        "最大管存下降",
        "maximum linepack decline",
    ),
    "linepack.maximum_continuous_decline_minutes": MetricDefinition(
        "linepack.maximum_continuous_decline_minutes",
        ("linepack_metrics", "maximum_continuous_decline_minutes"),
        "minimize",
        "管存最长连续下降分钟",
        "maximum continuous linepack-decline minutes",
    ),
    "linepack.insufficient_recovery_count": MetricDefinition(
        "linepack.insufficient_recovery_count",
        ("linepack_metrics", "insufficient_recovery_count"),
        "minimize",
        "管存恢复不足次数",
        "insufficient linepack-recovery count",
    ),
    "energy.delta_vs_baseline": MetricDefinition(
        "energy.delta_vs_baseline",
        ("energy_metrics", "delta_vs_baseline"),
        "minimize",
        "相对基线能耗变化",
        "energy delta versus baseline",
        ("energy_metrics", "unit"),
    ),
    "energy.total": MetricDefinition(
        "energy.total",
        ("energy_metrics", "total"),
        "minimize",
        "总能耗",
        "total energy",
        ("energy_metrics", "unit"),
    ),
}

DEFAULT_DECISION_POLICY: Dict[str, Any] = {
    "hard_constraints": ["no_constraint_failure"],
    "objectives": [
        {"metric": "warning.count", "direction": "minimize", "tolerance": 0.0},
        {"metric": "risk.rank", "direction": "minimize", "tolerance": 0.0},
        {
            "metric": "energy.delta_vs_baseline",
            "direction": "minimize",
            "tolerance": 0.0,
        },
    ],
}

SUPPORTED_HARD_CONSTRAINTS = {"no_constraint_failure"}
MAX_OBJECTIVES = 5
RISK_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _nested(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _number(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_decision_policy(
    raw_policy: Optional[Mapping[str, Any]],
) -> tuple[Dict[str, Any], List[str]]:
    source = (
        str(dict(raw_policy or {}).get("source") or "structured")
        if raw_policy is not None
        else "default"
    )
    source_policy = dict(raw_policy or DEFAULT_DECISION_POLICY)
    errors: List[str] = []
    raw_constraints = list(source_policy.get("hard_constraints") or [])
    constraints = [str(value) for value in raw_constraints]
    unsupported_constraints = [
        value for value in constraints if value not in SUPPORTED_HARD_CONSTRAINTS
    ]
    errors.extend(
        f"unsupported_hard_constraint:{value}" for value in unsupported_constraints
    )

    raw_objectives = list(source_policy.get("objectives") or [])
    if not raw_objectives:
        errors.append("decision_objectives_missing")
    if len(raw_objectives) > MAX_OBJECTIVES:
        errors.append(f"too_many_decision_objectives:{len(raw_objectives)}")

    objectives: List[Dict[str, Any]] = []
    seen = set()
    for raw in raw_objectives[:MAX_OBJECTIVES]:
        item = dict(raw or {})
        metric = str(item.get("metric") or "")
        definition = METRIC_CATALOG.get(metric)
        if definition is None:
            errors.append(f"unsupported_decision_metric:{metric or 'missing'}")
            continue
        if metric in seen:
            errors.append(f"duplicate_decision_metric:{metric}")
            continue
        seen.add(metric)
        direction = str(item.get("direction") or definition.direction)
        if direction != definition.direction:
            errors.append(
                f"invalid_metric_direction:{metric}:{direction}:expected_{definition.direction}"
            )
            continue
        tolerance = _number(item.get("tolerance"))
        if tolerance is None:
            tolerance = 0.0
        if tolerance < 0:
            errors.append(f"negative_metric_tolerance:{metric}")
            continue
        normalized = {
            "metric": metric,
            "direction": direction,
            "tolerance": tolerance,
            "label_zh": definition.label_zh,
            "label_en": definition.label_en,
        }
        if item.get("source_excerpt"):
            normalized["source_excerpt"] = str(item["source_excerpt"]).strip()
        if item.get("proxy_for"):
            normalized["proxy_for"] = str(item["proxy_for"])
        objectives.append(normalized)

    normalized_policy = {
            "source": source,
            "hard_constraints": constraints,
            "objectives": objectives,
        }
    if source_policy.get("source_excerpt"):
        normalized_policy["source_excerpt"] = str(source_policy["source_excerpt"])
    return normalized_policy, list(dict.fromkeys(errors))


def metric_evidence(
    candidate: Mapping[str, Any],
    metric: str,
) -> Optional[Dict[str, Any]]:
    definition = METRIC_CATALOG.get(metric)
    if definition is None:
        return None
    if metric == "risk.rank":
        raw_value = RISK_RANK.get(str(candidate.get("risk_level") or "").casefold())
    else:
        raw_value = _nested(candidate, definition.path)
    value = _number(raw_value)
    if value is None:
        return None

    result: Dict[str, Any] = {"value": value}
    metric_container = _nested(candidate, definition.path[:-1])
    if isinstance(metric_container, Mapping):
        if metric_container.get("variable"):
            result["variable"] = str(metric_container["variable"])
        if metric_container.get("metric"):
            result["source_metric"] = str(metric_container["metric"])
        if metric_container.get("status"):
            result["status"] = str(metric_container["status"])
    unit = _nested(candidate, definition.unit_path) if definition.unit_path else None
    if unit:
        result["unit"] = str(unit)
    return result


def collect_objective_evidence(
    candidates: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> tuple[Dict[str, Dict[str, Dict[str, Any]]], List[str]]:
    evidence: Dict[str, Dict[str, Dict[str, Any]]] = {}
    missing: List[str] = []
    objectives = list(policy.get("objectives") or [])
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        candidate_evidence: Dict[str, Dict[str, Any]] = {}
        for objective in objectives:
            metric = str(dict(objective).get("metric") or "")
            value = metric_evidence(candidate, metric)
            if value is None:
                missing.append(f"{candidate_id}:{metric}")
            else:
                candidate_evidence[metric] = value
        evidence[candidate_id] = candidate_evidence
    return evidence, missing


def rank_candidate_groups(
    candidate_ids: Iterable[str],
    policy: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> List[List[str]]:
    objectives = [dict(item) for item in policy.get("objectives") or []]
    groups = [sorted((str(value) for value in candidate_ids), key=str.casefold)]
    for objective in objectives:
        metric = str(objective["metric"])
        tolerance = float(objective.get("tolerance") or 0.0)
        descending = objective["direction"] == "maximize"
        refined: List[List[str]] = []
        for group in groups:
            ordered = sorted(
                group,
                key=lambda candidate_id: (
                    -float(evidence[candidate_id][metric]["value"])
                    if descending
                    else float(evidence[candidate_id][metric]["value"]),
                    candidate_id.casefold(),
                ),
            )
            buckets: List[List[str]] = []
            anchor: Optional[float] = None
            for candidate_id in ordered:
                value = float(evidence[candidate_id][metric]["value"])
                if anchor is None or abs(value - anchor) > tolerance:
                    buckets.append([candidate_id])
                    anchor = value
                else:
                    buckets[-1].append(candidate_id)
            refined.extend(buckets)
        groups = refined
    return [
        sorted(group, key=str.casefold)
        for group in groups
    ]


def rank_candidate_ids(
    candidate_ids: Iterable[str],
    policy: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> List[str]:
    return [
        candidate_id
        for group in rank_candidate_groups(candidate_ids, policy, evidence)
        for candidate_id in group
    ]
