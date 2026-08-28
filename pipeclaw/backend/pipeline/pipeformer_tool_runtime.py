from __future__ import annotations

import copy
import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from .condition_parser import (
    CATEGORY_ATTENTION_TARGETS,
    CATEGORY_OUTPUT_STATE_VARIABLES,
    DEFAULT_CONSTRAINT_VERIFICATION_TYPES,
    PIPEFORMER_TASK_SCHEMA_VERSION,
    parse_condition,
    targets_for_checks,
)
from .constraints.common import variables_matching
from .engineering_constraints import run_engineering_constraint_checks
from .evidence_extractor import summarize_variables, top_variables
from .forecast_result import (
    ForecastResult,
    compact_forecast_window,
    source_name,
)
from .pipeformer_inference import (
    PipeFormerInferenceConfig,
    PipeFormerInferenceEngine,
    resolve_pipeformer_environment,
)


REGISTRY_GROUP_RULES = {
    "nodes": {"equipment_types": {"node"}, "roles": {"output"}},
    "segments": {
        "equipment_types": {"pipeline_segment", "ball_valve"},
        "roles": {"output"},
    },
    "linepack": {"physical_quantities": {"linepack"}, "roles": {"output"}},
    "compressors": {
        "equipment_types": {"compressor", "compressor_power"},
        "roles": {"output"},
    },
    "pressure": {"physical_quantities": {"pressure"}, "roles": {"output"}},
    "flow": {"physical_quantities": {"flow"}, "roles": {"output"}},
    "compressor_load": {
        "physical_quantities": {"compressor_load"},
        "roles": {"output"},
    },
    "compressor": {
        "equipment_types": {"compressor", "compressor_power"},
        "roles": {"output"},
    },
    "compression_ratio": {
        "physical_quantities": {"compression_ratio"},
        "roles": {"output"},
    },
    "compressor_speed": {
        "physical_quantities": {"rotational_speed"},
        "roles": {"output"},
    },
    "compressor_power": {"physical_quantities": {"power"}, "roles": {"output"}},
    "power": {"physical_quantities": {"power"}, "roles": {"output"}},
    "energy": {"physical_quantities": {"power"}, "roles": {"output"}},
    "energy_consumption": {"physical_quantities": {"power"}, "roles": {"output"}},
    "energy_cost": {"physical_quantities": {"power"}, "roles": {"output"}},
    "operating_cost": {"physical_quantities": {"power"}, "roles": {"output"}},
    "valves": {"equipment_types": {"ball_valve"}},
    "pressure_regulators": {"equipment_types": {"pressure_regulator"}},
    "boundary_controls": {"roles": {"input"}, "controllable": True},
    "valve_opening": {"physical_quantities": {"valve_opening"}, "roles": {"output"}},
    "regulator_range": {
        "physical_quantities": {"regulator_range"},
        "roles": {"output"},
    },
    "boundary_control_adjustment": {"roles": {"input"}, "controllable": True},
    "dispatch_priority_audit": {"roles": {"output"}},
}
logger = logging.getLogger(__name__)


class BaselineForecastCache:
    """Small thread-safe LRU cache for unchanged forecast contexts."""

    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = max(1, int(max_entries))
        self._values: OrderedDict[tuple[Any, ...], Dict[str, Any]] = OrderedDict()
        self._lock = Lock()

    def get_or_compute(self, key: tuple[Any, ...], factory) -> Dict[str, Any]:
        with self._lock:
            if key in self._values:
                value = self._values.pop(key)
                self._values[key] = value
                return copy.deepcopy(value)

        value = factory()

        with self._lock:
            if key in self._values:
                cached = self._values.pop(key)
                self._values[key] = cached
                return copy.deepcopy(cached)
            self._values[key] = copy.deepcopy(value)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
            return copy.deepcopy(value)


def _integrated_energy_metrics(
    summaries: Dict[str, Dict[str, Any]],
    registry_entries: List[Dict[str, Any]],
    time_step_minutes: float,
    baseline_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    power_entries = [
        item
        for item in registry_entries
        if item.get("role") == "output" and item.get("physical_quantity") == "power"
    ]
    units = {str(item.get("unit") or "") for item in power_entries}
    energy_unit = (
        {
            "p.u.": "p.u.-hour",
            "MW": "MWh",
            "kW": "kWh",
        }.get(next(iter(units), ""))
        if len(units) == 1
        else None
    )
    usable = [
        str(item["variable"])
        for item in power_entries
        if (summaries.get(str(item["variable"])) or {}).get("predicted_values")
    ]
    if not energy_unit or len(usable) != len(power_entries) or not usable:
        return {
            "energy_consumption": None,
            "energy_consumption_delta": None,
            "energy_unit": energy_unit,
            "energy_variable_count": len(usable),
            "energy_evaluation_status": "not_evaluated",
        }

    factor = float(time_step_minutes) / 60.0
    total = (
        sum(
            sum(float(value) for value in summaries[variable]["predicted_values"])
            for variable in usable
        )
        * factor
    )
    baseline_total = None
    if baseline_summaries is not None and all(
        (baseline_summaries.get(variable) or {}).get("predicted_values")
        for variable in usable
    ):
        baseline_total = (
            sum(
                sum(
                    float(value)
                    for value in baseline_summaries[variable]["predicted_values"]
                )
                for variable in usable
            )
            * factor
        )
    return {
        "energy_consumption": round(total, 6),
        "energy_consumption_delta": (
            round(total - baseline_total, 6) if baseline_total is not None else None
        ),
        "energy_unit": energy_unit,
        "energy_variable_count": len(usable),
        "energy_evaluation_status": "evaluated",
    }


def _forecast_window_summary(forecast_context: Dict[str, Any]) -> Dict[str, Any]:
    real_rows = forecast_context.get("real_rows") or []
    predict_rows = forecast_context.get("predict_rows") or []
    labels = list(forecast_context.get("forecast_time_labels") or [])
    if not labels:
        labels = [
            str(getattr(row, "label", row))
            .removesuffix("_real")
            .removesuffix("_predict")
            for row in predict_rows or real_rows
        ]
    return compact_forecast_window(
        {
            "forecast_time_labels": labels,
            "time_step_minutes": forecast_context.get("time_step_minutes"),
            "real_rows": real_rows,
            "predict_rows": predict_rows,
        }
    )


def _clean_checks(requested_categories: Optional[List[str]]) -> List[str]:
    allowed = set(DEFAULT_CONSTRAINT_VERIFICATION_TYPES)
    checks = []
    for item in requested_categories or []:
        check = str(item).strip()
        if check and check in allowed and check not in checks:
            checks.append(check)
    return checks or DEFAULT_CONSTRAINT_VERIFICATION_TYPES.copy()


def _resolve_requested_variables(
    requested: List[str],
    variable_names: List[str],
    registry_entries: List[Dict[str, Any]],
) -> tuple[List[str], List[str]]:
    if not registry_entries:
        raise ValueError(
            "Variable registry metadata is required to resolve PipeFormer targets."
        )
    registry = {
        str(item.get("variable")): item
        for item in registry_entries
        if isinstance(item, dict) and item.get("variable")
    }
    resolved = []
    unresolved = []
    for raw in requested:
        target = str(raw).strip()
        matches: List[str] = []
        if target in variable_names:
            matches = [target]
        elif target in REGISTRY_GROUP_RULES:
            rule = REGISTRY_GROUP_RULES[target]
            matches = variables_matching(
                variable_names,
                registry=registry,
                physical_quantities=tuple(rule.get("physical_quantities") or ()),
                equipment_types=tuple(rule.get("equipment_types") or ()),
                roles=tuple(rule.get("roles") or ()),
                controllable=rule.get("controllable"),
            )
        else:
            matches = [
                name
                for name in variable_names
                if name.startswith(f"{target}_") or name.startswith(f"{target}:")
            ]
        if not matches:
            unresolved.append(target)
        for name in matches:
            if name not in resolved:
                resolved.append(name)
    return resolved, unresolved


def _normalize_vocabulary_provenance(
    values: Optional[List[Dict[str, Any]]],
    requested_targets: List[str],
) -> List[Dict[str, Any]]:
    normalized = []
    available_targets = set(requested_targets)
    for raw in values or []:
        if not isinstance(raw, dict):
            raise ValueError("Each vocabulary normalization must be an object.")
        requested_term = str(raw.get("requested_term") or "").strip()
        canonical_variables = list(
            dict.fromkeys(
                str(value).strip()
                for value in raw.get("canonical_variables") or []
                if str(value).strip()
            )
        )
        source = str(raw.get("normalization_source") or "").strip()
        if not requested_term or not canonical_variables:
            raise ValueError(
                "Vocabulary normalization requires requested_term and canonical_variables."
            )
        if source != "registry_search":
            raise ValueError(
                "Vocabulary normalization must use normalization_source='registry_search'."
            )
        missing = [
            value for value in canonical_variables if value not in available_targets
        ]
        if missing:
            raise ValueError(
                "Registry-normalized variables must also appear in attention_targets or "
                f"output_state_variables: {missing}"
            )
        normalized.append(
            {
                "requested_term": requested_term,
                "canonical_variables": canonical_variables,
                "normalization_source": source,
            }
        )
    return normalized


def _resolve_task_vocabulary(
    parsed_task: Dict[str, Any],
    variable_names: List[str],
    registry_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    resolved_attention, unresolved_attention = _resolve_requested_variables(
        parsed_task.get("attention_targets") or [],
        variable_names,
        registry_entries,
    )
    resolved_outputs, unresolved_outputs = _resolve_requested_variables(
        parsed_task.get("output_state_variables") or [],
        variable_names,
        registry_entries,
    )
    parsed_task["resolved_attention_variables"] = resolved_attention
    parsed_task["resolved_output_variables"] = resolved_outputs
    parsed_task["unresolved_attention_targets"] = unresolved_attention
    parsed_task["unresolved_output_state_variables"] = unresolved_outputs
    normalizations = []
    invalid_variables: List[str] = []
    for item in parsed_task.get("vocabulary_normalizations") or []:
        entry = dict(item)
        canonical: List[str] = []
        expanded_from_groups: Dict[str, List[str]] = {}
        for variable in entry.get("canonical_variables") or []:
            if variable in variable_names:
                if variable not in canonical:
                    canonical.append(variable)
                continue
            if variable in REGISTRY_GROUP_RULES:
                resolved_group, _unresolved_group = _resolve_requested_variables(
                    [variable],
                    variable_names,
                    registry_entries,
                )
                if resolved_group:
                    expanded_from_groups[variable] = list(resolved_group)
                    for name in resolved_group:
                        if name not in canonical:
                            canonical.append(name)
                    continue
            if variable not in invalid_variables:
                invalid_variables.append(variable)
        if expanded_from_groups:
            entry["canonical_variables"] = canonical
            entry["expanded_from_groups"] = expanded_from_groups
        normalizations.append(entry)
    if normalizations:
        parsed_task["vocabulary_normalizations"] = normalizations
    invalid_normalized_variables = list(dict.fromkeys(invalid_variables))
    parsed_task["invalid_normalized_variables"] = invalid_normalized_variables
    return {
        "resolved_attention_variables": resolved_attention,
        "resolved_output_variables": resolved_outputs,
        "unresolved_attention_targets": unresolved_attention,
        "unresolved_output_state_variables": unresolved_outputs,
        "invalid_normalized_variables": invalid_normalized_variables,
    }


def _counterfactual_comparison(
    baseline_rows: List[Any],
    disturbed_rows: List[Any],
    output_variables: List[str],
) -> Dict[str, Any]:
    comparisons = []
    for variable in output_variables:
        baseline = [
            float(row.values[variable])
            for row in baseline_rows
            if variable in row.values
        ]
        disturbed = [
            float(row.values[variable])
            for row in disturbed_rows
            if variable in row.values
        ]
        compared_steps = min(len(baseline), len(disturbed))
        if not compared_steps:
            continue
        deltas = [disturbed[index] - baseline[index] for index in range(compared_steps)]
        mean_delta = sum(deltas) / len(deltas)
        peak_index = max(range(len(deltas)), key=lambda index: abs(deltas[index]))
        comparisons.append(
            {
                "variable": variable,
                "mean_delta": round(mean_delta, 6),
                "final_delta": round(deltas[-1], 6),
                "max_abs_delta": round(abs(deltas[peak_index]), 6),
                "max_abs_delta_step_index": peak_index,
                "direction": "increase"
                if mean_delta > 0
                else "decrease"
                if mean_delta < 0
                else "unchanged",
            }
        )
    comparisons.sort(key=lambda item: item["max_abs_delta"], reverse=True)
    impacted = [item for item in comparisons if item["max_abs_delta"] > 1e-9]
    return {
        "mode": "baseline_vs_disturbed",
        "compared_step_count": min(len(baseline_rows), len(disturbed_rows)),
        "compared_output_variable_count": len(comparisons),
        "nonzero_impacted_variable_count": len(impacted),
        "top_impacted_variables": impacted[:5],
    }


def _build_unchanged_baseline_task(candidate_task: Dict[str, Any]) -> Dict[str, Any]:
    baseline_task = copy.deepcopy(candidate_task)
    baseline_task["disturbance_magnitude_percent"] = None
    baseline_task["disturbance_direction"] = "unknown"
    baseline_boundary = dict(baseline_task.get("boundary_conditions") or {})
    for key in (
        "setpoints",
        "percentage_changes",
        "disturbance_variable",
        "disturbance_direction",
        "disturbance_magnitude_percent",
    ):
        baseline_boundary.pop(key, None)
    baseline_boundary["keep_other_boundary_controls"] = True
    baseline_task["boundary_conditions"] = baseline_boundary
    return baseline_task


def _condition_number_from_case_id(case_id: Optional[str]) -> Optional[int]:
    if not case_id:
        return None
    digits = "".join(ch for ch in case_id if ch.isdigit())
    return int(digits) if digits else None


def _validate_forecast_horizon(value: Optional[int]) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "forecast_horizon_minutes must be an integer greater than or equal to 1."
        )


def _normalize_pipeformer_task(parsed: Dict[str, Any]) -> Dict[str, Any]:
    checks = _clean_checks(parsed.get("constraint_verification_types"))
    parsed["constraint_verification_types"] = checks

    parsed.setdefault("case_id", None)
    parsed.setdefault(
        "current_operating_condition_number",
        _condition_number_from_case_id(parsed.get("case_id")),
    )
    parsed.setdefault("forecast_horizon_minutes", None)
    _validate_forecast_horizon(parsed.get("forecast_horizon_minutes"))
    parsed.setdefault("disturbance_direction", "unknown")
    parsed.setdefault("disturbance_magnitude_percent", None)
    parsed.setdefault(
        "attention_targets", targets_for_checks(checks, CATEGORY_ATTENTION_TARGETS)
    )
    parsed.setdefault(
        "output_state_variables",
        targets_for_checks(checks, CATEGORY_OUTPUT_STATE_VARIABLES),
    )

    boundary_conditions = dict(parsed.get("boundary_conditions") or {})
    boundary_conditions.setdefault("keep_other_boundary_controls", True)
    boundary_conditions.setdefault(
        "disturbance_variable", parsed.get("disturbance_variable")
    )
    boundary_conditions.setdefault(
        "disturbance_direction", parsed.get("disturbance_direction")
    )
    boundary_conditions.setdefault(
        "disturbance_magnitude_percent", parsed.get("disturbance_magnitude_percent")
    )
    parsed["boundary_conditions"] = boundary_conditions

    parsed.setdefault("task_type", "prediction_and_verification")
    parsed.setdefault("parse_schema_version", PIPEFORMER_TASK_SCHEMA_VERSION)
    return parsed


def _merge_boundary_conditions(
    parsed_boundary: Dict[str, Any],
    explicit_boundary: Dict[str, Any],
) -> Dict[str, Any]:
    """Overlay explicit controls without dropping parsed disturbance controls."""
    merged = copy.deepcopy(dict(parsed_boundary or {}))
    for key, value in dict(explicit_boundary or {}).items():
        if key in {"percentage_changes", "setpoints"}:
            controls = dict(merged.get(key) or {})
            controls.update(dict(value or {}))
            merged[key] = controls
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _validate_disturbance_boundary_consistency(parsed: Dict[str, Any]) -> None:
    """Reject candidate corrections that reuse a background disturbance variable."""
    variable = str(parsed.get("disturbance_variable") or "")
    magnitude = parsed.get("disturbance_magnitude_percent")
    if not variable:
        return

    boundary = dict(parsed.get("boundary_conditions") or {})
    percentage_changes = dict(boundary.get("percentage_changes") or {})
    setpoints = dict(boundary.get("setpoints") or {})
    if str(parsed.get("disturbance_source") or "").casefold() == "external_condition":
        same_variable_candidate = variable in percentage_changes or (
            variable in setpoints and not variable.endswith(":ST")
        )
        if same_variable_candidate:
            raise ValueError(
                f"Candidate action cannot modify background disturbance variable {variable}. "
                "Keep it only in the disturbance fields and choose a different registered "
                "controllable input for the candidate action."
            )
        return

    if magnitude is None:
        return
    if variable not in percentage_changes:
        return

    direction = str(parsed.get("disturbance_direction") or "").casefold()
    expected = abs(float(magnitude)) * (-1.0 if direction == "down" else 1.0)
    explicit = float(percentage_changes[variable])
    if abs(explicit - expected) > 1e-9:
        raise ValueError(
            f"Boundary percentage change for {variable} ({explicit}%) conflicts with "
            f"disturbance {direction} {abs(float(magnitude))}%. Use one consistent action."
        )


def _validate_binary_state_controls(parsed: Dict[str, Any]) -> None:
    """Require discrete setpoints for binary ``:ST`` boundary variables."""
    boundary = dict(parsed.get("boundary_conditions") or {})
    percentage_changes = dict(boundary.get("percentage_changes") or {})
    invalid_percentages = sorted(
        str(variable)
        for variable in percentage_changes
        if str(variable).endswith(":ST")
    )
    disturbance_variable = str(parsed.get("disturbance_variable") or "")
    magnitude = parsed.get("disturbance_magnitude_percent")
    if disturbance_variable.endswith(":ST") and magnitude is not None:
        invalid_percentages.append(disturbance_variable)
    if invalid_percentages:
        raise ValueError(
            "Binary status variables do not accept percentage changes: "
            + ", ".join(sorted(set(invalid_percentages)))
            + ". Remove percentage_changes, use boundary_conditions.setpoints with exactly "
            "0 or 1, omit disturbance_magnitude_percent and retry."
        )

    invalid_setpoints = []
    for variable, raw_value in dict(boundary.get("setpoints") or {}).items():
        if not str(variable).endswith(":ST"):
            continue
        value = float(raw_value)
        if value not in {0.0, 1.0}:
            invalid_setpoints.append(f"{variable}={raw_value}")
    if invalid_setpoints:
        raise ValueError(
            "Binary status setpoints must be exactly 0 or 1: "
            + ", ".join(invalid_setpoints)
            + ". Replace each value with 0 or 1 and retry."
        )
    if disturbance_variable.endswith(":ST"):
        setpoints = dict(boundary.get("setpoints") or {})
        if disturbance_variable not in setpoints:
            raise ValueError(
                f"Binary status disturbance {disturbance_variable} requires an explicit "
                "boundary_conditions.setpoints value of 0 or 1. Omit "
                "disturbance_magnitude_percent and retry."
            )


def build_pipeformer_task(
    *,
    question: str,
    candidate_id: Optional[str] = None,
    case_id: Optional[str] = None,
    forecast_horizon_minutes: Optional[int] = None,
    current_operating_condition_number: Optional[int] = None,
    boundary_conditions: Optional[Dict[str, Any]] = None,
    disturbance_variable: Optional[str] = None,
    disturbance_setpoint: Optional[int] = None,
    disturbance_direction: Optional[str] = None,
    disturbance_magnitude_percent: Optional[float] = None,
    disturbance_assumption: Optional[str] = None,
    disturbance_source: Optional[str] = None,
    attention_targets: Optional[List[str]] = None,
    output_state_variables: Optional[List[str]] = None,
    vocabulary_normalizations: Optional[List[Dict[str, Any]]] = None,
    constraint_verification_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    parse_error: Optional[str] = None
    if question:
        try:
            parsed = parse_condition(question)
        except Exception as exc:
            parse_error = str(exc)

    parsed_direction = parsed.get("disturbance_direction")
    parsed_magnitude = parsed.get("disturbance_magnitude_percent")
    assumed_fields: List[str] = []

    if case_id is not None:
        parsed["case_id"] = case_id
    if current_operating_condition_number is not None:
        parsed["current_operating_condition_number"] = int(
            current_operating_condition_number
        )
    if boundary_conditions is not None:
        parsed["boundary_conditions"] = _merge_boundary_conditions(
            dict(parsed.get("boundary_conditions") or {}),
            dict(boundary_conditions),
        )
    if candidate_id is not None:
        parsed["candidate_id"] = str(candidate_id)
    if disturbance_variable is not None:
        parsed["disturbance_variable"] = disturbance_variable
    resolved_disturbance_variable = str(parsed.get("disturbance_variable") or "")
    if resolved_disturbance_variable.endswith(":ST"):
        if disturbance_setpoint is None:
            raise ValueError(
                f"Binary status disturbance {resolved_disturbance_variable} requires explicit "
                "disturbance_setpoint=0 or 1. Omit disturbance_magnitude_percent and retry."
            )
        if isinstance(disturbance_setpoint, bool):
            raise ValueError("disturbance_setpoint must be exactly 0 or 1.")
        setpoint_value = float(disturbance_setpoint)
        if setpoint_value not in {0.0, 1.0}:
            raise ValueError("disturbance_setpoint must be exactly 0 or 1.")
        parsed_boundary = dict(parsed.get("boundary_conditions") or {})
        setpoints = dict(parsed_boundary.get("setpoints") or {})
        existing_setpoint = setpoints.get(resolved_disturbance_variable)
        if existing_setpoint is not None and float(existing_setpoint) != setpoint_value:
            raise ValueError(
                f"disturbance_setpoint={int(setpoint_value)} conflicts with "
                f"boundary_conditions.setpoints[{resolved_disturbance_variable!r}]="
                f"{existing_setpoint}."
            )
        setpoints[resolved_disturbance_variable] = setpoint_value
        parsed_boundary["setpoints"] = setpoints
        parsed["boundary_conditions"] = parsed_boundary
        parsed["disturbance_setpoint"] = int(setpoint_value)
    elif disturbance_setpoint is not None:
        raise ValueError(
            "disturbance_setpoint is only valid for binary variables ending in :ST."
        )
    if disturbance_direction is not None:
        parsed["disturbance_direction"] = disturbance_direction
        if parsed_direction not in {"up", "down"}:
            assumed_fields.append("direction")
    if disturbance_magnitude_percent is not None:
        parsed["disturbance_magnitude_percent"] = float(disturbance_magnitude_percent)
        if parsed_magnitude is None:
            assumed_fields.append("magnitude_percent")
    elif str(parsed.get("disturbance_variable") or "").endswith(":ST"):
        # A percentage elsewhere in the question belongs to a candidate
        # action, not to the binary disturbance.
        parsed["disturbance_magnitude_percent"] = None
    if forecast_horizon_minutes is not None:
        _validate_forecast_horizon(forecast_horizon_minutes)
        parsed["forecast_horizon_minutes"] = forecast_horizon_minutes
    if attention_targets is not None:
        parsed["attention_targets"] = list(attention_targets)
    if output_state_variables is not None:
        parsed["output_state_variables"] = list(output_state_variables)
    if constraint_verification_types is not None:
        parsed["constraint_verification_types"] = _clean_checks(
            constraint_verification_types
        )

    parsed_boundary = dict(parsed.get("boundary_conditions") or {})
    parsed_boundary["disturbance_variable"] = parsed.get("disturbance_variable")
    parsed_boundary["disturbance_direction"] = parsed.get("disturbance_direction")
    parsed_boundary["disturbance_magnitude_percent"] = parsed.get(
        "disturbance_magnitude_percent"
    )
    parsed["boundary_conditions"] = parsed_boundary

    source = str(disturbance_source or "").strip().casefold()
    if source and source not in {"external_condition", "operator_action"}:
        raise ValueError(
            "disturbance_source must be 'external_condition' or 'operator_action'."
        )
    candidate_boundary = dict(parsed.get("boundary_conditions") or {})
    has_candidate_action = bool(
        candidate_boundary.get("percentage_changes")
        or candidate_boundary.get("setpoints")
    )
    parsed["disturbance_source"] = source or (
        "external_condition"
        if disturbance_assumption or (candidate_id and has_candidate_action)
        else "operator_action"
    )

    if assumed_fields and parsed["disturbance_source"] != "operator_action":
        statement = str(disturbance_assumption or "").strip()
        if not statement:
            direction = parsed.get("disturbance_direction")
            magnitude = parsed.get("disturbance_magnitude_percent")
            statement = (
                f"Provisional simulation assumption: {direction} {magnitude}%."
                if magnitude is not None
                else f"Provisional simulation assumption: direction {direction}."
            )
        parsed["disturbance_assumption"] = {
            "source": "llm_assumption",
            "assumed_fields": assumed_fields,
            "statement": statement,
        }
    elif parsed["disturbance_source"] == "operator_action":
        parsed.pop("disturbance_assumption", None)

    parsed = _normalize_pipeformer_task(parsed)
    parsed["vocabulary_normalizations"] = _normalize_vocabulary_provenance(
        vocabulary_normalizations,
        list(parsed.get("attention_targets") or [])
        + list(parsed.get("output_state_variables") or []),
    )
    if not parsed.get("disturbance_variable"):
        raise ValueError(
            f"PipeFormer forecast requires disturbance_variable. Parse error: {parse_error or 'not parsed'}"
        )
    if parsed.get("disturbance_magnitude_percent") is not None and parsed.get(
        "disturbance_direction"
    ) not in {"up", "down"}:
        raise ValueError(
            "PipeFormer forecast requires disturbance_direction to be 'up' or 'down' when disturbance_magnitude_percent is set."
        )
    _validate_disturbance_boundary_consistency(parsed)
    _validate_binary_state_controls(parsed)
    return parsed


class PipeFormerForecastService:
    """Coordinate task parsing, checkpoint inference, constraints, and evidence."""

    def __init__(
        self,
        backend_root: Path,
        *,
        baseline_cache: Optional[BaselineForecastCache] = None,
    ) -> None:
        self.backend_root = Path(backend_root).resolve()
        self.baseline_cache = (
            baseline_cache if baseline_cache is not None else BaselineForecastCache()
        )

    def analyze(self, **request: Any) -> Dict[str, Any]:
        result = _analyze_pipeformer_forecast(
            backend_root=self.backend_root,
            baseline_cache=self.baseline_cache,
            **request,
        )
        return (
            ForecastResult.from_payload(result).model_dump()
            if result.get("success") is True
            else result
        )


def run_pipeformer_forecast_analysis(
    *,
    question: str,
    backend_root: Path,
    candidate_id: Optional[str] = None,
    candidate_role: str = "candidate",
    case_id: Optional[str] = None,
    forecast_horizon_minutes: Optional[int] = None,
    current_operating_condition_number: Optional[int] = None,
    boundary_conditions: Optional[Dict[str, Any]] = None,
    disturbance_variable: Optional[str] = None,
    disturbance_setpoint: Optional[int] = None,
    disturbance_direction: Optional[str] = None,
    disturbance_magnitude_percent: Optional[float] = None,
    disturbance_assumption: Optional[str] = None,
    disturbance_source: Optional[str] = None,
    attention_targets: Optional[List[str]] = None,
    output_state_variables: Optional[List[str]] = None,
    vocabulary_normalizations: Optional[List[Dict[str, Any]]] = None,
    constraint_verification_types: Optional[List[str]] = None,
    include_baseline_comparison: Optional[bool] = None,
    baseline_cache: Optional[BaselineForecastCache] = None,
    pipeformer_root: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    static_dir: Optional[str] = None,
    mapping_csv: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Compatibility entry point; new callers should use PipeFormerForecastService."""
    request = locals().copy()
    service = PipeFormerForecastService(
        request.pop("backend_root"),
        baseline_cache=request.pop("baseline_cache"),
    )
    return service.analyze(**request)


def _analyze_pipeformer_forecast(
    *,
    question: str,
    backend_root: Path,
    candidate_id: Optional[str] = None,
    candidate_role: str = "candidate",
    case_id: Optional[str] = None,
    forecast_horizon_minutes: Optional[int] = None,
    current_operating_condition_number: Optional[int] = None,
    boundary_conditions: Optional[Dict[str, Any]] = None,
    disturbance_variable: Optional[str] = None,
    disturbance_setpoint: Optional[int] = None,
    disturbance_direction: Optional[str] = None,
    disturbance_magnitude_percent: Optional[float] = None,
    disturbance_assumption: Optional[str] = None,
    disturbance_source: Optional[str] = None,
    attention_targets: Optional[List[str]] = None,
    output_state_variables: Optional[List[str]] = None,
    vocabulary_normalizations: Optional[List[Dict[str, Any]]] = None,
    constraint_verification_types: Optional[List[str]] = None,
    include_baseline_comparison: Optional[bool] = None,
    baseline_cache: BaselineForecastCache,
    pipeformer_root: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    static_dir: Optional[str] = None,
    mapping_csv: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    logger.info("PipeFormer forecast tool started")
    candidate_role = str(candidate_role or "candidate").casefold()
    if candidate_role not in {"candidate", "baseline"}:
        raise ValueError("candidate_role must be 'candidate' or 'baseline'.")
    inference_config = PipeFormerInferenceConfig(
        backend_root=backend_root,
        checkpoint_dir=checkpoint_dir,
        pipeformer_root=pipeformer_root,
        data_dir=data_dir,
        static_dir=static_dir,
        mapping_path=mapping_csv,
        device=device,
    )
    parsed_task = build_pipeformer_task(
        question=question,
        candidate_id=candidate_id,
        case_id=case_id,
        forecast_horizon_minutes=forecast_horizon_minutes,
        current_operating_condition_number=current_operating_condition_number,
        boundary_conditions=boundary_conditions,
        disturbance_variable=disturbance_variable,
        disturbance_setpoint=disturbance_setpoint,
        disturbance_direction=disturbance_direction,
        disturbance_magnitude_percent=disturbance_magnitude_percent,
        disturbance_assumption=disturbance_assumption,
        disturbance_source=disturbance_source,
        attention_targets=attention_targets,
        output_state_variables=output_state_variables,
        vocabulary_normalizations=vocabulary_normalizations,
        constraint_verification_types=constraint_verification_types,
    )
    parsed_boundary = dict(parsed_task.get("boundary_conditions") or {})
    disturbance_magnitude = parsed_task.get("disturbance_magnitude_percent")
    has_disturbance = (
        disturbance_magnitude is not None and float(disturbance_magnitude) != 0.0
    )
    has_percentage_change = any(
        float(value) != 0.0
        for value in dict(parsed_boundary.get("percentage_changes") or {}).values()
    )
    has_setpoint = bool(parsed_boundary.get("setpoints"))
    if candidate_role == "baseline" and (
        has_disturbance or has_percentage_change or has_setpoint
    ):
        logger.warning(
            "Adjusted forecast cannot be a baseline; normalizing candidate_role to candidate."
        )
        candidate_role = "candidate"
    logger.info("PipeFormer parsed task: %s", parsed_task)
    environment = resolve_pipeformer_environment(inference_config)
    logger.info(
        "PipeFormer environment resolved: root=%s checkpoint=%s static=%s mapping=%s device=%s",
        environment.pipeformer_root,
        environment.checkpoint_dir,
        environment.static_dir,
        environment.mapping_path,
        environment.device,
    )
    inference_engine = PipeFormerInferenceEngine(environment)
    variable_names = environment.variable_names
    registry_entries = environment.registry_document.get("variables") or []
    task_resolution = _resolve_task_vocabulary(
        parsed_task,
        variable_names,
        registry_entries,
    )
    unresolved_attention = task_resolution["unresolved_attention_targets"]
    unresolved_outputs = task_resolution["unresolved_output_state_variables"]
    invalid_normalized_variables = task_resolution["invalid_normalized_variables"]
    if unresolved_attention or unresolved_outputs or invalid_normalized_variables:
        return {
            "success": False,
            "error_code": "unresolved_task_vocabulary",
            "retryable": True,
            "message": (
                "PipeFormer attention/output vocabulary must use canonical registry variables "
                "or supported canonical groups."
            ),
            "parsed_task": parsed_task,
            "task_resolution": task_resolution,
            "retry_guidance": (
                "Call search_pipeformer_registry for each unresolved term, then retry with only "
                "returned canonical variable IDs and record vocabulary_normalizations."
            ),
        }
    forecast_context = inference_engine.forecast(parsed_task)
    logger.info(
        "PipeFormer forecast context ready: mode=%s", forecast_context.get("mode")
    )
    parsed_task["forecast_time_step_minutes"] = forecast_context.get(
        "time_step_minutes"
    )
    parsed_task["_variable_registry"] = registry_entries
    parsed_task["_boundary_application_evidence"] = list(
        forecast_context.get("boundary_application_evidence") or []
    )
    variable_summaries = summarize_variables(
        forecast_context["real_rows"], forecast_context["predict_rows"]
    )
    logger.info(
        "PipeFormer variable summaries built: variables=%d", len(variable_summaries)
    )
    resolved_attention = task_resolution["resolved_attention_variables"]
    resolved_outputs = task_resolution["resolved_output_variables"]
    automatic_baseline = bool(candidate_id) and candidate_role == "candidate"
    baseline_enabled = (
        automatic_baseline
        if include_baseline_comparison is None
        else bool(include_baseline_comparison)
    )
    baseline_summaries = None
    baseline_reference = None
    counterfactual_comparison = None
    if baseline_enabled:
        baseline_task = _build_unchanged_baseline_task(parsed_task)
        baseline_key = (
            str(environment.checkpoint_dir),
            str(parsed_task.get("case_id") or ""),
            parsed_task.get("current_operating_condition_number"),
            parsed_task.get("forecast_horizon_minutes"),
            str(environment.disturbance_timing_mode or ""),
        )
        baseline_context = baseline_cache.get_or_compute(
            baseline_key,
            lambda: inference_engine.forecast(baseline_task),
        )
        baseline_reference = (
            "baseline_"
            + hashlib.sha256(
                "|".join(str(value) for value in baseline_key).encode("utf-8")
            ).hexdigest()[:12]
        )
        baseline_summaries = summarize_variables(
            baseline_context["real_rows"], baseline_context["predict_rows"]
        )
        comparison_variables = [
            str(item["variable"])
            for item in registry_entries
            if item.get("role") == "output"
        ]
        counterfactual_comparison = _counterfactual_comparison(
            baseline_context["predict_rows"],
            forecast_context["predict_rows"],
            comparison_variables,
        )
        counterfactual_comparison["baseline_reference"] = baseline_reference
        counterfactual_comparison["disturbance_variable"] = parsed_task.get(
            "disturbance_variable"
        )
        counterfactual_comparison["applied_disturbance"] = next(
            (
                item
                for item in forecast_context.get("applied_boundary_conditions") or []
                if item.get("variable") == parsed_task.get("disturbance_variable")
            ),
            None,
        )
    verification = run_engineering_constraint_checks(
        variable_summaries,
        parsed_task=parsed_task,
    )
    comparable_metrics = _integrated_energy_metrics(
        variable_summaries,
        registry_entries,
        float(forecast_context.get("time_step_minutes") or 1.0),
        baseline_summaries,
    )
    comparable_metrics["baseline_reference"] = baseline_reference
    verification["comparable_metrics"] = comparable_metrics
    logger.info(
        "PipeFormer constraint checks finished: overall_status=%s",
        verification.get("overall_status"),
    )
    priority_evidence_variables = []
    for finding in verification.get("priority_findings", []):
        for value in list(finding.get("offending_values") or []) + list(
            finding.get("evaluated_values") or []
        ):
            variable = value.get("variable")
            if variable and variable not in priority_evidence_variables:
                priority_evidence_variables.append(variable)
    disturbance_variable = parsed_task.get("disturbance_variable")
    output_variable_set = set(resolved_outputs)
    evidence_candidate_names = output_variable_set | set(priority_evidence_variables)
    if disturbance_variable:
        evidence_candidate_names.add(disturbance_variable)
    evidence_summaries = {
        variable: summary
        for variable, summary in variable_summaries.items()
        if variable in evidence_candidate_names
    }
    evidence_variables = top_variables(
        evidence_summaries,
        limit=3,
        preferred_variables=resolved_outputs,
        priority_variables=priority_evidence_variables,
    )
    observation_variables = top_variables(
        {
            variable: summary
            for variable, summary in variable_summaries.items()
            if variable in output_variable_set
        },
        limit=2,
        preferred_variables=resolved_attention,
        priority_variables=[
            variable
            for variable in priority_evidence_variables
            if variable in output_variable_set
        ],
    )
    logger.info(
        "PipeFormer evidence assembled: manual_intervention=%s top_variables=%s",
        verification.get("human_intervention_label"),
        [item.get("variable") for item in evidence_variables],
    )

    prediction_summary = {
        "forecast_mode": forecast_context["mode"],
        "case_id": parsed_task.get("case_id"),
        "current_operating_condition_number": parsed_task.get(
            "current_operating_condition_number"
        ),
        "forecast_horizon_minutes": parsed_task.get("forecast_horizon_minutes"),
        "disturbance_variable": parsed_task["disturbance_variable"],
        "disturbance_direction": parsed_task["disturbance_direction"],
        "disturbance_magnitude_percent": parsed_task["disturbance_magnitude_percent"],
        "disturbance_assumption": parsed_task.get("disturbance_assumption"),
        "disturbance_source": parsed_task.get("disturbance_source"),
        "attention_targets": parsed_task["attention_targets"],
        "output_state_variables": parsed_task["output_state_variables"],
        "constraint_verification_types": parsed_task["constraint_verification_types"],
        "resolved_attention_variables": resolved_attention,
        "resolved_output_variables": resolved_outputs,
        # ForecastResult owns the public field and variable selection. These
        # detailed summaries are transient inside this service only.
        "output_forecast_summary": variable_summaries,
        "top_watch_variables": evidence_variables,
    }
    if counterfactual_comparison is not None:
        prediction_summary["counterfactual_comparison"] = counterfactual_comparison
    forecast_metadata = {
        "mode": forecast_context["mode"],
        "disturbance_variable_mapping": forecast_context[
            "disturbance_variable_mapping"
        ],
        "forecast_window": _forecast_window_summary(forecast_context),
        "baseline_comparison_included": counterfactual_comparison is not None,
        "baseline_reference": baseline_reference,
    }
    for key in (
        "sequence_length",
        "time_step_offset",
        "requested_forecast_horizon_minutes",
        "requested_forecast_steps",
        "time_step_minutes",
        "actual_forecast_steps",
        "actual_forecast_horizon_minutes",
        "actual_forecast_horizon_source",
        "disturbance_timing_mode",
        "adjusted_input_step_count",
        "device",
        "model_input_projection_type",
        "data_provenance",
        "operating_condition_number_used",
        "applied_boundary_conditions",
        "boundary_application_evidence",
    ):
        if key in forecast_context:
            forecast_metadata[key] = forecast_context[key]

    for source_key, metadata_key in (
        ("checkpoint_dir", "checkpoint_id"),
        ("data_case_dir", "data_case_id"),
    ):
        if source_key in forecast_context:
            forecast_metadata[metadata_key] = source_name(forecast_context[source_key])

    parsed_task.pop("_variable_registry", None)
    parsed_task.pop("_boundary_application_evidence", None)
    result = {
        "success": True,
        "candidate_id": candidate_id,
        "candidate_role": candidate_role,
        "parsed_task": parsed_task,
        "prediction_summary": prediction_summary,
        "constraint_check": verification,
        "evidence": {
            "top_watch_variables": evidence_variables,
            "key_observation_variables": observation_variables,
            "boundary_application_evidence": forecast_context.get(
                "boundary_application_evidence"
            )
            or [],
        },
        "risk_level": verification["risk_level"],
        "manual_intervention_label": verification["human_intervention_label"],
        "dispatch_recommendation": verification.get("dispatch_recommendation"),
        "quality_flag": (
            "pass"
            if verification.get("verification_complete", True)
            else "needs_review"
        ),
        "forecast_metadata": forecast_metadata,
    }
    if counterfactual_comparison is not None:
        result["counterfactual_comparison"] = counterfactual_comparison
    return result
