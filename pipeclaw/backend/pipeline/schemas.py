from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ForecastRequest:
    question: str
    candidate_id: Optional[str] = None
    candidate_role: str = "candidate"
    case_id: Optional[str] = None
    forecast_horizon_minutes: Optional[int] = None
    current_operating_condition_number: Optional[int] = None
    boundary_conditions: Optional[Dict[str, Any]] = None
    disturbance_variable: Optional[str] = None
    disturbance_setpoint: Optional[int] = None
    disturbance_direction: Optional[str] = None
    disturbance_magnitude_percent: Optional[float] = None
    disturbance_assumption: Optional[str] = None
    disturbance_source: Optional[str] = None
    attention_targets: Optional[List[str]] = None
    output_state_variables: Optional[List[str]] = None
    vocabulary_normalizations: Optional[List[Dict[str, Any]]] = None
    constraint_verification_types: Optional[List[str]] = None
    include_baseline_comparison: Optional[bool] = None
    pipeformer_root: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    data_dir: Optional[str] = None
    static_dir: Optional[str] = None
    mapping_csv: Optional[str] = None
    device: Optional[str] = None
    _provided_fields: tuple[str, ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ForecastRequest":
        field_names = {
            field.name for field in fields(cls) if field.name != "_provided_fields"
        }
        unknown = set(values) - field_names
        if unknown:
            names = ", ".join(sorted(map(str, unknown)))
            raise TypeError(f"Unknown forecast request fields: {names}")
        instance = cls(**deepcopy(dict(values)))
        object.__setattr__(instance, "_provided_fields", tuple(values))
        return instance

    def to_mapping(self) -> Dict[str, Any]:
        names = self._provided_fields or tuple(
            field.name for field in fields(self) if field.name != "_provided_fields"
        )
        return deepcopy({name: getattr(self, name) for name in names})


@dataclass
class ForecastRow:
    label: str
    values: Dict[str, float]


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    category: str
    description: str
    priority: int
    metric: str
    physical_quantities: Tuple[str, ...] = ()
    equipment_types: Tuple[str, ...] = ()
    roles: Tuple[str, ...] = ()
    use_registry_limits: bool = False
    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    fail_low: Optional[float] = None
    fail_high: Optional[float] = None
    warning_threshold: Optional[float] = None
    fail_threshold: Optional[float] = None
    pass_flag: Optional[str] = None
    warning_flag: Optional[str] = None
    fail_flag: Optional[str] = None
