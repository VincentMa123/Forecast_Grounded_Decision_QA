from __future__ import annotations

import json
from pathlib import Path
import os
import re
from typing import Any, Dict, List, Optional

from .registry import register_tool
from pipeclaw.backend.pipeline.registry.search_service import (
    PipeFormerRegistrySearchService,
)
from pipeclaw.backend.pipeline.forecast.runtime import PipeFormerForecastService
from pipeclaw.backend.grounding.decision_policy import (
    METRIC_CATALOG,
    normalize_policy_tool_request,
)
from pipeclaw.backend.grounding.evidence.topology import build_topology_evidence_result

_REGISTERED = False
NODE_FILE_RE = re.compile(r"^\d{8}_node\.csv$", re.IGNORECASE)
PIPELINE_FILE_RE = re.compile(r"^\d{8}_pipeline\.csv$", re.IGNORECASE)


def _default_backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def register_pipeformer_tools(
    backend_root: Optional[Path] = None,
    *,
    registry_search_service: Optional[PipeFormerRegistrySearchService] = None,
) -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    resolved_backend_root = Path(backend_root).resolve() if backend_root else _default_backend_root()
    forecast_service = PipeFormerForecastService(resolved_backend_root)
    registry_search_service = registry_search_service or PipeFormerRegistrySearchService(
        resolved_backend_root
    )

    @register_tool(
        name="analyze_pipeline_topology",
        description=(
            "Deterministically analyze source reachability, shortest paths, direct inbound segments, "
            "and shared-gateway structure from one daily node CSV and pipeline CSV. Use this before "
            "answering topology reachability, source-count, shortest-path, or gateway questions."
        ),
        parameters={
            "type": "object",
            "properties": {
                "node_file": {
                    "type": "string",
                    "pattern": r"^\d{8}_node\.csv$",
                    "description": "Daily node filename, for example 20190211_node.csv.",
                },
                "pipeline_file": {
                    "type": "string",
                    "pattern": r"^\d{8}_pipeline\.csv$",
                    "description": "Daily pipeline filename, for example 20190211_pipeline.csv.",
                },
                "target_station": {"type": "string"},
                "pipeline_scope": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional pipeline names to restrict the graph traversal.",
                },
            },
            "required": ["node_file", "pipeline_file", "target_station"],
            "additionalProperties": False,
        },
        returns="Compact deterministic topology evidence with source counts and one shortest path.",
    )
    def analyze_pipeline_topology(
        node_file: str,
        pipeline_file: str,
        target_station: str,
        pipeline_scope: Optional[List[str]] = None,
        session_id: str = "",
        agent_id: str = "default",
    ) -> Dict[str, Any]:
        del session_id, agent_id
        if not NODE_FILE_RE.fullmatch(str(node_file)):
            return {"success": False, "error": "node_file must match YYYYMMDD_node.csv"}
        if not PIPELINE_FILE_RE.fullmatch(str(pipeline_file)):
            return {"success": False, "error": "pipeline_file must match YYYYMMDD_pipeline.csv"}
        scope = " ".join(str(value).strip() for value in pipeline_scope or [] if str(value).strip())
        request_text = (
            f"Use {node_file} and {pipeline_file}. {scope} "
            f"\u53cd\u5411\u8ffd\u5230 {str(target_station).strip()} \u7684\u53ef\u8fbe\u6c14\u6e90\u548c\u6700\u77ed\u8def\u5f84\u3002"
        )
        summary, failure = build_topology_evidence_result(
            request_text,
            pipeline_data_root=resolved_backend_root / "pipeline_data",
        )
        if not summary:
            return {
                "success": False,
                "error_code": failure.get("error_code", "topology_evidence_unavailable"),
                "error": failure.get(
                    "message",
                    "Topology evidence could not be built from the requested files and target.",
                ),
                "details": {
                    key: value
                    for key, value in failure.items()
                    if key not in {"error_code", "message"}
                },
                "retryable": failure.get("error_code") in {
                    "missing_target_station",
                    "missing_topology_file_reference",
                    "target_station_not_in_topology",
                },
            }
        return {
            "success": True,
            "evidence_kind": "file_content",
            "source_artifacts": [str(node_file), str(pipeline_file)],
            "evidence_excerpt": json.dumps(summary, ensure_ascii=False)[:2_000],
            "topology_summary": summary,
        }

    @register_tool(
        name="search_pipeformer_registry",
        description=(
            "Search the PipeFormer variable registry before every forecast and before selecting dispatch controls. "
            "Use role=input and controllable=true for action variables. Each bounded result page contains "
            "canonical variable IDs plus role and controllable flags; a forecast may use only variables "
            "that appear in preceding successful relevant search results."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Variable ID or physical/equipment/effect search terms."},
                "role": {"type": "string", "enum": ["input", "output"]},
                "controllable": {"type": "boolean"},
                "equipment_ids": {"type": "array", "items": {"type": "string"}},
                "equipment_types": {"type": "array", "items": {"type": "string"}},
                "physical_quantities": {"type": "array", "items": {"type": "string"}},
                "attention_targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Equipment or topology nodes used to rank nearby controls first.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "Zero-based result offset for deterministic pagination.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
            },
            "additionalProperties": False,
        },
        returns=(
            "A bounded ranked page with success, matched counts, offset, an optional next_offset, "
            "and variables containing only variable, role, and controllable."
        ),
    )
    def search_pipeformer_registry(
        query: str = "",
        role: Optional[str] = None,
        controllable: Optional[bool] = None,
        equipment_ids: Optional[List[str]] = None,
        equipment_types: Optional[List[str]] = None,
        physical_quantities: Optional[List[str]] = None,
        attention_targets: Optional[List[str]] = None,
        offset: int = 0,
        limit: int = 12,
        session_id: str = "",
        agent_id: str = "default",
    ) -> Dict[str, Any]:
        del session_id, agent_id
        return registry_search_service.search(
            query=query,
            role=role,
            controllable=controllable,
            equipment_ids=equipment_ids or [],
            equipment_types=equipment_types or [],
            physical_quantities=physical_quantities or [],
            attention_targets=attention_targets or [],
            offset=offset,
            limit=limit,
        )

    @register_tool(
        name="set_decision_policy",
        description=(
            "Translate the user's natural-language dispatch priorities into an ordered, "
            "machine-checkable decision policy before comparing multiple PipeFormer candidates. "
            "The LLM must infer the order from the current user request; do not ask the user to "
            "supply this schema and do not invent a proxy that the request did not authorize."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source_excerpt": {
                    "type": "string",
                    "description": (
                        "The exact short user phrase that establishes the objective order or "
                        "hard constraint."
                    ),
                },
                "hard_constraints": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["no_constraint_failure"],
                    },
                    "minItems": 1,
                    "maxItems": 1,
                },
                "objectives": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "enum": sorted(METRIC_CATALOG),
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["minimize", "maximize"],
                            },
                            "tolerance": {
                                "type": "number",
                                "minimum": 0,
                                "description": "Optional absolute tie tolerance.",
                            },
                            "source_excerpt": {
                                "type": "string",
                                "description": (
                                    "One exact contiguous phrase from the current user request "
                                    "that supports this objective. Do not join separate phrases."
                                ),
                            },
                            "proxy_for": {
                                "type": "string",
                                "description": (
                                    "Optional user-authorized proxy target. Omit unless the user "
                                    "explicitly accepts a proxy."
                                ),
                            },
                        },
                        "required": ["metric", "direction", "source_excerpt"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["hard_constraints", "objectives"],
            "additionalProperties": False,
        },
        returns=(
            "A validated LLM-extracted decision policy. Invalid metrics, directions, "
            "constraints, or tolerances return exact retry errors."
        ),
    )
    def set_decision_policy(
        hard_constraints: List[str],
        objectives: List[Dict[str, Any]],
        source_excerpt: str = "",
        session_id: str = "",
        agent_id: str = "default",
    ) -> Dict[str, Any]:
        del session_id, agent_id
        return normalize_policy_tool_request(
            hard_constraints=hard_constraints,
            objectives=objectives,
            source_excerpt=source_excerpt,
        )

    @register_tool(
        name="run_pipeformer_forecast",
        description=(
            "Run real PipeFormer checkpoint inference for forecast, what-if, risk, dispatch, "
            "or transient-operation questions. Organize the task with PDF terms such as "
            "disturbance_variable, disturbance_direction, disturbance_magnitude_percent, "
            "forecast_horizon_minutes, output_state_variables, and constraint_verification_types. "
            "Preconditions: a valid operating case, a canonical disturbance variable returned by "
            "a preceding successful relevant registry search, and every candidate control returned "
            "by a preceding role=input, controllable=true registry search."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The full user forecast or what-if question."},
                "candidate_id": {
                    "type": "string",
                    "description": (
                        "Stable identifier for a dispatch comparison item, for example candidate_1. "
                        "Omit for a single prediction. A named candidate requires a nonempty "
                        "boundary action; a disturbance-only reference must use candidate_role=baseline."
                    ),
                },
                "candidate_role": {
                    "type": "string",
                    "enum": ["candidate", "baseline"],
                    "description": (
                        "Explicit comparison role. Use candidate only when boundary_conditions "
                        "contains at least one distinct action setpoint or percentage change. Use "
                        "baseline for a no-action disturbance reference."
                    ),
                },
                "case_id": {"type": "string", "description": "Optional mock case id, for example mock_test_001."},
                "current_operating_condition_number": {
                    "type": "integer",
                    "description": "Current operating-condition number from the scenario, when available.",
                },
                "boundary_conditions": {
                    "type": "object",
                    "description": "Boundary-control assumptions and explicit setpoints or signed percentage changes.",
                    "properties": {
                        "keep_other_boundary_controls": {"type": "boolean"},
                        "setpoints": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                            "description": (
                                "Absolute boundary-control setpoints keyed by a preceding returned "
                                "controllable input. Variables ending in :ST must use exactly 0 or 1."
                            ),
                        },
                        "percentage_changes": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                            "description": (
                                "Signed percentage changes keyed by a preceding returned controllable "
                                "input. Never use percentage changes for variables ending in :ST."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
                "disturbance_variable": {
                    "type": "string",
                    "description": (
                        "Canonical PipeFormer variable to perturb. It must appear in a preceding "
                        "successful registry result authorized by an exact-ID search or a meaningful "
                        "search with registry normalization provenance."
                    ),
                },
                "disturbance_setpoint": {
                    "type": "integer",
                    "enum": [0, 1],
                    "description": (
                        "Required for a binary :ST disturbance and prohibited for continuous "
                        "disturbances. This is the background disturbance value, not a candidate action."
                    ),
                },
                "disturbance_direction": {"type": "string", "enum": ["up", "down"], "description": "Disturbance direction."},
                "disturbance_magnitude_percent": {
                    "type": "number",
                    "description": (
                        "Percent disturbance magnitude, for example 11 for 11%. For a binary :ST "
                        "disturbance, omit this field and apply exactly 0 or 1 through setpoints."
                    ),
                },
                "disturbance_assumption": {
                    "type": "string",
                    "description": (
                        "Required when direction or magnitude was inferred rather than stated by the user. "
                        "Briefly describe the provisional simulation assumption."
                    ),
                },
                "disturbance_source": {
                    "type": "string",
                    "enum": ["external_condition", "operator_action"],
                    "description": (
                        "Whether the disturbance is a background/exogenous condition or the operator action "
                        "being evaluated. Dispatch candidate calls normally use external_condition."
                    ),
                },
                "forecast_horizon_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Requested forecast horizon in minutes (at least 1).",
                },
                "attention_targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Nodes, segments, equipment, or risk targets requiring attention.",
                },
                "output_state_variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Canonical state-variable groups or exact output IDs. Resolve unknown user "
                        "terms with search_pipeformer_registry before calling this tool. "
                        "Decision-policy metrics are derived from forecast audit results; never "
                        "put catalog metrics such as flow.max_abs_supply_demand_gap or "
                        "energy.delta_vs_baseline here."
                    ),
                },
                "vocabulary_normalizations": {
                    "type": "array",
                    "description": (
                        "Provenance for free-form terms resolved by a preceding registry search. "
                        "Each canonical_variables entry must be a canonical registry variable ID "
                        "(for example TE_017_v000) or a supported registry group name (for example "
                        "power, energy, pressure, flow, linepack, compressor_load); never invent "
                        "names. The canonical variables must also appear in attention_targets or "
                        "output_state_variables."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "requested_term": {"type": "string"},
                            "canonical_variables": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                            "normalization_source": {
                                "type": "string",
                                "enum": ["registry_search"],
                            },
                        },
                        "required": [
                            "requested_term",
                            "canonical_variables",
                            "normalization_source",
                        ],
                        "additionalProperties": False,
                    },
                },
                "constraint_verification_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["pressure", "flow", "linepack", "compressor", "equipment_regulation", "abnormality_warning", "dispatch_priority"],
                    },
                    "description": "Engineering constraint categories to execute, using the PDF names.",
                },
                "include_baseline_comparison": {
                    "type": "boolean",
                    "description": (
                        "Override automatic baseline comparison. Dispatch calls with candidate_id automatically use "
                        "one shared unchanged baseline when this field is omitted."
                    ),
                },
                "device": {"type": "string", "description": "Optional Torch device override, for example cpu or cuda."},
            },
            "allOf": [
                {
                    "if": {
                        "required": ["disturbance_variable"],
                        "properties": {
                            "disturbance_variable": {
                                "type": "string",
                                "pattern": ":ST$",
                            }
                        },
                    },
                    "then": {"required": ["disturbance_setpoint"]},
                }
            ],
            "required": ["question", "disturbance_variable"],
            "additionalProperties": False,
        },
        returns="PipeFormer prediction summary, constraint checks, and evidence variables.",
    )
    def run_pipeformer_forecast(
        question: str,
        candidate_id: Optional[str] = None,
        candidate_role: str = "candidate",
        case_id: Optional[str] = None,
        current_operating_condition_number: Optional[int] = None,
        boundary_conditions: Optional[Dict[str, Any]] = None,
        disturbance_variable: Optional[str] = None,
        disturbance_setpoint: Optional[int] = None,
        disturbance_direction: Optional[str] = None,
        disturbance_magnitude_percent: Optional[float] = None,
        disturbance_assumption: Optional[str] = None,
        disturbance_source: Optional[str] = None,
        forecast_horizon_minutes: Optional[int] = None,
        attention_targets: Optional[List[str]] = None,
        output_state_variables: Optional[List[str]] = None,
        vocabulary_normalizations: Optional[List[Dict[str, Any]]] = None,
        constraint_verification_types: Optional[List[str]] = None,
        include_baseline_comparison: Optional[bool] = None,
        device: Optional[str] = None,
        pipeformer_root: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        static_dir: Optional[str] = None,
        mapping_csv: Optional[str] = None,
        session_id: str = "",
        agent_id: str = "default",
    ) -> Dict[str, Any]:
        return forecast_service.analyze(
            question=question,
            candidate_id=candidate_id,
            candidate_role=candidate_role,
            case_id=case_id,
            current_operating_condition_number=current_operating_condition_number,
            boundary_conditions=boundary_conditions,
            disturbance_variable=disturbance_variable,
            disturbance_setpoint=disturbance_setpoint,
            disturbance_direction=disturbance_direction,
            disturbance_magnitude_percent=disturbance_magnitude_percent,
            disturbance_assumption=disturbance_assumption,
            disturbance_source=disturbance_source,
            forecast_horizon_minutes=forecast_horizon_minutes,
            attention_targets=attention_targets,
            output_state_variables=output_state_variables,
            vocabulary_normalizations=vocabulary_normalizations,
            constraint_verification_types=constraint_verification_types,
            include_baseline_comparison=include_baseline_comparison,
            pipeformer_root=pipeformer_root,
            checkpoint_dir=checkpoint_dir,
            data_dir=data_dir,
            static_dir=static_dir,
            mapping_csv=mapping_csv,
            device=device,
        )

    _REGISTERED = True
