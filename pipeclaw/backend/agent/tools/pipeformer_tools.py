from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import register_tool
from pipeline.pipeformer_tool_runtime import PipeFormerForecastService

_REGISTERED = False


def _default_backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def register_pipeformer_tools(backend_root: Optional[Path] = None) -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    resolved_backend_root = Path(backend_root).resolve() if backend_root else _default_backend_root()
    forecast_service = PipeFormerForecastService(resolved_backend_root)

    @register_tool(
        name="run_pipeformer_forecast",
        description=(
            "Run real PipeFormer checkpoint inference for forecast, what-if, risk, dispatch, "
            "or transient-operation questions. Organize the task with PDF terms such as "
            "disturbance_variable, disturbance_direction, disturbance_magnitude_percent, "
            "forecast_horizon_minutes, output_state_variables, and constraint_verification_types."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The full user forecast or what-if question."},
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
                            "description": "Absolute boundary-control setpoints keyed by mapped PipeFormer variable.",
                        },
                        "percentage_changes": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                            "description": "Signed percentage changes keyed by mapped PipeFormer variable.",
                        },
                    },
                    "additionalProperties": False,
                },
                "disturbance_variable": {
                    "type": "string",
                    "description": "PipeFormer variable to perturb, for example T_001:BC000.",
                },
                "disturbance_direction": {"type": "string", "enum": ["up", "down"], "description": "Disturbance direction."},
                "disturbance_magnitude_percent": {
                    "type": "number",
                    "description": "Percent disturbance magnitude, for example 11 for 11%.",
                },
                "forecast_horizon_minutes": {"type": "integer", "description": "Requested forecast horizon in minutes."},
                "attention_targets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Nodes, segments, equipment, or risk targets requiring attention.",
                },
                "output_state_variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "State-variable groups that should be returned or emphasized.",
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
                        "Run an additional unchanged baseline forecast and return compact baseline-versus-disturbed "
                        "deltas. Use only when the question requires causal, propagation, or disturbance-impact claims."
                    ),
                },
                "device": {"type": "string", "description": "Optional Torch device override, for example cpu or cuda."},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        returns="PipeFormer prediction summary, constraint checks, and evidence variables.",
    )
    def run_pipeformer_forecast(
        question: str,
        case_id: Optional[str] = None,
        current_operating_condition_number: Optional[int] = None,
        boundary_conditions: Optional[Dict[str, Any]] = None,
        disturbance_variable: Optional[str] = None,
        disturbance_direction: Optional[str] = None,
        disturbance_magnitude_percent: Optional[float] = None,
        forecast_horizon_minutes: Optional[int] = None,
        attention_targets: Optional[List[str]] = None,
        output_state_variables: Optional[List[str]] = None,
        constraint_verification_types: Optional[List[str]] = None,
        include_baseline_comparison: bool = False,
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
            case_id=case_id,
            current_operating_condition_number=current_operating_condition_number,
            boundary_conditions=boundary_conditions,
            disturbance_variable=disturbance_variable,
            disturbance_direction=disturbance_direction,
            disturbance_magnitude_percent=disturbance_magnitude_percent,
            forecast_horizon_minutes=forecast_horizon_minutes,
            attention_targets=attention_targets,
            output_state_variables=output_state_variables,
            constraint_verification_types=constraint_verification_types,
            include_baseline_comparison=include_baseline_comparison,
            pipeformer_root=pipeformer_root,
            checkpoint_dir=checkpoint_dir,
            data_dir=data_dir,
            static_dir=static_dir,
            mapping_csv=mapping_csv,
            device=device,
        )
