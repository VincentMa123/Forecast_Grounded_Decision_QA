"""Small deterministic records consumed by the current evaluators."""

from copy import deepcopy


def _forecast_task() -> dict:
    return {
        "tool_call_id": "forecast-1",
        "case_id": "case-1",
        "current_operating_condition_number": 1,
        "disturbance_variable": "FLOW_001",
        "disturbance_direction": "up",
        "disturbance_magnitude_percent": 5.0,
        "forecast_horizon_minutes": 30,
        "constraint_verification_types": ["pressure"],
    }


def _forecast_output() -> dict:
    return {
        "success": True,
        "provenance": {"checkpoint_id": "checkpoint-1"},
        "prediction": {
            "forecast_mode": "checkpoint_inference",
            "disturbance_variable": "FLOW_001",
            "disturbance_direction": "up",
            "disturbance_magnitude_percent": 5.0,
            "forecast_horizon_minutes": 30,
            "actual_forecast_horizon_minutes": 30,
            "forecast_window": {"time_step_minutes": 5, "predict_row_count": 6},
        },
        "task_resolution": {
            "applied_boundary_conditions": [
                {"variable": "FLOW_001", "mode": "percent_change", "value": 5.0}
            ]
        },
        "verification": {
            "requested_categories": ["pressure"],
            "category_status": {"pressure": "pass"},
            "rule_status": {"pressure": "pass"},
            "verification_complete": True,
            "not_evaluated_rules": [],
        },
    }


def passing_teacher_record() -> dict:
    """A complete PipeFormer teacher record accepted by the native evaluator."""
    task = _forecast_task()
    return {
        "sample_id": "sample-1",
        "scenario_id": "scenario-1",
        "scenario_type": "pipeformer",
        "user_input": "Forecast the outcome.",
        "parsed_task": task,
        "tool_calls": [
            {
                "tool_call_id": "search-1",
                "name": "search_pipeformer_registry",
                "arguments": {"query": "FLOW_001"},
            },
            {"tool_call_id": "forecast-1", "name": "run_pipeformer_forecast", "arguments": task},
        ],
        "tool_outputs": [
            {
                "tool_call_id": "search-1",
                "name": "search_pipeformer_registry",
                "output": {"success": True, "variables": [{"variable": "FLOW_001"}]},
            },
            {"tool_call_id": "forecast-1", "name": "run_pipeformer_forecast", "output": _forecast_output()},
        ],
        "prediction_summary": {"forecast_mode": "checkpoint_inference"},
        "constraint_check": {"category_status": {"pressure": "pass"}},
        "evidence": {},
        "risk_level": "low",
        "manual_intervention_label": "not_required",
        "dispatch_recommendation": "monitor",
        "final_answer": "Forecast completed.",
        "quality_flag": "pass",
        "trace_status": "completed",
    }


def assumed_disturbance_record() -> dict:
    """A provisional teacher disturbance whose executed prediction is authoritative."""
    record = passing_teacher_record()
    task = record["parsed_task"]
    task.update(
        {
            "disturbance_direction": "down",
            "disturbance_magnitude_percent": 9.0,
            "disturbance_assumption": {
                "source": "llm_assumption",
                "assumed_fields": ["direction", "magnitude_percent"],
            },
        }
    )
    record["tool_calls"][1]["arguments"] = task
    record["final_answer"] = "Provisional assumption: a 5 percent increase was forecast."
    return record


def teacher_reference() -> dict:
    """A teacher source record for Task 2 oracle metrics."""
    return deepcopy(passing_teacher_record())


def reference_without_risk() -> dict:
    """A source whose oracle has no risk label, making risk inapplicable."""
    reference = teacher_reference()
    reference.pop("risk_level")
    reference["tool_outputs"][1]["output"]["verification"].pop("risk_level", None)
    return reference


def successful_rollout() -> dict:
    """A completed autonomous rollout with one successful forecast call."""
    task = _forecast_task()
    return {
        "trace_status": "completed",
        "tool_calls": [
            {
                "tool_call_id": "student-forecast-1",
                "name": "run_pipeformer_forecast",
                "arguments": task,
                "schema_valid": True,
                "execution_success": True,
            }
        ],
        "tool_outputs": [
            {
                "tool_call_id": "student-forecast-1",
                "name": "run_pipeformer_forecast",
                "output": _forecast_output(),
            }
        ],
        "final_answer": "Forecast completed.",
        "json_errors": [],
    }


def malformed_then_retried_rollout() -> dict:
    """A malformed forecast attempt followed by a successful retry."""
    rollout = successful_rollout()
    retry = rollout["tool_calls"][0]
    retry["tool_call_id"] = "student-forecast-2"
    rollout["tool_calls"].insert(
        0,
        {
            "tool_call_id": "student-forecast-1",
            "name": "run_pipeformer_forecast",
            "arguments": {"case_id": "case-1"},
            "schema_valid": False,
            "execution_success": False,
        },
    )
    rollout["tool_outputs"][0]["tool_call_id"] = "student-forecast-2"
    return rollout


def openclaw_artifact_rollout() -> dict:
    """An OpenClaw-style rollout that reads the requested artifact's content."""
    return {
        "tool_calls": [
            {
                "tool_call_id": "read-1",
                "name": "read_file",
                "arguments": {"path": "requested.csv"},
                "schema_valid": True,
                "execution_success": True,
            }
        ],
        "tool_outputs": [
            {
                "tool_call_id": "read-1",
                "name": "read_file",
                "output": {"success": True, "path": "requested.csv", "content": "value,42\n"},
            }
        ],
        "final_answer": "The requested artifact was read.",
    }
