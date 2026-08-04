"""Regression tests for the canonical shared evaluator checks."""

from __future__ import annotations

from copy import deepcopy
import unittest

from pipeclaw.backend.evaluator import EvaluationProfile, evaluate
from pipeclaw.tests.evaluation.fixtures import (
    malformed_then_retried_rollout,
    openclaw_artifact_rollout,
    successful_rollout,
    teacher_reference,
)


def _autonomous_report(reference: dict, rollout: dict):
    return evaluate(
        rollout,
        profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
        reference=reference,
    )


class AssumptionCheckTests(unittest.TestCase):
    def test_assumed_magnitude_matches_executed_student_prediction_not_sampled_oracle(self):
        reference = teacher_reference()
        reference["parsed_task"]["disturbance_assumption"] = {
            "source": "llm_assumption",
            "assumed_fields": ["magnitude_percent"],
        }
        rollout = successful_rollout()
        rollout["tool_calls"][0]["arguments"]["disturbance_magnitude_percent"] = 9.0
        output = rollout["tool_outputs"][0]["output"]
        output["prediction"]["disturbance_magnitude_percent"] = 9.0
        output["task_resolution"]["applied_boundary_conditions"][0]["value"] = 9.0

        report = _autonomous_report(reference, rollout)

        self.assertTrue(report.metrics["task_parsing"].passed)
        self.assertTrue(report.metrics["assumption_consistency"].passed)

    def test_explicit_magnitude_remains_strict_against_oracle(self):
        rollout = successful_rollout()
        rollout["tool_calls"][0]["arguments"]["disturbance_magnitude_percent"] = 9.0
        output = rollout["tool_outputs"][0]["output"]
        output["prediction"]["disturbance_magnitude_percent"] = 9.0
        output["task_resolution"]["applied_boundary_conditions"][0]["value"] = 9.0

        report = _autonomous_report(teacher_reference(), rollout)

        self.assertFalse(report.metrics["task_parsing"].passed)
        self.assertIn(
            "disturbance_magnitude_percent",
            report.metrics["task_parsing"].details["mismatched_fields"],
        )


class SharedMetricTests(unittest.TestCase):
    def test_successful_student_forecast_numeric_output_grounds_different_answer(self):
        rollout = successful_rollout()
        rollout["tool_outputs"][0]["output"]["prediction"][
            "student_specific_value"
        ] = 9.25
        rollout["final_answer"] = "The successful student forecast returned 9.25."

        report = _autonomous_report(teacher_reference(), rollout)

        metric = report.metrics["evidence_consistency"]
        self.assertTrue(metric.passed)
        self.assertEqual(metric.details["unsupported_numeric_values"], [])

    def test_failed_required_call_then_success_fails_tool_call_but_passes_recovery(self):
        report = _autonomous_report(
            teacher_reference(), malformed_then_retried_rollout()
        )

        self.assertFalse(report.metrics["tool_call"].passed)
        self.assertTrue(report.metrics["tool_recovery"].applicable)
        self.assertTrue(report.metrics["tool_recovery"].passed)
        self.assertFalse(report.metrics["tool_recovery"].included_in_score)

    def test_every_compact_forecast_requires_preceding_registry_authorization(self):
        rollout = successful_rollout()
        first_forecast = rollout["tool_calls"].pop(0)
        first_output = rollout["tool_outputs"].pop(0)
        rollout["tool_calls"].extend(
            [
                {
                    "tool_call_id": "search-1",
                    "name": "search_pipeformer_registry",
                    "arguments": {"query": "FLOW_001"},
                    "schema_valid": True,
                    "execution_success": True,
                },
                first_forecast,
                {
                    "tool_call_id": "forecast-2",
                    "name": "run_pipeformer_forecast",
                    "arguments": {
                        **first_forecast["arguments"],
                        "disturbance_variable": "FLOW_002",
                    },
                    "schema_valid": True,
                    "execution_success": True,
                },
            ]
        )
        second_output = deepcopy(first_output)
        second_output["tool_call_id"] = "forecast-2"
        second_output["output"]["prediction"]["disturbance_variable"] = "FLOW_002"
        second_output["output"]["task_resolution"]["applied_boundary_conditions"][0][
            "variable"
        ] = "FLOW_002"
        rollout["tool_outputs"].extend(
            [
                {
                    "tool_call_id": "search-1",
                    "name": "search_pipeformer_registry",
                    "output": {
                        "success": True,
                        "variables": [{"variable": "FLOW_001"}],
                    },
                },
                first_output,
                second_output,
            ]
        )

        report = _autonomous_report(teacher_reference(), rollout)

        metric = report.metrics["registry_ordering"]
        self.assertFalse(metric.passed)
        self.assertEqual(metric.details["unauthorized_forecast_call_ids"], ["forecast-2"])

    def test_openclaw_filename_mention_without_read_or_computation_is_not_evidence(self):
        source = {"user_input": "Read requested.csv before answering."}
        rollout = openclaw_artifact_rollout()
        rollout["tool_outputs"][0]["output"].pop("content")
        rollout["final_answer"] = "I used requested.csv."

        report = _autonomous_report(source, rollout)

        metric = report.metrics["artifact_evidence"]
        self.assertTrue(metric.applicable)
        self.assertFalse(metric.passed)
        self.assertEqual(metric.details["missing_artifacts"], ["requested.csv"])

    def test_current_pipeformer_output_shape_remains_supported(self):
        rollout = successful_rollout()
        output = rollout["tool_outputs"][0]["output"]
        output["prediction_summary"] = output.pop("prediction")
        output["constraint_check"] = output.pop("verification")

        report = _autonomous_report(teacher_reference(), rollout)

        self.assertTrue(report.metrics["checkpoint_inference"].passed)
        self.assertTrue(report.metrics["forecast_horizon"].passed)
        self.assertTrue(report.metrics["constraint_execution"].passed)
        self.assertTrue(report.metrics["verification_completeness"].passed)


if __name__ == "__main__":
    unittest.main()
