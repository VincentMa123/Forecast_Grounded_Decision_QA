"""Contract tests for the shared evaluation engine."""

from __future__ import annotations

import json
import unittest
from enum import Enum
from pathlib import Path

from pipeclaw.backend.evaluator import (
    AutonomousRolloutAdapter,
    EvaluationInputError,
    EvaluationProfile,
    MetricResult,
    TeacherTraceAdapter,
    build_teacher_oracle,
    evaluate,
    summarize,
)
from pipeclaw.backend.evaluator.engine import build_report
from pipeclaw.backend.evaluator.profiles import get_profile_policy
from pipeclaw.tests.evaluation.fixtures import (
    successful_rollout,
    teacher_reference,
)


class _DetailState(str, Enum):
    READY = "ready"


class EvaluationEngineTests(unittest.TestCase):
    def test_score_uses_only_applicable_included_metrics(self):
        metrics = [
            MetricResult("a", True, True, 2.0, True, True, {}),
            MetricResult("b", True, False, 1.0, False, True, {}),
            MetricResult("diagnostic", True, False, 100.0, True, False, {}),
            MetricResult("unused", False, False, 100.0, True, True, {}),
        ]

        report = build_report(EvaluationProfile.AUTONOMOUS_ROLLOUT, metrics)

        self.assertAlmostEqual(report.overall_score, 66.666667)
        self.assertTrue(report.hard_gate_passed)
        self.assertTrue(report.passed)
        self.assertEqual(report.failed_checks, ("b",))

    def test_no_applicable_metrics_fails_closed(self):
        report = build_report(EvaluationProfile.AUTONOMOUS_ROLLOUT, [])

        self.assertIsNone(report.overall_score)
        self.assertTrue(report.hard_gate_passed)
        self.assertFalse(report.passed)

    def test_autonomous_pass_uses_critical_gate_not_score_threshold(self):
        metrics = [
            MetricResult("task_parsing", True, True, 1.0, True, True, {}),
            MetricResult("answer_style", True, False, 9.0, False, True, {}),
        ]

        report = build_report(EvaluationProfile.AUTONOMOUS_ROLLOUT, metrics)

        self.assertEqual(report.overall_score, 10.0)
        self.assertTrue(report.passed)

    def test_applicable_critical_failure_closes_hard_gate(self):
        metrics = [
            MetricResult("task_parsing", True, False, 1.0, True, True, {}),
        ]

        report = build_report(EvaluationProfile.AUTONOMOUS_ROLLOUT, metrics)

        self.assertFalse(report.hard_gate_passed)
        self.assertFalse(report.passed)
        self.assertEqual(report.critical_failures, ("task_parsing",))

    def test_hard_input_issues_close_gate_and_are_diagnostic(self):
        metrics = [
            MetricResult("task_parsing", True, True, 1.0, True, True, {}),
        ]

        report = build_report(
            EvaluationProfile.AUTONOMOUS_ROLLOUT,
            metrics,
            hard_issues=["invalid_source"],
        )

        self.assertFalse(report.hard_gate_passed)
        self.assertFalse(report.passed)
        self.assertEqual(report.diagnostics["hard_issues"], ("invalid_source",))

    def test_teacher_pass_requires_default_or_overridden_threshold(self):
        metrics = [
            MetricResult("task_parsing", True, True, 80.0, True, True, {}),
            MetricResult("record_contract", True, False, 20.0, False, True, {}),
        ]

        default_report = build_report(EvaluationProfile.TEACHER_TRACE, metrics)
        overridden_report = build_report(
            EvaluationProfile.TEACHER_TRACE,
            metrics,
            minimum_score=75.0,
        )

        self.assertEqual(default_report.overall_score, 80.0)
        self.assertFalse(default_report.passed)
        self.assertTrue(overridden_report.passed)


class EvaluationModelTests(unittest.TestCase):
    def test_metric_and_report_serialization_are_strict_json(self):
        metric = MetricResult(
            "task_parsing",
            True,
            True,
            1.0,
            True,
            True,
            {
                "states": {_DetailState.READY},
                "path": Path("artifacts/result.json"),
                "coordinates": (1, 2),
            },
        )
        report = build_report(
            EvaluationProfile.AUTONOMOUS_ROLLOUT,
            [metric],
            diagnostics={"labels": ("one", "two")},
        )

        metric_payload = metric.to_dict()
        report_payload = report.to_dict()

        json.dumps(metric_payload, allow_nan=False)
        json.dumps(report_payload, allow_nan=False)
        self.assertEqual(metric_payload["details"]["states"], ["ready"])
        self.assertEqual(report_payload["profile"], "autonomous_rollout")
        self.assertEqual(report_payload["failed_checks"], [])


class EvaluationAdapterAndOracleTests(unittest.TestCase):
    def test_teacher_adapter_normalizes_a_native_record(self):
        source = teacher_reference()

        context = TeacherTraceAdapter().adapt(source)

        self.assertEqual(context.profile, EvaluationProfile.TEACHER_TRACE)
        self.assertEqual(context.record["sample_id"], "sample-1")
        self.assertEqual(context.oracle, {})
        self.assertIsNot(context.record, source)

    def test_autonomous_adapter_requires_teacher_reference(self):
        with self.assertRaises(EvaluationInputError):
            AutonomousRolloutAdapter().adapt(successful_rollout(), reference=None)

    def test_autonomous_adapter_omitted_reference_raises_typed_error(self):
        with self.assertRaises(EvaluationInputError):
            AutonomousRolloutAdapter().adapt(successful_rollout())

    def test_autonomous_adapter_builds_teacher_oracle(self):
        reference = teacher_reference()

        context = AutonomousRolloutAdapter().adapt(
            successful_rollout(),
            reference=reference,
        )

        self.assertEqual(context.profile, EvaluationProfile.AUTONOMOUS_ROLLOUT)
        self.assertEqual(context.oracle["task"]["case_id"], "case-1")
        self.assertEqual(context.oracle["risk_level"], "low")
        self.assertIsNot(context.reference, reference)

    def test_teacher_oracle_extracts_targets_without_score_fields(self):
        oracle = build_teacher_oracle(teacher_reference())

        self.assertEqual(oracle["required_constraints"], ["pressure"])
        self.assertEqual(
            oracle["teacher_tool_names"],
            ["search_pipeformer_registry", "run_pipeformer_forecast"],
        )
        self.assertNotIn("overall_score", oracle)
        self.assertNotIn("metrics", oracle)

    def test_public_evaluate_raises_typed_error_without_reference(self):
        with self.assertRaises(EvaluationInputError):
            evaluate(
                successful_rollout(),
                profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
                metrics=[],
            )


class EvaluationProfileTests(unittest.TestCase):
    def test_teacher_policy_retains_native_pipeformer_weights(self):
        policy = get_profile_policy(EvaluationProfile.TEACHER_TRACE)

        self.assertEqual(policy.metric("task_parsing").weight, 10.0)
        self.assertEqual(policy.metric("tool_call").weight, 15.0)
        self.assertEqual(policy.metric("record_contract").weight, 5.0)
        self.assertTrue(policy.metric("registry_ordering").critical)
        self.assertFalse(policy.metric("record_contract").critical)
        self.assertEqual(policy.minimum_score, 85.0)

    def test_teacher_policy_retains_native_generic_weights(self):
        policy = get_profile_policy(
            EvaluationProfile.TEACHER_TRACE,
            teacher_variant="generic",
        )

        self.assertEqual(policy.metric("trace_completed").weight, 25.0)
        self.assertEqual(policy.metric("answer_completeness").weight, 25.0)
        self.assertEqual(policy.metric("tool_call").weight, 20.0)
        self.assertEqual(policy.metric("record_contract").weight, 10.0)

    def test_autonomous_policy_equal_weights_and_diagnostics(self):
        policy = get_profile_policy(EvaluationProfile.AUTONOMOUS_ROLLOUT)

        task = policy.metric("task_parsing")
        recovery = policy.metric("tool_recovery")

        self.assertEqual(task.weight, 1.0)
        self.assertTrue(task.critical)
        self.assertTrue(task.included_in_score)
        self.assertEqual(recovery.weight, 0.0)
        self.assertFalse(recovery.critical)
        self.assertFalse(recovery.included_in_score)
        self.assertFalse(policy.metric("portability").included_in_score)
        self.assertFalse(policy.metric("hallucination").included_in_score)
        self.assertIsNone(policy.minimum_score)


class EvaluationAggregationTests(unittest.TestCase):
    def test_summary_uses_only_applicable_metric_denominators(self):
        passing = build_report(
            EvaluationProfile.AUTONOMOUS_ROLLOUT,
            [MetricResult("risk", True, True, 1.0, True, True, {})],
        )
        inapplicable = build_report(
            EvaluationProfile.AUTONOMOUS_ROLLOUT,
            [MetricResult("risk", False, False, 1.0, True, True, {})],
        )

        summary = summarize([passing, inapplicable])

        self.assertEqual(summary["schema_version"], "pipeclaw_evaluation_v2")
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["metrics"]["risk"]["numerator"], 1)
        self.assertEqual(summary["metrics"]["risk"]["denominator"], 1)
        self.assertEqual(summary["metrics"]["risk"]["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
