"""Characterize autonomous oracle metrics before their refactor."""

import unittest

from fixtures import (
    malformed_then_retried_rollout,
    openclaw_artifact_rollout,
    reference_without_risk,
    successful_rollout,
    teacher_reference,
)
from pipeclaw.task2_student.evaluator.oracle_metrics import aggregate_results, evaluate_rollout


class RolloutCharacterizationTests(unittest.TestCase):
    def test_successful_rollout_has_applicable_tool_metric(self):
        metrics = evaluate_rollout(teacher_reference(), successful_rollout())
        self.assertTrue(metrics["tool_call"]["applicable"])
        self.assertTrue(metrics["tool_call"]["record_pass"])

    def test_inapplicable_metric_does_not_enter_denominator(self):
        summary = aggregate_results(
            [{"metrics": evaluate_rollout(reference_without_risk(), successful_rollout())}]
        )
        self.assertEqual(summary["metrics"]["risk"]["denominator"], 0)

    def test_successful_retry_recovers_from_malformed_arguments(self):
        metrics = evaluate_rollout(teacher_reference(), malformed_then_retried_rollout())
        self.assertFalse(metrics["tool_call"]["record_pass"])
        self.assertTrue(metrics["tool_recovery"]["applicable"])
        self.assertTrue(metrics["tool_recovery"]["record_pass"])

    def test_openclaw_artifact_read_is_evidence(self):
        source = {"user_input": "Read requested.csv before answering."}
        metrics = evaluate_rollout(source, openclaw_artifact_rollout())
        self.assertTrue(metrics["artifact_evidence"]["applicable"])
        self.assertTrue(metrics["artifact_evidence"]["record_pass"])
