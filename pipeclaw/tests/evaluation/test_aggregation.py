"""Regression tests for canonical aggregation and compatibility aliases."""

from __future__ import annotations

import unittest
import sys
import subprocess
from pathlib import Path

from pipeclaw.backend.evaluator import EvaluationProfile, evaluate, summarize
from pipeclaw.backend.evaluator.scorer import NativeTraceEvaluator
from pipeclaw.tests.evaluation.fixtures import (
    passing_teacher_record,
    successful_rollout,
    teacher_reference,
)


class DerivedHallucinationTests(unittest.TestCase):
    def test_hallucination_is_an_unscored_copy_of_evidence_consistency(self):
        report = evaluate(
            successful_rollout(),
            profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
            reference=teacher_reference(),
        )

        payload = report.to_dict()
        evidence = payload["metrics"]["evidence_consistency"]
        hallucination = payload["metrics"]["hallucination"]

        self.assertEqual(hallucination["derived_from"], "evidence_consistency")
        self.assertFalse(hallucination["included_in_score"])
        for key in ("applicable", "passed", "weight", "critical", "details"):
            self.assertEqual(hallucination[key], evidence[key])
        self.assertNotIn("hallucination", report.failed_checks)

    def test_dataset_hallucination_rate_equals_evidence_failure_rate(self):
        passing = evaluate(
            successful_rollout(),
            profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
            reference=teacher_reference(),
        )
        failing_rollout = successful_rollout()
        failing_rollout["final_answer"] = "The forecast returned 987654.321."
        failing = evaluate(
            failing_rollout,
            profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
            reference=teacher_reference(),
        )

        summary = summarize([passing, failing])

        self.assertEqual(
            summary["hallucination_rate"],
            summary["metrics"]["evidence_consistency"]["failure_rate"],
        )
        self.assertNotIn("hallucination", summary["metrics"])
        self.assertEqual(
            summary["diagnostics"]["hallucination"]["failure_rate"],
            summary["hallucination_rate"],
        )


class NativeFacadeTests(unittest.TestCase):
    def test_native_facade_serializes_canonical_report_and_legacy_aliases(self):
        result = NativeTraceEvaluator().evaluate(passing_teacher_record())

        self.assertEqual(result["schema_version"], "pipeclaw_evaluation_v2")
        self.assertEqual(result["quality_score"], result["overall_score"])
        self.assertEqual(result["quality_flag"], "pass")
        self.assertEqual(result["quality_failed_checks"], result["failed_checks"])
        self.assertTrue(result["parsed_task_correct"])
        self.assertTrue(result["forecast_tool_succeeded"])
        self.assertTrue(result["answer_grounded"])
        self.assertIn("task_parsing", result["metrics"])

    def test_native_facade_works_through_package_import_without_backend_path(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pipeclaw.backend.evaluator.scorer import NativeTraceEvaluator; "
                    "from pipeclaw.tests.evaluation.fixtures import passing_teacher_record; "
                    "assert NativeTraceEvaluator().evaluate(passing_teacher_record())"
                    "['quality_flag'] == 'pass'"
                ),
            ],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_native_facade_imports_from_direct_backend_execution_path(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, 'backend'); import evaluator.scorer",
            ],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
