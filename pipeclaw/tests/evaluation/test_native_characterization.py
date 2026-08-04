"""Characterize the native teacher-trace evaluator before its refactor."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from evaluator.scorer import NativeTraceEvaluator
from fixtures import assumed_disturbance_record, passing_teacher_record


class NativeCharacterizationTests(unittest.TestCase):
    def test_assumed_disturbance_uses_executed_prediction(self):
        result = NativeTraceEvaluator().evaluate(assumed_disturbance_record())
        checks = {item["name"]: item for item in result["checks"]}
        self.assertEqual(checks["disturbance_applied_correctly"]["status"], "pass")

    def test_quality_aliases_are_stable(self):
        result = NativeTraceEvaluator().evaluate(passing_teacher_record())
        self.assertIn("quality_score", result)
        self.assertIn(result["quality_flag"], {"pass", "needs_review"})
