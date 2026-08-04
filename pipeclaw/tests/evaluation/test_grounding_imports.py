import unittest
from pathlib import Path


class GroundingImportTests(unittest.TestCase):
    def test_runtime_grounding_imports(self):
        from pipeclaw.backend.grounding.contract import GroundingContractBuilder
        from pipeclaw.backend.grounding.decision_policy import normalize_decision_policy
        from pipeclaw.backend.grounding.evidence.tool import classify_tool_evidence

        self.assertTrue(callable(normalize_decision_policy))
        self.assertTrue(callable(classify_tool_evidence))
        self.assertIsNotNone(GroundingContractBuilder)

    def test_runtime_modules_are_not_imported_from_evaluator(self):
        forbidden = tuple(
            "evaluator." + module
            for module in (
                "grounding_contract",
                "decision_policy",
                "decision_trace_state",
                "tool_evidence",
            )
        )
        for path in Path("pipeclaw").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, str(path))


if __name__ == "__main__":
    unittest.main()
