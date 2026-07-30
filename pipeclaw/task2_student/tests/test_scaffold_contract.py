from __future__ import annotations

import unittest
from pathlib import Path


TASK2_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK2_ROOT.parents[1]


class Task2ScaffoldContractTests(unittest.TestCase):
    def test_required_scaffold_paths_exist(self) -> None:
        expected = [
            TASK2_ROOT / "README.md",
            TASK2_ROOT / "requirements.txt",
            TASK2_ROOT / "configs" / "README.md",
            TASK2_ROOT / "data" / "README.md",
            TASK2_ROOT / "data" / "answer_only",
            TASK2_ROOT / "data" / "trace_level",
            TASK2_ROOT / "data" / "constraint_multitask",
            TASK2_ROOT / "data" / "manifests",
            TASK2_ROOT / "scripts" / "README.md",
            TASK2_ROOT / "outputs",
        ]
        self.assertEqual(
            [str(path.relative_to(REPO_ROOT)) for path in expected if not path.exists()],
            [],
        )

    def test_readme_names_authoritative_splits_and_models(self) -> None:
        readme = (TASK2_ROOT / "README.md").read_text(encoding="utf-8")
        for expected in (
            "teacher_trace_train.jsonl",
            "teacher_trace_valid.jsonl",
            "teacher_trace_test.jsonl",
            "902 / 124 / 114",
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen3.5-9B",
            "MS-SWIFT",
        ):
            self.assertIn(expected, readme)

    def test_ms_swift_dependency_stays_on_compatible_major_version(self) -> None:
        requirements = (TASK2_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertEqual(requirements.strip(), "ms-swift>=4.0,<5.0")


if __name__ == "__main__":
    unittest.main()
