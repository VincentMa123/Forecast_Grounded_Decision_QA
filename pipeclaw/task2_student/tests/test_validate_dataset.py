from __future__ import annotations

import copy
import json
import unittest

from pipeclaw.task2_student.scripts.validate_dataset import (
    DatasetValidationError,
    validate_projection_records,
    validate_source_records,
)


def _source_record(sample_id: str = "source::turn_001") -> dict:
    return {
        "sample_id": sample_id,
        "scenario_id": "scenario_001",
        "session_id": "session_001",
        "turn_id": 1,
        "scenario_type": "pipeformer",
        "state_before": {},
        "recent_turns": [],
        "user_input": "将 E_018:SNQ 上调 1e1%，并检查约束。",
        "parsed_task": {"disturbance_variable": "E_018:SNQ"},
        "tool_calls": [],
        "tool_outputs": [],
        "evidence": {},
        "decision_summary": {},
        "final_answer": "E_018:SNQ 已按 +10% 情景完成校核。",
    }


def _trace_record(split: str = "train") -> dict:
    return {
        "example_id": "source::turn_001",
        "source_sample_id": "source::turn_001",
        "scenario_id": "scenario_001",
        "session_id": "session_001",
        "turn_id": 1,
        "scenario_type": "pipeformer",
        "split": split,
        "projection": "trace_level",
        "tools": json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file.",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ]
        ),
        "messages": [
            {"role": "system", "content": "Use verified evidence only."},
            {"role": "user", "content": "读取数据。"},
            {
                "role": "tool_call",
                "content": json.dumps(
                    {"name": "read_file", "arguments": {"path": "input.csv"}}
                ),
                "loss": True,
            },
            {
                "role": "tool_response",
                "content": json.dumps({"success": True, "content": "数据"}),
                "loss": False,
            },
            {"role": "assistant", "content": "读取完成。", "loss": True},
        ],
    }


def _constraint_record(content: str) -> dict:
    record = {
        key: value
        for key, value in _trace_record().items()
        if key != "tools"
    }
    record["example_id"] = "source::turn_001::condition_parsing"
    record["projection"] = "constraint_multitask"
    record["task_type"] = "condition_parsing"
    record["messages"] = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "解析条件。"},
        {"role": "assistant", "content": content, "loss": True},
    ]
    return record


class SourceValidationTests(unittest.TestCase):
    def test_accepts_complete_unique_source_records(self) -> None:
        validate_source_records([_source_record()], split="train", expected_count=1)

    def test_rejects_duplicate_source_sample_ids(self) -> None:
        duplicate = _source_record()

        with self.assertRaisesRegex(DatasetValidationError, "duplicate sample_id"):
            validate_source_records(
                [_source_record(), duplicate],
                split="train",
                expected_count=2,
            )

    def test_rejects_source_count_mismatch(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "expected 2 records"):
            validate_source_records([_source_record()], split="train", expected_count=2)


class ProjectionValidationTests(unittest.TestCase):
    def test_accepts_well_formed_trace_projection(self) -> None:
        validate_projection_records(
            [_trace_record()],
            projection="trace_level",
            split="train",
            registered_tool_names={"read_file"},
        )

    def test_rejects_changed_or_leaked_split(self) -> None:
        record = _trace_record(split="test")

        with self.assertRaisesRegex(DatasetValidationError, "split"):
            validate_projection_records(
                [record],
                projection="trace_level",
                split="train",
                registered_tool_names={"read_file"},
            )

    def test_rejects_duplicate_derived_example_ids(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "duplicate example_id"):
            validate_projection_records(
                [_trace_record(), copy.deepcopy(_trace_record())],
                projection="trace_level",
                split="train",
                registered_tool_names={"read_file"},
            )

    def test_rejects_unregistered_tool_call(self) -> None:
        record = _trace_record()
        record["messages"][2]["content"] = json.dumps(
            {"name": "invented_tool", "arguments": {}}
        )

        with self.assertRaisesRegex(DatasetValidationError, "unregistered tool"):
            validate_projection_records(
                [record],
                projection="trace_level",
                split="train",
                registered_tool_names={"read_file"},
            )

    def test_rejects_tool_call_without_matching_response(self) -> None:
        record = _trace_record()
        del record["messages"][3]

        with self.assertRaisesRegex(DatasetValidationError, "tool_response"):
            validate_projection_records(
                [record],
                projection="trace_level",
                split="train",
                registered_tool_names={"read_file"},
            )

    def test_rejects_failed_tool_response(self) -> None:
        record = _trace_record()
        record["messages"][3]["content"] = json.dumps(
            {"success": False, "error": "failed"}
        )

        with self.assertRaisesRegex(DatasetValidationError, "successful"):
            validate_projection_records(
                [record],
                projection="trace_level",
                split="train",
                registered_tool_names={"read_file"},
            )

    def test_rejects_wrong_loss_flags(self) -> None:
        for message_index, expected_error in (
            (2, "tool_call must receive loss"),
            (3, "tool_response must not receive loss"),
            (4, "assistant must receive loss"),
        ):
            with self.subTest(message_index=message_index):
                record = _trace_record()
                record["messages"][message_index]["loss"] = not record["messages"][
                    message_index
                ]["loss"]
                with self.assertRaisesRegex(DatasetValidationError, expected_error):
                    validate_projection_records(
                        [record],
                        projection="trace_level",
                        split="train",
                        registered_tool_names={"read_file"},
                    )

    def test_rejects_missing_final_answer(self) -> None:
        record = _trace_record()
        record["messages"][-1]["content"] = ""

        with self.assertRaisesRegex(DatasetValidationError, "final assistant"):
            validate_projection_records(
                [record],
                projection="trace_level",
                split="train",
                registered_tool_names={"read_file"},
            )

    def test_rejects_invalid_structured_auxiliary_target(self) -> None:
        with self.assertRaisesRegex(
            DatasetValidationError,
            "structured assistant target",
        ):
            validate_projection_records(
                [_constraint_record("{invalid")],
                projection="constraint_multitask",
                split="train",
                registered_tool_names={"read_file"},
            )


if __name__ == "__main__":
    unittest.main()
