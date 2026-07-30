from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipeclaw.task2_student.scripts.prepare_dataset import (
    generate_datasets,
    load_registered_tool_schemas,
    project_answer_only,
    project_constraint_multitask,
    project_trace_level,
)
from pipeclaw.task2_student.scripts.validate_dataset import (
    DatasetValidationError,
    validate_release,
)


FORECAST_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_pipeformer_forecast",
        "description": "Run a verified transient forecast.",
        "parameters": {
            "type": "object",
            "properties": {
                "disturbance_variable": {"type": "string"},
                "disturbance_magnitude_percent": {"type": "number"},
            },
            "required": [
                "disturbance_variable",
                "disturbance_magnitude_percent",
            ],
        },
    },
}


def _source_record() -> dict:
    return {
        "sample_id": "dataset:scenario_001_session_001::turn_001",
        "scenario_id": "scenario_001",
        "session_id": "scenario_001_session_001",
        "turn_id": 1,
        "scenario_type": "pipeformer",
        "state_before": {
            "schema_version": "verified_decision_state_v1",
            "candidates": [],
        },
        "recent_turns": [
            {
                "turn_id": 0,
                "user_input": "保持其余边界不变。",
                "assistant_output": "已记录。",
            }
        ],
        "user_input": "将 E_018:SNQ 上调 1e1%，并检查压力和管存约束。",
        "parsed_task": {
            "disturbance_variable": "E_018:SNQ",
            "disturbance_direction": "up",
            "disturbance_magnitude_percent": 10.0,
            "constraint_verification_types": ["pressure", "linepack"],
        },
        "tool_calls": [
            {
                "tool_call_id": "call_001",
                "name": "run_pipeformer_forecast",
                "arguments": {
                    "disturbance_variable": "E_018:SNQ",
                    "disturbance_magnitude_percent": 10.0,
                },
            }
        ],
        "tool_outputs": [
            {
                "tool_call_id": "call_001",
                "name": "run_pipeformer_forecast",
                "output": {
                    "success": True,
                    "prediction": {"forecast_horizon_minutes": 120},
                    "verification": {
                        "category_status": {
                            "pressure": "pass",
                            "linepack": "warning",
                        },
                        "overall_status": "warning",
                        "verification_complete": True,
                        "risk_level": "medium",
                        "failure_count": 0,
                        "warning_count": 1,
                        "failed_rule_ids": [],
                        "warning_rule_ids": ["linepack_decline_and_recovery"],
                        "triggered_flags": ["linepack_warning"],
                        "human_intervention_label": "monitoring_only",
                        "dispatch_recommendation": "Priority 3: restore linepack margin.",
                    },
                    "evidence": {
                        "top_watch_variables": [
                            {"variable": "H_002_v000", "mean_prediction": 1.053482}
                        ]
                    },
                },
            }
        ],
        "evidence": {
            "top_watch_variables": [
                {"variable": "H_002_v000", "mean_prediction": 1.053482}
            ],
            "supporting_numeric_values": [10.0, 1.053482],
        },
        "decision_summary": {
            "risk_level": "medium",
            "manual_intervention_label": "monitoring_only",
        },
        "final_answer": (
            "E_018:SNQ 按 +10% 情景完成校核；压力通过，管存告警，建议持续监测。"
        ),
    }


class ProjectionTests(unittest.TestCase):
    def test_answer_only_preserves_identity_and_exact_text(self) -> None:
        source = _source_record()

        projected = project_answer_only(source, "train")

        self.assertEqual(projected["example_id"], source["sample_id"])
        self.assertEqual(projected["source_sample_id"], source["sample_id"])
        self.assertEqual(projected["split"], "train")
        self.assertEqual(projected["projection"], "answer_only")
        self.assertEqual(
            projected["messages"],
            [
                {"role": "user", "content": source["user_input"]},
                {
                    "role": "assistant",
                    "content": source["final_answer"],
                    "loss": True,
                },
            ],
        )
        self.assertIn("1e1", projected["messages"][0]["content"])
        self.assertIn("E_018:SNQ", projected["messages"][1]["content"])

    def test_trace_uses_standard_agent_roles_and_injected_schema(self) -> None:
        source = _source_record()

        projected = project_trace_level(source, "train", [FORECAST_SCHEMA])

        self.assertEqual(
            [message["role"] for message in projected["messages"]],
            ["system", "user", "tool_call", "tool_response", "assistant"],
        )
        self.assertEqual(json.loads(projected["tools"]), [FORECAST_SCHEMA])
        self.assertIn(
            '"disturbance_variable":"E_018:SNQ"',
            projected["messages"][2]["content"],
        )
        self.assertTrue(projected["messages"][2]["loss"])
        self.assertFalse(projected["messages"][3]["loss"])
        self.assertTrue(projected["messages"][4]["loss"])
        self.assertIn("保持其余边界不变", projected["messages"][0]["content"])
        self.assertEqual(
            json.loads(projected["messages"][3]["content"]),
            source["tool_outputs"][0]["output"],
        )

    def test_trace_rejects_failed_or_unregistered_tool_targets(self) -> None:
        failed = _source_record()
        failed["tool_outputs"][0]["output"]["success"] = False
        with self.assertRaisesRegex(DatasetValidationError, "successful"):
            project_trace_level(failed, "train", [FORECAST_SCHEMA])

        unknown = _source_record()
        unknown["tool_calls"][0]["name"] = "invented_tool"
        unknown["tool_outputs"][0]["name"] = "invented_tool"
        with self.assertRaisesRegex(DatasetValidationError, "registered"):
            project_trace_level(unknown, "train", [FORECAST_SCHEMA])

    def test_constraint_projection_emits_five_structured_tasks(self) -> None:
        source = _source_record()

        examples = project_constraint_multitask(
            source,
            "train",
            [FORECAST_SCHEMA],
        )

        self.assertEqual(
            [example["task_type"] for example in examples],
            [
                "condition_parsing",
                "tool_planning",
                "constraint_judgment",
                "evidence_extraction",
                "answer_generation",
            ],
        )
        self.assertEqual(
            [example["example_id"] for example in examples],
            [
                f"{source['sample_id']}::condition_parsing",
                f"{source['sample_id']}::tool_planning",
                f"{source['sample_id']}::constraint_judgment",
                f"{source['sample_id']}::evidence_extraction",
                f"{source['sample_id']}::answer_generation",
            ],
        )
        by_task = {example["task_type"]: example for example in examples}
        self.assertEqual(
            json.loads(by_task["condition_parsing"]["messages"][-1]["content"]),
            source["parsed_task"],
        )
        judgment = json.loads(
            by_task["constraint_judgment"]["messages"][-1]["content"]
        )
        self.assertEqual(
            judgment["judgments"][0]["human_intervention_label"],
            "monitoring_only",
        )
        self.assertEqual(
            json.loads(by_task["evidence_extraction"]["messages"][-1]["content"]),
            source["evidence"],
        )
        evidence_input = json.loads(
            by_task["evidence_extraction"]["messages"][-2]["content"]
        )
        self.assertNotIn("evidence", evidence_input["verified_context"])
        self.assertEqual(
            evidence_input["verified_context"]["tool_outputs"],
            source["tool_outputs"],
        )
        self.assertEqual(
            by_task["answer_generation"]["messages"][-1]["content"],
            source["final_answer"],
        )
        for example in examples:
            for message in example["messages"]:
                if message["role"] in {"assistant", "tool_call"}:
                    self.assertNotIn("<think>", message["content"].casefold())

    def test_constraint_projection_omits_only_empty_auxiliary_targets(self) -> None:
        source = copy.deepcopy(_source_record())
        source["parsed_task"] = {}
        source["tool_calls"] = []
        source["tool_outputs"] = []
        source["evidence"] = {}

        examples = project_constraint_multitask(
            source,
            "valid",
            [FORECAST_SCHEMA],
        )

        self.assertEqual(
            [example["task_type"] for example in examples],
            ["answer_generation"],
        )
        self.assertEqual(examples[0]["split"], "valid")
        self.assertEqual(examples[0]["source_sample_id"], source["sample_id"])


class DatasetGenerationTests(unittest.TestCase):
    def test_loads_exact_actual_pipeclaw_tool_registry(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]

        schemas = load_registered_tool_schemas(repo_root)

        self.assertEqual(
            [schema["function"]["name"] for schema in schemas],
            [
                "analyze_pipeline_topology",
                "edit_file",
                "read_file",
                "run_command",
                "run_pipeformer_forecast",
                "search_pipeformer_registry",
                "set_decision_policy",
                "write_file",
            ],
        )

    def test_generates_deterministic_valid_files_without_mutating_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            source_root = temporary_root / "source"
            output_root = temporary_root / "derived"
            manifest_path = output_root / "manifests" / "manifest.json"
            source_root.mkdir()
            for split_index, split in enumerate(("train", "valid", "test"), start=1):
                record = _source_record()
                record["sample_id"] = f"dataset:{split}::turn_001"
                record["scenario_id"] = f"scenario_{split_index:03d}"
                record["session_id"] = f"session_{split_index:03d}"
                path = source_root / f"teacher_trace_{split}.jsonl"
                path.write_text(
                    json.dumps(record, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            source_bytes_before = {
                path.name: path.read_bytes()
                for path in sorted(source_root.glob("*.jsonl"))
            }

            manifest = generate_datasets(
                source_root=source_root,
                output_root=output_root,
                manifest_path=manifest_path,
                expected_counts={"train": 1, "valid": 1, "test": 1},
                created_at="2026-07-30T00:00:00Z",
            )
            first_manifest_bytes = manifest_path.read_bytes()
            second_manifest = generate_datasets(
                source_root=source_root,
                output_root=output_root,
                manifest_path=manifest_path,
                expected_counts={"train": 1, "valid": 1, "test": 1},
                created_at="2026-07-30T00:00:00Z",
            )

            self.assertEqual(manifest, second_manifest)
            self.assertEqual(first_manifest_bytes, manifest_path.read_bytes())
            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in sorted(source_root.glob("*.jsonl"))
                },
                source_bytes_before,
            )
            self.assertEqual(
                {
                    path.relative_to(output_root).as_posix()
                    for path in output_root.glob("*/*.jsonl")
                },
                {
                    f"{projection}/{split}.jsonl"
                    for projection in (
                        "answer_only",
                        "trace_level",
                        "constraint_multitask",
                    )
                    for split in ("train", "valid", "test")
                },
            )
            self.assertEqual(manifest["schema_version"], "task2_ms_swift_manifest_v1")
            self.assertEqual(
                manifest["projections"]["answer_only"]["train"]["record_count"],
                1,
            )
            self.assertEqual(
                manifest["projections"]["trace_level"]["test"]["record_count"],
                1,
            )
            self.assertEqual(
                manifest["projections"]["constraint_multitask"]["valid"][
                    "record_count"
                ],
                5,
            )
            self.assertEqual(
                manifest["projections"]["constraint_multitask"]["valid"][
                    "task_counts"
                ],
                {
                    "answer_generation": 1,
                    "condition_parsing": 1,
                    "constraint_judgment": 1,
                    "evidence_extraction": 1,
                    "tool_planning": 1,
                },
            )
            for projection in manifest["projections"].values():
                for split_details in projection.values():
                    self.assertEqual(len(split_details["sha256"]), 64)
            validated = validate_release(
                source_root=source_root,
                output_root=output_root,
                manifest_path=manifest_path,
                expected_counts={"train": 1, "valid": 1, "test": 1},
                registered_tool_names={
                    "analyze_pipeline_topology",
                    "edit_file",
                    "read_file",
                    "run_command",
                    "run_pipeformer_forecast",
                    "search_pipeformer_registry",
                    "set_decision_policy",
                    "write_file",
                },
            )
            self.assertEqual(validated["validated_projection_files"], 9)

            tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tampered_manifest["tool_schemas"]["sha256"] = "0" * 64
            manifest_path.write_text(
                json.dumps(tampered_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DatasetValidationError,
                "tool schema checksum",
            ):
                validate_release(
                    source_root=source_root,
                    output_root=output_root,
                    manifest_path=manifest_path,
                    expected_counts={"train": 1, "valid": 1, "test": 1},
                    registered_tool_names={
                        "analyze_pipeline_topology",
                        "edit_file",
                        "read_file",
                        "run_command",
                        "run_pipeformer_forecast",
                        "search_pipeformer_registry",
                        "set_decision_policy",
                        "write_file",
                    },
                )
            manifest_path.write_bytes(first_manifest_bytes)

            answer_path = output_root / "answer_only" / "train.jsonl"
            answer_bytes = answer_path.read_bytes()
            answer_record = json.loads(answer_bytes.decode("utf-8"))
            answer_record["messages"][-1]["content"] = "tampered answer"
            answer_path.write_text(
                json.dumps(
                    answer_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            tampered_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            tampered_manifest["projections"]["answer_only"]["train"]["sha256"] = (
                hashlib.sha256(answer_path.read_bytes()).hexdigest()
            )
            manifest_path.write_text(
                json.dumps(tampered_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetValidationError, "final answer changed"):
                validate_release(
                    source_root=source_root,
                    output_root=output_root,
                    manifest_path=manifest_path,
                    expected_counts={"train": 1, "valid": 1, "test": 1},
                    registered_tool_names={
                        "analyze_pipeline_topology",
                        "edit_file",
                        "read_file",
                        "run_command",
                        "run_pipeformer_forecast",
                        "search_pipeformer_registry",
                        "set_decision_policy",
                        "write_file",
                    },
                )
            answer_path.write_bytes(answer_bytes)
            manifest_path.write_bytes(first_manifest_bytes)

            trace_path = output_root / "trace_level" / "train.jsonl"
            trace_path.write_bytes(b" " + trace_path.read_bytes())
            with self.assertRaisesRegex(DatasetValidationError, "checksum"):
                validate_release(
                    source_root=source_root,
                    output_root=output_root,
                    manifest_path=manifest_path,
                    expected_counts={"train": 1, "valid": 1, "test": 1},
                    registered_tool_names={
                        "analyze_pipeline_topology",
                        "edit_file",
                        "read_file",
                        "run_command",
                        "run_pipeformer_forecast",
                        "search_pipeformer_registry",
                        "set_decision_policy",
                        "write_file",
                    },
                )


if __name__ == "__main__":
    unittest.main()
