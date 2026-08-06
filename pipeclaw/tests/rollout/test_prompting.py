"""Tests for prompt construction and teacher-future suppression."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeclaw.backend.agent.prompt_builder import PromptBuilder
from pipeclaw.task2_student.rollout.prompting import (
    PromptCaseBuilder,
    parse_tool_schemas,
    strip_teacher_future_messages,
)


SEARCH_SCHEMA = {
    "type": "function",
    "function": {"name": "search", "parameters": {"type": "object"}},
}

TEACHER_MESSAGES = [
    {"role": "system", "content": "trace system prompt"},
    {"role": "user", "content": "earlier request"},
    {"role": "assistant", "content": "earlier answer"},
    {"role": "user", "content": "current request"},
    {"role": "tool_call", "content": '{"name": "search", "arguments": {}}'},
    {"role": "tool_response", "content": '{"success": true}'},
    {"role": "assistant", "content": "TEACHER ANSWER"},
]


class StubPromptBuilder:
    """Stand in for the production PromptBuilder without touching a workspace."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return "BUILT SYSTEM PROMPT"


class StripTeacherFutureMessagesTests(unittest.TestCase):
    def test_keeps_context_through_the_last_user_turn(self) -> None:
        kept = strip_teacher_future_messages(TEACHER_MESSAGES)

        self.assertEqual(
            [message["role"] for message in kept],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(kept[-1]["content"], "current request")

    def test_does_not_mutate_the_source_messages(self) -> None:
        kept = strip_teacher_future_messages(TEACHER_MESSAGES)
        kept[0]["content"] = "mutated"

        self.assertEqual(TEACHER_MESSAGES[0]["content"], "trace system prompt")

    def test_without_a_user_turn_everything_is_kept(self) -> None:
        messages = [{"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]

        self.assertEqual(strip_teacher_future_messages(messages), messages)


class ParseToolSchemasTests(unittest.TestCase):
    def test_none_yields_no_schemas(self) -> None:
        self.assertEqual(parse_tool_schemas(None), [])

    def test_json_string_is_decoded(self) -> None:
        self.assertEqual(parse_tool_schemas(json.dumps([SEARCH_SCHEMA])), [SEARCH_SCHEMA])

    def test_single_mapping_is_wrapped(self) -> None:
        self.assertEqual(parse_tool_schemas(SEARCH_SCHEMA), [SEARCH_SCHEMA])

    def test_malformed_json_yields_no_schemas(self) -> None:
        self.assertEqual(parse_tool_schemas("{not json"), [])

    def test_non_mapping_items_are_dropped(self) -> None:
        self.assertEqual(parse_tool_schemas([SEARCH_SCHEMA, 3]), [SEARCH_SCHEMA])


class PromptCaseBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = PromptCaseBuilder()
        self.prompt_builder = StubPromptBuilder()
        self.workspace = Path("workspaces") / "sample" / "workspace-autonomous-evaluation"

    def build(self, record: Mapping[str, Any], **kwargs: Any):
        return self.builder.build(
            record,
            workspace_root=self.workspace,
            prompt_builder=self.prompt_builder,
            **kwargs,
        )

    def test_prompt_uses_runner_workspace_environment_name(self) -> None:
        prompt = PromptBuilder(self.workspace).build(memory_payload={})

        self.assertIn("WORKSPACE_DIR", prompt)
        self.assertNotIn("WORKSPACE_ROOT", prompt)

    def test_teacher_future_messages_are_hidden(self) -> None:
        case = self.build(
            {
                "sample_id": "s1",
                "scenario_id": "sc1",
                "scenario_type": "pipeformer",
                "messages": TEACHER_MESSAGES,
                "tools": [SEARCH_SCHEMA],
            }
        )

        serialized = json.dumps(case.messages)
        self.assertNotIn("TEACHER ANSWER", serialized)
        self.assertNotIn("tool_response", serialized)
        self.assertEqual(case.messages[-1], {"role": "user", "content": "current request"})

    def test_prompt_builder_system_prompt_replaces_the_trace_prompt(self) -> None:
        case = self.build(
            {
                "sample_id": "s1",
                "messages": TEACHER_MESSAGES,
                "state_before": {"verified": True},
                "recent_turns": [{"role": "user", "content": "x"}],
            }
        )

        self.assertEqual(case.messages[0], {"role": "system", "content": "BUILT SYSTEM PROMPT"})
        self.assertEqual(
            [message["content"] for message in case.messages if message["role"] == "system"],
            ["BUILT SYSTEM PROMPT"],
        )
        built = self.prompt_builder.calls[0]
        self.assertEqual(built["verified_state"], {"verified": True})
        self.assertEqual(built["recent_turns"], [{"role": "user", "content": "x"}])

    def test_system_prompt_is_inserted_when_the_trace_has_none(self) -> None:
        case = self.build(
            {"sample_id": "s1", "messages": [{"role": "user", "content": "ask"}]}
        )

        self.assertEqual(
            [message["role"] for message in case.messages], ["system", "user"]
        )

    def test_user_input_field_is_used_without_messages(self) -> None:
        case = self.build({"sample_id": "s1", "user_input": "direct question"})

        self.assertEqual(
            case.messages,
            [
                {"role": "system", "content": "BUILT SYSTEM PROMPT"},
                {"role": "user", "content": "direct question"},
            ],
        )

    def test_user_input_is_recovered_from_messages(self) -> None:
        case = self.build({"sample_id": "s1", "messages": TEACHER_MESSAGES})

        self.assertEqual(case.messages[-1]["content"], "current request")

    def test_missing_user_input_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            self.build({"sample_id": "s1", "messages": [{"role": "system", "content": "s"}]})

    def test_non_mapping_record_is_a_type_error(self) -> None:
        with self.assertRaises(TypeError):
            self.build(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_tool_schemas_override_the_record_field(self) -> None:
        override = [{"type": "function", "function": {"name": "other"}}]

        case = self.build(
            {"sample_id": "s1", "user_input": "ask", "tools": [SEARCH_SCHEMA]},
            tool_schemas=override,
        )

        self.assertEqual(case.tools, override)

    def test_identity_fields_and_workspace_are_preserved(self) -> None:
        record = {
            "sample_id": "s1",
            "scenario_id": "sc1",
            "scenario_type": "openclaw",
            "user_input": "ask",
        }

        case = self.build(record)

        self.assertEqual(case.sample_id, "s1")
        self.assertEqual(case.scenario_id, "sc1")
        self.assertEqual(case.scenario_type, "openclaw")
        self.assertEqual(case.workspace_root, self.workspace)
        self.assertEqual(case.source_record, record)

    def test_example_id_is_accepted_as_the_sample_id(self) -> None:
        case = self.build({"example_id": "e9", "user_input": "ask"})

        self.assertEqual(case.sample_id, "e9")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
