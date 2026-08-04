"""Tests for tool-call parsing, schema handling, and allow-listed dispatch."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from pipeclaw.task2_student.rollout.models import ToolCall
from pipeclaw.task2_student.rollout.tools import (
    ToolDispatcher,
    append_tool_exchange,
    coerce_schema_value,
    parse_tool_calls,
    schema_error,
)


SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "deep": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class RecordingRegistry:
    def __init__(self, result: Any = None) -> None:
        self.result = result if result is not None else {"success": True}
        self.executions: list[tuple[str, dict[str, Any]]] = []

    def execute(self, name: str, **kwargs: Any) -> Any:
        self.executions.append((name, dict(kwargs)))
        return self.result


class ParseToolCallsTests(unittest.TestCase):
    def test_plain_text_is_a_final_answer(self) -> None:
        text, calls, errors = parse_tool_calls("just an answer")

        self.assertEqual(text, "just an answer")
        self.assertEqual(calls, [])
        self.assertEqual(errors, [])

    def test_tagged_json_call(self) -> None:
        response = 'prefix <tool_call>{"name": "search", "arguments": {"query": "x"}}</tool_call>'

        text, calls, errors = parse_tool_calls(response)

        self.assertEqual(text, "prefix")
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "search")
        self.assertEqual(calls[0].arguments, {"query": "x"})

    def test_openai_style_message_object(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "native-7",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "x"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

        text, calls, errors = parse_tool_calls(response)

        self.assertEqual(text, "")
        self.assertEqual(errors, [])
        self.assertEqual(calls[0].call_id, "native-7")
        self.assertEqual(calls[0].arguments, {"query": "x"})

    def test_qwen_text_wins_over_duplicate_native_call(self) -> None:
        response = {
            "message": {
                "content": (
                    "<function=search>"
                    '<parameter=tags>["a", "b"]</parameter>'
                    "</function>"
                ),
                "tool_calls": [
                    {
                        "id": "native-3",
                        "function": {
                            "name": "search",
                            "arguments": '{"tags": "[\\"a\\", \\"b\\"]"}',
                        },
                    }
                ],
            }
        }

        _, calls, errors = parse_tool_calls(response)

        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)
        # The typed text carries the real list; the native id is preserved.
        self.assertEqual(calls[0].arguments, {"tags": ["a", "b"]})
        self.assertEqual(calls[0].call_id, "native-3")

    def test_qwen_syntax_inside_tool_call_tags_is_not_double_parsed(self) -> None:
        response = (
            "<tool_call><function=search>"
            "<parameter=query>x</parameter>"
            "</function></tool_call>"
        )

        text, calls, errors = parse_tool_calls(response)

        self.assertEqual(text, "")
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {"query": "x"})

    def test_duplicate_qwen_parameter_is_an_error(self) -> None:
        response = (
            "<function=search>"
            "<parameter=query>a</parameter>"
            "<parameter=query>b</parameter>"
            "</function>"
        )

        _, calls, errors = parse_tool_calls(response)

        self.assertEqual(calls, [])
        self.assertIn("duplicate parameter query", errors[0])

    def test_malformed_tagged_call_is_not_an_answer(self) -> None:
        text, calls, errors = parse_tool_calls("<tool_call>{oops}</tool_call>")

        self.assertEqual(text, "")
        self.assertEqual(calls, [])
        self.assertTrue(errors)

    def test_bare_json_object_call(self) -> None:
        text, calls, errors = parse_tool_calls(
            '{"name": "search", "arguments": {"query": "x"}}'
        )

        self.assertEqual(text, "")
        self.assertEqual(errors, [])
        self.assertEqual(calls[0].name, "search")


class SchemaTests(unittest.TestCase):
    parameters = SEARCH_SCHEMA["function"]["parameters"]

    def test_valid_arguments_pass(self) -> None:
        self.assertIsNone(schema_error({"query": "x", "limit": 5}, self.parameters))

    def test_missing_required_field(self) -> None:
        error = schema_error({"limit": 5}, self.parameters)

        self.assertIsNotNone(error)
        self.assertIn("missing required field", str(error))

    def test_additional_property_rejected(self) -> None:
        error = schema_error({"query": "x", "extra": 1}, self.parameters)

        self.assertIn("unexpected field", str(error))

    def test_numeric_bounds(self) -> None:
        self.assertIn("above maximum", str(schema_error({"query": "x", "limit": 99}, self.parameters)))
        self.assertIn("below minimum", str(schema_error({"query": "x", "limit": 0}, self.parameters)))

    def test_array_item_type(self) -> None:
        error = schema_error({"query": "x", "tags": [1]}, self.parameters)

        self.assertIn("expected string", str(error))

    def test_coercion_respects_declared_types(self) -> None:
        coerced = coerce_schema_value(
            {"query": "123", "limit": "7", "deep": "true", "tags": '["a"]'},
            self.parameters,
        )

        self.assertEqual(coerced["query"], "123")
        self.assertEqual(coerced["limit"], 7)
        self.assertIs(coerced["deep"], True)
        self.assertEqual(coerced["tags"], ["a"])

    def test_coercion_leaves_unrecoverable_values_alone(self) -> None:
        coerced = coerce_schema_value({"limit": "many", "deep": "maybe"}, self.parameters)

        self.assertEqual(coerced["limit"], "many")
        self.assertEqual(coerced["deep"], "maybe")


class ToolDispatcherTests(unittest.TestCase):
    def dispatcher(self, registry: RecordingRegistry, **kwargs: Any) -> ToolDispatcher:
        return ToolDispatcher(registry, schemas=[SEARCH_SCHEMA], **kwargs)

    def test_successful_dispatch_adds_execution_context(self) -> None:
        registry = RecordingRegistry()
        dispatcher = self.dispatcher(registry, execution_context={"session_id": "s1"})

        result = dispatcher.dispatch(ToolCall("c1", "search", {"query": "x"}, None))

        self.assertTrue(result["success"])
        name, kwargs = registry.executions[0]
        self.assertEqual(name, "search")
        self.assertEqual(kwargs, {"query": "x", "session_id": "s1"})

    def test_tool_outside_allow_list_is_unknown(self) -> None:
        registry = RecordingRegistry()
        dispatcher = self.dispatcher(registry, allowed_names=set())

        result = dispatcher.dispatch(ToolCall("c1", "search", {"query": "x"}, None))

        self.assertEqual(result["error_code"], "unknown_tool")
        self.assertEqual(registry.executions, [])

    def test_invalid_arguments_are_rejected_before_execution(self) -> None:
        registry = RecordingRegistry()
        dispatcher = self.dispatcher(registry)

        result = dispatcher.dispatch(ToolCall("c1", "search", {"limit": 5}, None))

        self.assertEqual(result["error_code"], "invalid_arguments")
        self.assertEqual(registry.executions, [])

    def test_authorization_callback_can_block_a_call(self) -> None:
        registry = RecordingRegistry()
        seen: list[int] = []

        def authorize(call: ToolCall, completed):
            seen.append(len(completed))
            return {"error_code": "policy", "error": "blocked"}

        dispatcher = self.dispatcher(registry, authorization_callback=authorize)

        result = dispatcher.dispatch(ToolCall("c1", "search", {"query": "x"}, None))

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "policy")
        self.assertEqual(registry.executions, [])
        self.assertEqual(seen, [0])
        self.assertEqual(len(dispatcher.completed_tool_calls), 1)

    def test_execution_arguments_callback_rewrites_arguments(self) -> None:
        registry = RecordingRegistry()
        dispatcher = self.dispatcher(
            registry,
            execution_arguments_callback=lambda call: {**call.arguments, "query": "rewritten"},
        )

        dispatcher.dispatch(ToolCall("c1", "search", {"query": "x"}, None))

        self.assertEqual(registry.executions[0][1]["query"], "rewritten")

    def test_execution_arguments_callback_failure_is_recorded(self) -> None:
        registry = RecordingRegistry()

        def boom(call: ToolCall):
            raise RuntimeError("bad transform")

        dispatcher = self.dispatcher(registry, execution_arguments_callback=boom)

        result = dispatcher.dispatch(ToolCall("c1", "search", {"query": "x"}, None))

        self.assertEqual(result["error_code"], "tool_execution_error")
        self.assertEqual(registry.executions, [])

    def test_error_payload_without_success_becomes_a_failure(self) -> None:
        registry = RecordingRegistry({"error": "not found"})
        dispatcher = self.dispatcher(registry)

        result = dispatcher.dispatch(ToolCall("c1", "search", {"query": "x"}, None))

        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "tool_execution_error")

    def test_async_tool_result_is_awaited(self) -> None:
        class AsyncRegistry:
            async def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
                del name, kwargs
                return {"success": True, "async": True}

        dispatcher = ToolDispatcher(AsyncRegistry(), schemas=[SEARCH_SCHEMA])

        result = dispatcher.dispatch(ToolCall("c1", "search", {"query": "x"}, None))

        self.assertTrue(result["async"])

    def test_reset_history_and_workspace_setup(self) -> None:
        registry = RecordingRegistry()
        seen: list[Path] = []
        dispatcher = self.dispatcher(registry, workspace_setup=seen.append)

        dispatcher.dispatch(ToolCall("c1", "search", {"query": "x"}, None))
        self.assertEqual(len(dispatcher.completed_tool_calls), 1)
        dispatcher.reset_history()
        self.assertEqual(dispatcher.completed_tool_calls, [])

        dispatcher.set_case_workspace(Path("workspace-autonomous-evaluation"))
        self.assertEqual(seen, [Path("workspace-autonomous-evaluation")])


class AppendToolExchangeTests(unittest.TestCase):
    def test_appends_agent_template_roles(self) -> None:
        messages: list[dict[str, Any]] = []

        append_tool_exchange(
            messages,
            ToolCall("c1", "search", {"query": "x"}, None),
            {"success": True},
            assistant_content="thinking",
        )

        self.assertEqual(
            [message["role"] for message in messages],
            ["assistant", "tool_call", "tool_response"],
        )
        self.assertEqual(
            json.loads(messages[1]["content"]),
            {"name": "search", "arguments": {"query": "x"}},
        )
        self.assertEqual(json.loads(messages[2]["content"]), {"success": True})

    def test_empty_assistant_content_is_omitted(self) -> None:
        messages: list[dict[str, Any]] = []

        append_tool_exchange(
            messages, ToolCall("c1", "search", {}, None), {"success": True}
        )

        self.assertEqual(
            [message["role"] for message in messages], ["tool_call", "tool_response"]
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
