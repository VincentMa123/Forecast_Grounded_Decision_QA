"""Runner tests using a fake generator and a fake tool registry.

These tests deliberately exercise ``RolloutRunner`` without model weights, CUDA,
or PipeFormer inference: the rollout core must be verifiable on any machine.
"""

from __future__ import annotations

import unittest
from typing import Any, Mapping, Sequence

from pipeclaw.task2_student.rollout.models import (
    PromptCase,
    RolloutConfig,
)
from pipeclaw.task2_student.rollout.runner import RolloutRunner
from pipeclaw.task2_student.rollout.tools import ToolDispatcher


ECHO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "echo",
        "parameters": {
            "type": "object",
            "properties": {
                "flag": {"type": "boolean"},
                "items": {"type": "array", "items": {"type": "string"}},
                "text": {"type": "string"},
            },
        },
    },
}


class ScriptedGenerator:
    """Return queued responses, then raise if the runner asks for more."""

    def __init__(self, responses: Sequence[Any], *, repeat_last: bool = False) -> None:
        self.responses = list(responses)
        self.repeat_last = repeat_last
        self.calls: list[list[dict[str, Any]]] = []

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        del tools, max_tokens, temperature
        self.calls.append([dict(message) for message in messages])
        if not self.responses:
            raise AssertionError("generator called more times than scripted")
        if self.repeat_last and len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


class FailingGenerator:
    def generate(self, messages, tools, *, max_tokens, temperature):
        del messages, tools, max_tokens, temperature
        raise RuntimeError("engine exploded")


class FakeRegistry:
    """Minimal ``execute``-style registry recording the executed arguments."""

    def __init__(self, result: Any = None, *, raises: bool = False) -> None:
        self.result = result if result is not None else {"success": True, "ok": True}
        self.raises = raises
        self.executions: list[tuple[str, dict[str, Any]]] = []

    def execute(self, name: str, **kwargs: Any) -> Any:
        self.executions.append((name, dict(kwargs)))
        if self.raises:
            raise RuntimeError("tool exploded")
        return self.result


def make_case(**overrides: Any) -> PromptCase:
    payload: dict[str, Any] = {
        "sample_id": "sample-1",
        "scenario_id": "scenario-1",
        "scenario_type": "pipeformer",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        "tools": [ECHO_SCHEMA],
        "source_record": {},
    }
    payload.update(overrides)
    return PromptCase(**payload)


def make_config(**overrides: Any) -> RolloutConfig:
    payload: dict[str, Any] = {
        "max_turns": 4,
        "max_new_tokens": 128,
        "temperature": 0.0,
    }
    payload.update(overrides)
    return RolloutConfig(**payload)


def make_runner(
    responses: Sequence[Any],
    *,
    registry: FakeRegistry | None = None,
    repeat_last: bool = False,
) -> tuple[RolloutRunner, FakeRegistry, ScriptedGenerator]:
    registry = registry or FakeRegistry()
    generator = ScriptedGenerator(responses, repeat_last=repeat_last)
    dispatcher = ToolDispatcher(registry, schemas=[ECHO_SCHEMA])
    return RolloutRunner(generator, dispatcher), registry, generator


class RolloutRunnerTests(unittest.TestCase):
    def test_immediate_final_answer(self) -> None:
        runner, _, generator = make_runner(["the final answer"])

        result = runner.run(make_case(), make_config())

        self.assertEqual(result.trace_status, "completed")
        self.assertEqual(result.final_answer, "the final answer")
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(result.messages[-1]["role"], "assistant")

    def test_one_tool_call_then_final_answer(self) -> None:
        runner, registry, _ = make_runner(
            [
                '<tool_call>{"name": "echo", "arguments": {"text": "hi"}}</tool_call>',
                "done",
            ]
        )

        result = runner.run(make_case(), make_config())

        self.assertEqual(result.trace_status, "completed")
        self.assertEqual(result.final_answer, "done")
        self.assertEqual(result.turns, 2)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["name"], "echo")
        self.assertTrue(result.tool_calls[0]["schema_valid"])
        self.assertTrue(result.tool_calls[0]["execution_success"])
        self.assertEqual(registry.executions[0][0], "echo")
        roles = [message["role"] for message in result.messages]
        self.assertIn("tool_call", roles)
        self.assertIn("tool_response", roles)

    def test_qwen_tagged_call_coerces_stringified_values(self) -> None:
        response = (
            "<function=echo>"
            "<parameter=flag>True</parameter>"
            '<parameter=items>["a", "b"]</parameter>'
            "</function>"
        )
        runner, registry, _ = make_runner([response, "done"])

        result = runner.run(make_case(), make_config())

        self.assertEqual(result.trace_status, "completed")
        # The record preserves what the model literally emitted; the dispatcher
        # coerces it against the declared schema before executing the tool.
        arguments = result.tool_calls[0]["arguments"]
        self.assertEqual(arguments["flag"], "True")
        self.assertEqual(arguments["items"], ["a", "b"])
        self.assertTrue(result.tool_calls[0]["schema_valid"])
        executed = registry.executions[0][1]
        self.assertIs(executed["flag"], True)
        self.assertEqual(executed["items"], ["a", "b"])

    def test_numeric_qwen_parameter_for_a_string_field_fails_validation(self) -> None:
        # The Qwen parameter decoder is JSON-first, so ``123`` arrives as an int.
        # Schema-guided coercion never widens an int into a string, so the call is
        # recorded as schema-invalid rather than silently executed.
        response = "<function=echo><parameter=text>123</parameter></function>"
        runner, registry, _ = make_runner([response, "done"])

        result = runner.run(make_case(), make_config())

        self.assertEqual(result.tool_calls[0]["arguments"], {"text": 123})
        self.assertFalse(result.tool_calls[0]["schema_valid"])
        self.assertEqual(registry.executions, [])

    def test_unquoted_qwen_parameter_stays_a_string(self) -> None:
        response = "<function=echo><parameter=text>hello world</parameter></function>"
        runner, registry, _ = make_runner([response, "done"])

        result = runner.run(make_case(), make_config())

        self.assertTrue(result.tool_calls[0]["schema_valid"])
        self.assertEqual(registry.executions[0][1]["text"], "hello world")

    def test_malformed_call_recorded_as_completed_invalid(self) -> None:
        runner, registry, _ = make_runner(["<tool_call>{not json}</tool_call>"])

        result = runner.run(make_case(), make_config())

        self.assertEqual(result.trace_status, "completed")
        self.assertEqual(result.final_answer, "")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(registry.executions, [])
        self.assertTrue(result.json_errors)
        self.assertTrue(
            result.json_errors[0].startswith("malformed_tool_call_json")
        )

    def test_tool_failure_retained_in_partial_state(self) -> None:
        registry = FakeRegistry(raises=True)
        runner, _, _ = make_runner(
            ['<tool_call>{"name": "echo", "arguments": {"text": "hi"}}</tool_call>'],
            registry=registry,
        )

        result = runner.run(make_case(), make_config(max_turns=1))

        self.assertEqual(result.trace_status, "max_turns_exceeded")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertTrue(result.tool_calls[0]["schema_valid"])
        self.assertFalse(result.tool_calls[0]["execution_success"])
        self.assertEqual(
            result.tool_outputs[0]["output"]["error_code"], "tool_execution_error"
        )

    def test_unknown_tool_is_schema_invalid(self) -> None:
        runner, registry, _ = make_runner(
            ['<tool_call>{"name": "nope", "arguments": {}}</tool_call>']
        )

        result = runner.run(make_case(), make_config(max_turns=1))

        self.assertFalse(result.tool_calls[0]["schema_valid"])
        self.assertFalse(result.tool_calls[0]["execution_success"])
        self.assertEqual(registry.executions, [])

    def test_generation_exception_becomes_generation_error(self) -> None:
        dispatcher = ToolDispatcher(FakeRegistry(), schemas=[ECHO_SCHEMA])
        runner = RolloutRunner(FailingGenerator(), dispatcher)

        result = runner.run(make_case(), make_config())

        self.assertEqual(result.trace_status, "generation_error")
        self.assertEqual(result.generation_error, "engine exploded")
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.final_answer, "")

    def test_repeated_tool_calls_exhaust_max_turns(self) -> None:
        runner, registry, _ = make_runner(
            ['<tool_call>{"name": "echo", "arguments": {"text": "hi"}}</tool_call>'],
            repeat_last=True,
        )

        result = runner.run(make_case(), make_config(max_turns=3))

        self.assertEqual(result.trace_status, "max_turns_exceeded")
        self.assertEqual(result.turns, 3)
        self.assertEqual(len(result.tool_calls), 3)
        self.assertEqual(len(registry.executions), 3)
        self.assertEqual(result.final_answer, "")

    def test_empty_response_status(self) -> None:
        runner, _, _ = make_runner(["   "])

        result = runner.run(make_case(), make_config())

        self.assertEqual(result.trace_status, "empty_response")
        self.assertEqual(result.final_answer, "")

    def test_zero_max_turns_never_calls_the_generator(self) -> None:
        runner, _, generator = make_runner([])

        result = runner.run(make_case(), make_config(max_turns=0))

        self.assertEqual(result.trace_status, "max_turns_exceeded")
        self.assertEqual(result.turns, 0)
        self.assertEqual(generator.calls, [])

    def test_raw_capture_only_when_enabled(self) -> None:
        script = [
            '<tool_call>{"name": "echo", "arguments": {"text": "hi"}}</tool_call>',
            "done",
        ]
        runner, _, _ = make_runner(list(script))

        default_result = runner.run(make_case(), make_config())

        self.assertIsNone(default_result.raw_responses)
        self.assertIsNone(default_result.raw_tool_outputs)
        self.assertNotIn("raw_responses", default_result.to_dict())
        self.assertNotIn("raw_tool_outputs", default_result.to_dict())

        runner, _, _ = make_runner(list(script))
        captured = runner.run(
            make_case(),
            make_config(capture_raw_responses=True, capture_raw_tool_outputs=True),
        )

        self.assertEqual(len(captured.raw_responses or []), 2)
        self.assertEqual(len(captured.raw_tool_outputs or []), 1)
        payload = captured.to_dict()
        self.assertIn("raw_responses", payload)
        self.assertIn("raw_tool_outputs", payload)

    def test_dispatcher_receives_case_context(self) -> None:
        runner, _, _ = make_runner(["done"])
        dispatcher = runner.dispatcher
        dispatcher.completed_tool_calls.append({"stale": True})

        runner.run(make_case(), make_config())

        self.assertEqual(dispatcher.completed_tool_calls, [])
        self.assertEqual(dispatcher.current_user_request, "question")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
