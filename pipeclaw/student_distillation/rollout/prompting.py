from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import PromptCase, jsonable


def strip_teacher_future_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep context through the last user turn and hide teacher future actions.

    This is intentionally conservative: earlier system/user context is retained,
    but all messages after the current user request (tool calls, tool responses,
    and the teacher answer) are removed.
    """

    copied = [dict(message) for message in messages]
    last_user = max(
        (
            index
            for index, message in enumerate(copied)
            if message.get("role") == "user"
        ),
        default=len(copied) - 1,
    )
    return copied[: last_user + 1]


def parse_tool_schemas(value: Any) -> list[dict[str, Any]]:
    """Normalize a record's ``tools`` field into a list of OpenAI schemas."""

    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(m) for item in value if isinstance((m := jsonable(item)), Mapping)]


class PromptCaseBuilder:
    """Build production-shaped prompts without exposing teacher future messages."""

    def build(
        self,
        record: Mapping[str, Any],
        *,
        workspace_root: Path,
        prompt_builder: Any | None = None,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
    ) -> PromptCase:
        if not isinstance(record, Mapping):
            raise TypeError("evaluation record must be a mapping")
        user_input = self._user_input(record)

        if prompt_builder is None:
            from pipeclaw.backend.agent.prompt_builder import PromptBuilder

            prompt_builder = PromptBuilder(workspace_root)

        memory_payload = {
            "control_files": [],
            "assets": [],
            "trace_meta": {},
            "verified_evidence_summaries": [],
        }
        system_prompt = prompt_builder.build(
            memory_payload=memory_payload,
            skills_section="",
            verified_state=record.get("state_before") or {},
            recent_turns=record.get("recent_turns") or [],
        )
        messages = self._messages(record, user_input, system_prompt)
        schemas = list(tool_schemas or parse_tool_schemas(record.get("tools")))
        return PromptCase(
            sample_id=str(
                record.get("sample_id") or record.get("example_id") or "unknown"
            ),
            scenario_id=str(record.get("scenario_id") or ""),
            scenario_type=str(record.get("scenario_type") or ""),
            messages=messages,
            tools=schemas,
            source_record=dict(record),
            workspace_root=Path(workspace_root),
        )

    @staticmethod
    def _user_input(record: Mapping[str, Any]) -> str:
        user_input = record.get("user_input")
        if not isinstance(user_input, str) or not user_input.strip():
            raw_messages = record.get("messages")
            if isinstance(raw_messages, Sequence) and not isinstance(
                raw_messages, (str, bytes)
            ):
                user_messages = [
                    message.get("content")
                    for message in raw_messages
                    if isinstance(message, Mapping) and message.get("role") == "user"
                ]
                if user_messages and isinstance(user_messages[-1], str):
                    user_input = user_messages[-1]
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("evaluation record requires a non-empty user_input")
        return user_input

    @staticmethod
    def _messages(
        record: Mapping[str, Any],
        user_input: str,
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        raw_messages = record.get("messages")
        if not isinstance(raw_messages, Sequence) or isinstance(
            raw_messages, (str, bytes)
        ):
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        # The source trace may include a system prompt and several turns.  We
        # still strip every teacher-generated future message before use.
        messages = strip_teacher_future_messages(raw_messages)
        if not any(message.get("role") == "user" for message in messages):
            messages.append({"role": "user", "content": user_input})
        # PromptBuilder is the authoritative system prompt for autonomous eval.
        system_index = next(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "system"
            ),
            None,
        )
        if system_index is None:
            messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            messages[system_index] = {"role": "system", "content": system_prompt}
        return messages
