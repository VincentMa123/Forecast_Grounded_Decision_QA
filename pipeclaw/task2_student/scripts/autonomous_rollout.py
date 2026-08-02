"""Core helpers for autonomous evaluation of a Task 2 student model.

The training traces in this repository contain the teacher's future tool calls and
answer.  This module deliberately builds a prompt from the pre-action state only,
then provides a small, allow-listed tool loop for evaluation.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


@dataclass
class PromptCase:
    """A prompt-only evaluation case plus its source record."""

    sample_id: str
    scenario_id: str
    scenario_type: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    source_record: dict[str, Any]
    workspace_root: Path | None = None


@dataclass(frozen=True)
class ToolCall:
    """A normalized function call emitted by a model."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    raw: Any


class Generator(Protocol):
    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        """Generate one response for the current conversation."""


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, Mapping):
                return dumped
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
            if isinstance(dumped, Mapping):
                return dumped
        except Exception:
            pass
    # Lightweight SDK response objects are sometimes plain classes rather than
    # pydantic models.  Expose the small set of fields needed by the parser.
    fields = (
        "id",
        "type",
        "function",
        "name",
        "arguments",
        "content",
        "tool_calls",
        "choices",
        "message",
    )
    attrs = {field: getattr(value, field) for field in fields if hasattr(value, field)}
    if attrs:
        return attrs
    return None


def _get(value: Any, key: str, default: Any = None) -> Any:
    mapped = _as_mapping(value)
    if mapped is not None:
        return mapped.get(key, default)
    return getattr(value, key, default)


def _jsonable(value: Any) -> Any:
    """Convert SDK response objects into JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    mapped = _as_mapping(value)
    if mapped is not None:
        return {str(key): _jsonable(item) for key, item in mapped.items()}
    return str(value)


def strip_teacher_future_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep context through the last user turn and hide teacher future actions.

    This is intentionally conservative: earlier system/user context is retained,
    but all messages after the current user request (tool calls, tool responses,
    and the teacher answer) are removed.
    """

    copied = [dict(message) for message in messages]
    last_user = max(
        (index for index, message in enumerate(copied) if message.get("role") == "user"),
        default=len(copied) - 1,
    )
    return copied[: last_user + 1]


def _parse_tools(value: Any) -> list[dict[str, Any]]:
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
    return [dict(_jsonable(item)) for item in value if isinstance(_jsonable(item), Mapping)]


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
        user_input = record.get("user_input")
        if not isinstance(user_input, str) or not user_input.strip():
            raw_messages = record.get("messages")
            if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes)):
                user_messages = [
                    message.get("content")
                    for message in raw_messages
                    if isinstance(message, Mapping) and message.get("role") == "user"
                ]
                if user_messages and isinstance(user_messages[-1], str):
                    user_input = user_messages[-1]
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("evaluation record requires a non-empty user_input")

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

        raw_messages = record.get("messages")
        if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes)):
            # The source trace may include a system prompt and several turns.  We
            # still strip every teacher-generated future message before use.
            messages = strip_teacher_future_messages(raw_messages)
            if not any(message.get("role") == "user" for message in messages):
                messages.append({"role": "user", "content": user_input})
            # PromptBuilder is the authoritative system prompt for autonomous eval.
            system_index = next(
                (index for index, message in enumerate(messages) if message.get("role") == "system"),
                None,
            )
            if system_index is None:
                messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                messages[system_index] = {"role": "system", "content": system_prompt}
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]

        schemas = list(tool_schemas or _parse_tools(record.get("tools")))
        source_record = dict(record)
        return PromptCase(
            sample_id=str(record.get("sample_id") or record.get("example_id") or "unknown"),
            scenario_id=str(record.get("scenario_id") or ""),
            scenario_type=str(record.get("scenario_type") or ""),
            messages=messages,
            tools=schemas,
            source_record=source_record,
            workspace_root=Path(workspace_root),
        )


_TAGGED_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _parse_call_payload(payload: Any, *, default_id: str, raw: Any) -> tuple[ToolCall | None, str | None]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            return None, f"malformed_tool_call_json: {exc.msg}"
    mapped = _as_mapping(payload)
    if mapped is None:
        return None, "malformed_tool_call_payload: expected object"

    function = mapped.get("function")
    function_map = _as_mapping(function) or {}
    name = mapped.get("name") or function_map.get("name")
    arguments = mapped.get("arguments", function_map.get("arguments", {}))
    if not isinstance(name, str) or not name:
        return None, "malformed_tool_call_payload: missing function name"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return None, f"malformed_tool_arguments_json: {exc.msg}"
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        return None, "malformed_tool_arguments: expected object"
    call_id = str(mapped.get("id") or default_id)
    return ToolCall(call_id, name, dict(arguments), raw), None


def parse_tool_calls(response: Any) -> tuple[str, list[ToolCall], list[str]]:
    """Normalize tagged, OpenAI-style, and SDK response objects.

    Returns ``(visible_text, tool_calls, errors)``.  A response containing a
    malformed tagged call is not treated as a final answer, preventing malformed
    tool JSON from silently becoming an apparently successful rollout.
    """

    errors: list[str] = []
    message = response
    choices = _get(response, "choices")
    if choices:
        try:
            first_choice = choices[0]
        except (IndexError, TypeError):
            first_choice = None
        message = _get(first_choice, "message", first_choice)
    message = _get(response, "message", message)

    raw_calls = _get(message, "tool_calls")
    content = response if isinstance(response, str) else _get(message, "content", "")
    calls: list[ToolCall] = []
    if raw_calls:
        for index, raw_call in enumerate(raw_calls):
            call, error = _parse_call_payload(raw_call, default_id=f"call-{index + 1}", raw=raw_call)
            if call is not None:
                calls.append(call)
            if error:
                errors.append(error)

    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)

    tagged = list(_TAGGED_TOOL_CALL.finditer(content))
    if tagged:
        for index, match in enumerate(tagged):
            call, error = _parse_call_payload(
                match.group(1), default_id=f"call-{len(calls) + index + 1}", raw=match.group(0)
            )
            if call is not None:
                calls.append(call)
            if error:
                errors.append(error)
        visible_text = _TAGGED_TOOL_CALL.sub("", content).strip()
        # If every visible character was a malformed tool block, it is not an answer.
        if errors and not visible_text and not calls:
            visible_text = ""
        return visible_text, calls, errors

    if not calls:
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, Mapping) and ("name" in decoded or "function" in decoded):
                call, error = _parse_call_payload(decoded, default_id="call-1", raw=decoded)
                if call is not None:
                    calls.append(call)
                if error:
                    errors.append(error)
                return "", calls, errors
    return content.strip(), calls, errors


def _schema_index(schemas: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for schema in schemas:
        function = schema.get("function", schema) if isinstance(schema, Mapping) else {}
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            index[function["name"]] = schema
    return index


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    return True


def _schema_error(value: Any, schema: Mapping[str, Any], *, path: str = "arguments") -> str | None:
    expected_type = schema.get("type")
    if expected_type and not _type_matches(value, str(expected_type)):
        return f"{path}: expected {expected_type}"
    if "enum" in schema and value not in schema.get("enum", []):
        return f"{path}: value is not in enum"
    if isinstance(value, str):
        pattern = schema.get("pattern")
        if pattern and re.fullmatch(str(pattern), value) is None:
            return f"{path}: pattern mismatch"
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            return f"{path}: shorter than minLength"
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            return f"{path}: longer than maxLength"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: above maximum"
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            return f"{path}: fewer than minItems"
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            return f"{path}: more than maxItems"
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                error = _schema_error(item, item_schema, path=f"{path}[{index}]")
                if error:
                    return error
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            return f"{path}: missing required field(s) {', '.join(map(str, missing))}"
        if schema.get("additionalProperties") is False:
            extras = [name for name in value if name not in properties]
            if extras:
                return f"{path}: unexpected field(s) {', '.join(map(str, extras))}"
        if isinstance(properties, Mapping):
            for name, item in value.items():
                child_schema = properties.get(name)
                if isinstance(child_schema, Mapping):
                    error = _schema_error(item, child_schema, path=f"{path}.{name}")
                    if error:
                        return error
        for condition in schema.get("allOf", []) or []:
            if not isinstance(condition, Mapping):
                continue
            if_schema = condition.get("if")
            then_schema = condition.get("then")
            if not isinstance(if_schema, Mapping) or not isinstance(then_schema, Mapping):
                continue
            if _schema_error(value, if_schema, path=path) is None:
                error = _schema_error(value, then_schema, path=path)
                if error:
                    return error
    return None


class ToolDispatcher:
    """Validate and execute only explicitly allow-listed evaluation tools."""

    def __init__(
        self,
        registry: Any,
        *,
        schemas: Sequence[Mapping[str, Any]],
        allowed_names: set[str] | None = None,
        authorization_callback: Callable[[ToolCall, Sequence[Mapping[str, Any]]], Mapping[str, Any] | None] | None = None,
        execution_arguments_callback: Callable[[ToolCall], Mapping[str, Any]] | None = None,
        execution_context: Mapping[str, Any] | None = None,
        workspace_setup: Callable[[Path], None] | None = None,
    ) -> None:
        self.registry = registry
        self.schemas = _schema_index(schemas)
        self.allowed_names = set(allowed_names) if allowed_names is not None else set(self.schemas)
        self.authorization_callback = authorization_callback
        self.execution_arguments_callback = execution_arguments_callback
        self.execution_context = dict(execution_context or {})
        self.workspace_setup = workspace_setup
        self.completed_tool_calls: list[dict[str, Any]] = []
        self.current_user_request = ""

    def _validate(self, call: ToolCall) -> str | None:
        schema = self.schemas.get(call.name)
        if schema is None or call.name not in self.allowed_names:
            return "unknown_tool"
        function = schema.get("function", schema)
        parameters = function.get("parameters", {}) if isinstance(function, Mapping) else {}
        if not isinstance(parameters, Mapping) or not isinstance(call.arguments, Mapping):
            return "invalid_arguments"
        return "invalid_arguments" if _schema_error(call.arguments, parameters) else None

    @staticmethod
    def _await_if_needed(value: Any) -> Any:
        if not inspect.isawaitable(value):
            return value
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(value)
        # A running event loop cannot be nested in the same thread.  Execute the
        # coroutine in a short-lived worker so a notebook/async caller still gets
        # the same synchronous dispatcher contract.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, value).result()

    def dispatch(self, call: ToolCall) -> dict[str, Any]:
        error_code = self._validate(call)
        if error_code:
            result = {
                "success": False,
                "error_code": error_code,
                "error": f"Tool call rejected: {call.name}",
            }
            self._record(call, result)
            return result
        if self.authorization_callback is not None:
            authorization_error = self.authorization_callback(call, self.completed_tool_calls)
            if authorization_error:
                result = dict(authorization_error)
                result.setdefault("success", False)
                self._record(call, result)
                return result
        execution_arguments = dict(call.arguments)
        if self.execution_arguments_callback is not None:
            try:
                transformed = self.execution_arguments_callback(call)
                if not isinstance(transformed, Mapping):
                    raise TypeError("execution argument callback must return a mapping")
                execution_arguments = dict(transformed)
            except Exception as exc:
                result = {
                    "success": False,
                    "error_code": "tool_execution_error",
                    "error": f"Tool argument transformation failed: {exc}",
                }
                self._record(call, result)
                return result
        try:
            if hasattr(self.registry, "execute"):
                result = self.registry.execute(call.name, **dict(execution_arguments, **self.execution_context))
            else:
                tool = self.registry.get(call.name)
                if tool is None:
                    raise KeyError(call.name)
                result = tool(**dict(execution_arguments, **self.execution_context))
            result = self._await_if_needed(result)
            if isinstance(result, Mapping):
                result = dict(result)
                if result.get("error") and "success" not in result:
                    result["success"] = False
                    result.setdefault("error_code", "tool_execution_error")
                self._record(call, result)
                return result
            result = {"success": True, "result": result}
            self._record(call, result)
            return result
        except Exception as exc:  # evaluation records the failure; it must not abort the suite
            result = {
                "success": False,
                "error_code": "tool_execution_error",
                "error": str(exc),
            }
            self._record(call, result)
            return result

    def _record(self, call: ToolCall, output: Mapping[str, Any]) -> None:
        self.completed_tool_calls.append(
            {
                "tool_call_id": call.call_id,
                "name": call.name,
                "arguments": dict(call.arguments),
                "output": dict(output),
            }
        )

    def reset_history(self) -> None:
        """Clear per-case authorization history before a new rollout."""

        self.completed_tool_calls.clear()

    def set_current_user_request(self, request: str) -> None:
        self.current_user_request = str(request or "")

    def set_case_workspace(self, workspace_root: Path) -> None:
        if self.workspace_setup is not None:
            self.workspace_setup(Path(workspace_root))


def append_tool_exchange(
    messages: list[dict[str, Any]],
    call: ToolCall,
    result: Mapping[str, Any],
    *,
    assistant_content: str = "",
) -> None:
    """Append the MS-SWIFT agent-template tool-call/tool-response pair.

    MS-SWIFT's agent templates consume ``tool_call`` and ``tool_response``
    messages (the same roles used by the training projection), while the parser
    above still accepts OpenAI-style response objects from the engine.
    """

    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})
    messages.append(
        {
            "role": "tool_call",
            "content": json.dumps(
                {"name": call.name, "arguments": call.arguments},
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
    )
    messages.append(
        {
            "role": "tool_response",
            "content": json.dumps(_jsonable(result), ensure_ascii=False, sort_keys=True),
        }
    )
