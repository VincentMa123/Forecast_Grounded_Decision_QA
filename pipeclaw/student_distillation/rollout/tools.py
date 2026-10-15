from __future__ import annotations

import asyncio
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipeclaw.protocols.tool_calls import (
    ToolCall,
    coerce_schema_value,
    jsonable,
    schema_error,
)

from .models import RolloutResult


def _schema_index(
    schemas: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for schema in schemas:
        function = schema.get("function", schema) if isinstance(schema, Mapping) else {}
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            index[function["name"]] = schema
    return index


class ToolDispatcher:
    """Validate and execute only explicitly allow-listed evaluation tools."""

    def __init__(
        self,
        registry: Any,
        *,
        schemas: Sequence[Mapping[str, Any]],
        allowed_names: set[str] | None = None,
        authorization_callback: Callable[
            [ToolCall, Sequence[Mapping[str, Any]]], Mapping[str, Any] | None
        ]
        | None = None,
        execution_arguments_callback: Callable[[ToolCall], Mapping[str, Any]]
        | None = None,
        execution_context: Mapping[str, Any] | None = None,
        workspace_setup: Callable[[Path], None] | None = None,
    ) -> None:
        self.registry = registry
        self.schemas = _schema_index(schemas)
        self.allowed_names = (
            set(allowed_names) if allowed_names is not None else set(self.schemas)
        )
        self.authorization_callback = authorization_callback
        self.execution_arguments_callback = execution_arguments_callback
        self.execution_context = dict(execution_context or {})
        self.workspace_setup = workspace_setup
        self.completed_tool_calls: list[dict[str, Any]] = []
        self.current_user_request = ""

    def _validate(self, call: ToolCall) -> str | None:
        schema = self.schemas.get(call.name)
        if schema is None:
            return "unknown_tool"
        if call.name not in self.allowed_names:
            return "tool_not_allowed"
        function = schema.get("function", schema)
        parameters = (
            function.get("parameters", {}) if isinstance(function, Mapping) else {}
        )
        if not isinstance(parameters, Mapping) or not isinstance(
            call.arguments, Mapping
        ):
            return "invalid_arguments"
        return "invalid_arguments" if schema_error(call.arguments, parameters) else None

    def schema_normalized_call(self, call: ToolCall) -> ToolCall:
        schema = self.schemas.get(call.name)
        if schema is None:
            return call
        function = schema.get("function", schema)
        parameters = (
            function.get("parameters", {}) if isinstance(function, Mapping) else {}
        )
        if not isinstance(parameters, Mapping):
            return call
        normalized = coerce_schema_value(call.arguments, parameters)
        if not isinstance(normalized, Mapping):
            return call
        return ToolCall(call.call_id, call.name, dict(normalized), call.raw)

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
        normalized_call = self.schema_normalized_call(call)
        error_code = self._validate(normalized_call)
        if error_code:
            result = {
                "success": False,
                "error_code": error_code,
                "error": f"Tool call rejected: {call.name}",
            }
            self._record(normalized_call, result)
            return result
        if self.authorization_callback is not None:
            authorization_error = self.authorization_callback(
                normalized_call, self.completed_tool_calls
            )
            if authorization_error:
                result = dict(authorization_error)
                result.setdefault("success", False)
                self._record(normalized_call, result)
                return result
        execution_arguments = dict(normalized_call.arguments)
        if self.execution_arguments_callback is not None:
            try:
                transformed = self.execution_arguments_callback(normalized_call)
                if not isinstance(transformed, Mapping):
                    raise TypeError("execution argument callback must return a mapping")
                execution_arguments = dict(transformed)
            except Exception as exc:
                result = {
                    "success": False,
                    "error_code": "tool_execution_error",
                    "error": f"Tool argument transformation failed: {exc}",
                }
                self._record(normalized_call, result)
                return result
        try:
            if hasattr(self.registry, "execute"):
                result = self.registry.execute(
                    normalized_call.name,
                    **dict(execution_arguments, **self.execution_context),
                )
            else:
                tool = self.registry.get(normalized_call.name)
                if tool is None:
                    raise KeyError(call.name)
                result = tool(**dict(execution_arguments, **self.execution_context))
            result = self._await_if_needed(result)
            if isinstance(result, Mapping):
                result = dict(result)
                if result.get("error") and "success" not in result:
                    result["success"] = False
                    result.setdefault("error_code", "tool_execution_error")
                self._record(normalized_call, result)
                return result
            result = {"success": True, "result": result}
            self._record(normalized_call, result)
            return result
        except (
            Exception
        ) as exc:  # evaluation records the failure; it must not abort the suite
            result = {
                "success": False,
                "error_code": "tool_execution_error",
                "error": str(exc),
            }
            self._record(normalized_call, result)
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
            "content": json.dumps(jsonable(result), ensure_ascii=False, sort_keys=True),
        }
    )


# ---------------------------------------------------------------------------
# Episode bookkeeping: dispatch parsed calls and record the exchange
# ---------------------------------------------------------------------------

_SCHEMA_ERROR_CODES = {"unknown_tool", "invalid_arguments", "tool_not_allowed"}
_EXECUTION_ERROR_CODES = {
    "tool_execution_error",
    "forecast_registry_precondition_failed",
}
_PORTABILITY_RECORD_KEYS = ("cwd_rebased", "portable_path_normalization")


def schema_valid(result: Mapping[str, Any]) -> bool:
    """Return whether a tool result passed schema/allow-list validation."""

    return result.get("error_code") not in _SCHEMA_ERROR_CODES


def execution_success(result: Mapping[str, Any]) -> bool:
    """Return whether a tool execution completed without a domain failure."""

    return (
        result.get("success", True) is not False
        and not result.get("error")
        and result.get("error_code") not in _EXECUTION_ERROR_CODES
        and result.get("exit_code") in (None, 0)
    )


def dispatch_and_record(
    calls: Sequence[ToolCall],
    *,
    dispatcher: ToolDispatcher,
    result: RolloutResult,
    record_arguments: Callable[[ToolCall], Mapping[str, Any]],
    compact_result: Callable[[ToolCall, Any, Mapping[str, Any]], Any],
    portability_metadata: Callable[[ToolCall], Mapping[str, Any]] | None = None,
    assistant_content: str = "",
) -> None:
    """Dispatch calls and append their records/messages without replacing lists.

    Rollout loops own parsing, turn limits, and terminal decisions.  This
    operation owns only the semantic call exchange shared by those lifecycles.
    List mutation remains in-place for snapshot consumers.
    """

    portability_for = portability_metadata or (lambda _call: {})
    for index, call in enumerate(calls):
        normalized = dispatcher.schema_normalized_call(call)
        portability = dict(portability_for(normalized) or {})
        tool_result = dispatcher.dispatch(normalized)
        if not isinstance(tool_result, Mapping):
            tool_result = {"success": True, "result": tool_result}

        call_record: dict[str, Any] = {
            "tool_call_id": call.call_id,
            "name": call.name,
            "arguments": dict(record_arguments(normalized)),
            "schema_valid": schema_valid(tool_result),
            "execution_success": execution_success(tool_result),
        }
        call_record.update(
            {
                key: portability[key]
                for key in _PORTABILITY_RECORD_KEYS
                if portability.get(key)
            }
        )
        result.tool_calls.append(call_record)

        # Policy projections consume the schema-normalized call; the original
        # call is still used for transcript serialization below.
        compact = compact_result(normalized, tool_result, portability)
        result.tool_outputs.append(
            {"tool_call_id": call.call_id, "name": call.name, "output": compact}
        )
        if result.raw_tool_outputs is not None:
            raw_entry: dict[str, Any] = {
                "tool_call_id": call.call_id,
                "name": call.name,
                "output": jsonable(tool_result),
            }
            if portability:
                raw_entry["diagnostics"] = dict(portability)
            result.raw_tool_outputs.append(raw_entry)
        append_tool_exchange(
            result.messages,
            call,
            compact,
            assistant_content=assistant_content if index == 0 else "",
        )
