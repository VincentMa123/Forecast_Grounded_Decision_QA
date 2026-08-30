"""Shared parsed-call dispatch and episode transcript bookkeeping."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from pipeclaw.protocols.tool_calls import ToolCall, jsonable

from .tools import ToolDispatcher, append_tool_exchange

__all__ = [
    "dispatch_and_record",
    "execution_success",
    "schema_valid",
]


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
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    tool_outputs: list[dict[str, Any]],
    record_arguments: Callable[[ToolCall], Mapping[str, Any]],
    compact_result: Callable[[ToolCall, Any, Mapping[str, Any]], Any],
    normalize_call: Callable[[ToolCall], ToolCall] | None = None,
    portability_metadata: Callable[[ToolCall], Mapping[str, Any]] | None = None,
    raw_tool_outputs: list[dict[str, Any]] | None = None,
    assistant_content: str = "",
    dispatch: Callable[[ToolCall], Mapping[str, Any]] | None = None,
    on_result: Callable[[ToolCall, Mapping[str, Any]], None] | None = None,
) -> None:
    """Dispatch calls and append their records/messages without replacing lists.

    Rollout loops own parsing, turn limits, and terminal decisions.  This
    operation owns only the semantic call exchange shared by those lifecycles.
    Callers may supply a dispatch wrapper (for example, Swift's lock) and a
    result hook for lifecycle-specific diagnostics while list mutation remains
    in-place for snapshot consumers.
    """

    normalize = normalize_call or dispatcher.schema_normalized_call
    execute = dispatch or dispatcher.dispatch
    portability_for = portability_metadata or (lambda _call: {})
    for index, call in enumerate(calls):
        normalized = normalize(call)
        portability = dict(portability_for(normalized) or {})
        tool_result = execute(normalized)
        if not isinstance(tool_result, Mapping):
            tool_result = {"success": True, "result": tool_result}
        if on_result is not None:
            on_result(normalized, tool_result)

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
        tool_calls.append(call_record)

        # Policy projections consume the schema-normalized call.  This keeps
        # Swift's callback contract aligned with the pre-refactor scheduler;
        # the original call is still used for transcript serialization below.
        compact = compact_result(normalized, tool_result, portability)
        tool_outputs.append(
            {"tool_call_id": call.call_id, "name": call.name, "output": compact}
        )
        if raw_tool_outputs is not None:
            raw_entry: dict[str, Any] = {
                "tool_call_id": call.call_id,
                "name": call.name,
                "output": jsonable(tool_result),
            }
            if portability:
                raw_entry["diagnostics"] = dict(portability)
            raw_tool_outputs.append(raw_entry)
        append_tool_exchange(
            messages,
            call,
            compact,
            assistant_content=assistant_content if index == 0 else "",
        )
