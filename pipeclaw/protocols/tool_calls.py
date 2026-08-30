"""Shared parsing and schema mechanics for model tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "ToolCall",
    "as_mapping",
    "coerce_schema_value",
    "get_field",
    "jsonable",
    "parse_tool_calls",
    "schema_error",
]


@dataclass(frozen=True)
class ToolCall:
    """A normalized function call emitted by a model."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    raw: Any


def as_mapping(value: Any) -> Mapping[str, Any] | None:
    """View an SDK response object as a mapping when possible."""

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


def get_field(value: Any, key: str, default: Any = None) -> Any:
    """Read one field from a mapping-like or attribute-based response object."""

    mapped = as_mapping(value)
    if mapped is not None:
        return mapped.get(key, default)
    return getattr(value, key, default)


def jsonable(value: Any) -> Any:
    """Convert SDK response objects into JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    mapped = as_mapping(value)
    if mapped is not None:
        return {str(key): jsonable(item) for key, item in mapped.items()}
    return str(value)


_TAGGED_TOOL_CALL = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_QWEN_FUNCTION_CALL = re.compile(
    r"<function=(?P<name>[^>\s]+)>\s*(?P<body>.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
_QWEN_PARAMETER = re.compile(
    r"<parameter=(?P<name>[^>\s]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def _parse_call_payload(
    payload: Any, *, default_id: str, raw: Any
) -> tuple[ToolCall | None, str | None]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            return None, f"malformed_tool_call_json: {exc.msg}"
    mapped = as_mapping(payload)
    if mapped is None:
        return None, "malformed_tool_call_payload: expected object"

    function = mapped.get("function")
    function_map = as_mapping(function) or {}
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


def _parse_qwen_parameter_value(value: str) -> Any:
    """Decode one Qwen3.5 parameter while preserving unquoted text strings."""

    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] in '[{"-0123456789' or stripped in {"true", "false", "null"}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return stripped


def _parse_qwen_function_call(
    match: re.Match[str], *, default_id: str
) -> tuple[ToolCall | None, str | None]:
    """Parse Qwen3.5's ``<function>/<parameter>`` text tool-call format."""

    name = match.group("name").strip()
    arguments: dict[str, Any] = {}
    for parameter in _QWEN_PARAMETER.finditer(match.group("body")):
        parameter_name = parameter.group("name").strip()
        if parameter_name in arguments:
            return (
                None,
                f"malformed_qwen_tool_call: duplicate parameter {parameter_name}",
            )
        arguments[parameter_name] = _parse_qwen_parameter_value(
            parameter.group("value")
        )
    return ToolCall(default_id, name, arguments, match.group(0)), None


def parse_tool_calls(response: Any) -> tuple[str, list[ToolCall], list[str]]:
    """Normalize tagged, OpenAI-style, and SDK response objects.

    Returns ``(visible_text, tool_calls, errors)``.  A response containing a
    malformed tagged call is not treated as a final answer, preventing malformed
    tool JSON from silently becoming an apparently successful rollout.  Qwen3.5
    may expose both its typed ``<function>/<parameter>`` text and a duplicate
    native ``tool_calls`` object; the typed text is authoritative in that case.
    """

    errors: list[str] = []
    message = response
    choices = get_field(response, "choices")
    if choices:
        try:
            first_choice = choices[0]
        except (IndexError, TypeError):
            first_choice = None
        message = get_field(first_choice, "message", first_choice)
    message = get_field(response, "message", message)

    raw_calls = get_field(message, "tool_calls")
    content = (
        response if isinstance(response, str) else get_field(message, "content", "")
    )
    native_calls: list[ToolCall] = []
    native_errors: list[str] = []
    if raw_calls:
        for index, raw_call in enumerate(raw_calls):
            call, error = _parse_call_payload(
                raw_call, default_id=f"call-{index + 1}", raw=raw_call
            )
            if call is not None:
                native_calls.append(call)
            if error:
                native_errors.append(error)

    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)

    qwen_matches = list(_QWEN_FUNCTION_CALL.finditer(content))
    qwen_calls: list[ToolCall] = []
    qwen_errors: list[str] = []
    for index, match in enumerate(qwen_matches):
        call, error = _parse_qwen_function_call(match, default_id=f"call-{index + 1}")
        if call is not None:
            qwen_calls.append(call)
        if error:
            qwen_errors.append(error)

    # The Qwen text contains the original typed values.  Prefer it over the
    # duplicate native representation, whose adapter may stringify arrays,
    # objects, and numbers.  Preserve a matching native id when available.
    calls = qwen_calls if qwen_calls else native_calls
    if qwen_calls:
        for index, call in enumerate(calls):
            if index < len(native_calls) and native_calls[index].name == call.name:
                calls[index] = ToolCall(
                    native_calls[index].call_id, call.name, call.arguments, call.raw
                )
        errors.extend(qwen_errors)
    else:
        errors.extend(native_errors)
        errors.extend(qwen_errors)

    tagged = list(_TAGGED_TOOL_CALL.finditer(content))
    if tagged or qwen_matches:
        for index, match in enumerate(tagged):
            # Qwen3.5 wraps its parameter syntax in <tool_call>; it has already
            # been parsed above and must not be fed to the JSON-only parser.
            if _QWEN_FUNCTION_CALL.search(match.group(1)):
                continue
            call, error = _parse_call_payload(
                match.group(1),
                default_id=f"call-{len(calls) + index + 1}",
                raw=match.group(0),
            )
            if call is not None:
                calls.append(call)
            if error:
                errors.append(error)
        visible_text = _TAGGED_TOOL_CALL.sub("", content).strip()
        visible_text = _QWEN_FUNCTION_CALL.sub("", visible_text).strip()
        return visible_text, calls, errors

    if not calls:
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, Mapping) and (
                "name" in decoded or "function" in decoded
            ):
                call, error = _parse_call_payload(
                    decoded, default_id="call-1", raw=decoded
                )
                if call is not None:
                    calls.append(call)
                if error:
                    errors.append(error)
                return "", calls, errors
    return content.strip(), calls, errors


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


def schema_error(
    value: Any, schema: Mapping[str, Any], *, path: str = "arguments"
) -> str | None:
    """Return the first JSON-schema violation, or ``None`` when the value fits."""

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
                error = schema_error(item, item_schema, path=f"{path}[{index}]")
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
                    error = schema_error(item, child_schema, path=f"{path}.{name}")
                    if error:
                        return error
        for condition in schema.get("allOf", []) or []:
            if not isinstance(condition, Mapping):
                continue
            if_schema = condition.get("if")
            then_schema = condition.get("then")
            if not isinstance(if_schema, Mapping) or not isinstance(
                then_schema, Mapping
            ):
                continue
            if schema_error(value, if_schema, path=path) is None:
                error = schema_error(value, then_schema, path=path)
                if error:
                    return error
    return None


def coerce_schema_value(value: Any, schema: Mapping[str, Any]) -> Any:
    """Coerce JSON-like strings only when the declared schema permits it.

    Qwen3.5's tagged ``<parameter>`` format can emit Python-style literals
    (``True``/``False``) and native adapters may stringify otherwise typed
    values.  The schema is the only safe source of type information: a field
    declared as a string must remain a string, while a boolean/integer/array
    field may be recovered from its serialized representation.
    """

    expected_type = str(schema.get("type") or "")
    if isinstance(value, str):
        stripped = value.strip()
        if expected_type == "boolean":
            lowered = stripped.casefold()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        elif expected_type == "integer" and re.fullmatch(r"[+-]?\d+", stripped):
            try:
                return int(stripped)
            except ValueError:
                pass
        elif expected_type == "number":
            try:
                decoded = json.loads(stripped)
            except (TypeError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, (int, float)) and not isinstance(decoded, bool):
                return decoded
        elif expected_type in {"array", "object"}:
            try:
                decoded = json.loads(stripped)
            except (TypeError, json.JSONDecodeError):
                decoded = None
            if expected_type == "array" and isinstance(decoded, list):
                value = decoded
            elif expected_type == "object" and isinstance(decoded, Mapping):
                value = decoded

    if expected_type == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            return [coerce_schema_value(item, item_schema) for item in value]
        return value

    if expected_type == "object" and isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return dict(value)
        return {
            name: coerce_schema_value(item, properties[name])
            if isinstance(properties.get(name), Mapping)
            else item
            for name, item in value.items()
        }

    return value
