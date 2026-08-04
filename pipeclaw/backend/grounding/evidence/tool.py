from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Tuple


DATA_FILE_REFERENCE = re.compile(
    r"(?i)(?<![\w.-])[\w.-]+\.(?:csv|jsonl?|xlsx?|parquet)(?![\w.-])"
)
NO_EVIDENCE_OUTPUT = re.compile(
    r"(?im)^\s*(?:NOT_FOUND|NO_MATCH|File Not Found)\s*$"
    r"|cannot find (?:the )?(?:file|path) specified"
    r"|no such file or directory"
    r"|找不到(?:指定的)?(?:文件|路径)|未找到(?:请求的)?(?:文件|数据)"
)
SERIALIZED_NO_EVIDENCE_OUTPUT = re.compile(
    r'(?i)"(?:stdout|stderr)"\s*:\s*"(?:NOT_FOUND|NO_MATCH|File Not Found)(?:\\r\\n|\\n)?"'
)
LOCATOR_COMMAND = re.compile(
    r"(?i)\b(?:where|dir|get-childitem|rg\s+--files|find\s+)\b"
)
CONTENT_COMMAND = re.compile(
    r"(?i)\b(?:type|get-content|import-csv|select-string|python)\b"
    r"|\bopen\s*\(|csv\.(?:reader|dictreader)"
)


class ToolEvidenceState(str, Enum):
    EXECUTION_FAILED = "execution_failed"
    NO_EVIDENCE = "no_evidence"
    LOCATOR_ONLY = "locator_only"
    CONTENT_EVIDENCE = "content_evidence"


@dataclass(frozen=True)
class ToolEvidenceAssessment:
    state: ToolEvidenceState
    reason: str
    matched_artifacts: Tuple[str, ...] = ()

    @property
    def evidence_found(self) -> bool:
        return self.state is ToolEvidenceState.CONTENT_EVIDENCE


def requested_artifacts(text: str) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(value.casefold() for value in DATA_FILE_REFERENCE.findall(text)))


def _tool_arguments(wrapper: dict[str, Any]) -> dict[str, Any]:
    value = wrapper.get("arguments") or wrapper.get("args") or {}
    return value if isinstance(value, dict) else {}


def _source_artifact_names(wrapper: dict[str, Any], output: dict[str, Any], tool_name: str) -> Tuple[str, ...]:
    arguments = _tool_arguments(wrapper)
    candidates = []
    candidates.extend(output.get("source_artifacts") or [])
    if tool_name == "read_file":
        candidates.extend([output.get("path"), output.get("abs_path"), arguments.get("path")])
    elif tool_name == "run_command":
        command = output.get("cmd") or arguments.get("cmd") or []
        if isinstance(command, list):
            candidates.extend(command)
        else:
            candidates.append(command)
    names = []
    for candidate in candidates:
        names.extend(DATA_FILE_REFERENCE.findall(str(candidate or "")))
    return tuple(dict.fromkeys(value.casefold() for value in names))


def attach_tool_arguments(tool_outputs: Iterable[dict[str, Any]], tool_calls: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    calls_by_id = {
        str(item.get("tool_call_id") or ""): item
        for item in tool_calls
        if item.get("tool_call_id")
    }
    result = []
    for item in tool_outputs:
        enriched = dict(item)
        call = calls_by_id.get(str(item.get("tool_call_id") or ""))
        if call:
            enriched["arguments"] = call.get("arguments") or call.get("args") or {}
        result.append(enriched)
    return result


def classify_tool_evidence(
    item: Any,
    *,
    requested: Iterable[str] = (),
) -> ToolEvidenceAssessment:
    wrapper = item if isinstance(item, dict) else {}
    output = wrapper.get("output") if "output" in wrapper else item
    if not isinstance(output, dict):
        if isinstance(output, str) and NO_EVIDENCE_OUTPUT.search(output):
            return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "negative_result_sentinel")
        if output not in (None, ""):
            return ToolEvidenceAssessment(ToolEvidenceState.CONTENT_EVIDENCE, "nonempty_tool_result")
        return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "empty_tool_result")

    if (
        output.get("success") is False
        or bool(output.get("error"))
        or output.get("exit_code") not in (None, 0)
    ):
        return ToolEvidenceAssessment(ToolEvidenceState.EXECUTION_FAILED, "tool_execution_failed")

    text = "\n".join(
        str(output.get(key) or "")
        for key in ("stdout", "stderr", "content", "message", "evidence_excerpt")
    )
    if NO_EVIDENCE_OUTPUT.search(text) or SERIALIZED_NO_EVIDENCE_OUTPUT.search(text):
        return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "negative_result_sentinel")

    tool_name = str(wrapper.get("name") or output.get("tool") or "").casefold()
    source_names = _source_artifact_names(wrapper, output, tool_name)
    requested_set = {str(value).casefold() for value in requested}
    matched = tuple(sorted(value for value in requested_set if value in source_names))
    if output.get("evidence_kind") in {"file_content", "command_content"}:
        if requested_set and not matched:
            return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "requested_artifact_not_read")
        if not text.strip():
            return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "empty_compact_evidence", matched)
        return ToolEvidenceAssessment(ToolEvidenceState.CONTENT_EVIDENCE, "compact_content_evidence", matched)

    if requested_set and tool_name in {"read_file", "run_command"} and not matched:
        return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "requested_artifact_not_read")

    arguments = _tool_arguments(wrapper)
    command_value = output.get("cmd") or arguments.get("cmd") or []
    if isinstance(command_value, list):
        command = " ".join(str(value) for value in command_value)
    else:
        command = str(command_value)
    if tool_name == "read_file":
        content = str(output.get("content") or output.get("evidence_excerpt") or "").strip()
        if content:
            return ToolEvidenceAssessment(ToolEvidenceState.CONTENT_EVIDENCE, "file_content_read", matched)
        return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "read_returned_no_content", matched)
    if tool_name == "run_command":
        if LOCATOR_COMMAND.search(command) and not CONTENT_COMMAND.search(command):
            return ToolEvidenceAssessment(ToolEvidenceState.LOCATOR_ONLY, "artifact_location_only", matched)
        if requested_set and not CONTENT_COMMAND.search(command):
            return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "command_did_not_read_requested_data", matched)
        if CONTENT_COMMAND.search(command) and not text.strip():
            return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "command_returned_no_content", matched)
        if not text.strip():
            return ToolEvidenceAssessment(ToolEvidenceState.CONTENT_EVIDENCE, "command_action_confirmed", matched)
        return ToolEvidenceAssessment(ToolEvidenceState.CONTENT_EVIDENCE, "command_content_or_computation", matched)
    if requested_set:
        return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "requested_artifact_not_read")
    if tool_name in {"write_file", "edit_file"}:
        return ToolEvidenceAssessment(ToolEvidenceState.CONTENT_EVIDENCE, "file_action_confirmed")
    payload_keys = set(output) - {"success", "tool", "session_id", "timestamp", "error", "warnings", "workspace"}
    if not text.strip() and not payload_keys:
        return ToolEvidenceAssessment(ToolEvidenceState.NO_EVIDENCE, "empty_tool_result")
    return ToolEvidenceAssessment(ToolEvidenceState.CONTENT_EVIDENCE, "successful_tool_result", matched)


def tool_output_failed(output: Any) -> bool:
    """Compatibility predicate: only evidence-bearing output can ground a trace."""
    return not classify_tool_evidence(output).evidence_found
