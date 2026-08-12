from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from pipeclaw.backend.grounding.evidence.tool import tool_output_failed


@dataclass(frozen=True)
class QualityContext:
    answer: str
    question: str
    pipeformer: Optional[Dict[str, Any]]
    conversation_context: tuple[Dict[str, Any], ...]
    tool_outputs: tuple[Dict[str, Any], ...]
    trusted_tool_outputs: tuple[Dict[str, Any], ...]
    record_evidence: Dict[str, Any]
    grounding_evidence: Dict[str, Any]


def build_quality_context(
    *,
    answer: str,
    question: str,
    pipeformer: Optional[Dict[str, Any]],
    conversation_context: Optional[Iterable[Dict[str, Any]]] = None,
    tool_outputs: Optional[Iterable[Dict[str, Any]]] = None,
    record_evidence: Optional[Dict[str, Any]] = None,
) -> QualityContext:
    """Build the immutable trusted view used by live and offline validation."""

    trusted_context = trusted_conversation_context(conversation_context or [])
    normalized_outputs = tuple(dict(item) for item in tool_outputs or [])
    trusted_tool_outputs = tuple(
        item for item in normalized_outputs if not tool_output_failed(item)
    )
    normalized_pipeformer = dict(pipeformer) if pipeformer else None
    normalized_record_evidence = dict(record_evidence or {})
    return QualityContext(
        answer=answer,
        question=question,
        pipeformer=normalized_pipeformer,
        conversation_context=tuple(trusted_context),
        tool_outputs=normalized_outputs,
        trusted_tool_outputs=trusted_tool_outputs,
        record_evidence=normalized_record_evidence,
        grounding_evidence={
            "pipeformer": normalized_pipeformer or {},
            "conversation_context": trusted_context,
            "tool_outputs": list(trusted_tool_outputs),
            "record_evidence": normalized_record_evidence,
        },
    )


def trusted_conversation_context(
    items: Iterable[Dict[str, Any]],
    *,
    verified_evidence_only: bool = False,
) -> List[Dict[str, Any]]:
    """Retain trusted context, optionally projecting only verified evidence."""

    if not verified_evidence_only:
        return [
            dict(item)
            if item.get("grounding_verified") is True
            else {key: value for key, value in item.items() if key != "assistant_output"}
            for item in items
        ]
    context = []
    for item in items:
        if item.get("grounding_verified") is True:
            context.append(dict(item))
            continue
        summary = item.get("verified_evidence_summary")
        legacy_verified_summary = (
            "tool_evidence_verified" not in item
            and isinstance(summary, dict)
            and bool(summary)
        )
        if item.get("tool_evidence_verified") is True or legacy_verified_summary:
            context.append({
                "tool_evidence_verified": True,
                "evidence_artifacts": list(item.get("evidence_artifacts") or []),
                "verified_evidence_summary": summary,
            })
    return context
