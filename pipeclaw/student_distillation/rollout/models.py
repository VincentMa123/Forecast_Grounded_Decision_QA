from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass
class PromptCase:
    """A prompt-only evaluation case plus its source record."""

    sample_id: str
    scenario_id: str
    scenario_type: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    source_record: Mapping[str, Any]
    workspace_root: Path | None = None


@dataclass(frozen=True)
class RolloutConfig:
    """Bounded generation limits for one rollout."""

    max_turns: int
    max_new_tokens: int
    temperature: float
    capture_raw_responses: bool = False
    capture_raw_tool_outputs: bool = False


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
        raise NotImplementedError
@dataclass
class RolloutResult:
    """The complete trajectory of one bounded model/tool conversation.

    Partial state is always preserved: a generation failure or a rejected tool
    call is recorded on the result instead of aborting the surrounding suite.
    """

    sample_id: str
    scenario_id: str
    scenario_type: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    trace_status: str = ""
    json_errors: list[str] = field(default_factory=list)
    turns: int = 0
    raw_responses: list[Any] | None = None
    raw_tool_outputs: list[dict[str, Any]] | None = None
    generation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize in the released rollout record shape."""

        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "tool_calls": self.tool_calls,
            "tool_outputs": self.tool_outputs,
            "final_answer": self.final_answer,
            "trace_status": self.trace_status,
            "json_errors": self.json_errors,
            "messages": self.messages,
            "turns": self.turns,
        }
        if self.raw_responses is not None:
            payload["raw_responses"] = self.raw_responses
        if self.raw_tool_outputs is not None:
            payload["raw_tool_outputs"] = self.raw_tool_outputs
        if self.generation_error is not None:
            payload["generation_error"] = self.generation_error
        return payload
