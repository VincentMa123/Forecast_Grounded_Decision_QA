"""Model/tool execution for Task 2 autonomous rollouts.

A rollout only produces a trajectory.  Scoring is the responsibility of
``pipeclaw.backend.evaluator``, so nothing in the execution core imports
evaluation, metric, or teacher-oracle code.  ``.suite`` is the one module that
joins the two, and it is imported explicitly rather than re-exported here.

The re-exports here are limited to the hardware-free execution core.  Import
``.scenarios`` for PipeFormer/OpenClaw dispatchers, ``.swift_generator`` for
model loading, and ``.suite`` for dataset orchestration directly, so that
neither the PipeFormer backend nor torch is pulled in by simply importing this
package.
"""

from __future__ import annotations

from .models import (
    Generator,
    PromptCase,
    RolloutConfig,
    RolloutResult,
    ToolCall,
    as_mapping,
    get_field,
    jsonable,
)
from .prompting import (
    PromptCaseBuilder,
    parse_tool_schemas,
    strip_teacher_future_messages,
)
from .runner import PassthroughPolicy, RolloutPolicy, RolloutRunner
from .tools import (
    ToolDispatcher,
    append_tool_exchange,
    coerce_schema_value,
    parse_tool_calls,
    schema_error,
)

__all__ = [
    "Generator",
    "PassthroughPolicy",
    "PromptCase",
    "PromptCaseBuilder",
    "RolloutConfig",
    "RolloutPolicy",
    "RolloutResult",
    "RolloutRunner",
    "ToolCall",
    "ToolDispatcher",
    "append_tool_exchange",
    "as_mapping",
    "coerce_schema_value",
    "get_field",
    "jsonable",
    "parse_tool_calls",
    "parse_tool_schemas",
    "schema_error",
    "strip_teacher_future_messages",
]
