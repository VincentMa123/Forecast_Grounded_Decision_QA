"""
Agent data models for the trace-first runtime.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TraceSummary(BaseModel):
    trace_path: str
    status: str
    updated_at: Optional[str] = None
    messages_count: int = 0
    tool_calls_count: int = 0
    artifacts_count: int = 0


class SessionHistoryItem(BaseModel):
    session_id: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    status: str = "unknown"
    first_user_message: str = ""
    messages_count: int = 0


class SessionHistoryResponse(BaseModel):
    agent_id: str
    sessions: List[SessionHistoryItem] = Field(default_factory=list)
    total_count: int = 0


class OneTurnAfterRunAgent(BaseModel):
    return_message: str = Field(..., description="Assistant markdown message")
    memory_updates: Optional[List[str]] = Field(default=None)
    generated_artifacts: Optional[List[str]] = Field(default=None)
    trace_summary: TraceSummary


class UIContext(BaseModel):
    date: Optional[str] = None
    selected: Optional[Dict[str, str]] = None
    viewport: Optional[Dict[str, Any]] = None


class ChatOptions(BaseModel):
    mode: Literal["analysis", "dispatch", "auto"] = "auto"
    max_tool_calls: int = 100
    enable_streaming: bool = False
    enable_skills: bool = False


class AgentChatRequest(BaseModel):
    agent_id: str = Field(default="default", description="Agent workspace identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier inside context_trace")
    message: str
    ui_context: Optional[UIContext] = None
    options: Optional[ChatOptions] = Field(default_factory=ChatOptions)
    event_time: Optional[str] = Field(default=None, description="Optional ISO timestamp override used by offline evaluation runs")


class AgentChatResponse(BaseModel):
    agent_id: str
    session_id: str
    message_markdown: str
    trace_summary: TraceSummary
    memory_updates: Optional[List[str]] = None
    generated_artifacts: Optional[List[str]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SessionMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class TraceData(BaseModel):
    session_id: str
    agent_id: str
    created_at: str
    updated_at: str
    status: str
    messages: List[SessionMessage] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    context_injection: Dict[str, Any] = Field(default_factory=dict)
    decision_log: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    memory_commits: List[str] = Field(default_factory=list)


class MemorySummaryResponse(BaseModel):
    agent_id: str
    workspace_root: str
    timeline_files: List[str] = Field(default_factory=list)
    latest_timeline_path: Optional[str] = None
    latest_timeline_excerpt: str = ""
    latest_trace_summary: Optional[TraceSummary] = None


class WorkspaceTreeNode(BaseModel):
    name: str
    path: str
    node_type: Literal["file", "directory"]
    size_bytes: Optional[int] = None
    children: List["WorkspaceTreeNode"] = Field(default_factory=list)


class WorkspaceTreeResponse(BaseModel):
    agent_id: str
    workspace_root: str
    requested_path: str
    total_files: int
    total_directories: int
    tree: WorkspaceTreeNode


class WorkspaceFileResponse(BaseModel):
    agent_id: str
    workspace_root: str
    path: str
    size_bytes: int
    is_text: bool
    truncated: bool = False
    content: str = ""


class CreateAgentRequest(BaseModel):
    agent_id: Optional[str] = Field(default=None, description="Optional custom agent id")


class CreateAgentResponse(BaseModel):
    agent_id: str
    workspace_root: str
    memory_root: str
    assets_root: str
    trace_root: str
    temporary_dir: str
    reports_dir: str
    plan_path: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


WorkspaceTreeNode.model_rebuild()
