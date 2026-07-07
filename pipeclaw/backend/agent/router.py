"""
Agent API routes for the trace-first runtime.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from . import orchestrator as orchestrator_module
from .agent_workspace_manager import AgentWorkspaceManager
from .schemas import (
    AgentChatRequest,
    CreateAgentRequest,
    CreateAgentResponse,
    MemorySummaryResponse,
    SessionHistoryItem,
    SessionHistoryResponse,
    TraceData,
    TraceSummary,
    WorkspaceFileResponse,
    WorkspaceTreeNode,
    WorkspaceTreeResponse,
)
from .services.memory_manager import MemoryManager
from .trace_writer import TraceWriter

router = APIRouter(prefix="/api/agent", tags=["agent"])
_SESSION_COUNTER_FILE = Path(__file__).resolve().parent.parent / ".openclaw" / ".session_counter"
_AGENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TEXT_FILE_SUFFIXES = {
    ".md",
    ".txt",
    ".json",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".css",
    ".html",
    ".yml",
    ".yaml",
    ".csv",
    ".log",
    ".sh",
    ".env",
}
_MAX_FILE_PREVIEW_BYTES = 120_000
_SESSION_TITLE_MAX_LENGTH = 80


def _get_next_session_id() -> int:
    _SESSION_COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _SESSION_COUNTER_FILE.exists():
        try:
            counter = int(_SESSION_COUNTER_FILE.read_text().strip())
        except Exception:
            counter = 0
    else:
        counter = 0
    next_id = counter + 1
    _SESSION_COUNTER_FILE.write_text(str(next_id), encoding="utf-8")
    return next_id


def _generate_session_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_get_next_session_id()}"


def _generate_agent_id() -> str:
    return f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _resolve_agent_id(raw_agent_id: Optional[str]) -> str:
    if raw_agent_id is None:
        return _generate_agent_id()
    agent_id = raw_agent_id.strip()
    if not agent_id:
        return _generate_agent_id()
    if not _AGENT_ID_PATTERN.fullmatch(agent_id):
        raise HTTPException(status_code=400, detail="agent_id 只能包含字母、数字、下划线和中划线，且长度不能超过 64")
    return agent_id


def _get_workspace_manager(agent_id: str) -> AgentWorkspaceManager:
    backend_root = Path(__file__).resolve().parent.parent
    return AgentWorkspaceManager(backend_root, agent_id)


def _resolve_workspace_path(workspace_root: Path, relative_path: str) -> Path:
    normalized = (relative_path or ".").strip().replace("\\", "/")
    candidate = (workspace_root / normalized).resolve()
    workspace_resolved = workspace_root.resolve()
    try:
        candidate.relative_to(workspace_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法路径，不能越过 workspace 根目录") from exc
    return candidate


def _count_tree(node: WorkspaceTreeNode) -> tuple[int, int]:
    if node.node_type == "file":
        return 1, 0
    files = 0
    directories = 1
    for child in node.children:
        child_files, child_directories = _count_tree(child)
        files += child_files
        directories += child_directories
    return files, directories


def _build_tree_node(path: Path, workspace_root: Path, depth: int) -> WorkspaceTreeNode:
    relative_path = "." if path == workspace_root else path.relative_to(workspace_root).as_posix()
    if path.is_file():
        return WorkspaceTreeNode(
            name=path.name,
            path=relative_path,
            node_type="file",
            size_bytes=path.stat().st_size,
            children=[],
        )

    children: List[WorkspaceTreeNode] = []
    if depth != 0:
        next_depth = depth - 1 if depth > 0 else -1
        entries = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        for child in entries:
            children.append(_build_tree_node(child, workspace_root, next_depth))
    return WorkspaceTreeNode(
        name=path.name if path != workspace_root else workspace_root.name,
        path=relative_path,
        node_type="directory",
        size_bytes=None,
        children=children,
    )


def _normalize_session_title(content: str) -> str:
    normalized = " ".join((content or "").split())
    if len(normalized) <= _SESSION_TITLE_MAX_LENGTH:
        return normalized
    return normalized[:_SESSION_TITLE_MAX_LENGTH].rstrip() + "..."


def _summarize_trace_file(trace_path: Path) -> Optional[SessionHistoryItem]:
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    messages = payload.get("messages", [])
    first_user_message = ""
    for item in messages:
        if item.get("role") == "user":
            first_user_message = _normalize_session_title(str(item.get("content") or ""))
            break

    stat = trace_path.stat()
    updated_at = payload.get("updated_at") or datetime.fromtimestamp(stat.st_mtime).isoformat()
    created_at = payload.get("created_at") or datetime.fromtimestamp(stat.st_ctime).isoformat()

    return SessionHistoryItem(
        session_id=str(payload.get("session_id") or trace_path.stem),
        created_at=created_at,
        updated_at=updated_at,
        status=str(payload.get("status") or "unknown"),
        first_user_message=first_user_message,
        messages_count=len(messages),
    )


def _load_template_questions() -> list[dict]:
    llm_root = Path(__file__).resolve().parent.parent / "llm_data"
    for candidate in [
        llm_root / "question_template_80.json",
        llm_root / "question_template_all.json",
        llm_root / "question_all.json",
    ]:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and isinstance(data.get("questions"), list):
                return data["questions"]
    raise FileNotFoundError("未找到可用的问题模板文件")


def _default_placeholder_value(var_name: str, date: Optional[str], selected_type: Optional[str], selected_id: Optional[str]) -> str:
    defaults = {
        "date": date or "2019-01-01",
        "pipeline_division": selected_id if selected_type == "system" and selected_id else "中俄东线",
        "station_name": selected_id if selected_type == "node" and selected_id else "临河站",
        "supply_point": selected_id if selected_type == "node" and selected_id else "霍尔果斯首站",
        "province": "江苏",
        "K": "10",
        "top_k": "10",
        "N": "7",
        "P": "95",
        "z": "2.0",
        "pct_threshold": "0.3",
        "pct": "0",
        "start_date": "2019-01-01",
        "end_date": date or "2019-01-14",
        "date_a": "2019-01-01",
        "date_b": date or "2019-01-02",
        "lon": "105.0",
        "lat": "35.0",
        "export_amount": "100",
        "critical_user_share_pct": "8.0",
        "min_abs_flow": "0.0",
        "min_active_consumption": "0.1",
        "history_days": "14",
        "zero_flow_eps": "0.0",
        "days": "7",
        "K_users": "5",
        "K_points": "3",
        "supply_drop_pct": "20",
    }
    return defaults.get(var_name, "示例值")


def _format_sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def get_orchestrator():
    return orchestrator_module.orchestrator


@router.post("/create", response_model=CreateAgentResponse)
async def create_agent(request: Optional[CreateAgentRequest] = None):
    orchestrator = get_orchestrator()
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent 服务未初始化")
    agent_id = _resolve_agent_id(request.agent_id if request else None)
    backend_root = Path(__file__).resolve().parent.parent
    workspace_root = backend_root / ".openclaw" / f"workspace-{agent_id}"
    if workspace_root.exists():
        raise HTTPException(status_code=409, detail=f"agent_id 已存在: {agent_id}")
    workspace_manager = AgentWorkspaceManager(backend_root, agent_id)
    workspace_info = workspace_manager.as_dict()
    return CreateAgentResponse(agent_id=agent_id, **workspace_info)


@router.post("/chat")
async def chat(request: AgentChatRequest):
    orchestrator = get_orchestrator()
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Agent 服务未初始化")
    session_id = request.session_id or _generate_session_id()
    session_orchestrator = orchestrator_module.AgentOrchestrator(
        data_loader=orchestrator.data_loader,
        agent_id=request.agent_id,
        session_id=session_id,
        enable_skills=bool(request.options.enable_skills) if request.options else False,
    )

    def event_stream():
        try:
            for payload in session_orchestrator.run_agent_stream(request):
                yield _format_sse_event(payload["event"], payload["data"])
        except Exception as exc:
            yield _format_sse_event("error", {"message": f"流式对话失败: {exc}", "timestamp": datetime.now().isoformat()})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sessions/{agent_id}", response_model=SessionHistoryResponse)
async def list_sessions(agent_id: str):
    workspace_manager = _get_workspace_manager(agent_id)
    sessions: List[SessionHistoryItem] = []
    for trace_path in workspace_manager.trace_root.glob("*.json"):
        summary = _summarize_trace_file(trace_path)
        if summary is not None:
            sessions.append(summary)
    sessions.sort(key=lambda item: (item.updated_at or item.created_at or "", item.session_id), reverse=True)
    return SessionHistoryResponse(agent_id=agent_id, sessions=sessions, total_count=len(sessions))


@router.get("/trace/{agent_id}/{session_id}", response_model=TraceData)
async def get_trace(agent_id: str, session_id: str):
    workspace_manager = _get_workspace_manager(agent_id)
    trace_writer = TraceWriter(workspace_manager.trace_root, agent_id, workspace_manager.plan_path)
    return TraceData(**trace_writer.load_trace(session_id))


@router.get("/memory/{agent_id}/summary", response_model=MemorySummaryResponse)
async def get_memory_summary(agent_id: str, session_id: Optional[str] = None):
    workspace_manager = _get_workspace_manager(agent_id)
    memory_manager = MemoryManager(workspace_manager.workspace_root)
    summary = memory_manager.get_summary()
    latest_trace_summary = None
    if session_id:
        trace_writer = TraceWriter(workspace_manager.trace_root, agent_id, workspace_manager.plan_path)
        latest_trace_summary = TraceSummary(**trace_writer.summarize(session_id))
    return MemorySummaryResponse(agent_id=agent_id, latest_trace_summary=latest_trace_summary, **summary)


@router.get("/workspace/{agent_id}/tree", response_model=WorkspaceTreeResponse)
async def get_workspace_tree(agent_id: str, path: str = ".", max_depth: int = 4):
    workspace_manager = _get_workspace_manager(agent_id)
    target = _resolve_workspace_path(workspace_manager.workspace_root, path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在: {path}")
    if max_depth < -1 or max_depth > 8:
        raise HTTPException(status_code=400, detail="max_depth 必须在 -1 到 8 之间")
    tree = _build_tree_node(target, workspace_manager.workspace_root, max_depth)
    total_files, total_directories = _count_tree(tree)
    return WorkspaceTreeResponse(
        agent_id=agent_id,
        workspace_root=workspace_manager.workspace_root.as_posix(),
        requested_path="." if target == workspace_manager.workspace_root else target.relative_to(workspace_manager.workspace_root).as_posix(),
        total_files=total_files,
        total_directories=total_directories,
        tree=tree,
    )


@router.get("/workspace/{agent_id}/file", response_model=WorkspaceFileResponse)
async def get_workspace_file(agent_id: str, path: str):
    workspace_manager = _get_workspace_manager(agent_id)
    target = _resolve_workspace_path(workspace_manager.workspace_root, path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")

    size_bytes = target.stat().st_size
    suffix = target.suffix.lower()
    is_text = suffix in _TEXT_FILE_SUFFIXES
    content = ""
    truncated = False

    if is_text:
        raw = target.read_text(encoding="utf-8", errors="replace")
        if len(raw.encode("utf-8")) > _MAX_FILE_PREVIEW_BYTES:
            encoded = raw.encode("utf-8")[:_MAX_FILE_PREVIEW_BYTES]
            content = encoded.decode("utf-8", errors="replace")
            truncated = True
        else:
            content = raw

    return WorkspaceFileResponse(
        agent_id=agent_id,
        workspace_root=workspace_manager.workspace_root.as_posix(),
        path=target.relative_to(workspace_manager.workspace_root).as_posix(),
        size_bytes=size_bytes,
        is_text=is_text,
        truncated=truncated,
        content=content,
    )


@router.get("/suggestions")
async def get_suggestions(date: Optional[str] = None, selected_type: Optional[str] = None, selected_id: Optional[str] = None):
    try:
        questions = _load_template_questions()
        suggestions = []
        for idx, question in enumerate(questions):
            category = str(question.get("category", "未分类"))
            if "基础问题" not in category and "当前运行情况" not in category:
                continue
            question_text = question.get("template_question") or question.get("question") or ""
            placeholders = sorted(set(re.findall(r"\{([A-Za-z0-9_]+)\}", question_text)))
            for var_name in placeholders:
                value = _default_placeholder_value(var_name, date, selected_type, selected_id)
                question_text = question_text.replace(f"{{{var_name}}}", value)
            suggestions.append(
                {
                    "id": question.get("question_id") or question.get("template_id") or f"suggestion_{idx}",
                    "category": category,
                    "question": question_text,
                    "intent": question.get("template_id") or question.get("question_id") or "template",
                }
            )
        return {"suggestions": suggestions[:10], "total": len(suggestions)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取问题建议失败: {str(exc)}")


@router.get("/templates")
async def get_question_templates():
    try:
        return {"questions": _load_template_questions()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"获取问题模板失败: {str(exc)}")


@router.get("/health")
async def health_check():
    orchestrator = get_orchestrator()
    if not orchestrator:
        return {
            "status": "not_initialized",
            "mode": "mock",
            "config": {"has_api_key": False, "api_key_prefix": None, "has_custom_base": False, "api_base": "", "model": ""},
            "data": {"available_dates": 0, "date_range": {"start": None, "end": None}},
        }
    try:
        available_dates = orchestrator.data_loader.get_available_dates("node_flow")
    except Exception:
        available_dates = []
    api_key = orchestrator.api_key or ""
    return {
        "status": "ok",
        "mode": "production" if api_key else "mock",
        "config": {
            "has_api_key": bool(api_key),
            "api_key_prefix": (api_key[:8] + "...") if api_key else None,
            "has_custom_base": bool(orchestrator.api_base),
            "api_base": orchestrator.api_base or "",
            "model": orchestrator.model,
        },
        "data": {
            "available_dates": len(available_dates),
            "date_range": {"start": available_dates[0] if available_dates else None, "end": available_dates[-1] if available_dates else None},
        },
    }
