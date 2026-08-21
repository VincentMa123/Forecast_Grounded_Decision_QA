from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_timestamp(timestamp: Optional[str] = None) -> str:
    return timestamp or datetime.now().isoformat()


class TraceWriter:
    def __init__(self, trace_root: Path, agent_id: str, plan_path: Path):
        self.trace_root = Path(trace_root)
        self.agent_id = agent_id
        self.plan_path = Path(plan_path)
        self.llm_request_root = self.trace_root / 'llm_requests'
        self.trace_root.mkdir(parents=True, exist_ok=True)
        self.llm_request_root.mkdir(parents=True, exist_ok=True)

    def reset_turn(self) -> None:
        if self.plan_path.exists():
            self.plan_path.unlink()

    def ensure_trace(self, session_id: str, *, created_at: Optional[str] = None) -> Path:
        trace_path = self.trace_root / f"{session_id}.json"
        if not trace_path.exists():
            created_timestamp = _resolve_timestamp(created_at)
            payload = {
                "session_id": session_id,
                "agent_id": self.agent_id,
                "created_at": created_timestamp,
                "updated_at": created_timestamp,
                "status": "active",
                "messages": [],
                "tool_calls": [],
                "audit_tool_calls": [],
                "llm_calls": [],
                "context_injection": {
                    "control_files": [],
                    "memory_files": [],
                    "assets": [],
                    "notes": "",
                },
                "decision_log": [],
                "artifacts": [],
                "memory_commits": [],
            }
            trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return trace_path

    def load_trace(self, session_id: str) -> Dict[str, Any]:
        trace_path = self.ensure_trace(session_id)
        return json.loads(trace_path.read_text(encoding="utf-8"))

    def save_trace(self, session_id: str, payload: Dict[str, Any], *, updated_at: Optional[str] = None) -> Path:
        trace_path = self.ensure_trace(session_id, created_at=payload.get("created_at"))
        payload["updated_at"] = _resolve_timestamp(updated_at)
        trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return trace_path

    def append_message(self, session_id: str, *, role: str, content: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        payload = self.load_trace(session_id)
        resolved_timestamp = _resolve_timestamp(timestamp)
        payload.setdefault("messages", []).append({
            "role": role,
            "content": content,
            "timestamp": resolved_timestamp,
        })
        self.save_trace(session_id, payload, updated_at=resolved_timestamp)
        return payload

    def append_tool_call(
        self,
        session_id: str,
        tool_name: str,
        args: Dict[str, Any],
        result_summary: str,
        extra: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self.load_trace(session_id)
        resolved_timestamp = _resolve_timestamp(timestamp)
        record = {
            "tool_name": tool_name,
            "args": args,
            "result_summary": result_summary,
            "timestamp": resolved_timestamp,
        }
        if extra:
            record.update(extra)
        record["sequence"] = len(payload.get("tool_calls") or []) + len(
            payload.get("audit_tool_calls") or []
        ) + 1
        payload.setdefault("tool_calls", []).append(record)
        self.save_trace(session_id, payload, updated_at=resolved_timestamp)
        return payload

    def append_audit_tool_call(
        self,
        session_id: str,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        *,
        tool_call_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retain failed/internal calls for audit without exporting them to SFT."""
        payload = self.load_trace(session_id)
        resolved_timestamp = _resolve_timestamp(timestamp)
        record = {
            "tool_name": tool_name,
            "args": args,
            "result": result,
            "timestamp": resolved_timestamp,
            "teacher_trace_eligible": False,
            "sequence": len(payload.get("tool_calls") or []) + len(
                payload.get("audit_tool_calls") or []
            ) + 1,
        }
        if tool_call_id:
            record["tool_call_id"] = tool_call_id
        payload.setdefault("audit_tool_calls", []).append(record)
        self.save_trace(
            session_id,
            payload,
            updated_at=resolved_timestamp,
        )
        return payload

    def append_llm_call(
        self,
        session_id: str,
        *,
        step: int,
        phase: str,
        payload_data: Dict[str, Any],
        summary: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        trace_payload = self.load_trace(session_id)
        llm_calls = trace_payload.setdefault("llm_calls", [])
        call_index = len(llm_calls) + 1
        resolved_timestamp = _resolve_timestamp(timestamp)
        session_root = self.llm_request_root / session_id
        session_root.mkdir(parents=True, exist_ok=True)
        payload_path = session_root / f"{call_index:03d}_step_{step:02d}_{phase}.json"
        payload_path.write_text(json.dumps(payload_data, ensure_ascii=False, indent=2), encoding="utf-8")
        record = {
            "index": call_index,
            "step": step,
            "phase": phase,
            "timestamp": resolved_timestamp,
            "payload_path": payload_path.as_posix(),
        }
        if summary:
            record.update(summary)
        llm_calls.append(record)
        self.save_trace(session_id, trace_payload, updated_at=resolved_timestamp)
        return record

    def set_context_injection(self, session_id: str, injection: Dict[str, Any], *, timestamp: Optional[str] = None) -> Dict[str, Any]:
        payload = self.load_trace(session_id)
        payload["context_injection"] = injection
        self.save_trace(session_id, payload, updated_at=timestamp)
        return payload

    def append_decision(self, session_id: str, summary: str, artifact_paths: Optional[List[str]] = None, *, timestamp: Optional[str] = None) -> Dict[str, Any]:
        payload = self.load_trace(session_id)
        resolved_timestamp = _resolve_timestamp(timestamp)
        payload.setdefault("decision_log", []).append({
            "summary": summary,
            "artifact_paths": artifact_paths or [],
            "timestamp": resolved_timestamp,
        })
        self.save_trace(session_id, payload, updated_at=resolved_timestamp)
        return payload

    def extend_artifacts(self, session_id: str, artifacts: List[str], *, timestamp: Optional[str] = None) -> Dict[str, Any]:
        payload = self.load_trace(session_id)
        existing = payload.setdefault("artifacts", [])
        for artifact in artifacts:
            if artifact and artifact not in existing:
                existing.append(artifact)
        self.save_trace(session_id, payload, updated_at=timestamp)
        return payload

    def extend_memory_commits(self, session_id: str, memory_commits: List[str], *, timestamp: Optional[str] = None) -> Dict[str, Any]:
        payload = self.load_trace(session_id)
        existing = payload.setdefault("memory_commits", [])
        for item in memory_commits:
            if item and item not in existing:
                existing.append(item)
        self.save_trace(session_id, payload, updated_at=timestamp)
        return payload

    def set_status(self, session_id: str, status: str, *, timestamp: Optional[str] = None) -> Dict[str, Any]:
        payload = self.load_trace(session_id)
        payload["status"] = status
        self.save_trace(session_id, payload, updated_at=timestamp)
        return payload

    def summarize(self, session_id: str) -> Dict[str, Any]:
        payload = self.load_trace(session_id)
        return {
            "trace_path": (self.trace_root / f"{session_id}.json").as_posix(),
            "status": payload.get("status", "unknown"),
            "updated_at": payload.get("updated_at"),
            "messages_count": len(payload.get("messages", [])),
            "tool_calls_count": len(payload.get("tool_calls", [])),
            "llm_calls_count": len(payload.get("llm_calls", [])),
            "artifacts_count": len(payload.get("artifacts", [])),
        }
