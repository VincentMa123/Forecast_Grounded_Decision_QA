"""
Trace-first agent orchestrator.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .schemas import AgentChatRequest, AgentChatResponse, OneTurnAfterRunAgent, TraceSummary
from .services import MemoryAssembler, MemoryManager
from .agent_workspace_manager import AgentWorkspaceManager
from .prompt_builder import PromptBuilder
from .trace_writer import TraceWriter
from .tools.registry import tool_registry
from .skills.skill_manager import SkillManager
from executor.runner import get_runner

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    def __init__(
        self,
        data_loader=None,
        *,
        agent_id: str = "default",
        session_id: str = "default",
        enable_skills: Optional[bool] = None,
        request_timeout_seconds: Optional[float] = None,
        workspace_root_base: Optional[Path] = None,
    ):
        self.data_loader = data_loader
        self.agent_id = agent_id or "default"
        self.session_id = session_id
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.max_steps = int(os.getenv("MAX_AGENT_STEPS", "30"))
        raw_timeout = request_timeout_seconds
        if raw_timeout is None:
            raw_timeout = os.getenv("OPENAI_TIMEOUT_SECONDS")
        self.request_timeout_seconds = float(raw_timeout) if raw_timeout not in (None, "", 0, "0") else None
        self.enable_skills = bool(enable_skills)
        self.backend_root = Path(__file__).resolve().parent.parent
        self.workspace_root_base = Path(workspace_root_base) if workspace_root_base is not None else (self.backend_root / ".openclaw")
        self.workspace_manager = AgentWorkspaceManager(self.backend_root, self.agent_id, workspace_root_base=self.workspace_root_base)
        self.workspace_dir = self.workspace_manager.workspace_root
        self.trace_writer = TraceWriter(self.workspace_manager.trace_root, self.agent_id, self.workspace_manager.plan_path)
        self.memory_manager = MemoryManager(self.workspace_dir)
        self.memory_assembler = MemoryAssembler(self.workspace_dir)
        self.prompt_builder = PromptBuilder(self.workspace_dir)
        self.skill_manager = SkillManager(self.backend_root / "agent" / "skills") if self.enable_skills else None
        runner = get_runner()
        runner.set_workspace_root(self.workspace_root_base)
        runner.set_active_agent(self.agent_id)
        self._init_tools()

    def _init_tools(self) -> None:
        from .tools.workspace_tools import WorkspaceTools
        from .tools.pipeformer_tools import register_pipeformer_tools
        WorkspaceTools(self.session_id)
        register_pipeformer_tools(self.backend_root)

    def _build_history_block(self, trace_messages: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for item in trace_messages[-12:]:
            role = str(item.get("role", "")).lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _build_user_context(self, request: AgentChatRequest, trace_payload: Dict[str, Any]) -> str:
        parts: List[str] = []
        if request.ui_context:
            parts.append("## UI Context")
            if request.ui_context.date:
                parts.append(f"- date: {request.ui_context.date}")
            if request.ui_context.selected:
                parts.append(f"- selected: {request.ui_context.selected.get('type')} {request.ui_context.selected.get('id')}")
        history = self._build_history_block(trace_payload.get("messages", []))
        if history:
            parts.append("\n## Trace History")
            parts.append(history)
        parts.append("\n## Current User Request")
        parts.append(request.message)
        return "\n".join(parts).strip()

    def _safe_parse_args(self, raw_args: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(raw_args) if raw_args else {}
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _tool_loop_stream(self, system_prompt: str, user_context: str, *, event_time: Optional[str] = None):
        if not self.api_key:
            message = "Agent service is missing OPENAI_API_KEY; cannot run a live model call."
            timestamp = event_time or datetime.now().isoformat()
            self.trace_writer.append_message(self.session_id, role="assistant", content=message, timestamp=timestamp)
            yield {"event": "assistant_message", "data": {"content": message, "timestamp": timestamp, "final": True}}
            return message, "error"

        client_kwargs = {"api_key": self.api_key, "base_url": self.api_base}
        if self.request_timeout_seconds is not None:
            client_kwargs["timeout"] = self.request_timeout_seconds
        client = OpenAI(**client_kwargs)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_context}]
        tools_schema = tool_registry.openai_tools_schema()
        last_request_snapshot: Optional[Dict[str, Any]] = None

        try:
            for step_index in range(self.max_steps):
                request_payload = {
                    "model": self.model,
                    "messages": messages,
                    "tools": tools_schema,
                    "tool_choice": "auto",
                    "temperature": 0.2,
                }
                last_request_snapshot = self.trace_writer.append_llm_call(
                    self.session_id,
                    step=step_index + 1,
                    phase="request",
                    payload_data=request_payload,
                    summary={"model": self.model},
                    timestamp=event_time,
                )
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=0.2,
                )
                choice = response.choices[0]
                message = choice.message
                finish_reason = choice.finish_reason
                tool_calls = message.tool_calls or []
                content = (message.content or "").strip()
                try:
                    response_trace = response.model_dump(mode="json")
                except Exception as exc:
                    response_trace = {
                        "serialization_error": str(exc),
                        "response_repr": repr(response),
                    }
                self.trace_writer.append_llm_call(
                    self.session_id,
                    step=step_index + 1,
                    phase="response",
                    payload_data={"finish_reason": finish_reason, "response": response_trace},
                    summary={"model": self.model, "finish_reason": finish_reason},
                    timestamp=event_time,
                )

                if finish_reason == "tool_calls" and not tool_calls:
                    logger.warning("Model returned finish_reason=tool_calls without tool_calls; retrying without tools")
                    last_request_snapshot = self.trace_writer.append_llm_call(
                        self.session_id,
                        step=step_index + 1,
                        phase="fallback_request",
                        payload_data={"model": self.model, "messages": messages, "temperature": 0.2},
                        summary={"model": self.model},
                        timestamp=event_time,
                    )
                    fallback_response = client.chat.completions.create(model=self.model, messages=messages, temperature=0.2)
                    fallback_choice = fallback_response.choices[0]
                    response = fallback_response
                    message = fallback_choice.message
                    finish_reason = fallback_choice.finish_reason
                    tool_calls = message.tool_calls or []
                    content = (message.content or "").strip()
                    try:
                        response_trace = response.model_dump(mode="json")
                    except Exception as exc:
                        response_trace = {
                            "serialization_error": str(exc),
                            "response_repr": repr(response),
                        }
                    self.trace_writer.append_llm_call(
                        self.session_id,
                        step=step_index + 1,
                        phase="fallback_response",
                        payload_data={"finish_reason": finish_reason, "response": response_trace},
                        summary={"model": self.model, "finish_reason": finish_reason},
                        timestamp=event_time,
                    )

                if content:
                    event_timestamp = event_time or datetime.now().isoformat()
                    yield {
                        "event": "assistant_message",
                        "data": {
                            "content": content,
                            "timestamp": event_timestamp,
                            "final": not bool(tool_calls),
                        },
                    }

                if not tool_calls:
                    final_text = content
                    if not final_text and finish_reason == "tool_calls":
                        final_text = "Model reported tool_calls without returning tool_calls. Check whether the current gateway supports OpenAI function calling."
                    if not final_text:
                        final_text = "Model did not return any displayable content."
                    self.trace_writer.append_message(self.session_id, role="assistant", content=final_text, timestamp=event_time)
                    if not content or final_text != content:
                        event_timestamp = event_time or datetime.now().isoformat()
                        yield {
                            "event": "assistant_message",
                            "data": {
                                "content": final_text,
                                "timestamp": event_timestamp,
                                "final": True,
                            },
                        }
                    return final_text, "completed"

                assistant_tool_calls = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": assistant_tool_calls,
                    }
                )

                for call in tool_calls:
                    args = self._safe_parse_args(call.function.arguments)
                    tool_input: Dict[str, Any] = args if args is not None else {"_raw_arguments": call.function.arguments}
                    tool_timestamp = event_time or datetime.now().isoformat()
                    yield {
                        "event": "tool_start",
                        "data": {
                            "tool_call_id": call.id,
                            "tool": call.function.name,
                            "input": tool_input,
                            "timestamp": tool_timestamp,
                        },
                    }

                    if args is None:
                        result = {
                            "success": False,
                            "error": "Invalid JSON arguments",
                            "tool": call.function.name,
                            "raw_arguments": call.function.arguments,
                        }
                    else:
                        try:
                            result = tool_registry.execute(call.function.name, session_id=self.session_id, agent_id=self.agent_id, **args)
                        except Exception as exc:
                            result = {"success": False, "error": str(exc), "tool": call.function.name, "params": args}

                    result_summary = json.dumps(result, ensure_ascii=False)[:4000]
                    trace_extra = {"tool_call_id": call.id, "result": result}
                    if isinstance(result, dict) and result.get("error"):
                        trace_extra["failure_context"] = {
                            "finish_reason": finish_reason,
                            "assistant_message": {
                                "content": content,
                                "tool_calls": assistant_tool_calls,
                            },
                            "llm_response": response_trace,
                        }
                    self.trace_writer.append_tool_call(
                        self.session_id,
                        call.function.name,
                        args or {},
                        result_summary,
                        extra=trace_extra,
                        timestamp=event_time,
                    )
                    tool_end_timestamp = event_time or datetime.now().isoformat()
                    yield {
                        "event": "tool_end",
                        "data": {
                            "tool_call_id": call.id,
                            "tool": call.function.name,
                            "output": result_summary,
                            "timestamp": tool_end_timestamp,
                            "success": not isinstance(result, dict) or not bool(result.get("error")),
                        },
                    }
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
        except Exception as exc:
            last_request_path = last_request_snapshot.get("payload_path") if last_request_snapshot else None
            error_message = f"LLM call failed: {exc}" + (f" | last_request={last_request_path}" if last_request_path else "")
            logger.exception("LLM tool loop failed")
            self.trace_writer.append_message(self.session_id, role="assistant", content=error_message, timestamp=event_time)
            yield {"event": "error", "data": {"message": error_message, "timestamp": event_time or datetime.now().isoformat(), "last_request_path": last_request_path}}
            return error_message, "error"

        timeout_message = "Tool-call step limit reached; workflow stopped."
        timeout_timestamp = event_time or datetime.now().isoformat()
        self.trace_writer.append_message(self.session_id, role="assistant", content=timeout_message, timestamp=timeout_timestamp)
        yield {
            "event": "assistant_message",
            "data": {
                "content": timeout_message,
                "timestamp": timeout_timestamp,
                "final": True,
            },
        }
        return timeout_message, "timeout"

    def run_agent(self, request: AgentChatRequest) -> OneTurnAfterRunAgent:
        stream = self.run_agent_stream(request)
        while True:
            try:
                next(stream)
            except StopIteration as stop:
                if stop.value is None:
                    raise RuntimeError("run_agent_stream did not return a result")
                return stop.value

    def run_agent_stream(self, request: AgentChatRequest):
        event_time = request.event_time
        event_timestamp = event_time or datetime.now().isoformat()
        self.trace_writer.reset_turn()
        self.trace_writer.ensure_trace(self.session_id, created_at=event_time)
        self.trace_writer.set_status(self.session_id, "running", timestamp=event_time)
        self.trace_writer.append_message(self.session_id, role="user", content=request.message, timestamp=event_time)
        yield {
            "event": "session_created",
            "data": {
                "agent_id": self.agent_id,
                "session_id": self.session_id,
                "timestamp": event_timestamp,
            },
        }

        trace_payload = self.trace_writer.load_trace(self.session_id)
        memory_payload = self.memory_assembler.build(self.session_id)
        self.trace_writer.set_context_injection(
            self.session_id,
            {
                "control_files": [item["name"] for item in memory_payload.get("control_files", [])],
                "memory_files": [item["name"] for item in memory_payload.get("memory_content_blocks", [])],
                "assets": [item["path"] for item in memory_payload.get("assets", [])],
                "notes": "full memory concatenation",
            },
            timestamp=event_time,
        )
        skills_section = self.skill_manager.render_skills_section() if self.skill_manager else ""
        system_prompt = self.prompt_builder.build(memory_payload=memory_payload, skills_section=skills_section)
        user_context = self._build_user_context(request, trace_payload)
        assistant_message, final_status = yield from self._tool_loop_stream(system_prompt, user_context, event_time=event_time)
        memory_commit = self.memory_manager.commit_turn(
            session_id=self.session_id,
            user_message=request.message,
            assistant_message=assistant_message,
            event_time=event_time,
        )
        memory_updates = [memory_commit["memory_commit_path"]]
        self.trace_writer.extend_memory_commits(self.session_id, memory_updates, timestamp=event_time)
        self.trace_writer.append_decision(self.session_id, summary=assistant_message[:500], artifact_paths=[], timestamp=event_time)
        self.trace_writer.set_status(self.session_id, final_status, timestamp=event_time)
        trace_summary = TraceSummary(**self.trace_writer.summarize(self.session_id))
        result = OneTurnAfterRunAgent(return_message=assistant_message, memory_updates=memory_updates, generated_artifacts=[], trace_summary=trace_summary)
        yield {
            "event": "done",
            "data": AgentChatResponse(
                agent_id=self.agent_id,
                session_id=self.session_id,
                message_markdown=result.return_message,
                trace_summary=result.trace_summary,
                memory_updates=result.memory_updates,
                generated_artifacts=result.generated_artifacts,
                timestamp=event_timestamp,
            ).model_dump(),
        }
        return result


orchestrator: Optional[AgentOrchestrator] = None


def init_orchestrator(data_loader, session_id: str = "default", enable_skills: Optional[bool] = None, agent_id: str = "default"):
    global orchestrator
    orchestrator = AgentOrchestrator(data_loader, agent_id=agent_id, session_id=session_id, enable_skills=enable_skills)
    return orchestrator
