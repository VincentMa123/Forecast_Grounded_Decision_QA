from __future__ import annotations

from pathlib import Path
from typing import Dict, List


class PromptBuilder:
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)

    def build(self, *, memory_payload: Dict[str, List[Dict[str, str]]], skills_section: str = "") -> str:
        sections: List[str] = []
        tool_calling_contract = "\n".join([
            "## Tool Calling Contract",
            "- You are using native OpenAI function calling. Never simulate tool calls in plain text.",
            "- Every tool call must contain exactly one complete JSON object for that tool's arguments.",
            "- Never concatenate two JSON objects. Never output `}{`. Never append a previous tool's arguments to the next tool call.",
            "- Treat each tool call as isolated. Start from a fresh argument object every time.",
            "- Prefer at most one tool call per assistant turn unless multiple independent calls are absolutely necessary.",
            "- If you do need multiple tool calls, each call must still have its own complete arguments object, with no shared or reused argument text.",
            "- After a tool call, wait for the tool result before deciding the next tool call unless the calls are truly independent.",
            "- Only use keys defined by the tool schema. Do not invent extra keys.",
            "- For run_command, cmd must be a JSON array of strings, not a shell string.",
            "- This environment is Windows-first. Prefer cmd or powershell. Do not use bash unless the user explicitly asks for it or you have confirmed it exists.",
            "- If you are unsure about tool arguments, make a smaller valid tool call first instead of emitting a large risky one.",
        ])
        sections.append(tool_calling_contract)
        pipeformer_routing = "\n".join([
            "## PipeFormer Routing",
            "- For forecast, what-if, risk, dispatch, or transient-operation questions about future pipeline states, call `run_pipeformer_forecast` before answering.",
            "- This includes requests that mention PipeFormer, future prediction, boundary/control perturbations, mock_test cases, pressure/flow/linepack checks, compressor load, or energy checks.",
            "- For historical lookup, current-state retrieval, ranking, aggregation, or visualization questions that do not ask for future prediction, use the workspace/data workflow instead of PipeFormer.",
            "- Do not simulate PipeFormer tool calls in text; use the registered tool and base the final answer on its returned evidence.",
        ])
        sections.append(pipeformer_routing)
        control_files = memory_payload.get("control_files", [])
        if control_files:
            sections.append("## Control Plane Files\n" + "\n\n".join(
                f"### {item['name']}\n{item['content']}" for item in control_files if item.get("content")
            ))
        if skills_section:
            sections.append(skills_section)
        workspace_contract = "\n".join([
            "## Workspace Contract",
            f"- WORKSPACE_ROOT: {self.workspace_root.as_posix()}",
            f"- MEMORY_ROOT: {(self.workspace_root / 'memory').as_posix()}",
            f"- ASSETS_ROOT: {(self.workspace_root / 'assets').as_posix()}",
            f"- TRACE_ROOT: {(self.workspace_root / 'context_trace').as_posix()}",
            f"- TEMPORARY_DIR: {(self.workspace_root / 'temporary_dir').as_posix()}",
            f"- REPORTS_DIR: {(self.workspace_root / 'reports').as_posix()}",
            f"- PLAN_PATH: {(self.workspace_root / 'plan.md').as_posix()}",
            "- Do not rely on any sessions directory or per-session workspace root.",
            "- plan.md is temporary and recreated for each turn.",
            "- Write intermediate artifacts under TEMPORARY_DIR.",
            "- Write reports and final deliverables under REPORTS_DIR.",
            "- Do not scatter generated files directly under WORKSPACE_ROOT unless they are control-plane files.",
        ])
        sections.append(workspace_contract)
        memory_blocks = memory_payload.get("memory_content_blocks", [])
        if memory_blocks:
            sections.append("## Full Memory Blocks\n" + "\n\n".join(
                f"### {item['name']}\n{item['content']}" for item in memory_blocks if item.get("content")
            ))
        assets = memory_payload.get("assets", [])
        if assets:
            sections.append("## Assets\n" + "\n".join(f"- {item['path']}" for item in assets))
        trace_meta = memory_payload.get("trace_meta", {})
        if trace_meta:
            lines = ["## Trace Context Summary"]
            for key, value in trace_meta.items():
                lines.append(f"- {key}: {value}")
            sections.append("\n".join(lines))
        return "\n\n".join(section.strip() for section in sections if section and section.strip())
