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
            "- Treat the returned PipeFormer result as the sole evidence for the answer. Do not claim prior runs, repeated reproduction, stability across runs, execution times, or historical agreement unless those facts are explicitly present in the tool result.",
            "- Do not invent numerical values, thresholds, model details, timestamps, or operational facts that are absent from the tool result.",
            "- Never describe a `not_evaluated` rule or category as passing. If `verification_complete` is false, state briefly which requested checks lacked required variables and treat the risk conclusion as incomplete.",
            "- Follow the user's requested answer scope exactly. If the user excludes dispatch actions or another topic, do not add it.",
            "- Keep the final answer compact: use at most 8 short bullets and no more than 180 English words or about 300 Chinese characters.",
            "- Do not use Markdown tables, code fences, emoji, repeated summaries, or decorative headings in the final answer.",
            "- Report the overall result, non-pass constraints with grounded values, requested watch variables, intervention label, and only the recommendation explicitly returned by the tool and requested by the user.",
            "- When the user asks for watch variables or key evidence variables, copy `evidence.top_watch_variables` and `evidence.key_observation_variables` exactly in their returned order; do not rerank or replace them.",
            "- Do not call a variable, constraint, or warning unique/only unless the returned structured result explicitly proves that no other item has the same status.",
            "- Treat PipeFormer variable identifiers as opaque unless the tool returns explicit variable metadata. Do not infer that a `B_`, `R_`, `T_`, or other proxy is a physical flow, causal bridge, transmission hub, or specific equipment measurement.",
            "- For watch and evidence variables, report only the variable id and numerical fields actually returned by the tool; do not add causal explanations or unreturned engineering meanings.",
            "- Keep safety, equipment, and energy conclusions distinct: pressure/flow/linepack and abnormality warnings are safety, compressor and equipment regulation are equipment, and energy consumption/cost is energy. Never relabel an equipment warning as an energy warning.",
            "- Only answer a user-requested conditional section when the structured result proves its condition; otherwise state briefly that the condition was not met.",
            "- Summarize passing categories in one sentence instead of listing every passing rule.",
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
