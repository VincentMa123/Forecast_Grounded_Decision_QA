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
        sections.append("\n".join([
            "## Evidence Grounding",
            "- Base factual and numerical claims only on the current request, visible conversation history, and successful tool results.",
            "- If a requested file or record is unavailable, say so plainly. Do not substitute another date, case, file, or pipeline unless the user authorizes it.",
            "- Do not infer stability, capacity, bottlenecks, causality, or measurement faults from rankings or aggregate balance alone; label what remains unresolved.",
            "- Keep historical/current-state answers concise and avoid repeating raw tool output.",
        ]))
        pipeformer_routing = "\n".join([
            "## PipeFormer Routing",
            "- For questions about future pipeline states, forecasts, what-if disturbances, risk, dispatch, or transient operation, call `run_pipeformer_forecast` before answering. Use the workspace/data workflow for purely historical, current-state, ranking, aggregation, or visualization requests.",
            "- Use only registered tool keys. Put `keep_other_boundary_controls` inside `boundary_conditions`, never at the top level. Treat the returned result as the sole evidence; a `not_evaluated` check never passes.",
            "- Set `include_baseline_comparison=true` only when the user asks what the disturbance caused, changed, propagated to, or affected. Make those claims only from `counterfactual_comparison`; otherwise report the disturbed forecast without causal wording.",
            "- When a prediction-and-verification request asks for the operating result, intervention, watch variables, and a safety-energy conditional, return exactly four short bullets and no heading: (1) summarize passing categories once and copy every `priority_findings` warning/failure with returned values and thresholds; (2) copy `risk_level` and `human_intervention_label`; (3) copy `top_watch_variables` in order using only `variable`, `mean_prediction`, and `mean_abs_delta_vs_observed`; (4) follow `verification.safety_energy_comparison`.",
            "- In bullet 4, when `comparison_complete` is true and `consistent` is true, say `安全侧与能耗侧结论一致` and do not report an audit constraint or key observations. When `consistent` is false, report the first priority audit constraint and copy `key_observation_variables` using identifiers and numerical fields only. When comparison is incomplete, say so without inferring a result.",
            "- For a narrower follow-up, answer only the requested slots instead of forcing all four bullets. User prohibitions override tool fields; omit dispatch wording when the user says `不要给调度动作`, `不要给调度建议`, or equivalent.",
            "- Never add physical meanings to evidence identifiers unless the tool returns metadata for that exact variable. Never claim uniqueness, prior runs, propagation, causality, or relationships between variables unless the corresponding structured evidence is returned.",
            "- Keep the answer within 180 English words or 500 total characters for Chinese/mixed answers. Do not use headings, tables, code fences, emoji, repeated summaries, or decorative formatting.",
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
