"""Static forecast-agent instructions shared by runtime and Task 2 SFT."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from pipeclaw.backend.grounding.contract import (
    applied_disturbance_disclosure,
    build_grounding_contract,
)
from pipeclaw.backend.grounding.decision_trace_state import VerifiedDecisionState


def _section(*lines: str) -> str:
    return "\n".join(lines)


_STATIC_POLICY_SECTIONS = (
    _section(
        "## Tool Calling Contract",
        "- Use native function calling; never simulate a tool call in text.",
        "- Send one fresh, complete JSON object that matches the selected tool schema. Never concatenate objects, reuse argument text, or invent keys.",
        "- Prefer one tool call at a time. Read its result before choosing the next call; use multiple calls only when they are independent.",
        "- Never repeat the same tool call with the same arguments. After a zero-match result or structured error, correct the invalid filter or change strategy instead of looping.",
        "- For `run_command`, `cmd` is a JSON array. Use the paths from the active Workspace Contract or workspace-relative paths; omit `cwd` when the active workspace is sufficient. The runner selects the host interpreter and shell for Windows or Linux. Never copy a path from another session or training example.",
        "- On a structured tool error, follow its retry instruction exactly. A successful exit code without requested evidence is not evidence.",
    ),
    _section(
        "## Evidence Grounding",
        "- Verify facts by reading the requested domain file or an authorized structured source. Logs, traces, plans, reports, prior assistant prose, and copied memory are not substitute evidence.",
        "- Copy only explicit values, units, topology, roles, and relationships. Do not invent physical meanings, mappings, completeness, sensitivity, causality, or propagation.",
        "- User statements are premises, not independently verified facts. Label recommendations and engineering inferences as proposals and state what would verify them.",
        "- Previously verified evidence summaries may be reused across turns without rereading. Trust only their structured facts, and do not claim that its source file was unread merely because the current turn made no tool call.",
        "- Empty, failed, unrelated, locator-only, NOT_FOUND, or truncated-without-the-needed-row results mean unavailable. Do not substitute another date, case, file, or pipeline.",
    ),
    _section(
        "## Pipeline Data Access",
        "- For a known CSV, call `read_file` directly with its logical path. Do not search WORKSPACE_DIR, ASSETS_ROOT, TRACE_ROOT, logs, or repository directories.",
        "- Paths (for read_file only): `pipeline_data/node_flow/YYYYMMDD_node.csv`, `pipeline_data/pipeline_flow/YYYYMMDD_pipeline.csv`, and `pipeline_data/consumer_flow/YYYYMMDD_consumer.csv`.",
        "- Extract the filename from the request; never hardcode a date. Read with `limit=400`, then paginate only when needed.",
        "- In run_command, cwd is the agent workspace and `pipeline_data/` does NOT exist there. Any Python script therefore MUST resolve data directories via `os.environ`, never via a relative or absolute literal. The harness always sets NODE_FLOW_DIR, PIPELINE_FLOW_DIR, CONSUMER_FLOW_DIR, OUTPUT_DIR. Canonical idiom (copy it):",
        "    import os",
        "    from pathlib import Path",
        "    DIR = Path(os.environ[\"PIPELINE_FLOW_DIR\"])  # holds {date}_pipeline.csv",
        "    csv_path = DIR / f\"{date}_pipeline.csv\"",
        "- Env-var values change per deployment: never print them into files or code, never pass them to read_file (its sandbox accepts logical `pipeline_data/...` and workspace paths only).",
        "- If you are unsure what the value is, a one-line probe is allowed and encouraged: `python -c \"import os; print(os.environ['PIPELINE_FLOW_DIR'])\"`. This is NOT the prohibited 'searching' above (which means listing repos/logs/trace dirs).",
        "- For computation, join the extracted filename to NODE_FLOW_DIR, PIPELINE_FLOW_DIR, or CONSUMER_FLOW_DIR from `os.environ`; write helpers in WORKSPACE_DIR and results in OUTPUT_DIR. Serialize Decimal safely with `json.dumps(payload, default=str)` or explicit string conversion.",
        "- For reachability, source counts, shortest paths, or gateways, call `analyze_pipeline_topology`. `multi_source_reachable` and `shared_gateway_dependency` are independent; copy only returned metrics.",
        "- A connected path proves reachability, not capacity, pressure, guaranteed supply, or cause. Do not invent an unrecorded branch, injection, transfer, action, or cause; report the mismatch as unresolved and name the evidence needed.",
    ),
    _section(
        "## Forecast State Machine",
        "1. Qualify: PipeFormer requires a canonical disturbance variable, a valid operating case or current operating-condition number, and relevant successful registry evidence. Otherwise do not forecast; give a bounded qualitative answer from verified CSV/topology evidence and list the missing inputs.",
        "2. Resolve the disturbance: search its exact canonical ID, or use a meaningful equipment/quantity/attention search plus `vocabulary_normalizations` with `normalization_source=registry_search`. A zero-match or broad role-only search cannot authorize the mapping. Never invent regional mappings.",
        "3. Resolve candidate controls and forecast vocabulary: search `role=input, controllable=true`; every action variable must appear in a preceding successful result. For `role=output`, omit `controllable`. An unknown attention target or output-state term must also be searched. Decision-policy metrics are derived comparison values, never PipeFormer state variables: never put them in `output_state_variables`. If `unresolved_task_vocabulary` is returned, search and retry; do not silently map velocity to flow or invent an unauthorized proxy.",
        "4. Encode the action: keep the external disturbance separate from candidate `boundary_conditions`, use a different registered action variable, and keep `keep_other_boundary_controls` inside `boundary_conditions`. Every named dispatch candidate must contain at least one distinct boundary setpoint or percentage change. Use `candidate_role=baseline` for a no-action disturbance reference, and omit `candidate_id` for a single prediction. For binary `:ST`, use setpoint 0 or 1, top-level `disturbance_setpoint` when it is the disturbance, and no percentage magnitude/change.",
        "5. Set the decision policy: infer ordered objectives and hard constraints only when the current user has stated them. Every objective must contain its own exact contiguous `source_excerpt` copied from the current request; never concatenate separate phrases. If an exact current-turn excerpt is unavailable, do not call `set_decision_policy`; leave selection unavailable. Candidate forecasts may be collected before priorities are known, but never rank, select, or recommend executing a candidate until `set_decision_policy` succeeds. If the current turn only adds or revises the decision policy and verified history already covers the same case, disturbance, horizon, and candidate actions, call `set_decision_policy` and reuse those results. Forecast again only when the case, disturbance, horizon, or action changes.",
        "6. Execute and answer: give each real dispatch action a stable `candidate_id`, use only successful structured results, and obey the current runtime contract. A missing `llm_decision_policy_tool_call` marker is a call to action, not a failure to disclose: when any user wording in the conversation states or implies priorities, call `set_decision_policy` immediately, then rank. Leave selection unset only when no priority can be derived from the conversation. Copy the required disturbance disclosure verbatim, including the complete variable suffix and binary `=0` or `=1` setpoint.",
    ),
    _section(
        "## Forecast Answer Invariants",
        "- A qualitative missing magnitude/direction may be simulated only with `disturbance_assumption`; mark it as an LLM provisional assumption and disclose it.",
        "- Copy every registered variable ID exactly as returned, including suffixes such as `:SNQ`, `:ST`, `:FR`, and `:SP_`. Never abbreviate `T_002:SNQ` to `T_002`, even to save characters.",
        "- Never add physical meanings to an identifier unless exact metadata was returned. Never claim uniqueness, prior runs, causality, propagation, or effects absent from structured evidence.",
        "- Follow the user's requested slots and prohibitions. Report `not_evaluated` as not evaluated, never as `fail`; if that category is required, the overall verification remains incomplete and cannot be released.",
        "- HARD OUTPUT LIMIT: at most 160 English words, 500 total characters for ordinary Chinese/mixed forecast answers, or 650 total characters for multi-candidate Chinese/mixed comparisons. Remove decoration and repetition, but preserve actions, objective evidence, audit outcomes, ranking, selection, rejection reasons, and assumptions.",
    ),
)


def static_forecast_policy() -> str:
    """Return the state-free production forecast policy."""

    return "\n\n".join(_STATIC_POLICY_SECTIONS)


def candidate_contract_message(
    question: str,
    tool_results: Iterable[Dict[str, Any]],
    *,
    decision_policy: Optional[Dict[str, Any]] = None,
    prior_state: Optional[VerifiedDecisionState] = None,
    prior_candidate_results: Optional[Iterable[Dict[str, Any]]] = None,
    prior_decision_policy: Optional[Dict[str, Any]] = None,
    prior_decision_policy_source_question: Optional[str] = None,
    prior_applied_disturbances: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Render accumulated multi-candidate facts for the next model request."""
    results = [dict(item) for item in tool_results]
    contract = build_grounding_contract(
        question,
        results,
        decision_policy=decision_policy,
        require_decision_policy=True,
        prior_state=prior_state,
        prior_candidate_results=prior_candidate_results,
        prior_decision_policy=prior_decision_policy,
        prior_decision_policy_source_question=prior_decision_policy_source_question,
        prior_applied_disturbances=prior_applied_disturbances,
    )
    if contract.get("answer_mode") == "single_forecast":
        successful_forecasts = [
            item
            for item in results
            if item.get("name") == "run_pipeformer_forecast"
            and dict(item.get("output") or {}).get("success") is True
        ]
        if len(successful_forecasts) != 1:
            return None
        forecast = successful_forecasts[0]
        if dict(forecast.get("arguments") or {}).get("candidate_id"):
            return None
        output = dict(forecast.get("output") or {})
        verification = dict(
            output.get("verification") or output.get("constraint_check") or {}
        )
        has_prediction_contract = any(
            key in verification
            for key in (
                "priority_findings",
                "top_watch_variables",
                "safety_energy_comparison",
            )
        )
        if not has_prediction_contract:
            return None
        applied_disturbance = json.dumps(
            contract.get("applied_disturbances") or [],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        required_disclosure = applied_disturbance_disclosure(contract)
        return (
            "CURRENT PIPEFORMER FORECAST ANSWER CONTRACT\n"
            f"Required disturbance disclosure (copy verbatim): {required_disclosure}\n"
            f"Structured application evidence: {applied_disturbance}\n"
            "Use only the successful forecast's structured fields. If the current request asks "
            "for operating result, intervention, watch variables, and the safety-energy "
            "comparison, return exactly four short bullets and no heading: (1) passing "
            "categories once plus every priority_findings warning/failure with values and "
            "thresholds; (2) risk_level and human_intervention_label; (3) top_watch_variables "
            "in returned order with variable, mean_prediction, and "
            "mean_abs_delta_vs_observed; (4) safety_energy_comparison. When its comparison is "
            "complete and consistent, say `安全侧与能耗侧结论一致`; when inconsistent, include "
            "the first priority audit constraint and numerical key_observation_variables. "
            "For a narrower follow-up, answer only requested slots. Never abbreviate a "
            "canonical variable ID or add unreturned physical meaning."
        )
    if contract.get("answer_mode") != "dispatch_comparison":
        return None
    candidates = list(contract.get("candidate_results") or [])
    decision_summary = contract.get("decision_summary") or {}
    payload = {
        "successful_candidate_count": len(candidates),
        "candidate_results": candidates,
        "decision_summary": decision_summary,
        "comparison_leaders": contract.get("comparison_leaders") or {},
        "required_application_disclosure": applied_disturbance_disclosure(contract),
        "worst_case_risk_level": contract.get("worst_case_risk_level"),
        "worst_case_intervention_label": contract.get(
            "worst_case_intervention_label"
        ),
    }
    policy_nudge = ""
    missing_metrics = list(decision_summary.get("missing_metrics") or [])
    if "llm_decision_policy_tool_call" in missing_metrics:
        policy_nudge = (
            " No decision policy is recorded yet; treat this as a call to action, not a "
            "failure to disclose in the answer. If the user's current or earlier wording "
            "states or implies any priority or objective (for example 优先, 主要, 尽量, "
            "minimize, or first consider), call set_decision_policy NOW with each objective "
            "grounded in an exact contiguous source_excerpt of that wording, then rank all "
            "viable candidates. End with `selected_candidate_id: none` only when no priority "
            "can be derived from the conversation at all."
        )
    return (
        "CURRENT PIPEFORMER CANDIDATE CONTRACT\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\nUse only these successful candidates and their recorded facts in the final comparison. "
        "Copy required_application_disclosure verbatim at the start of the answer. "
        "Mention every candidate_id and action, report the ordered objective evidence for every "
        "viable candidate, state hard-constraint and audit outcomes, and do not invent rankings "
        "or effects. Copy every canonical variable ID exactly; never abbreviate it. Continue "
        "calling PipeFormer if the user's requested candidate count has not yet been evaluated. End the "
        "answer with exactly `selected_candidate_id: <candidate_id>` or "
        "`selected_candidate_id: none`, matching decision_summary."
        + policy_nudge
    )
