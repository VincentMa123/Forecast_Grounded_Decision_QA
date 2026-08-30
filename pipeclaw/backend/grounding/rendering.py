from __future__ import annotations

from typing import Any, Dict, List, Optional

from .answer_limits import (
    ENGLISH_COMPARISON_MAX_CHARS,
    ENGLISH_MAX_WORDS,
    chinese_comparison_max_chars,
)
from .construction import is_chinese
from .decision_policy import nested_value, number_value
from .validation import (
    _AUDIT_CATEGORIES,
    canonical_applied_disturbance_lines,
    format_number,
)


def _format_action(action: Dict[str, Any], chinese: bool) -> str:
    parts: List[str] = []
    for variable, value in dict(action.get("percentage_changes") or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            parts.append(f"{variable}={value}")
            continue
        if chinese:
            direction = "\u4e0a\u8c03" if number >= 0 else "\u4e0b\u8c03"
            parts.append(f"{variable} {direction} {format_number(abs(number))}%")
        else:
            direction = "increase" if number >= 0 else "decrease"
            parts.append(f"{direction} {variable} by {format_number(abs(number))}%")
    for variable, value in dict(action.get("setpoints") or {}).items():
        parts.append(f"{variable}={format_number(value)}")
    if parts:
        return ", ".join(parts)
    return "\u672a\u8bb0\u5f55" if chinese else "not recorded"


def _candidate_line(candidate: Dict[str, Any], chinese: bool) -> str:
    candidate_id = str(candidate.get("candidate_id") or "candidate")
    action = _format_action(dict(candidate.get("action") or {}), chinese)
    failed_rules = ", ".join(candidate.get("failed_rule_ids") or []) or "none"
    warning_rules = ", ".join(candidate.get("warning_rule_ids") or []) or "none"
    if not chinese:
        return (
            f"- {candidate_id}: action {action}; "
            f"{candidate.get('failure_count', 0)} failures ({failed_rules}); "
            f"{candidate.get('warning_count', 0)} warnings ({warning_rules}); "
            f"risk {candidate.get('risk_level', 'unknown')}; "
            f"intervention {candidate.get('manual_intervention_label', 'unknown')}."
        )

    pressure = number_value(
        nested_value(
            dict(candidate.get("pressure_metrics") or {}),
            ("minimum_operating_window_margin", "value"),
        )
    )
    linepack = dict(candidate.get("linepack_metrics") or {})
    decline = number_value(
        nested_value(linepack, ("maximum_decline_from_start", "value"))
    )
    duration = linepack.get("maximum_continuous_decline_minutes")
    recovery = linepack.get("insufficient_recovery_count")
    energy = dict(candidate.get("energy_metrics") or {}).get("delta_vs_baseline")
    metrics = []
    if pressure is not None:
        metrics.append(f"\u538b\u88d5{format_number(pressure)}")
    if decline is not None:
        linepack_text = f"\u7ba1\u5b58\u964d{format_number(decline)}"
        if duration is not None:
            linepack_text += f"/{format_number(duration)}min"
        if recovery is not None:
            linepack_text += f"/\u6062\u590d\u4e0d\u8db3{format_number(recovery)}"
        metrics.append(linepack_text)
    if energy is not None:
        metrics.append(f"\u80fd\u8017\u0394{format_number(energy)}")
    metric_text = "\uff1b".join(metrics) or "\u65e0\u53ef\u6bd4\u6307\u6807"
    return (
        f"- {candidate_id}\uff08{action}\uff09\uff1aF{candidate.get('failure_count', 0)}/"
        f"W{candidate.get('warning_count', 0)}\uff1b{metric_text}\uff1b"
        f"\u89c4\u5219{failed_rules if failed_rules != 'none' else warning_rules}\u3002"
    )


def _outcome_key(candidate: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate.get("failure_count", 0),
        tuple(candidate.get("failed_rule_ids") or []),
        candidate.get("warning_count", 0),
        tuple(candidate.get("warning_rule_ids") or []),
        candidate.get("risk_level", "unknown"),
        candidate.get("manual_intervention_label", "unknown"),
    )



def _finalize_comparison_answer(
    lines: List[str],
    selected_candidate_id: str,
    contract: Dict[str, Any],
    *,
    maximum_chars: int = 500,
    maximum_words: Optional[int] = None,
) -> str:
    body = "\n".join(str(line).strip() for line in lines if str(line).strip())
    answer = body + f"\nselected_candidate_id: {selected_candidate_id or 'none'}"
    over_budget = len(answer) > maximum_chars
    if maximum_words is not None and len(answer.split()) > maximum_words:
        over_budget = True
    if over_budget:
        contract["answer_render_status"] = "answer_budget_insufficient"
    else:
        contract.pop("answer_render_status", None)
    return answer


def _shared_outcome_line(candidate: Dict[str, Any], chinese: bool) -> str:
    failed_rules = ", ".join(candidate.get("failed_rule_ids") or []) or "none"
    warning_rules = ", ".join(candidate.get("warning_rule_ids") or []) or "none"
    if chinese:
        return (
            f"\u5171\u540c\u6821\u6838\u7ed3\u679c\uff1a\u5931\u8d25 {candidate.get('failure_count', 0)}\uff08{failed_rules}\uff09\uff1b"
            f"\u544a\u8b66 {candidate.get('warning_count', 0)}\uff08{warning_rules}\uff09\uff1b"
            f"\u98ce\u9669 {candidate.get('risk_level', 'unknown')}\uff1b"
            f"\u4eba\u5de5\u5e72\u9884 {candidate.get('manual_intervention_label', 'unknown')}\u3002"
        )
    return (
        f"Shared result: {candidate.get('failure_count', 0)} failures ({failed_rules}); "
        f"{candidate.get('warning_count', 0)} warnings ({warning_rules}); "
        f"risk {candidate.get('risk_level', 'unknown')}; "
        f"intervention {candidate.get('manual_intervention_label', 'unknown')}."
    )


def _compact_action(action: Dict[str, Any]) -> str:
    parts: List[str] = []
    for variable, raw_value in dict(action.get("percentage_changes") or {}).items():
        try:
            value = float(raw_value)
            parts.append(f"{variable}{value:+g}%")
        except (TypeError, ValueError):
            parts.append(f"{variable}={raw_value}")
    for variable, raw_value in dict(action.get("setpoints") or {}).items():
        parts.append(f"{variable}={format_number(raw_value)}")
    return ",".join(parts) or "未记录"


def applied_disturbance_disclosure(
    contract: Dict[str, Any],
) -> str:
    """Return the exact machine-verifiable application evidence block."""
    return "\n".join(canonical_applied_disturbance_lines(contract))


def _compact_assumption_lines(contract: Dict[str, Any], chinese: bool) -> List[str]:
    lines: List[str] = []
    seen = set()
    for item in contract.get("provisional_assumptions") or []:
        variable = str(item.get("variable") or "disturbance")
        direction = str(item.get("direction") or "unknown")
        magnitude = item.get("magnitude_percent")
        setpoint = item.get("setpoint")
        key = (
            variable.casefold(),
            direction.casefold(),
            format_number(magnitude),
            format_number(setpoint),
        )
        if key in seen:
            continue
        seen.add(key)
        if variable.endswith(":ST") and setpoint is not None:
            if chinese:
                lines.append(f"LLM暂设：{variable}={format_number(setpoint)}。")
            else:
                lines.append(
                    f"Provisional LLM assumption: {variable}={format_number(setpoint)}."
                )
            continue
        if chinese:
            direction_label = {"up": "上调", "down": "下调"}.get(
                direction.casefold(), direction
            )
            if magnitude is None:
                lines.append(f"LLM临时假设：{variable}{direction_label}。")
            else:
                lines.append(
                    f"LLM临时假设：{variable}{direction_label}{format_number(magnitude)}%。"
                )
        else:
            if magnitude is None:
                lines.append(f"Provisional LLM assumption: {variable} {direction}.")
            else:
                lines.append(
                    f"Provisional LLM assumption: {variable} {direction} {format_number(magnitude)}%."
                )
    return lines


def _objective_label(objective: Dict[str, Any], chinese: bool) -> str:
    label = str(
        objective.get("label_zh" if chinese else "label_en")
        or objective.get("metric")
        or "metric"
    )
    proxy = str(objective.get("proxy_for") or "")
    if proxy:
        if chinese:
            proxy_label = (
                "末压代理" if proxy == "terminal_pressure_preservation" else "代理"
            )
            label += f"({proxy_label})"
        else:
            label += f" (proxy for {proxy.replace('_', ' ')})"
    return label


def _objective_token(
    objective: Dict[str, Any],
    evidence: Dict[str, Any],
    chinese: bool,
) -> str:
    label = _objective_label(objective, chinese)
    value = format_number(evidence.get("value"))
    variable = str(evidence.get("variable") or "")
    unit = str(evidence.get("unit") or "")
    detail = value
    if variable:
        detail += f"({variable})"
    if unit:
        detail += f" {unit}"
    return f"{label}={detail}"


def _audit_category_summary(
    candidates: List[Dict[str, Any]],
    chinese: bool,
) -> str:
    present = [
        (category, zh, en)
        for category, key, zh, en, _ in _AUDIT_CATEGORIES
        if any(dict(candidate.get(key) or {}) for candidate in candidates)
    ]
    labels = [zh if chinese else en for _, zh, en in present]
    statuses = {
        str(dict(candidate.get("category_status") or {}).get(category) or "evidence")
        for candidate in candidates
        for category, _, _ in present
    }
    status = next(iter(statuses)) if len(statuses) == 1 else "mixed"
    if chinese:
        status_label = {"evidence": "有据", "mixed": "状态见候选"}.get(status, status)
        return f"共同审核：{'/'.join(labels)}均{status_label}。"
    status_label = {"evidence": "evidenced", "mixed": "candidate-specific"}.get(
        status,
        status,
    )
    return f"Shared audit: {'/'.join(labels)}={status_label}."


def _selected_comparison_answer(
    question: str,
    contract: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    decision: Dict[str, Any],
) -> str:
    chinese = is_chinese(question)
    ranking = [str(value) for value in decision.get("ranked_candidate_ids") or []]
    by_id = {str(item.get("candidate_id") or ""): item for item in candidates}
    ranked = [by_id[candidate_id] for candidate_id in ranking if candidate_id in by_id]
    ranked.extend(item for item in candidates if item not in ranked)
    policy = dict(decision.get("ranking_policy") or {})
    objectives = [dict(item) for item in policy.get("objectives") or []]
    objective_evidence = dict(decision.get("objective_evidence") or {})
    selected_id = str(decision.get("selected_candidate_id") or "none")
    direction_labels = {"minimize": "↓", "maximize": "↑"}
    policy_line = (
        ("目标：" if chinese else "Objectives: ")
        + " > ".join(
            _objective_label(objective, chinese)
            + direction_labels.get(str(objective.get("direction") or ""), "")
            for objective in objectives
        )
        + ("；硬约束F=0。" if chinese else "; hard constraint: zero failures.")
    )
    candidate_lines: List[str] = []
    for candidate in ranked:
        candidate_id = str(candidate.get("candidate_id") or "")
        action = _compact_action(dict(candidate.get("action") or {}))
        evidence = dict(objective_evidence.get(candidate_id) or {})
        metric_tokens = [
            _objective_token(
                objective, dict(evidence.get(objective["metric"]) or {}), chinese
            )
            for objective in objectives
        ]
        if chinese:
            candidate_lines.append(
                f"{candidate_id}[{action}]：F{candidate.get('failure_count', 0)}/"
                f"W{candidate.get('warning_count', 0)}；{'；'.join(metric_tokens)}。"
            )
        else:
            candidate_lines.append(
                f"{candidate_id} [{action}]: F{candidate.get('failure_count', 0)}/"
                f"W{candidate.get('warning_count', 0)}; {'; '.join(metric_tokens)}."
            )

    ranked_groups = [
        [str(candidate_id) for candidate_id in group if str(candidate_id) in by_id]
        for group in decision.get("ranked_candidate_groups") or []
    ]
    ranked_groups = [group for group in ranked_groups if group]
    if not ranked_groups:
        ranked_groups = [[candidate_id] for candidate_id in ranking]
    ranking_text = ">".join("=".join(group) for group in ranked_groups)
    ranking_line = (
        ("排序：" if chinese else "Ranking: ")
        + ranking_text
        + ("。" if chinese else ".")
    )
    eliminated = list(decision.get("eliminated_candidates") or [])
    eliminated_ids = {str(item.get("candidate_id") or "") for item in eliminated}
    rejected = [
        candidate_id
        for candidate_id in ranking
        if candidate_id != selected_id and candidate_id not in eliminated_ids
    ]
    selected_group = next(
        (group for group in ranked_groups if selected_id in group),
        [selected_id],
    )
    tied_with_selected = [
        candidate_id for candidate_id in rejected if candidate_id in selected_group
    ]
    lower_ranked = [
        candidate_id for candidate_id in rejected if candidate_id not in selected_group
    ]
    eliminated_text = ";".join(
        f"{item.get('candidate_id')}={','.join(item.get('failed_rules') or ['rule_failure'])}"
        for item in eliminated
    )
    if chinese:
        rejection_parts = [
            f"淘汰：{eliminated_text}" if eliminated_text else "淘汰：无"
        ]
        if lower_ranked:
            rejection_parts.append(f"未选：{','.join(lower_ranked)}仅因目标次优")
        if tied_with_selected:
            rejection_parts.append(
                f"并列未选：{','.join(tied_with_selected)}与首选目标相同，"
                "仅按候选ID稳定排序"
            )
        rejection_line = "；".join(rejection_parts) + "。"
    else:
        rejection_parts = [
            f"Eliminated: {eliminated_text}" if eliminated_text else "Eliminated: none"
        ]
        if lower_ranked:
            rejection_parts.append(
                "not selected because lower-ranked on objectives: "
                + ", ".join(lower_ranked)
            )
        if tied_with_selected:
            rejection_parts.append(
                "tied with the selected candidate on requested objectives; "
                "stable candidate-ID tie-break: " + ", ".join(tied_with_selected)
            )
        rejection_line = "; ".join(rejection_parts) + "."

    lines = [
        *canonical_applied_disturbance_lines(contract),
        *_compact_assumption_lines(contract, chinese),
        policy_line,
        *candidate_lines,
        _audit_category_summary(ranked, chinese),
        ranking_line,
        rejection_line,
    ]
    return _finalize_comparison_answer(
        lines,
        selected_id,
        contract,
        maximum_chars=(
            chinese_comparison_max_chars(len(ranked))
            if chinese
            else ENGLISH_COMPARISON_MAX_CHARS
        ),
        maximum_words=None if chinese else ENGLISH_MAX_WORDS,
    )


def grounded_fallback_answer(question: str, contract: Dict[str, Any]) -> str:
    """Render a compact, instruction-complete answer from deterministic facts."""
    decision = dict(contract.get("decision_summary") or {})
    candidates = list(contract.get("candidate_results") or [])
    chinese = is_chinese(question)
    ranking = [str(value) for value in decision.get("ranked_candidate_ids") or []]
    if ranking:
        positions = {candidate_id: index for index, candidate_id in enumerate(ranking)}
        candidates.sort(
            key=lambda item: positions.get(
                str(item.get("candidate_id") or ""), len(positions)
            )
        )
    if decision.get("status") == "selected":
        return _selected_comparison_answer(question, contract, candidates, decision)

    assumption_lines = [
        *canonical_applied_disturbance_lines(contract),
        *_compact_assumption_lines(contract, chinese),
    ]
    shared_outcome = (
        not ranking
        and len(candidates) > 1
        and len({_outcome_key(candidate) for candidate in candidates}) == 1
    )
    lines = [
        *assumption_lines,
        ("\u5019\u9009\u52a8\u4f5c\uff1a" if chinese else "Candidate actions:")
        if shared_outcome
        else (
            "\u5019\u9009\u52a8\u4f5c\u6bd4\u8f83\uff1a"
            if chinese
            else "Candidate comparison:"
        ),
    ]
    if shared_outcome:
        lines.extend(
            f"- {candidate.get('candidate_id')}: "
            f"{_format_action(dict(candidate.get('action') or {}), chinese)}"
            for candidate in candidates
        )
        lines.append(_shared_outcome_line(candidates[0], chinese))
    else:
        lines.extend(_candidate_line(candidate, chinese) for candidate in candidates)
    eliminated = list(decision.get("eliminated_candidates") or [])
    missing = [str(value) for value in decision.get("missing_metrics") or []]

    all_eliminated = bool(candidates) and len(eliminated) == len(candidates)
    failed_rules = sorted(
        {str(rule) for item in eliminated for rule in item.get("failed_rules") or []}
    )
    if chinese:
        conclusion = (
            "\u7ed3\u8bba\uff1a\u6ca1\u6709\u6ee1\u8db3\u7ea6\u675f\u7684\u5019\u9009\u52a8\u4f5c\uff0c"
            "\u5f53\u524d\u4e0d\u80fd\u8fdb\u5165\u6267\u884c\u9636\u6bb5\u3002"
            if all_eliminated
            else "\u7ed3\u8bba\uff1a\u73b0\u6709\u53ef\u6bd4\u8bc1\u636e\u4e0d\u8db3\uff0c\u4e0d\u80fd\u9009\u51fa\u6392\u540d\u7b2c\u4e00\u7684\u52a8\u4f5c\u3002"
        )
        conditions = []
        if failed_rules:
            conditions.append(
                "\u5fc5\u987b\u5148\u8c03\u6574\u52a8\u4f5c\u5e76\u6d88\u9664\u4e0a\u8ff0\u5931\u8d25\u89c4\u5219"
                if shared_outcome
                else "\u5fc5\u987b\u5148\u8c03\u6574\u52a8\u4f5c\u5e76\u6d88\u9664\u5931\u8d25\u89c4\u5219 "
                + ", ".join(failed_rules)
            )
        if missing and not all_eliminated:
            conditions.append(
                "\u9700\u8865\u5145\u53ef\u6bd4\u8bc1\u636e " + ", ".join(missing)
            )
        lines.extend(
            [
                conclusion,
                "\u9002\u7528\u524d\u63d0\uff1a" + "\uff1b".join(conditions) + "\u3002",
            ]
        )
    else:
        conclusion = (
            "Conclusion: no candidate satisfies the constraints, so none can proceed to execution."
            if all_eliminated
            else "Conclusion: the comparable evidence is insufficient to select a first-ranked action."
        )
        conditions = []
        if failed_rules:
            conditions.append(
                "adjust the actions and clear the failed rules above"
                if shared_outcome
                else "adjust the actions and clear failed rules "
                + ", ".join(failed_rules)
            )
        if missing and not all_eliminated:
            conditions.append("provide comparable evidence " + ", ".join(missing))
        lines.extend([conclusion, "Applicability: " + "; ".join(conditions) + "."])
    return _finalize_comparison_answer(
        lines,
        "none",
        contract,
        maximum_chars=(
            chinese_comparison_max_chars(len(candidates))
            if chinese
            else ENGLISH_COMPARISON_MAX_CHARS
        ),
        maximum_words=None if chinese else ENGLISH_MAX_WORDS,
    )
