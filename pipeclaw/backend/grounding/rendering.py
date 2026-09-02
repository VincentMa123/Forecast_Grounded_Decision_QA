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


def _format_action(
    action: Dict[str, Any], chinese: bool, *, compact: bool = False
) -> str:
    """Format a boundary action for either the full or compact view."""
    parts: List[str] = []
    for variable, raw_value in dict(action.get("percentage_changes") or {}).items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            parts.append(f"{variable}={raw_value}")
        else:
            if compact:
                parts.append(f"{variable}{value:+g}%")
            elif chinese:
                direction = "上调" if value >= 0 else "下调"
                parts.append(f"{variable} {direction} {format_number(abs(value))}%")
            else:
                direction = "increase" if value >= 0 else "decrease"
                parts.append(f"{direction} {variable} by {format_number(abs(value))}%")
    parts.extend(
        f"{variable}={format_number(value)}"
        for variable, value in dict(action.get("setpoints") or {}).items()
    )
    if not parts:
        return "未记录" if chinese or compact else "not recorded"
    return ("," if compact else ", ").join(parts)


def _rule_labels(candidate: Dict[str, Any]) -> tuple[str, str]:
    return tuple(
        ", ".join(candidate.get(key) or []) or "none"
        for key in ("failed_rule_ids", "warning_rule_ids")
    )


def _candidate_line(candidate: Dict[str, Any], chinese: bool) -> str:
    candidate_id = str(candidate.get("candidate_id") or "candidate")
    action = _format_action(dict(candidate.get("action") or {}), chinese)
    failed_rules, warning_rules = _rule_labels(candidate)
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
    decline = number_value(nested_value(linepack, ("maximum_decline_from_start", "value")))
    metrics = []
    if pressure is not None:
        metrics.append(f"压裕{format_number(pressure)}")
    if decline is not None:
        suffix = "".join(
            f"/{format_number(linepack[key])}{label}"
            for key, label in (
                ("maximum_continuous_decline_minutes", "min"),
                ("insufficient_recovery_count", "恢复不足"),
            )
            if linepack.get(key) is not None
        )
        metrics.append(f"管存降{format_number(decline)}{suffix}")
    energy = dict(candidate.get("energy_metrics") or {}).get("delta_vs_baseline")
    if energy is not None:
        metrics.append(f"能耗Δ{format_number(energy)}")
    metric_text = "；".join(metrics) or "无可比指标"
    return (
        f"- {candidate_id}（{action}）：F{candidate.get('failure_count', 0)}/"
        f"W{candidate.get('warning_count', 0)}；{metric_text}；"
        f"规则{failed_rules if failed_rules != 'none' else warning_rules}。"
    )


def _finalize_comparison_answer(
    lines: List[str],
    selected_candidate_id: str,
    contract: Dict[str, Any],
    *,
    maximum_chars: int = 500,
    maximum_words: Optional[int] = None,
) -> str:
    answer = "\n".join(str(line).strip() for line in lines if str(line).strip())
    answer += f"\nselected_candidate_id: {selected_candidate_id or 'none'}"
    over_budget = len(answer) > maximum_chars or (
        maximum_words is not None and len(answer.split()) > maximum_words
    )
    if over_budget:
        contract["answer_render_status"] = "answer_budget_insufficient"
    else:
        contract.pop("answer_render_status", None)
    return answer


def _shared_outcome_line(candidate: Dict[str, Any], chinese: bool) -> str:
    failed_rules, warning_rules = _rule_labels(candidate)
    failures = candidate.get("failure_count", 0)
    warnings = candidate.get("warning_count", 0)
    risk = candidate.get("risk_level", "unknown")
    intervention = candidate.get("manual_intervention_label", "unknown")
    if chinese:
        return (
            f"共同校核结果：失败 {failures}（{failed_rules}）；"
            f"告警 {warnings}（{warning_rules}）；风险 {risk}；人工干预 {intervention}。"
        )
    return (
        f"Shared result: {failures} failures ({failed_rules}); "
        f"{warnings} warnings ({warning_rules}); risk {risk}; "
        f"intervention {intervention}."
    )


def _compact_action(action: Dict[str, Any]) -> str:
    return _format_action(action, False, compact=True)


def applied_disturbance_disclosure(contract: Dict[str, Any]) -> str:
    """Return the exact machine-verifiable application evidence block."""
    return "\n".join(canonical_applied_disturbance_lines(contract))


def _compact_assumption_lines(contract: Dict[str, Any], chinese: bool) -> List[str]:
    lines = []
    seen = set()
    for item in contract.get("provisional_assumptions") or []:
        variable = str(item.get("variable") or "disturbance")
        direction = str(item.get("direction") or "unknown")
        magnitude = item.get("magnitude_percent")
        setpoint = item.get("setpoint")
        key = (variable.casefold(), direction.casefold(), format_number(magnitude), format_number(setpoint))
        if key in seen:
            continue
        seen.add(key)
        if variable.endswith(":ST") and setpoint is not None:
            lines.append(
                f"LLM暂设：{variable}={format_number(setpoint)}。"
                if chinese
                else f"Provisional LLM assumption: {variable}={format_number(setpoint)}."
            )
            continue
        if chinese:
            direction_label = {"up": "上调", "down": "下调"}.get(direction.casefold(), direction)
            suffix = "" if magnitude is None else f"{format_number(magnitude)}%"
            lines.append(f"LLM临时假设：{variable}{direction_label}{suffix}。")
        else:
            suffix = "" if magnitude is None else f" {format_number(magnitude)}%"
            lines.append(f"Provisional LLM assumption: {variable} {direction}{suffix}.")
    return lines


def _objective_label(objective: Dict[str, Any], chinese: bool) -> str:
    label = str(objective.get("label_zh" if chinese else "label_en") or objective.get("metric") or "metric")
    proxy = str(objective.get("proxy_for") or "")
    if proxy:
        proxy_label = (
            "末压代理"
            if proxy == "terminal_pressure_preservation"
            else "代理"
        )
        label += (
            f"({proxy_label})"
            if chinese
            else f" (proxy for {proxy.replace('_', ' ')})"
        )
    return label


def _objective_token(
    objective: Dict[str, Any], evidence: Dict[str, Any], chinese: bool
) -> str:
    detail = format_number(evidence.get("value"))
    variable = str(evidence.get("variable") or "")
    unit = str(evidence.get("unit") or "")
    if variable:
        detail += f"({variable})"
    if unit:
        detail += f" {unit}"
    return f"{_objective_label(objective, chinese)}={detail}"


def _audit_category_summary(candidates: List[Dict[str, Any]], chinese: bool) -> str:
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
    status_label = {"evidence": "evidenced", "mixed": "candidate-specific"}.get(status, status)
    return f"Shared audit: {'/'.join(labels)}={status_label}."


def _comparison_prefix(contract: Dict[str, Any], chinese: bool) -> List[str]:
    return [*canonical_applied_disturbance_lines(contract), *_compact_assumption_lines(contract, chinese)]


def _finalize_for_locale(
    lines: List[str], selected_id: str, contract: Dict[str, Any], chinese: bool, count: int
) -> str:
    return _finalize_comparison_answer(
        lines,
        selected_id,
        contract,
        maximum_chars=chinese_comparison_max_chars(count) if chinese else ENGLISH_COMPARISON_MAX_CHARS,
        maximum_words=None if chinese else ENGLISH_MAX_WORDS,
    )


def _policy_line(objectives: List[Dict[str, Any]], chinese: bool) -> str:
    directions = {"minimize": "↓", "maximize": "↑"}
    text = " > ".join(
        _objective_label(item, chinese)
        + directions.get(str(item.get("direction") or ""), "")
        for item in objectives
    )
    prefix = "目标：" if chinese else "Objectives: "
    suffix = "；硬约束F=0。" if chinese else "; hard constraint: zero failures."
    return prefix + text + suffix


def _comparison_candidate_lines(
    candidates: List[Dict[str, Any]],
    objectives: List[Dict[str, Any]],
    objective_evidence: Dict[str, Any],
    chinese: bool,
) -> List[str]:
    lines = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        evidence = dict(objective_evidence.get(candidate_id) or {})
        tokens = [
            _objective_token(
                item,
                dict(evidence.get(item["metric"]) or {}),
                chinese,
            )
            for item in objectives
        ]
        metrics = ("；" if chinese else "; ").join(tokens)
        action = _compact_action(dict(candidate.get("action") or {}))
        failures = candidate.get("failure_count", 0)
        warnings = candidate.get("warning_count", 0)
        if chinese:
            lines.append(
                f"{candidate_id}[{action}]：F{failures}/W{warnings}；{metrics}。"
            )
        else:
            lines.append(
                f"{candidate_id} [{action}]: F{failures}/W{warnings}; {metrics}."
            )
    return lines


def _ranking_line(groups: List[List[str]], ranking: List[str], chinese: bool) -> str:
    text = ">".join("=".join(group) for group in groups or [[candidate_id] for candidate_id in ranking])
    return ("排序：" if chinese else "Ranking: ") + text + ("。" if chinese else ".")


def _rejection_line(
    selected_id: str,
    ranking: List[str],
    groups: List[List[str]],
    eliminated: List[Dict[str, Any]],
    chinese: bool,
) -> str:
    eliminated_ids = {str(item.get("candidate_id") or "") for item in eliminated}
    rejected = [
        candidate_id
        for candidate_id in ranking
        if candidate_id != selected_id and candidate_id not in eliminated_ids
    ]
    selected_group = next((group for group in groups if selected_id in group), [selected_id])
    tied = [candidate_id for candidate_id in rejected if candidate_id in selected_group]
    lower = [candidate_id for candidate_id in rejected if candidate_id not in selected_group]
    eliminated_text = ";".join(
        f"{item.get('candidate_id')}="
        f"{','.join(item.get('failed_rules') or ['rule_failure'])}"
        for item in eliminated
    )
    if chinese:
        parts = [f"淘汰：{eliminated_text or '无'}"]
        if lower:
            parts.append(f"未选：{','.join(lower)}仅因目标次优")
        if tied:
            parts.append(f"并列未选：{','.join(tied)}与首选目标相同，仅按候选ID稳定排序")
        return "；".join(parts) + "。"
    parts = [f"Eliminated: {eliminated_text or 'none'}"]
    if lower:
        parts.append("not selected because lower-ranked on objectives: " + ", ".join(lower))
    if tied:
        parts.append(
            "tied with the selected candidate on requested objectives; "
            "stable candidate-ID tie-break: " + ", ".join(tied)
        )
    return "; ".join(parts) + "."


def _conclusion_lines(
    chinese: bool,
    shared_outcome: bool,
    all_eliminated: bool,
    failed_rules: List[str],
    missing: List[str],
) -> List[str]:
    if chinese:
        conclusion = (
            "结论：没有满足约束的候选动作，当前不能进入执行阶段。"
            if all_eliminated
            else "结论：现有可比证据不足，不能选出排名第一的动作。"
        )
        conditions = []
        if failed_rules:
            conditions.append(
                "必须先调整动作并消除上述失败规则"
                if shared_outcome
                else "必须先调整动作并消除失败规则 " + ", ".join(failed_rules)
            )
        if missing and not all_eliminated:
            conditions.append("需补充可比证据 " + ", ".join(missing))
        return [conclusion, "适用前提：" + "；".join(conditions) + "。"]
    conclusion = (
        "Conclusion: no candidate satisfies the constraints, so none can proceed "
        "to execution."
        if all_eliminated
        else "Conclusion: the comparable evidence is insufficient to select a "
        "first-ranked action."
    )
    conditions = []
    if failed_rules:
        conditions.append(
            "adjust the actions and clear the failed rules above"
            if shared_outcome
            else "adjust the actions and clear failed rules " + ", ".join(failed_rules)
        )
    if missing and not all_eliminated:
        conditions.append("provide comparable evidence " + ", ".join(missing))
    return [conclusion, "Applicability: " + "; ".join(conditions) + "."]


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
    evidence = dict(decision.get("objective_evidence") or {})
    selected_id = str(decision.get("selected_candidate_id") or "none")
    groups = [
        [str(candidate_id) for candidate_id in group if str(candidate_id) in by_id]
        for group in decision.get("ranked_candidate_groups") or []
    ]
    groups = [group for group in groups if group] or [[candidate_id] for candidate_id in ranking]
    eliminated = list(decision.get("eliminated_candidates") or [])
    lines = [
        *_comparison_prefix(contract, chinese),
        _policy_line(objectives, chinese),
        *_comparison_candidate_lines(ranked, objectives, evidence, chinese),
        _audit_category_summary(ranked, chinese),
        _ranking_line(groups, ranking, chinese),
        _rejection_line(selected_id, ranking, groups, eliminated, chinese),
    ]
    return _finalize_for_locale(lines, selected_id, contract, chinese, len(ranked))


def _same_outcome(candidates: List[Dict[str, Any]]) -> bool:
    return len({
        (
            candidate.get("failure_count", 0),
            tuple(candidate.get("failed_rule_ids") or []),
            candidate.get("warning_count", 0),
            tuple(candidate.get("warning_rule_ids") or []),
            candidate.get("risk_level", "unknown"),
            candidate.get("manual_intervention_label", "unknown"),
        )
        for candidate in candidates
    }) == 1


def grounded_fallback_answer(question: str, contract: Dict[str, Any]) -> str:
    """Render a compact, instruction-complete answer from deterministic facts."""
    decision = dict(contract.get("decision_summary") or {})
    candidates = list(contract.get("candidate_results") or [])
    chinese = is_chinese(question)
    ranking = [str(value) for value in decision.get("ranked_candidate_ids") or []]
    if ranking:
        positions = {candidate_id: index for index, candidate_id in enumerate(ranking)}
        candidates.sort(key=lambda item: positions.get(str(item.get("candidate_id") or ""), len(positions)))
    if decision.get("status") == "selected":
        return _selected_comparison_answer(question, contract, candidates, decision)
    shared_outcome = not ranking and len(candidates) > 1 and _same_outcome(candidates)
    title = (
        ("候选动作：" if chinese else "Candidate actions:")
        if shared_outcome
        else ("候选动作比较：" if chinese else "Candidate comparison:")
    )
    lines = [*_comparison_prefix(contract, chinese), title]
    if shared_outcome:
        lines.extend(
            f"- {candidate.get('candidate_id')}: {_format_action(dict(candidate.get('action') or {}), chinese)}"
            for candidate in candidates
        )
        lines.append(_shared_outcome_line(candidates[0], chinese))
    else:
        lines.extend(_candidate_line(candidate, chinese) for candidate in candidates)
    eliminated = list(decision.get("eliminated_candidates") or [])
    missing = [str(value) for value in decision.get("missing_metrics") or []]
    all_eliminated = bool(candidates) and len(eliminated) == len(candidates)
    failed_rules = sorted({str(rule) for item in eliminated for rule in item.get("failed_rules") or []})
    lines.extend(_conclusion_lines(chinese, shared_outcome, all_eliminated, failed_rules, missing))
    return _finalize_for_locale(lines, "none", contract, chinese, len(candidates))
