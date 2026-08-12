from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .answer_limits import (
    CHINESE_SINGLE_FORECAST_MAX_CHARS,
    GENERIC_MAX_CHARS,
    chinese_comparison_max_chars,
)
from .contract import (
    GroundingContractBuilder,
    _canonical_applied_disturbance_lines,
    _forecast_views,
    _format_number,
    grounded_fallback_answer,
    normalize_not_evaluated_wording,
)
from .evidence.tool import attach_tool_arguments


_LINEPACK_DECLINE_REQUEST = re.compile(
    r"管存.{0,12}(?:持续|下降|走低|消耗)|(?:持续|下降|走低).{0,12}管存|linepack.{0,12}(?:declin|fall|decreas)",
    re.IGNORECASE,
)
_UNSUPPORTED_UNIT_TOKEN = re.compile(
    r"(?:万方/日|万立方米/日|立方米/秒|m³/d|m3/d|m³/s|m3/s|MPa|kPa|bar|MW|kW)",
    re.IGNORECASE,
)


def _successful_pipeformer_results(
    tool_results: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in tool_results
        if item.get("name") == "run_pipeformer_forecast"
        and dict(item.get("output") or {}).get("success") is True
    ]


def _status_summary(category_status: Dict[str, Any], chinese: bool) -> str:
    groups: Dict[str, List[str]] = {}
    for category, status in category_status.items():
        groups.setdefault(str(status), []).append(str(category))
    if chinese:
        labels = {"pass": "通过", "warning": "告警", "fail": "失败", "not_evaluated": "未评估"}
        return "；".join(
            f"{'/'.join(values)}{labels.get(status, status)}"
            for status, values in groups.items()
        )
    return "; ".join(f"{'/'.join(values)}={status}" for status, values in groups.items())


def _finding_summary(findings: List[Dict[str, Any]], chinese: bool) -> str:
    parts: List[str] = []
    for finding in findings[:2]:
        name = str(finding.get("name") or "unknown_rule")
        status = str(finding.get("status") or "unknown")
        variables = ", ".join(str(value) for value in finding.get("affected_variables") or [])
        detail = ""
        values = list(finding.get("evaluated_values") or [])
        if values:
            evaluated = dict(values[0])
            metric = str(evaluated.get("metric") or "value")
            detail = f"; {metric}={_format_number(evaluated.get('value'))}"
        parts.append(f"{name}({status}{'; ' + variables if variables else ''}{detail})")
    return "，".join(parts) if chinese else ", ".join(parts)


def _watch_summary(output: Dict[str, Any]) -> List[str]:
    evidence = dict(output.get("evidence") or {})
    watch = list(evidence.get("top_watch_variables") or [])
    if not watch:
        prediction, _ = _forecast_views(output)
        summaries = dict(prediction.get("output_forecast_summary") or {})
        watch = [{"variable": variable, **dict(summary or {})} for variable, summary in list(summaries.items())[:3]]
    result = []
    for item in watch[:3]:
        variable = str(item.get("variable") or "")
        value = item.get("mean_prediction")
        result.append(f"{variable}={_format_number(value)}" if value is not None else variable)
    return [value for value in result if value]


def _single_forecast_answer(question: str, tool_result: Dict[str, Any]) -> str:
    output = dict(tool_result.get("output") or {})
    prediction, verification = _forecast_views(output)
    chinese = any("一" <= character <= "鿿" for character in question)
    contract = GroundingContractBuilder().build(question, [tool_result])
    lines: List[str] = _canonical_applied_disturbance_lines(contract)
    comparison = dict(prediction.get("counterfactual_comparison") or {})
    impact_count = comparison.get("nonzero_impacted_variable_count")
    linepack = dict(dict(verification.get("engineering_evidence") or {}).get("linepack") or {})
    linepack_status = dict(verification.get("category_status") or {}).get("linepack")
    decline_minutes = linepack.get("maximum_continuous_decline_minutes")
    if _LINEPACK_DECLINE_REQUEST.search(question):
        if linepack_status == "pass" and (decline_minutes is None or float(decline_minutes) <= 0):
            lines.append("持续管存下降未在当前预测中出现。" if chinese else "Sustained linepack decline did not appear in the current forecast.")
        elif linepack_status in {"warning", "fail"} or (decline_minutes is not None and float(decline_minutes) > 0):
            duration = _format_number(decline_minutes) if decline_minutes is not None else "unknown"
            lines.append(f"预测检测到管存下降，最长连续下降 {duration} 分钟。" if chinese else f"The forecast detected linepack decline; maximum continuous duration was {duration} minutes.")
    if impact_count is not None:
        lines.append(f"基线对比检出 {int(impact_count)} 个变化输出变量。" if chinese else f"Baseline comparison found {int(impact_count)} changed output variables.")
    category_status = dict(verification.get("category_status") or {})
    if category_status:
        summary = _status_summary(category_status, chinese)
        lines.append(f"校核：{summary}。" if chinese else f"Verification: {summary}.")
    risk = str(verification.get("risk_level") or output.get("risk_level") or "unknown")
    intervention = str(verification.get("human_intervention_label") or output.get("manual_intervention_label") or "unknown")
    lines.append(f"风险 {risk}；人工干预 {intervention}。" if chinese else f"Risk: {risk}; intervention: {intervention}.")
    findings = [dict(item) for item in verification.get("priority_findings") or []]
    if findings:
        summary = _finding_summary(findings, chinese)
        lines.append(f"优先发现：{summary}。" if chinese else f"Priority findings: {summary}.")
    watch = _watch_summary(output)
    if watch:
        lines.append(f"关注变量：{', '.join(watch)}。" if chinese else f"Watch variables: {', '.join(watch)}.")
    return "".join(lines) if chinese else " ".join(lines)


def _strip_unsupported_units(answer: str) -> str:
    stripped = _UNSUPPORTED_UNIT_TOKEN.sub("", answer)
    return re.sub(r"[ \t]+([，。；、,.!?])", r"\1", stripped)


def _compact_answer(answer: str, maximum_chars: int) -> str:
    """Extractively compact an answer without inventing replacement facts."""
    cleaned = re.sub(r"(?m)^\s*(?:#{1,6}\s*|---+\s*$)", "", answer)
    cleaned = cleaned.replace("**", "").replace("```", "")
    lines: List[str] = []
    seen = set()
    for raw_line in cleaned.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or re.fullmatch(r"\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?", line):
            continue
        if line not in seen:
            seen.add(line)
            lines.append(line)
    compact = " ".join(lines)
    if len(compact) <= maximum_chars:
        return compact
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])\s*|(?<=[A-Za-z])\.\s+", compact) if part.strip()]
    if not sentences:
        return compact[:maximum_chars].rstrip(" ，,；;")
    ending = sentences[-1]
    selected: List[str] = []
    reserve = len(ending) + 1 if len(ending) < maximum_chars // 2 else 0
    for sentence in sentences[:-1]:
        candidate = " ".join(selected + [sentence])
        if len(candidate) + reserve > maximum_chars:
            break
        selected.append(sentence)
    if ending not in selected and len(" ".join(selected + [ending])) <= maximum_chars:
        selected.append(ending)
    compacted = " ".join(selected)
    if compacted:
        return compacted
    return compact[:maximum_chars].rstrip(" ，,；;")


def repair_grounded_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Repair supported answers from stored evidence without an LLM call."""
    repaired = dict(record)
    tool_results = attach_tool_arguments(repaired.get("tool_outputs") or [], repaired.get("tool_calls") or [])
    contract = GroundingContractBuilder().build(
        str(repaired.get("user_input") or ""),
        tool_results,
        decision_policy=dict(repaired.get("decision_policy") or {}) or None,
    )
    repaired["answer_mode"] = contract.get("answer_mode")
    repaired["grounding_contract"] = contract
    repaired["decision_summary"] = dict(contract.get("decision_summary") or {})
    issues = {str(value) for value in repaired.get("quality_issues") or []}
    legacy_method = str(dict(repaired.get("repair_provenance") or {}).get("method") or "")
    pipeformer_results = _successful_pipeformer_results(tool_results)

    if contract.get("answer_mode") == "dispatch_comparison":
        decision = dict(contract.get("decision_summary") or {})
        repaired["final_answer"] = grounded_fallback_answer(str(repaired.get("user_input") or ""), contract)
        repaired["risk_level"] = contract.get("worst_case_risk_level")
        repaired["manual_intervention_label"] = contract.get("worst_case_intervention_label")
        repaired["dispatch_recommendation"] = str(decision.get("selected_dispatch_recommendation") or "")
        repaired["repair_provenance"] = {"method": "deterministic_grounding_contract", "external_llm_calls": 0, "reason": "Multi-candidate answer rebuilt from stored tool evidence."}

    should_render_single = bool(pipeformer_results) and (legacy_method == "offline_deterministic_repair" or bool(issues & {"answer_too_long", "unsupported_unit_claim"}))
    if contract.get("answer_mode") != "dispatch_comparison" and should_render_single:
        repaired["final_answer"] = _single_forecast_answer(str(repaired.get("user_input") or ""), pipeformer_results[0])
        output = dict(pipeformer_results[0].get("output") or {})
        _, verification = _forecast_views(output)
        repaired["risk_level"] = verification.get("risk_level") or output.get("risk_level")
        repaired["manual_intervention_label"] = verification.get("human_intervention_label") or output.get("manual_intervention_label")
        repaired["dispatch_recommendation"] = str(verification.get("dispatch_recommendation") or "")
        repaired["repair_provenance"] = {"method": "scenario_aware_deterministic_repair", "external_llm_calls": 0, "reason": "Single-forecast answer rebuilt from stored tool evidence."}
    if "unsupported_unit_claim" in issues:
        stripped = _strip_unsupported_units(str(repaired.get("final_answer") or ""))
        if stripped != repaired.get("final_answer"):
            repaired["final_answer"] = stripped
            repaired["repair_provenance"] = {"method": "unsupported_unit_removal", "external_llm_calls": 0, "reason": "Unsupported unit labels removed without changing the numeric claims."}
    repaired["final_answer"] = normalize_not_evaluated_wording(str(repaired.get("final_answer") or ""))
    if "answer_too_long" in issues:
        maximum_chars = (
            chinese_comparison_max_chars(len(contract.get("candidate_results") or []))
            if contract.get("answer_mode") == "dispatch_comparison"
            else CHINESE_SINGLE_FORECAST_MAX_CHARS
            if repaired.get("scenario_type") == "pipeformer"
            else GENERIC_MAX_CHARS
        )
        compacted = _compact_answer(str(repaired.get("final_answer") or ""), maximum_chars)
        if compacted != repaired.get("final_answer"):
            repaired["final_answer"] = compacted
            repaired["repair_provenance"] = {"method": "extractive_answer_compaction", "external_llm_calls": 0, "reason": "Formatting and redundant text removed; retained content is extractive."}
    return repaired
