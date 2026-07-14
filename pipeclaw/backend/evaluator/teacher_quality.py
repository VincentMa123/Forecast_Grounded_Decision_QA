from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


UNSUPPORTED_HISTORY_CLAIM = re.compile(
    r"\b(?:reproduc(?:ed|ible|ibility|tion)?|previous runs?|prior runs?|stable across runs?|times stable)\b"
    r"|复现|此前.*(?:结果|运行)|前(?:几|[一二三四五六七八九十\d]+)次.*一致|稳定.*(?:复现|运行)",
    re.IGNORECASE,
)
NO_DISPATCH_REQUEST = re.compile(
    r"不要.{0,12}调度(?:动作|建议)"
    r"|(?:do\s+not|don't)\s+(?:give|provide|include).{0,30}dispatch",
    re.IGNORECASE,
)
DISPATCH_ADVICE = re.compile(
    r"\s*(?:调度建议|dispatch\s+recommendation)\s*[:：][^\n]*",
    re.IGNORECASE,
)
SAFETY_ENERGY_INCONSISTENCY_CLAIM = re.compile(
    r"安全侧与(?:能耗(?:/设备)?|设备)侧结论不一致"
    r"|\bsafety\b.{0,40}\b(?:energy|equipment)\b.{0,30}\b(?:inconsisten\w*|differ\w*)\b",
    re.IGNORECASE,
)
UNSUPPORTED_UNIQUENESS_CLAIM = re.compile(
    r"(?:\s*[,，]\s*)?(?:唯一(?:越限|告警|异常)?变量|the\s+only\s+(?:violating|warning|abnormal)\s+variable)",
    re.IGNORECASE,
)
UNSUPPORTED_PROPAGATION_CLAIM = re.compile(
    r"(?:未|没有|无).{0,16}(?:传导|影响)|传导|传播|因果"
    r"|\b(?:did not|does not)\s+(?:propagat\w*|affect)\b"
    r"|\bno\s+(?:propagation|effect|impact)\b|\bpropagat\w*\b|\bcausal(?:ity)?\b",
    re.IGNORECASE,
)
NO_IMPACT_COUNTERFACTUAL_CLAIM = re.compile(
    r"(?:未|没有|无).{0,16}(?:传导|影响)"
    r"|\b(?:did not|does not)\s+(?:propagat\w*|affect)\b"
    r"|\bno\s+(?:propagation|effect|impact)\b",
    re.IGNORECASE,
)
NUMERIC_CLAIM = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?")
VARIABLE_REFERENCE = re.compile(r"\b[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?\b")
EVIDENCE_DESCRIPTION_TERM = re.compile(
    r"代理|调压器|压缩机|压缩比|流量|压力|管存|阀门|球阀|节点"
    r"|\b(?:proxy|regulator|compressor|compression ratio|flow|pressure|linepack|valve|node|segment)\b",
    re.IGNORECASE,
)
ANSWER_FORMAT_VIOLATION = re.compile(r"(?m)^\s*#{1,6}\s|```|^\s*\|.*\|\s*$")
DATA_FILE_REFERENCE = re.compile(r"(?i)(?<![\w.-])[\w.-]+\.(?:csv|jsonl?|xlsx?|parquet)(?![\w.-])")
OPERATIONAL_INFERENCE_CLAIM = re.compile(
    r"稳定承载|真实受限|(?:不|无|没有).{0,8}(?:瓶颈|受限)|(?:局部)?(?:量测|测量)(?:问题|故障)|供需平衡"
    r"|\b(?:stable operation|capacity bottleneck|measurement (?:issue|fault)|supply.demand balance)\b",
    re.IGNORECASE,
)
INFERENCE_QUALIFIER = re.compile(
    r"仅凭|单凭|不足以|不能|无法|尚不能|待核实|待验证|初步|倾向|更像|可能"
    r"|\b(?:cannot|insufficient|unresolved|preliminary|may|might|likely|appears?)\b",
    re.IGNORECASE,
)


def answer_quality_issues(
    answer: str,
    question: str,
    pipeformer: Optional[Dict[str, Any]],
    *,
    conversation_context: Optional[List[Dict[str, Any]]] = None,
    tool_outputs: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    issues: List[str] = []
    if not answer.strip():
        issues.append("missing_llm_final_answer")
    if pipeformer and UNSUPPORTED_HISTORY_CLAIM.search(answer):
        issues.append("unsupported_execution_history_or_repeatability_claim")
    if NO_DISPATCH_REQUEST.search(question) and DISPATCH_ADVICE.search(answer):
        issues.append("unrequested_dispatch_recommendation")
    if pipeformer and safety_and_energy_checks_pass(pipeformer):
        if SAFETY_ENERGY_INCONSISTENCY_CLAIM.search(answer):
            issues.append("unsupported_safety_energy_inconsistency_claim")
    if pipeformer and UNSUPPORTED_UNIQUENESS_CLAIM.search(answer):
        if not _proves_unique_nonpass_variable(pipeformer):
            issues.append("unsupported_uniqueness_claim")
    if (
        pipeformer
        and UNSUPPORTED_PROPAGATION_CLAIM.search(answer)
        and not _counterfactual_supports_claim(answer, pipeformer)
    ):
        issues.append("unsupported_causal_or_propagation_claim")
    trusted_context = [
        {
            key: value
            for key, value in item.items()
            if key != "assistant_output" or item.get("quality_flag") == "pass"
        }
        for item in conversation_context or []
    ]
    trusted_tool_outputs = [
        item
        for item in tool_outputs or []
        if not tool_output_failed(item.get("output"))
    ]
    grounding_evidence: Any = pipeformer or {
        "conversation_context": trusted_context,
        "tool_outputs": trusted_tool_outputs,
    }
    if not numeric_claims_are_grounded(answer, question, grounding_evidence):
        issues.append("unsupported_numerical_claim")
    requested_files = {value.lower() for value in DATA_FILE_REFERENCE.findall(question)}
    answer_files = {value.lower() for value in DATA_FILE_REFERENCE.findall(answer)}
    if requested_files and answer_files - requested_files:
        issues.append("unauthorized_source_substitution")
    if (
        not pipeformer
        and OPERATIONAL_INFERENCE_CLAIM.search(answer)
        and not INFERENCE_QUALIFIER.search(answer)
        and not _operational_inference_is_grounded(answer, grounding_evidence)
    ):
        issues.append("unsupported_operational_inference_claim")
    if pipeformer and not _variable_references_are_grounded(answer, pipeformer):
        issues.append("unsupported_variable_reference")
    if pipeformer and _has_unsupported_evidence_description(answer, pipeformer):
        issues.append("unsupported_evidence_variable_description")
    if len(answer) > (500 if pipeformer else 1200):
        issues.append("answer_too_long")
    if pipeformer and (
        ANSWER_FORMAT_VIOLATION.search(answer)
        or any(symbol in answer for symbol in ("⚠", "✅", "❌", "📊", "👀", "🟡", "🔑", "📋", "🥇", "🥈", "🥉"))
    ):
        issues.append("answer_format_contract_violation")
    return issues


def evaluate_teacher_quality(
    *,
    answer: str,
    question: str,
    pipeformer: Optional[Dict[str, Any]],
    trace_status: Optional[str],
    pipeformer_call_count: int,
    pipeformer_outputs: List[Dict[str, Any]],
    conversation_context: Optional[List[Dict[str, Any]]] = None,
    tool_outputs: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, List[str]]:
    outputs = tool_outputs or []
    issues = answer_quality_issues(
        answer,
        question,
        pipeformer,
        conversation_context=conversation_context,
        tool_outputs=outputs,
    )
    failed_indices = [
        index
        for index, item in enumerate(outputs)
        if tool_output_failed(item.get("output"))
    ]
    successful_outputs = [
        item
        for item in outputs
        if not tool_output_failed(item.get("output"))
    ]
    if failed_indices and not successful_outputs:
        issues.append("tool_execution_failed")
    issues = list(dict.fromkeys(issues))
    forecasts_pass = (
        (pipeformer_call_count == 0 or bool(pipeformer_outputs))
        and all(output.get("quality_flag") == "pass" for output in pipeformer_outputs)
    )
    quality_flag = (
        "pass"
        if trace_status == "completed" and forecasts_pass and not issues
        else "needs_review"
    )
    return quality_flag, issues


def safety_and_energy_checks_pass(pipeformer: Optional[Dict[str, Any]]) -> bool:
    constraint_check = (pipeformer or {}).get("constraint_check") or {}
    comparison = constraint_check.get("safety_energy_comparison") or {}
    if comparison.get("comparison_complete"):
        return comparison.get("safety_status") == "pass" and comparison.get("energy_status") == "pass"
    checks = constraint_check.get("checks") or []
    safety_checks = [
        item
        for item in checks
        if item.get("category") in {"pressure", "flow", "linepack", "abnormality_warning"}
    ]
    energy_checks = [item for item in checks if item.get("name") == "energy_consumption_cost"]
    return (
        bool(safety_checks)
        and all(item.get("status") == "pass" for item in safety_checks)
        and bool(energy_checks)
        and all(item.get("status") == "pass" for item in energy_checks)
    )


def numeric_claims_are_grounded(answer: str, question: str, evidence: Dict[str, Any]) -> bool:
    claimed = _numbers_in_text(answer)
    supported = _numbers_in_text(question)
    supported.extend(_numbers_in_value(evidence))
    return all(
        any(abs(value - candidate) <= max(0.001, abs(candidate) * 0.0005) for candidate in supported)
        for value in claimed
    )


def tool_output_failed(output: Any) -> bool:
    return isinstance(output, dict) and (
        output.get("success") is False
        or bool(output.get("error"))
        or output.get("exit_code") not in (None, 0)
    )


def _operational_inference_is_grounded(answer: str, evidence: Any) -> bool:
    evidence_text = json.dumps(evidence, ensure_ascii=False).casefold()
    claims = [match.group(0).casefold() for match in OPERATIONAL_INFERENCE_CLAIM.finditer(answer)]
    return bool(claims) and all(claim in evidence_text for claim in claims)


def _counterfactual_supports_claim(answer: str, pipeformer: Dict[str, Any]) -> bool:
    comparison = pipeformer.get("counterfactual_comparison")
    if not isinstance(comparison, dict):
        return False
    try:
        impacted_count = int(comparison.get("nonzero_impacted_variable_count"))
    except (TypeError, ValueError):
        return False

    no_impact_claim = bool(NO_IMPACT_COUNTERFACTUAL_CLAIM.search(answer))
    if impacted_count == 0:
        return no_impact_claim
    if no_impact_claim:
        # The compact comparison does not retain every zero-delta variable, so
        # a selective no-impact claim cannot be verified when other impacts exist.
        return False

    impacted_variables = {
        str(item.get("variable"))
        for item in comparison.get("top_impacted_variables") or []
        if item.get("variable")
    }
    if not impacted_variables:
        return False
    disturbance_variable = str(comparison.get("disturbance_variable") or "")
    allowed_variables = impacted_variables | ({disturbance_variable} if disturbance_variable else set())
    claim_sentences = [
        sentence
        for sentence in re.split(r"[。！？.!?\n]+", answer)
        if UNSUPPORTED_PROPAGATION_CLAIM.search(sentence)
    ]
    claimed_variables = {
        variable
        for sentence in claim_sentences
        for variable in VARIABLE_REFERENCE.findall(sentence)
    }
    return not claimed_variables or claimed_variables <= allowed_variables


def _proves_unique_nonpass_variable(pipeformer: Dict[str, Any]) -> bool:
    constraint_check = dict(pipeformer.get("constraint_check") or {})
    findings = list(constraint_check.get("priority_findings") or [])
    variables = {
        str(value.get("variable"))
        for finding in findings
        for value in list(finding.get("evaluated_values") or []) + list(finding.get("offending_values") or [])
        if value.get("variable")
    }
    return len(findings) == 1 and len(variables) == 1


def _numbers_in_text(value: str) -> List[float]:
    numbers = []
    for match in NUMERIC_CLAIM.finditer(value):
        line_start = value.rfind("\n", 0, match.start()) + 1
        prefix = value[line_start:match.start()]
        suffix = value[match.end():match.end() + 1]
        if not prefix.strip(" \t-*") and suffix in {".", ")", "）", "、"}:
            continue
        numbers.append(float(match.group(0)))
    return numbers


def _numbers_in_value(value: Any) -> List[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return _numbers_in_text(value)
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers_in_value(item)]
    if isinstance(value, list):
        return [number for item in value for number in _numbers_in_value(item)]
    return []


def _variable_references_are_grounded(answer: str, pipeformer: Dict[str, Any]) -> bool:
    claimed = set(VARIABLE_REFERENCE.findall(answer))
    supported = set(VARIABLE_REFERENCE.findall(json.dumps(pipeformer, ensure_ascii=False)))
    return claimed <= supported


def _has_unsupported_evidence_description(answer: str, pipeformer: Dict[str, Any]) -> bool:
    evidence = pipeformer.get("evidence") or {}
    finding_variables = {
        str(value.get("variable"))
        for finding in (pipeformer.get("constraint_check") or {}).get("priority_findings") or []
        for value in list(finding.get("evaluated_values") or []) + list(finding.get("offending_values") or [])
        if value.get("variable")
    }
    disturbance = str((pipeformer.get("parsed_task") or {}).get("disturbance_variable") or "")
    variables = {
        str(item.get("variable"))
        for key in ("top_watch_variables", "key_observation_variables")
        for item in evidence.get(key) or []
        if item.get("variable")
    }
    for variable in variables - finding_variables - {disturbance}:
        description = re.compile(rf"`?{re.escape(variable)}`?\s*[（(]([^）)\n]*)[）)]")
        if any(EVIDENCE_DESCRIPTION_TERM.search(match.group(1)) for match in description.finditer(answer)):
            return True
    return False


llm_answer_quality_issues = answer_quality_issues
