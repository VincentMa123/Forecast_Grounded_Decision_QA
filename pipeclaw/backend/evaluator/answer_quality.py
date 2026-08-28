from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pipeclaw.backend.grounding.answer_limits import (
    CHINESE_SINGLE_FORECAST_MAX_CHARS,
    ENGLISH_MAX_WORDS,
    GENERIC_MAX_CHARS,
    chinese_comparison_max_chars,
)
from pipeclaw.backend.grounding.contract import (
    GroundingContractBuilder,
    answer_without_machine_disclosure,
    comparison_answer_issues,
    provisional_assumption_disclosed,
    record_grounding_contract,
)
from pipeclaw.backend.grounding.decision_trace_state import VerifiedDecisionState
from pipeclaw.backend.grounding.evidence.tool import (
    attach_tool_arguments,
    requested_data_retrieved,
    tool_evidence_quality_issues,
    tool_output_failed,
)
from pipeclaw.backend.grounding.evidence.topology import (
    topology_quality_issues,
    topology_summary_from_tool_outputs,
    topology_tool_required,
)
from pipeclaw.backend.evaluator.numeric_grounding import (
    numeric_claims_are_grounded,
)
from pipeclaw.backend.evaluator.quality_references import (
    VARIABLE_REFERENCE,
    file_references,
    numeric_claim_values,
)
from pipeclaw.backend.evaluator.quality_context import (
    QualityContext,
    build_quality_context,
)

UNSUPPORTED_HISTORY_CLAIM = re.compile(
    r"\b(?:reproduced (?:in|across|during)|reproducible across|previous runs?|prior runs?|stable across runs?|times stable)\b"
    r"|(?:已|曾|多次|稳定|结果).{0,12}复现|此前.*(?:结果|运行)|前(?:几|[一二三四五六七八九十\d]+)次.*一致|稳定.*运行",
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
ANSWER_FORMAT_VIOLATION = re.compile(r"(?m)^\s*#{1,6}\s|```|^\s*\|.*\|\s*$")
OPERATIONAL_INFERENCE_CLAIM = re.compile(
    r"稳定承载|真实受限|(?:不|无|没有).{0,8}(?:瓶颈|受限)|(?:局部)?(?:量测|测量)(?:问题|故障)|供需平衡"
    r"|\b(?:stable operation|capacity bottleneck|measurement (?:issue|fault)|supply.demand balance)\b",
    re.IGNORECASE,
)
SUPPLY_DEMAND_BALANCE_CLAIM = re.compile(
    r"供需平衡|\bsupply.demand balance\b",
    re.IGNORECASE,
)
INFERENCE_QUALIFIER = re.compile(
    r"仅凭|单凭|不足以|不能|无法|尚不能|不代表|待核实|待验证|初步|倾向|更像|可能"
    r"|\b(?:cannot|insufficient|unresolved|preliminary|may|might|likely|appears?)\b",
    re.IGNORECASE,
)
UNSUPPORTED_UNIT_CLAIM = re.compile(
    r"(?i)(?:万方/日|万立方米/日|立方米/秒|m³/d|m3/d|m³/s|m3/s|MPa|kPa|bar|MW|kW)"
)
UNSUPPORTED_ABSOLUTE_ASSERTION = re.compile(
    r"最敏感|不丢失(?:任何)?(?:关键)?信息|就能判断|必然(?:导致|说明)|"
    r"合并为|整段已合并|"
    r"\bmost sensitive\b|\b(?:lose|loses) no (?:important )?information\b|"
    r"\bcan determine\b|\bmerged into\b|\bguarantees?\b",
    re.IGNORECASE,
)


def answer_quality_issues(
    answer: str,
    question: str,
    pipeformer: Optional[Dict[str, Any]],
    *,
    conversation_context: Optional[List[Dict[str, Any]]] = None,
    tool_outputs: Optional[List[Dict[str, Any]]] = None,
    record_evidence: Optional[Dict[str, Any]] = None,
) -> List[str]:
    return _answer_quality_issues(
        build_quality_context(
            answer=answer,
            question=question,
            pipeformer=pipeformer,
            conversation_context=conversation_context,
            tool_outputs=tool_outputs,
            record_evidence=record_evidence,
        )
    )


def _answer_quality_issues(context: QualityContext) -> List[str]:
    answer = context.answer
    question = context.question
    pipeformer = context.pipeformer
    conversation_context = context.conversation_context
    tool_outputs = context.tool_outputs
    record_evidence = context.record_evidence
    grounding_evidence = context.grounding_evidence
    issues: List[str] = []
    if not answer.strip():
        issues.append("missing_llm_final_answer")
    if pipeformer and UNSUPPORTED_HISTORY_CLAIM.search(answer):
        issues.append("unsupported_execution_history_or_repeatability_claim")
    if pipeformer and not provisional_assumption_disclosed(
        answer,
        pipeformer,
        numeric_claim_values(answer),
    ):
        issues.append("undisclosed_disturbance_assumption")
    if NO_DISPATCH_REQUEST.search(question) and DISPATCH_ADVICE.search(answer):
        issues.append("unrequested_dispatch_recommendation")
    if pipeformer and safety_and_energy_checks_pass(pipeformer) and SAFETY_ENERGY_INCONSISTENCY_CLAIM.search(answer):
        issues.append("unsupported_safety_energy_inconsistency_claim")
    if pipeformer and UNSUPPORTED_UNIQUENESS_CLAIM.search(answer) and not _proves_unique_nonpass_variable(pipeformer):
        issues.append("unsupported_uniqueness_claim")
    if UNSUPPORTED_PROPAGATION_CLAIM.search(answer) and (
        not _counterfactual_supports_claim(answer, pipeformer)
        if pipeformer
        else not INFERENCE_QUALIFIER.search(answer)
        and not _operational_inference_is_grounded(answer, grounding_evidence)
    ):
        issues.append("unsupported_causal_or_propagation_claim")
    if not requested_data_retrieved(question, tool_outputs or [], conversation_context or []):
        issues.append("requested_evidence_not_retrieved")
    if topology_tool_required(question) and not topology_summary_from_tool_outputs(tool_outputs or []):
        issues.append("required_topology_tool_not_called")
    if not numeric_claims_are_grounded(answer, question, grounding_evidence):
        issues.append("unsupported_numerical_claim")
    issues.extend(
        topology_quality_issues(
            answer,
            dict((record_evidence or {}).get("topology_summary") or {}),
        )
    )
    requested_files = {value.lower() for value in file_references(question)}
    answer_files = {value.lower() for value in file_references(answer)}
    if requested_files and answer_files - requested_files:
        issues.append("unauthorized_source_substitution")
    if (
        not pipeformer
        and OPERATIONAL_INFERENCE_CLAIM.search(answer)
        and not INFERENCE_QUALIFIER.search(answer)
        and not _operational_inference_is_grounded(answer, grounding_evidence)
    ):
        issues.append("unsupported_operational_inference_claim")
    evidence_text = json.dumps(grounding_evidence, ensure_ascii=False).casefold()
    if any(
        match.group(0).casefold() not in evidence_text
        for match in UNSUPPORTED_UNIT_CLAIM.finditer(answer)
    ):
        issues.append("unsupported_unit_claim")
    if UNSUPPORTED_ABSOLUTE_ASSERTION.search(answer) and not _absolute_assertion_is_grounded(
        answer, grounding_evidence
    ):
        issues.append("unsupported_absolute_or_topology_claim")
    if pipeformer and not _variable_references_are_grounded(answer, grounding_evidence):
        issues.append("unsupported_variable_reference")
    if pipeformer and _has_unsupported_evidence_description(answer, pipeformer):
        issues.append("unsupported_evidence_variable_description")
    has_forecast_result = bool(
        pipeformer
        and any(
            pipeformer.get(key)
            for key in (
                "prediction",
                "prediction_summary",
                "verification",
                "constraint_check",
                "parsed_task",
                "candidate_forecasts",
            )
        )
    )
    successful_forecast_count = sum(
        item.get("name") == "run_pipeformer_forecast"
        and isinstance(item.get("output"), dict)
        and item["output"].get("success") is True
        for item in tool_outputs or []
    )
    saved_candidates = list(
        dict(pipeformer or {}).get("candidate_forecasts") or []
    )
    comparison_candidate_count = max(
        successful_forecast_count,
        len(saved_candidates),
    )
    is_forecast_comparison = comparison_candidate_count > 1
    budgeted_answer = answer_without_machine_disclosure(answer)
    has_chinese = any(
        "\u4e00" <= character <= "\u9fff"
        for character in budgeted_answer
    )
    if has_forecast_result or successful_forecast_count:
        if has_chinese and len(budgeted_answer) > (
            chinese_comparison_max_chars(comparison_candidate_count)
            if is_forecast_comparison
            else CHINESE_SINGLE_FORECAST_MAX_CHARS
        ):
            issues.append("answer_too_long")
        if not has_chinese and len(budgeted_answer.split()) > ENGLISH_MAX_WORDS:
            issues.append("answer_too_long")
    elif len(budgeted_answer) > GENERIC_MAX_CHARS:
        issues.append("answer_too_long")
    if pipeformer and (
        ANSWER_FORMAT_VIOLATION.search(answer)
        or any(symbol in answer for symbol in ("⚠", "✅", "❌", "📊", "👀", "🟡", "🔑", "📋", "🥇", "🥈", "🥉"))
    ):
        issues.append("answer_format_contract_violation")
    return issues


def evaluate_answer_quality(
    *,
    answer: str,
    question: str,
    pipeformer: Optional[Dict[str, Any]],
    trace_status: Optional[str],
    pipeformer_call_count: int,
    pipeformer_outputs: List[Dict[str, Any]],
    conversation_context: Optional[List[Dict[str, Any]]] = None,
    tool_outputs: Optional[List[Dict[str, Any]]] = None,
    record_evidence: Optional[Dict[str, Any]] = None,
    grounding_contract: Optional[Dict[str, Any]] = None,
) -> tuple[str, List[str]]:
    outputs = tool_outputs or []
    issues = answer_quality_issues(
        answer=answer,
        question=question,
        pipeformer=pipeformer,
        conversation_context=conversation_context,
        tool_outputs=outputs,
        record_evidence=record_evidence,
    )
    contract = (
        dict(grounding_contract)
        if grounding_contract is not None
        else GroundingContractBuilder().build(question, outputs)
    )
    issues.extend(comparison_answer_issues(answer, contract))
    issues.extend(tool_evidence_quality_issues(outputs))
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


def record_answer_quality_issues(record: Dict[str, Any]) -> List[str]:
    """Recompute grounding from saved evidence instead of trusting an old flag."""
    tool_outputs = attach_tool_arguments(
        record.get("tool_outputs") or [],
        record.get("tool_calls") or [],
    )
    contract = record_grounding_contract(record, tool_outputs)
    contract_issues = comparison_answer_issues(
        str(record.get("final_answer") or ""),
        contract,
    )
    successful_pipeformer_outputs = [
        dict(item.get("output") or {})
        for item in tool_outputs
        if item.get("name") == "run_pipeformer_forecast"
        and not tool_output_failed(item)
    ]
    state = VerifiedDecisionState.from_dict(
        dict(record.get("state_before") or {})
    )
    pipeformer = None
    if successful_pipeformer_outputs:
        pipeformer = dict(successful_pipeformer_outputs[0])
        if len(successful_pipeformer_outputs) > 1:
            pipeformer["candidate_forecasts"] = successful_pipeformer_outputs
    if pipeformer is None and str(record.get("scenario_type") or "").casefold() == "pipeformer":
        saved_pipeformer = {
            "parsed_task": record.get("parsed_task") or {},
            "prediction_summary": record.get("prediction_summary") or {},
            "constraint_check": record.get("constraint_check") or {},
            "evidence": record.get("evidence") or {},
        }
        if state.candidates:
            # Candidate actions and their compact metrics are verified output
            # from prior successful forecasts, not unverified prose.
            saved_pipeformer["candidate_forecasts"] = state.candidates
        pipeformer = saved_pipeformer
    contract_candidates = list(contract.get("candidate_results") or [])
    if pipeformer is not None and len(contract_candidates) > 1:
        # A multi-turn comparison may execute only one new forecast in the
        # current turn.  The contract's candidates are verified state, so
        # answer budgeting and grounding must still treat it as a comparison.
        pipeformer["candidate_forecasts"] = contract_candidates

    record_evidence = dict(record.get("evidence") or {})
    single_forecast_snapshot = dict(
        state.verified_evidence.get("single_forecast_snapshot") or {}
    )
    if single_forecast_snapshot:
        record_evidence["single_forecast_snapshot"] = (
            single_forecast_snapshot
        )
    if not record_evidence.get("topology_summary"):
        topology_summary = topology_summary_from_tool_outputs(tool_outputs)
        if topology_summary:
            record_evidence["topology_summary"] = topology_summary

    context = build_quality_context(
        answer=str(record.get("final_answer") or ""),
        question=str(record.get("user_input") or ""),
        pipeformer=pipeformer,
        conversation_context=list(record.get("conversation_context") or []),
        tool_outputs=tool_outputs,
        record_evidence=record_evidence,
    )
    issues = _answer_quality_issues(context)
    issues.extend(contract_issues)
    issues.extend(tool_evidence_quality_issues(tool_outputs))
    return list(dict.fromkeys(issues))


def _absolute_assertion_is_grounded(answer: str, evidence: Any) -> bool:
    evidence_text = json.dumps(evidence, ensure_ascii=False).casefold()
    return all(
        match.group(0).casefold() in evidence_text
        for match in UNSUPPORTED_ABSOLUTE_ASSERTION.finditer(answer)
    )


def _supply_demand_balance_claim_is_grounded(answer: str, evidence: Any) -> bool:
    statuses: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "supply_demand_balance_status" and item is not None:
                    statuses.add(str(item).casefold())
                elif key == "supply_demand_balance" and isinstance(item, dict):
                    status = item.get("status")
                    if status is not None:
                        statuses.add(str(status).casefold())
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(evidence)
    if not statuses:
        return False

    claim = SUPPLY_DEMAND_BALANCE_CLAIM.search(answer)
    if not claim:
        return False
    nearby = answer[claim.start():claim.end() + 16].casefold()
    if re.search(r"fail|violation|失败|失衡|不平衡", nearby):
        return bool(statuses & {"fail", "violation"})
    if re.search(r"warn|warning|告警|预警", nearby):
        return bool(statuses & {"warning", "warn"})
    if re.search(r"pass|通过|正常", nearby):
        return "pass" in statuses
    return True


def _operational_inference_is_grounded(answer: str, evidence: Any) -> bool:
    evidence_text = json.dumps(evidence, ensure_ascii=False).casefold()
    claims = [match.group(0).casefold() for match in OPERATIONAL_INFERENCE_CLAIM.finditer(answer)]
    if not claims:
        return False
    return all(
        _supply_demand_balance_claim_is_grounded(answer, evidence)
        if SUPPLY_DEMAND_BALANCE_CLAIM.fullmatch(claim)
        else claim in evidence_text
        for claim in claims
    )


def _counterfactual_supports_claim(answer: str, pipeformer: Dict[str, Any]) -> bool:
    comparison = pipeformer.get("counterfactual_comparison")
    if not isinstance(comparison, dict):
        comparison = dict(pipeformer.get("prediction") or {}).get(
            "counterfactual_comparison"
        )
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


def _variable_references_are_grounded(answer: str, grounding_evidence: Any) -> bool:
    claimed = set(VARIABLE_REFERENCE.findall(answer))
    supported = set(
        VARIABLE_REFERENCE.findall(json.dumps(grounding_evidence, ensure_ascii=False))
    )
    return claimed <= supported


def _has_unsupported_evidence_description(
    answer: str,
    pipeformer: Dict[str, Any],
) -> bool:
    """Validate typed variable descriptions across all candidate forecasts."""
    forecasts = [
        pipeformer,
        *[
            dict(item)
            for item in pipeformer.get("candidate_forecasts") or []
            if isinstance(item, dict)
        ],
    ]
    typed_categories: Dict[str, set[str]] = {}
    contextual_variables: set[str] = set()
    for forecast in forecasts:
        verification = dict(
            forecast.get("verification")
            or forecast.get("constraint_check")
            or {}
        )
        engineering = dict(
            verification.get("engineering_evidence") or {}
        )
        for category, payload in engineering.items():
            for variable in VARIABLE_REFERENCE.findall(
                json.dumps(payload, ensure_ascii=False)
            ):
                typed_categories.setdefault(variable, set()).add(
                    str(category)
                )
        evidence = dict(forecast.get("evidence") or {})
        for key in ("top_watch_variables", "key_observation_variables"):
            for item in evidence.get(key) or []:
                variable = str(dict(item or {}).get("variable") or "")
                if variable:
                    contextual_variables.add(variable)
    category_terms = {
        "pressure": re.compile(r"压力|\bpressure\b", re.IGNORECASE),
        "flow": re.compile(r"流量|\bflow\b", re.IGNORECASE),
        "linepack": re.compile(r"管存|\blinepack\b", re.IGNORECASE),
        "compressor": re.compile(
            r"压缩机|压缩比|\bcompressor\b|\bcompression ratio\b",
            re.IGNORECASE,
        ),
        "equipment_regulation": re.compile(
            r"调压器|阀门|球阀|\bregulator\b|\bvalve\b",
            re.IGNORECASE,
        ),
    }
    variables = contextual_variables | set(typed_categories)
    for variable in variables:
        description = re.compile(
            rf"`?{re.escape(variable)}`?\s*[（(]([^）)\n]*)[）)]"
        )
        for match in description.finditer(answer):
            text = match.group(1)
            mentioned_categories = {
                category
                for category, pattern in category_terms.items()
                if pattern.search(text)
            }
            if not mentioned_categories:
                continue
            if not (
                mentioned_categories
                & typed_categories.get(variable, set())
            ):
                return True
    return False
