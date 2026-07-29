from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .answer_limits import (
    CHINESE_SINGLE_FORECAST_MAX_CHARS,
    ENGLISH_MAX_WORDS,
    GENERIC_MAX_CHARS,
    chinese_comparison_max_chars,
)
from .csv_evidence import build_csv_evidence
from .decision_trace_state import VerifiedDecisionState
from .grounding_contract import (
    GroundingContractBuilder,
    answer_without_machine_disclosure,
    comparison_answer_issues,
)
from .tool_evidence import (
    DATA_FILE_REFERENCE,
    ToolEvidenceState,
    attach_tool_arguments,
    classify_tool_evidence,
    requested_artifacts,
    tool_output_failed,
)
from .topology_evidence import (
    topology_quality_issues,
    topology_summary_from_tool_outputs,
    topology_tool_required,
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
NUMERIC_CLAIM = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][-+]?\d+)?"
)
NUMERIC_SIGN_TRANSLATION = str.maketrans({
    "\u2212": "-",
    "\ufe63": "-",
    "\uff0d": "-",
})
CANDIDATE_IDENTIFIER = re.compile(
    r"\bcandidate_[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*\b",
    re.IGNORECASE,
)
CHINESE_ORDINAL_REFERENCE = re.compile(
    r"第\s*\d+(?:\s*[、,，/]\s*\d+)*\s*"
    r"(?:名|位|段|个|项|条|种|组|候选|方案|动作|管段|用户)?"
)
ENGLISH_ORDINAL_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_])\d+(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)
ENGLISH_RANK_REFERENCE = re.compile(
    r"\b(?:rank(?:ed)?|position)\s*(?:#|no\.?\s*)?"
    r"\d+(?:\s*[/,]\s*\d+)*",
    re.IGNORECASE,
)
NEGATED_NUMERIC_REFERENCE = re.compile(
    r"(?:不是|并非|不应为|\bnot\b|\bis\s+not\b)"
    r"[\s:*_`]{0,12}[-+]?\d+(?:\.\d+)?%?",
    re.IGNORECASE,
)
CASE_IDENTIFIER_NUMBER = re.compile(
    r"(?:mock_test|case)[_-]0*(\d+)\b",
    re.IGNORECASE,
)
TIME_RANGE = re.compile(
    r"(?P<first>\d+(?:\.\d+)?)\s*(?:-|–|—|~|～|to|through|到|至)\s*"
    r"(?P<second>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>hours?|hrs?|hr|h|小时|小時|minutes?|mins?|min|分钟|分鐘)",
    re.IGNORECASE,
)
TIME_VALUE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>hours?|hrs?|hr|h|小时|小時|minutes?|mins?|min|分钟|分鐘)",
    re.IGNORECASE,
)
_FULL_DATE_PATTERN = (
    r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"\d{4}\u5e74\d{1,2}\u6708\d{1,2}\u65e5)"
)
_SHORT_DATE_PATTERN = (
    r"(?:\d{1,2}[-/.]\d{1,2}|\d{1,2}(?:\u65e5|\u53f7))"
)
DATE_RANGE_REFERENCE = re.compile(
    rf"(?<!\d){_FULL_DATE_PATTERN}\s*"
    r"(?:-|\u2013|\u2014|~|\uff5e|to|through|\u5230|\u81f3)\s*"
    rf"(?:{_FULL_DATE_PATTERN}|{_SHORT_DATE_PATTERN})(?!\d)",
    re.IGNORECASE,
)
DATE_REFERENCE = re.compile(rf"(?<!\d){_FULL_DATE_PATTERN}(?!\d)")
COMPACT_DATE_REFERENCE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?!\d)"
)
YEAR_REFERENCE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?=年|[-/.]\d)")
VARIABLE_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_])[A-Z]+_\d+(?::[A-Za-z0-9_]+|_[A-Za-z0-9_]+)?(?![A-Za-z0-9_:])"
)
EVIDENCE_DESCRIPTION_TERM = re.compile(
    r"代理|调压器|压缩机|压缩比|流量|压力|管存|阀门|球阀|节点"
    r"|\b(?:proxy|regulator|compressor|compression ratio|flow|pressure|linepack|valve|node|segment)\b",
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


def _disturbance_assumption_magnitudes(pipeformer: Dict[str, Any]) -> List[float]:
    magnitudes: List[float] = []

    def collect(value: Any) -> None:
        if not isinstance(value, dict):
            return
        assumption = value.get("disturbance_assumption")
        if isinstance(assumption, dict) and assumption.get("source") == "llm_assumption":
            magnitude = value.get("disturbance_magnitude_percent")
            if magnitude is not None:
                magnitudes.append(abs(float(magnitude)))

    collect(pipeformer.get("parsed_task"))
    collect(pipeformer.get("prediction_summary"))
    collect(pipeformer.get("prediction"))
    for candidate in pipeformer.get("candidate_forecasts") or []:
        if not isinstance(candidate, dict):
            continue
        collect(candidate)
        collect(candidate.get("parsed_task"))
        collect(candidate.get("prediction_summary"))
        collect(candidate.get("prediction"))
    return list(dict.fromkeys(magnitudes))


def _disturbance_assumption_is_disclosed(answer: str, pipeformer: Dict[str, Any]) -> bool:
    magnitudes = _disturbance_assumption_magnitudes(pipeformer)
    if not magnitudes:
        return True
    if not re.search(
        r"(?:假设|临时假设|暂按|暂定|暂设|本次按|LLM假设|LLM暂定|LLM暂设|LLM临时假设"
        r"|assum(?:e|ed|ption)|provisional)",
        answer,
        re.IGNORECASE,
    ):
        return False
    answer_numbers = _numbers_in_text(answer)
    return all(
        any(abs(abs(claimed) - expected) < 1e-6 for claimed in answer_numbers)
        for expected in magnitudes
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
    issues: List[str] = []
    if not answer.strip():
        issues.append("missing_llm_final_answer")
    if pipeformer and UNSUPPORTED_HISTORY_CLAIM.search(answer):
        issues.append("unsupported_execution_history_or_repeatability_claim")
    if pipeformer and not _disturbance_assumption_is_disclosed(answer, pipeformer):
        issues.append("undisclosed_disturbance_assumption")
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
    trusted_context = []
    for item in conversation_context or []:
        trusted = dict(item)
        if item.get("grounding_verified") is not True:
            trusted.pop("assistant_output", None)
        trusted_context.append(trusted)
    trusted_tool_outputs = [
        item
        for item in tool_outputs or []
        if not tool_output_failed(item)
    ]
    grounding_evidence: Any = {
        "pipeformer": pipeformer or {},
        "conversation_context": trusted_context,
        "tool_outputs": trusted_tool_outputs,
        "record_evidence": record_evidence or {},
    }
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
    if not pipeformer and UNSUPPORTED_PROPAGATION_CLAIM.search(answer):
        if not INFERENCE_QUALIFIER.search(answer) and not _operational_inference_is_grounded(answer, grounding_evidence):
            issues.append("unsupported_causal_or_propagation_claim")
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
    record_evidence: Optional[Dict[str, Any]] = None,
    grounding_contract: Optional[Dict[str, Any]] = None,
) -> tuple[str, List[str]]:
    outputs = tool_outputs or []
    issues = answer_quality_issues(
        answer,
        question,
        pipeformer,
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
    assessments = [classify_tool_evidence(item) for item in outputs]
    if assessments and not any(item.evidence_found for item in assessments):
        if any(item.state is ToolEvidenceState.EXECUTION_FAILED for item in assessments):
            issues.append("tool_execution_failed")
        if any(
            item.state in {ToolEvidenceState.NO_EVIDENCE, ToolEvidenceState.LOCATOR_ONLY}
            for item in assessments
        ):
            issues.append("tool_evidence_unavailable")
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


def grounded_numeric_claim_values(
    answer: str,
    question: str,
    evidence: Dict[str, Any],
) -> List[float]:
    """Return supported claims after scanning the evidence exactly once."""
    claimed = _numbers_in_text(answer)
    supported = _numbers_in_text(question)
    supported.extend(_numbers_in_value(evidence))
    supported.extend(_derived_numbers_in_value(evidence))
    claimed_times = _time_values_in_minutes(answer)
    supported_times = [
        minutes
        for _, minutes in (
            _time_values_in_minutes(question)
            + _time_values_in_minutes(json.dumps(evidence, ensure_ascii=False))
        )
    ]
    return [
        value
        for value in claimed
        if (
            _number_is_supported(value, supported)
            or _number_is_deterministically_derived(value, supported)
        )
        or any(
            _numbers_match(value, raw_value)
            and _number_is_supported(minutes, supported_times)
            for raw_value, minutes in claimed_times
        )
    ]


def numeric_claims_are_grounded(answer: str, question: str, evidence: Dict[str, Any]) -> bool:
    claimed = _numbers_in_text(answer)
    grounded = grounded_numeric_claim_values(answer, question, evidence)
    return len(grounded) == len(claimed)


def numeric_claim_values(answer: str) -> List[float]:
    """Return numeric claims while excluding dates and list/table numbering."""
    return _numbers_in_text(answer)


def numeric_grounding_evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    """Build the shared numeric-grounding view used by Task 1 and SFT export."""
    question = str(record.get("user_input") or "")
    requested = requested_artifacts(question)
    outputs = attach_tool_arguments(
        record.get("tool_outputs") or [],
        record.get("tool_calls") or [],
    )
    successful_outputs = [
        item
        for item in outputs
        if classify_tool_evidence(item, requested=requested).evidence_found
    ]
    successful_call_ids = {
        str(item.get("tool_call_id") or "")
        for item in successful_outputs
        if item.get("tool_call_id")
    }
    record_evidence = dict(record.get("evidence") or {})
    rebuilt_csv_evidence = build_csv_evidence(
        record.get("tool_calls") or [],
        record.get("tool_outputs") or [],
        str(record.get("final_answer") or ""),
        scope_text=question,
    )
    if rebuilt_csv_evidence:
        record_evidence["csv_evidence"] = {
            **dict(record_evidence.get("csv_evidence") or {}),
            **rebuilt_csv_evidence,
        }
    verified_state = VerifiedDecisionState.from_dict(
        dict(record.get("state_before") or {})
    )
    trusted_context = []
    for item in record.get("conversation_context") or []:
        if item.get("grounding_verified") is True:
            trusted_context.append(item)
            continue
        summary = item.get("verified_evidence_summary")
        legacy_verified_summary = (
            "tool_evidence_verified" not in item
            and isinstance(summary, dict)
            and bool(summary)
        )
        if item.get("tool_evidence_verified") is True or legacy_verified_summary:
            trusted_context.append({
                "tool_evidence_verified": True,
                "evidence_artifacts": list(item.get("evidence_artifacts") or []),
                "verified_evidence_summary": summary,
            })
    return {
        "prediction_summary": record.get("prediction_summary") or {},
        "constraint_check": record.get("constraint_check") or {},
        "evidence": record_evidence,
        "parsed_task": record.get("parsed_task") or {},
        "decision_summary": record.get("decision_summary") or {},
        "tool_calls": [
            item
            for item in record.get("tool_calls") or []
            if str(item.get("tool_call_id") or "") in successful_call_ids
        ],
        "tool_outputs": successful_outputs,
        "conversation_context": trusted_context,
        # Follow-up turns can legitimately cite a candidate action or a
        # forecast observation obtained on an earlier turn.  The bounded
        # VerifiedDecisionState is the verified source for that evidence;
        # raw conversation history is deliberately not reintroduced here.
        "verified_state": verified_state.to_dict(),
    }


def _numbers_match(value: float, candidate: float) -> bool:
    return abs(value - candidate) <= max(0.01, abs(candidate) * 0.005)


def _number_is_supported(value: float, supported: List[float]) -> bool:
    return any(_numbers_match(value, candidate) for candidate in supported)


def _number_is_deterministically_derived(
    value: float,
    supported: List[float],
) -> bool:
    """Recognize sign changes and simple sums/differences using bounded evidence."""
    if _number_is_supported(-value, supported):
        return True
    bounded = supported[:2_000]
    rounded_counts: Dict[float, int] = {}
    for candidate in bounded:
        key = round(candidate, 6)
        rounded_counts[key] = rounded_counts.get(key, 0) + 1

    def has_two_operands(first: float, second: float) -> bool:
        first_key = round(first, 6)
        second_key = round(second, 6)
        if second_key not in rounded_counts:
            return False
        return (
            first_key != second_key
            or rounded_counts.get(first_key, 0) >= 2
        )

    for candidate in dict.fromkeys(bounded):
        if has_two_operands(candidate, value - candidate):
            return True
        if has_two_operands(candidate, candidate - value):
            return True
        if value and has_two_operands(
            candidate,
            candidate * 100.0 / value,
        ):
            return True
    return False


def _time_values_in_minutes(value: str) -> List[tuple[float, float]]:
    values: List[tuple[float, float]] = []

    def append(raw_value: str, unit: str) -> None:
        number = float(raw_value)
        unit_key = unit.casefold()
        multiplier = 60.0 if unit_key in {"hour", "hours", "hr", "hrs", "h", "小时", "小時"} else 1.0
        values.append((number, number * multiplier))

    for match in TIME_RANGE.finditer(value):
        append(match.group("first"), match.group("unit"))
        append(match.group("second"), match.group("unit"))
    for match in TIME_VALUE.finditer(value):
        append(match.group("value"), match.group("unit"))
    return values


def requested_data_retrieved(
    question: str,
    tool_outputs: List[Dict[str, Any]],
    conversation_context: List[Dict[str, Any]],
) -> bool:
    requested = set(requested_artifacts(question))
    if not requested:
        return True

    verified_artifacts = {
        str(artifact).casefold()
        for item in conversation_context
        if (
            item.get("grounding_verified") is True
            or item.get("tool_evidence_verified") is True
        )
        for artifact in item.get("evidence_artifacts") or []
    }
    for item in conversation_context:
        if not (
            item.get("grounding_verified") is True
            or item.get("tool_evidence_verified") is True
        ):
            continue
        summary = dict(item.get("verified_evidence_summary") or {})
        csv_evidence = dict(summary.get("csv_evidence") or {})
        verified_artifacts.update(
            str(source_file).casefold()
            for source_file in csv_evidence.get("source_files") or []
            if source_file
        )
    unresolved = requested - verified_artifacts
    if not unresolved:
        return True

    for item in tool_outputs:
        assessment = classify_tool_evidence(item, requested=unresolved)
        if assessment.state is ToolEvidenceState.CONTENT_EVIDENCE:
            unresolved -= set(assessment.matched_artifacts)
    return not unresolved


def _policy_unavailable_at_generation(record: Dict[str, Any]) -> bool:
    """True when generation deliberately left the decision unsupported.

    The runtime contract requires a successful set_decision_policy call before
    ranking. When no priority wording existed, generation records
    status='insufficient_evidence' with missing 'llm_decision_policy_tool_call'
    and stores no decision_policy. Evaluation must respect that recorded state
    instead of recomputing a default-policy selection the answer never made.
    """
    if record.get("decision_policy"):
        return False
    stored_decision = dict(record.get("decision_summary") or {})
    if stored_decision.get("status") != "insufficient_evidence":
        return False
    missing = list(stored_decision.get("missing_metrics") or [])
    return "llm_decision_policy_tool_call" in missing


def record_grounding_contract(
    record: Dict[str, Any],
    tool_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Rebuild the current-turn contract from tools plus verified prior state."""
    outputs = (
        list(tool_outputs)
        if tool_outputs is not None
        else attach_tool_arguments(
            record.get("tool_outputs") or [],
            record.get("tool_calls") or [],
        )
    )
    state = VerifiedDecisionState.from_dict(
        dict(record.get("state_before") or {})
    )
    # ``decision_policy`` is also projected onto follow-up records for
    # convenience.  It is not a new policy declaration on those turns.
    # Prefer the state-carried policy so its original source question is used
    # for provenance validation.  Current set_decision_policy results are
    # discovered directly from ``outputs`` by GroundingContractBuilder.
    stored_policy = dict(record.get("decision_policy") or {}) or None
    explicit_legacy_policy = (
        stored_policy if not state.decision_policy else None
    )
    return GroundingContractBuilder().build(
        str(record.get("user_input") or ""),
        outputs,
        decision_policy=explicit_legacy_policy,
        require_decision_policy=_policy_unavailable_at_generation(record),
        prior_candidate_results=state.candidates,
        prior_decision_policy=state.decision_policy,
        prior_decision_policy_source_question=(
            state.decision_policy_source_question
        ),
        prior_applied_disturbances=state.applied_disturbances,
    )


def record_quality_issues(record: Dict[str, Any]) -> List[str]:
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

    issues = answer_quality_issues(
        str(record.get("final_answer") or ""),
        str(record.get("user_input") or ""),
        pipeformer,
        conversation_context=list(record.get("conversation_context") or []),
        tool_outputs=tool_outputs,
        record_evidence=record_evidence,
    )
    issues.extend(contract_issues)
    assessments = [classify_tool_evidence(item) for item in tool_outputs]
    if assessments and not any(item.evidence_found for item in assessments):
        if any(item.state is ToolEvidenceState.EXECUTION_FAILED for item in assessments):
            issues.append("tool_execution_failed")
        if any(
            item.state in {ToolEvidenceState.NO_EVIDENCE, ToolEvidenceState.LOCATOR_ONLY}
            for item in assessments
        ):
            issues.append("tool_evidence_unavailable")
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


def _numbers_in_text(value: str) -> List[float]:
    value = value.translate(NUMERIC_SIGN_TRANSLATION)
    numbers = []
    ignored_spans = [match.span() for match in DATE_RANGE_REFERENCE.finditer(value)]
    ignored_spans.extend(match.span() for match in DATE_REFERENCE.finditer(value))
    ignored_spans.extend(match.span() for match in COMPACT_DATE_REFERENCE.finditer(value))
    ignored_spans.extend(match.span() for match in DATA_FILE_REFERENCE.finditer(value))
    ignored_spans.extend(match.span() for match in YEAR_REFERENCE.finditer(value))
    ignored_spans.extend(match.span() for match in CANDIDATE_IDENTIFIER.finditer(value))
    ignored_spans.extend(
        match.span() for match in CHINESE_ORDINAL_REFERENCE.finditer(value)
    )
    ignored_spans.extend(
        match.span() for match in ENGLISH_ORDINAL_REFERENCE.finditer(value)
    )
    ignored_spans.extend(
        match.span() for match in ENGLISH_RANK_REFERENCE.finditer(value)
    )
    ignored_spans.extend(
        match.span() for match in NEGATED_NUMERIC_REFERENCE.finditer(value)
    )
    for match in NUMERIC_CLAIM.finditer(value):
        if any(start <= match.start() and match.end() <= end for start, end in ignored_spans):
            continue
        previous = value[match.start() - 1:match.start()]
        following = value[match.end():match.end() + 1]
        if previous in {"(", "（"} and following in {")", "）"}:
            continue
        line_start = value.rfind("\n", 0, match.start()) + 1
        prefix = value[line_start:match.start()]
        remainder = value[match.end():]
        suffix = remainder[:1]
        if not prefix.strip(" \t-*") and suffix in {".", ")", "）", "、"}:
            continue
        if prefix.strip() == "|" and remainder.lstrip().startswith("|"):
            continue
        if re.match(r"\s*(?:个?字|characters?\b)", remainder, re.IGNORECASE):
            continue
        numbers.append(float(match.group(0).replace(",", "")))
    return numbers


def _numbers_in_value(value: Any) -> List[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        values = _numbers_in_text(value)
        values.extend(
            float(match.group(1))
            for match in CASE_IDENTIFIER_NUMBER.finditer(value)
        )
        return values
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers_in_value(item)]
    if isinstance(value, list):
        return [number for item in value for number in _numbers_in_value(item)]
    return []


def _derived_numbers_in_value(value: Any) -> List[float]:
    """Compute bounded counts and grouped scalar sums from structured evidence."""
    derived: List[float] = []
    if isinstance(value, dict):
        for item in value.values():
            derived.extend(_derived_numbers_in_value(item))
        return derived
    if not isinstance(value, list):
        return derived
    derived.append(float(len(value)))
    rows = [dict(item) for item in value if isinstance(item, dict)]
    if rows and len(rows) <= 2_000:
        keys = sorted({key for row in rows for key in row})[:40]
        numeric_keys = [
            key
            for key in keys
            if any(
                isinstance(row.get(key), (int, float))
                and not isinstance(row.get(key), bool)
                for row in rows
            )
        ]
        category_keys = [
            key
            for key in keys
            if any(isinstance(row.get(key), str) for row in rows)
        ][:8]
        for numeric_key in numeric_keys:
            values = [
                float(row[numeric_key])
                for row in rows
                if isinstance(row.get(numeric_key), (int, float))
                and not isinstance(row.get(numeric_key), bool)
            ]
            if values:
                derived.append(sum(values))
                derived.append(float(sum(number > 0 for number in values)))
            for category_key in category_keys:
                grouped: Dict[str, float] = {}
                for row in rows:
                    category = row.get(category_key)
                    number = row.get(numeric_key)
                    if (
                        not isinstance(category, str)
                        or not isinstance(number, (int, float))
                        or isinstance(number, bool)
                    ):
                        continue
                    grouped[category] = (
                        grouped.get(category, 0.0) + float(number)
                    )
                if len(grouped) <= 200:
                    derived.extend(grouped.values())
    for item in value:
        derived.extend(_derived_numbers_in_value(item))
    return derived


def _variable_references_are_grounded(answer: str, grounding_evidence: Any) -> bool:
    claimed = set(VARIABLE_REFERENCE.findall(answer))
    supported = set(
        VARIABLE_REFERENCE.findall(json.dumps(grounding_evidence, ensure_ascii=False))
    )
    return claimed <= supported


def _legacy_unsupported_evidence_description(
    answer: str,
    pipeformer: Dict[str, Any],
) -> bool:
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
    finding_variables: set[str] = set()
    disturbance_variables: set[str] = set()
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
        for finding in verification.get("priority_findings") or []:
            finding_item = dict(finding or {})
            for value in [
                *list(finding_item.get("evaluated_values") or []),
                *list(finding_item.get("offending_values") or []),
            ]:
                variable = str(dict(value or {}).get("variable") or "")
                if variable:
                    finding_variables.add(variable)
        evidence = dict(forecast.get("evidence") or {})
        for key in ("top_watch_variables", "key_observation_variables"):
            for item in evidence.get(key) or []:
                variable = str(dict(item or {}).get("variable") or "")
                if variable:
                    contextual_variables.add(variable)
        parsed_task = dict(forecast.get("parsed_task") or {})
        disturbance = str(
            parsed_task.get("disturbance_variable") or ""
        )
        if disturbance:
            disturbance_variables.add(disturbance)
        for application in evidence.get(
            "boundary_application_evidence"
        ) or []:
            variable = str(dict(application or {}).get("variable") or "")
            if variable:
                disturbance_variables.add(variable)

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
                if (
                    EVIDENCE_DESCRIPTION_TERM.search(text)
                    and variable not in finding_variables
                    and variable not in disturbance_variables
                    and variable not in typed_categories
                ):
                    return True
                continue
            if not (
                mentioned_categories
                & typed_categories.get(variable, set())
            ):
                return True
    return False


llm_answer_quality_issues = answer_quality_issues
