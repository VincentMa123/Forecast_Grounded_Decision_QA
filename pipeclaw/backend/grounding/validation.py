from __future__ import annotations

from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .construction import provisional_assumptions


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    formatted = f"{number:.9g}"
    return format(Decimal(formatted), "f") if "e" in formatted.casefold() else formatted


_NUMBER_TOKEN = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[-+]?\d+)?(?!\w)",
    re.IGNORECASE,
)


def _reported_numbers(answer: str) -> List[tuple[str, Decimal]]:
    numbers = []
    for token in _NUMBER_TOKEN.findall(answer):
        try:
            numbers.append((token, Decimal(token)))
        except (InvalidOperation, ValueError):
            continue
    return numbers


def _number_disclosed(
    answer: str,
    value: Any,
    numbers: Optional[Sequence[tuple[str, Decimal]]] = None,
) -> bool:
    try:
        expected = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return format_number(value) in answer
    for token, reported in numbers if numbers is not None else _reported_numbers(answer):
        if reported == expected:
            return True
        mantissa = token.casefold().split("e", 1)[0]
        if "e" not in token.casefold() and "." in mantissa:
            precision = Decimal(1).scaleb(-len(mantissa.rsplit(".", 1)[1]))
            try:
                if expected.quantize(precision) == reported:
                    return True
            except InvalidOperation:
                continue
    return False


_CANDIDATE_REFERENCE = re.compile(r"\bcandidate_[A-Za-z0-9_-]+\b", re.IGNORECASE)
_SELECTED_CANDIDATE = re.compile(
    r"selected_candidate_id\s*[:=]\s*([A-Za-z][A-Za-z0-9_-]*|none|null)",
    re.IGNORECASE,
)


def _contract_action_variables(contract: Dict[str, Any]) -> set[str]:
    return {
        str(variable)
        for candidate in contract.get("candidate_results") or []
        for key in ("percentage_changes", "setpoints")
        for variable in dict(dict(candidate.get("action") or {}).get(key) or {})
        if str(variable)
    }


def _contains_bare_action_prefix(answer: str, action_variables: Iterable[str]) -> bool:
    """Detect `T_002` when the registered action ID is `T_002:SNQ`."""
    return any(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(variable.split(':', 1)[0])}(?![A-Za-z0-9_:])",
            answer,
            re.IGNORECASE,
        )
        for variable in action_variables
        if ":" in variable
    )


def _canonical_number(value: Any) -> str:
    try:
        number = Decimal(str(value).strip())
    except Exception:
        return str(value)
    if not number.is_finite():
        return str(value)
    if number == 0:
        return "0"
    rendered = format(number, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


_CANONICAL_DISCLOSURE_PREFIXES = (
    "Applied disturbance:",
    "Applied setpoint:",
    "Application status:",
)
_CANONICAL_ASSUMPTION_PREFIX = "Assumption source:"
_PROVISIONAL_ASSUMPTION_DISCLOSURE = re.compile(
    r"(?:假设|临时假设|暂按|暂定|暂设|本次按|LLM假设|LLM暂定|LLM暂设|LLM临时假设"
    r"|assum(?:e|ed|ption)|provisional)",
    re.IGNORECASE,
)
_NOT_EVALUATED_DISCLOSURE = "not_evaluated（未执行该项校核，不能判定 pass/fail）"
_AMBIGUOUS_NOT_EVALUATED = re.compile(
    r"(?<![A-Za-z0-9_])not_evaluated(?![A-Za-z0-9_])(?:"
    r"\s*[（(][^）)\r\n]{0,60}未通过[^）)\r\n]{0,60}[）)]"
    r"|[，,]\s*(?:即|属|视为|按)\s*[“\"]?未通过"
    r"(?:校核项|校核|评估|处理|对待|项)?[”\"]?"
    r"(?:\s*(?:处理|看待|对待))?[）)]?"
    r")"
)


def normalize_not_evaluated_wording(answer: str) -> str:
    """Keep not-evaluated evidence distinct from an actual failed check."""
    return _AMBIGUOUS_NOT_EVALUATED.sub(_NOT_EVALUATED_DISCLOSURE, answer)


def _answer_lines(answer: str) -> List[str]:
    return answer.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _canonical_sequence_value(values: Any, fallback: Any) -> str:
    rendered = [_canonical_number(item) for item in list(values or [])]
    if not rendered:
        return _canonical_number(fallback)
    if len(set(rendered)) == 1:
        return rendered[0]
    return json.dumps(rendered, ensure_ascii=False, separators=(",", ":"))


def canonical_applied_disturbance_lines(contract: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    seen = set()
    for raw_item in contract.get("applied_disturbances") or []:
        item = dict(raw_item or {})
        variable = str(item.get("variable") or "")
        mode = str(item.get("mode") or "")
        if not variable or not mode or item.get("verified") is not True:
            continue
        requested = item.get("requested_value")
        if mode == "percent_change":
            try:
                number = Decimal(str(requested).strip())
            except Exception:
                number = None
            if number is None or not number.is_finite():
                value = _canonical_number(requested)
            else:
                direction = str(item.get("direction") or "").casefold()
                if direction == "down":
                    number = -abs(number)
                elif direction == "up":
                    number = abs(number)
                value = _canonical_number(number)
                if number > 0:
                    value = f"+{value}"
            primary = f"Applied disturbance: {variable}={value}%"
        elif mode == "setpoint":
            primary = f"Applied setpoint: {variable}={_canonical_number(requested)}"
        else:
            primary = f"Applied disturbance: {variable}={_canonical_number(requested)}"
        if primary not in seen:
            seen.add(primary)
            lines.append(primary)
        if item.get("no_op") is True:
            status = (
                f"Application status: {variable}=no-op; "
                f"prior={_canonical_sequence_value(item.get('input_values_before'), requested)}; "
                f"applied={_canonical_sequence_value(item.get('input_values_applied'), requested)}"
            )
            if status not in seen:
                seen.add(status)
                lines.append(status)
    return lines


def answer_without_machine_disclosure(answer: str) -> str:
    """Return natural answer prose without deterministic metadata lines."""
    return "\n".join(
        line
        for line in _answer_lines(answer)
        if not line.strip().startswith(
            (*_CANONICAL_DISCLOSURE_PREFIXES, _CANONICAL_ASSUMPTION_PREFIX)
        )
    ).strip()


def _without_embedded_required_disclosures(prose: str, required_lines: Sequence[str]) -> str:
    cleaned = []
    for line in prose.split("\n"):
        positions = [line.find(required) for required in required_lines if required and required in line]
        line = line[:min(positions)].rstrip() if positions else line
        if line.strip():
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def _canonical_assumption_source_lines(contract: Dict[str, Any]) -> List[str]:
    assignments = {}
    for line in canonical_applied_disturbance_lines(contract):
        if line.startswith(("Applied disturbance:", "Applied setpoint:")):
            assignment = line.split(":", 1)[1].strip()
            assignments.setdefault(assignment.split("=", 1)[0], assignment)
    lines = []
    for raw_item in contract.get("provisional_assumptions") or []:
        variable = str(dict(raw_item or {}).get("variable") or "")
        assignment = assignments.get(variable)
        line = f"{_CANONICAL_ASSUMPTION_PREFIX} LLM provisional; {assignment}" if assignment else ""
        if line and line not in lines:
            lines.append(line)
    return lines


def provisional_assumption_disclosed(
    answer: str,
    pipeformer: Dict[str, Any],
    numeric_values: List[float],
) -> bool:
    """Check canonical disclosures while retaining legacy teacher-trace wording."""
    forecasts = [pipeformer] + [
        item
        for item in pipeformer.get("candidate_forecasts") or []
        if isinstance(item, dict)
    ]
    assumptions = provisional_assumptions([{"output": forecast} for forecast in forecasts])
    magnitudes = list(dict.fromkeys(
        abs(float(item["magnitude_percent"]))
        for item in assumptions
        if item.get("magnitude_percent") is not None
    ))
    if not magnitudes:
        return True
    canonical = any(
        line.strip().startswith(
            f"{_CANONICAL_ASSUMPTION_PREFIX} LLM provisional;"
        )
        for line in _answer_lines(answer)
    )
    disclosed = canonical or _PROVISIONAL_ASSUMPTION_DISCLOSURE.search(answer)
    return bool(disclosed) and all(
        any(abs(abs(value) - expected) < 1e-6 for value in numeric_values)
        for expected in magnitudes
    )


def finalize_applied_disturbance_disclosure(answer: str, contract: Dict[str, Any]) -> str:
    """Serialize the canonical application block without rewriting prose."""
    required = [*canonical_applied_disturbance_lines(contract), *_canonical_assumption_source_lines(contract)]
    prose = _without_embedded_required_disclosures(
        normalize_not_evaluated_wording(answer_without_machine_disclosure(answer)),
        required,
    )
    disclosure = "\n".join(required)
    return f"{disclosure}\n{prose}" if disclosure and prose else disclosure or prose


_COMPARISON_ACTIVITY_PATTERNS = (
    re.compile(r"\bcandidate[_-]?\d+\s*(?:>|<|>=|<=)\s*candidate[_-]?\d+\b", re.IGNORECASE),
    re.compile(r"(?:recommend|select|choose|建议|推荐|选择).{0,24}\bcandidate[_-]?\d+\b", re.IGNORECASE),
)


def comparison_requirements_active(answer: str, contract: Dict[str, Any]) -> bool:
    """Escalate comparison validation only for operational turns or claims."""
    return bool(
        int(contract.get("current_candidate_forecast_count") or 0) > 0
        or int(contract.get("current_decision_policy_call_count") or 0) > 0
        or _SELECTED_CANDIDATE.search(answer)
        or any(pattern.search(answer) for pattern in _COMPARISON_ACTIVITY_PATTERNS)
    )


_AUDIT_CATEGORIES = (
    ("pressure", "pressure_metrics", "压力", "pressure", r"压力|压裕|pressure"),
    ("flow", "flow_metrics", "流量", "flow", r"流量|供需|flow|supply.?demand"),
    ("linepack", "linepack_metrics", "管存", "linepack", r"管存|linepack"),
    ("compressor", "compressor_metrics", "压缩机", "compressor", r"压缩机|负荷|compressor|load"),
    ("energy", "energy_metrics", "能耗", "energy", r"能耗|energy"),
)

_HARD_CONSTRAINT_OUTCOME = re.compile(
    r"(?:F|失败|failure)\s*[:=]?\s*0|failure_count\s*[:=]\s*0|无(?:规则)?失败|"
    r"硬约束(?:均|全部|全都|已)?(?:通过|满足)|"
    r"(?:全部|所有|均).{0,16}(?<![未不])(?:通过|满足).{0,12}硬约束|"
    r"hard constraints?\s*(?:(?:all|are|were|have been)\s*)*(?:pass(?:ed)?|satisf(?:ied|y))|"
    r"(?:all|every)\s+hard constraints?\s+(?:pass(?:ed)?|are\s+satisfied)",
    re.IGNORECASE,
)
_REJECTION_REASON = re.compile(
    r"次优|未选|拒选|淘汰|lower-ranked|not selected|eliminated|rejected",
    re.IGNORECASE,
)


def _comparison_checks(
    answer: str, contract: Dict[str, Any]
) -> tuple[List[str], List[str], List[str]]:
    """Run all comparison checks from one normalized answer/context snapshot."""
    action_variables = _contract_action_variables(contract)
    lines = [line.strip() for line in _answer_lines(answer) if line.strip()]
    required = canonical_applied_disturbance_lines(contract)
    actual = [line for line in lines if line.startswith(_CANONICAL_DISCLOSURE_PREFIXES)]
    disclosure = []
    if _contains_bare_action_prefix(answer, action_variables):
        disclosure.append("canonical_action_variable_abbreviated")
    primary = [line for line in required if not line.startswith("Application status:")]
    statuses = [line for line in required if line.startswith("Application status:")]
    if any(line not in actual for line in primary):
        disclosure.append("applied_disturbance_disclosure_missing")
    if any(line not in actual for line in statuses):
        disclosure.append("disturbance_no_op_disclosure_missing")
    if (actual != required and actual) or sum(answer.count(line) for line in required) != len(required):
        disclosure.append("unexpected_applied_disturbance_disclosure")
    if required and lines[:len(required)] != required:
        disclosure.append("canonical_disclosure_block_not_at_start")

    candidates = [
        str(item.get("candidate_id") or "")
        for item in contract.get("candidate_results") or []
        if item.get("candidate_id")
    ]
    selection = _SELECTED_CANDIDATE.search(answer)
    comparison_text = _SELECTED_CANDIDATE.sub("", answer)
    folded = comparison_text.casefold()
    selection_issues = []
    if any(candidate.casefold() not in folded for candidate in candidates):
        selection_issues.append("candidate_comparison_incomplete")
    known = {candidate.casefold() for candidate in candidates}
    referenced = {match.group(0).casefold() for match in _CANDIDATE_REFERENCE.finditer(answer)}
    if referenced - known:
        selection_issues.append("unknown_candidate_reference")
    if selection is None:
        selection_issues.append("candidate_selection_missing")
    else:
        actual_id = selection.group(1).casefold()
        expected = str((contract.get("decision_summary") or {}).get("selected_candidate_id") or "none").casefold()
        if actual_id == "null":
            actual_id = "none"
        if actual_id != expected:
            selection_issues.append("candidate_selection_contradicts_contract")

    evidence_issues = []
    decision = dict(contract.get("decision_summary") or {})
    if decision.get("status") == "selected":
        objective_evidence = dict(decision.get("objective_evidence") or {})
        numbers = _reported_numbers(answer)
        missing = [
            f"{candidate_id}:{metric}"
            for candidate_id, metrics in objective_evidence.items()
            if candidate_id.casefold() in folded
            for metric, evidence in dict(metrics or {}).items()
            if dict(evidence or {}).get("value") is not None
            and not _number_disclosed(answer, dict(evidence or {}).get("value"), numbers)
        ]
        if missing:
            evidence_issues.append("decision_objective_evidence_incomplete")
        if any(variable.casefold() not in folded for variable in action_variables):
            evidence_issues.append("candidate_action_mapping_incomplete")
        present_categories = {
            category
            for candidate in contract.get("candidate_results") or []
            for category, key, *_ in _AUDIT_CATEGORIES
            if dict(candidate.get(key) or {})
        }
        if any(
            not re.search(pattern, answer, re.IGNORECASE)
            for category, _, _, _, pattern in _AUDIT_CATEGORIES
            if category in present_categories
        ):
            evidence_issues.append("candidate_audit_evidence_incomplete")
        if not _HARD_CONSTRAINT_OUTCOME.search(answer):
            evidence_issues.append("hard_constraint_outcome_missing")
        if len(candidates) > 1 and not _REJECTION_REASON.search(answer):
            evidence_issues.append("candidate_rejection_reason_missing")
    if contract.get("answer_render_status") == "answer_budget_insufficient":
        evidence_issues.append("answer_budget_insufficient")
    return disclosure, selection_issues, evidence_issues


def comparison_answer_issues(answer: str, contract: Dict[str, Any]) -> List[str]:
    """Validate semantic comparison coverage and the typed selection claim."""
    disclosure, selection, evidence = _comparison_checks(answer, contract)
    issues = list(disclosure)
    if contract.get("answer_mode") == "dispatch_comparison" and comparison_requirements_active(answer, contract):
        issues.extend(selection)
        issues.extend(evidence)
    return list(dict.fromkeys(issues))
