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

def _number_disclosed(answer: str, value: Any) -> bool:
    try:
        expected = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return format_number(value) in answer
    for match in _NUMBER_TOKEN.finditer(answer):
        token = match.group(0)
        try:
            reported = Decimal(token)
        except (InvalidOperation, ValueError):
            continue
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
    for variable in action_variables:
        if ":" not in variable:
            continue
        prefix = variable.split(":", 1)[0]
        pattern = (
            rf"(?<![A-Za-z0-9_]){re.escape(prefix)}"
            rf"(?![A-Za-z0-9_:])"
        )
        if re.search(pattern, answer, re.IGNORECASE):
            return True
    return False

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
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered

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

def _canonical_sequence_value(values: Any, fallback: Any) -> str:
    items = list(values or [])
    if not items:
        return _canonical_number(fallback)
    rendered = [_canonical_number(item) for item in items]
    if len(set(rendered)) == 1:
        return rendered[0]
    return json.dumps(rendered, ensure_ascii=False, separators=(",", ":"))

def canonical_applied_disturbance_lines(
    contract: Dict[str, Any],
) -> List[str]:
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
            direction = str(item.get("direction") or "").casefold()
            if number is not None and number.is_finite():
                if direction == "down":
                    number = -abs(number)
                elif direction == "up":
                    number = abs(number)
                value = _canonical_number(number)
                if number > 0:
                    value = f"+{value}"
            else:
                value = _canonical_number(requested)
            primary = f"Applied disturbance: {variable}={value}%"
        elif mode == "setpoint":
            primary = f"Applied setpoint: {variable}={_canonical_number(requested)}"
        else:
            primary = f"Applied disturbance: {variable}={_canonical_number(requested)}"
        if primary not in seen:
            seen.add(primary)
            lines.append(primary)

        if item.get("verified") is True and item.get("no_op") is True:
            prior = _canonical_sequence_value(
                item.get("input_values_before"), requested
            )
            applied = _canonical_sequence_value(
                item.get("input_values_applied"), requested
            )
            status = (
                f"Application status: {variable}=no-op; "
                f"prior={prior}; applied={applied}"
            )
            if status not in seen:
                seen.add(status)
                lines.append(status)
    return lines

def _canonical_disclosure_lines_in_answer(answer: str) -> List[str]:
    return [
        line.strip()
        for line in answer.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip().startswith(_CANONICAL_DISCLOSURE_PREFIXES)
    ]

def answer_without_machine_disclosure(answer: str) -> str:
    """Return natural answer prose without deterministic metadata lines."""
    return "\n".join(
        line
        for line in answer.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if not line.strip().startswith(
            (*_CANONICAL_DISCLOSURE_PREFIXES, _CANONICAL_ASSUMPTION_PREFIX)
        )
    ).strip()

def _without_embedded_required_disclosures(
    prose: str,
    required_lines: Sequence[str],
) -> str:
    cleaned: List[str] = []
    for line in prose.split("\n"):
        positions = [
            line.find(required)
            for required in required_lines
            if required and required in line
        ]
        if positions:
            line = line[: min(positions)].rstrip()
        if line.strip():
            cleaned.append(line)
    return "\n".join(cleaned).strip()

def _canonical_assumption_source_lines(
    contract: Dict[str, Any],
) -> List[str]:
    assignments: Dict[str, str] = {}
    for line in canonical_applied_disturbance_lines(contract):
        if not line.startswith(("Applied disturbance:", "Applied setpoint:")):
            continue
        assignment = line.split(":", 1)[1].strip()
        variable = assignment.split("=", 1)[0]
        assignments.setdefault(variable, assignment)

    lines: List[str] = []
    for raw_item in contract.get("provisional_assumptions") or []:
        item = dict(raw_item or {})
        variable = str(item.get("variable") or "")
        assignment = assignments.get(variable)
        if not assignment:
            continue
        line = f"{_CANONICAL_ASSUMPTION_PREFIX} LLM provisional; {assignment}"
        if line not in lines:
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
    assumptions = provisional_assumptions(
        [{"output": forecast} for forecast in forecasts]
    )
    magnitudes = list(
        dict.fromkeys(
            abs(float(item["magnitude_percent"]))
            for item in assumptions
            if item.get("magnitude_percent") is not None
        )
    )
    if not magnitudes:
        return True
    disclosed = any(
        line.strip().startswith(f"{_CANONICAL_ASSUMPTION_PREFIX} LLM provisional;")
        for line in answer.splitlines()
    ) or _PROVISIONAL_ASSUMPTION_DISCLOSURE.search(answer)
    return bool(disclosed) and all(
        any(abs(abs(value) - expected) < 1e-6 for value in numeric_values)
        for expected in magnitudes
    )

def finalize_applied_disturbance_disclosure(
    answer: str,
    contract: Dict[str, Any],
) -> str:
    """Serialize the canonical application block without rewriting prose."""
    required = [
        *canonical_applied_disturbance_lines(contract),
        *_canonical_assumption_source_lines(contract),
    ]
    prose = normalize_not_evaluated_wording(answer_without_machine_disclosure(answer))
    prose = _without_embedded_required_disclosures(prose, required)
    disclosure = "\n".join(required)
    if disclosure and prose:
        return f"{disclosure}\n{prose}"
    return disclosure or prose

def comparison_requirements_active(
    answer: str,
    contract: Dict[str, Any],
) -> bool:
    """Escalate comparison validation only for operational turns or claims."""
    return bool(
        int(contract.get("current_candidate_forecast_count") or 0) > 0
        or int(contract.get("current_decision_policy_call_count") or 0) > 0
        or _SELECTED_CANDIDATE.search(answer)
        or re.search(
            r"\bcandidate[_-]?\d+\s*(?:>|<|>=|<=)\s*candidate[_-]?\d+\b",
            answer,
            re.IGNORECASE,
        )
        or re.search(
            r"(?:recommend|select|choose|建议|推荐|选择).{0,24}\bcandidate[_-]?\d+\b",
            answer,
            re.IGNORECASE,
        )
    )

_AUDIT_CATEGORIES = (
    ("pressure", "pressure_metrics", "压力", "pressure", r"压力|压裕|pressure"),
    ("flow", "flow_metrics", "流量", "flow", r"流量|供需|flow|supply.?demand"),
    ("linepack", "linepack_metrics", "管存", "linepack", r"管存|linepack"),
    (
        "compressor",
        "compressor_metrics",
        "压缩机",
        "compressor",
        r"压缩机|负荷|compressor|load",
    ),
    ("energy", "energy_metrics", "能耗", "energy", r"能耗|energy"),
)

def comparison_answer_issues(answer: str, contract: Dict[str, Any]) -> List[str]:
    """Validate semantic comparison coverage and the typed selection claim."""
    action_variables = _contract_action_variables(contract)
    issues: List[str] = []
    if _contains_bare_action_prefix(answer, action_variables):
        issues.append("canonical_action_variable_abbreviated")
    required_disclosure = canonical_applied_disturbance_lines(contract)
    actual_disclosure = _canonical_disclosure_lines_in_answer(answer)
    normalized_answer_lines = [
        line.strip()
        for line in answer.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]
    required_primary = [
        line
        for line in required_disclosure
        if not line.startswith("Application status:")
    ]
    required_status = [
        line for line in required_disclosure if line.startswith("Application status:")
    ]
    if any(line not in actual_disclosure for line in required_primary):
        issues.append("applied_disturbance_disclosure_missing")
    if any(line not in actual_disclosure for line in required_status):
        issues.append("disturbance_no_op_disclosure_missing")
    disclosure_occurrences = sum(answer.count(line) for line in required_disclosure)
    if (
        actual_disclosure != required_disclosure and actual_disclosure
    ) or disclosure_occurrences != len(required_disclosure):
        issues.append("unexpected_applied_disturbance_disclosure")
    if (
        required_disclosure
        and normalized_answer_lines[: len(required_disclosure)] != required_disclosure
    ):
        issues.append("canonical_disclosure_block_not_at_start")
    if contract.get("answer_mode") != "dispatch_comparison":
        return list(dict.fromkeys(issues))
    if not comparison_requirements_active(answer, contract):
        return list(dict.fromkeys(issues))
    candidates = [
        str(item.get("candidate_id") or "")
        for item in contract.get("candidate_results") or []
        if item.get("candidate_id")
    ]
    selection = _SELECTED_CANDIDATE.search(answer)
    comparison_text = _SELECTED_CANDIDATE.sub("", answer)
    answer_folded = comparison_text.casefold()
    if any(candidate.casefold() not in answer_folded for candidate in candidates):
        issues.append("candidate_comparison_incomplete")

    known = {candidate.casefold() for candidate in candidates}
    referenced = {
        match.group(0).casefold() for match in _CANDIDATE_REFERENCE.finditer(answer)
    }
    if referenced - known:
        issues.append("unknown_candidate_reference")

    if selection is None:
        issues.append("candidate_selection_missing")
    else:
        actual = selection.group(1).casefold()
        expected = str(
            (contract.get("decision_summary") or {}).get("selected_candidate_id")
            or "none"
        ).casefold()
        if actual == "null":
            actual = "none"
        if actual != expected:
            issues.append("candidate_selection_contradicts_contract")

    decision = dict(contract.get("decision_summary") or {})
    if decision.get("status") == "selected":
        objective_evidence = dict(decision.get("objective_evidence") or {})
        missing_objective_values = []
        for candidate_id, metrics in objective_evidence.items():
            if candidate_id.casefold() not in answer_folded:
                continue
            for metric, evidence in dict(metrics or {}).items():
                expected_value = dict(evidence or {}).get("value")
                if expected_value is not None and not _number_disclosed(
                    answer, expected_value
                ):
                    missing_objective_values.append(f"{candidate_id}:{metric}")
        if missing_objective_values:
            issues.append("decision_objective_evidence_incomplete")

        if any(
            variable.casefold() not in answer_folded for variable in action_variables
        ):
            issues.append("candidate_action_mapping_incomplete")

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
            issues.append("candidate_audit_evidence_incomplete")

        if not re.search(
            r"(?:F|失败|failure)\s*[:=]?\s*0|failure_count\s*[:=]\s*0|无(?:规则)?失败|"
            r"硬约束(?:均|全部|全都|已)?(?:通过|满足)|"
            r"(?:全部|所有|均).{0,16}(?<![未不])(?:通过|满足).{0,12}硬约束|"
            r"hard constraints?\s*(?:(?:all|are|were|have been)\s*)*"
            r"(?:pass(?:ed)?|satisf(?:ied|y))|"
            r"(?:all|every)\s+hard constraints?\s+"
            r"(?:pass(?:ed)?|are\s+satisfied)",
            answer,
            re.IGNORECASE,
        ):
            issues.append("hard_constraint_outcome_missing")
        if len(candidates) > 1 and not re.search(
            r"次优|未选|拒选|淘汰|lower-ranked|not selected|eliminated|rejected",
            answer,
            re.IGNORECASE,
        ):
            issues.append("candidate_rejection_reason_missing")
    if contract.get("answer_render_status") == "answer_budget_insufficient":
        issues.append("answer_budget_insufficient")
    return list(dict.fromkeys(issues))
