from __future__ import annotations

from typing import Any, Dict, List


INTERVENTION_REASONS = {
    "no_intervention": "No manual intervention is required by the engineering constraint library.",
    "monitoring_only": "Continue monitoring because at least one non-safety rule returned a warning.",
    "operator_attention_required": "Operator attention is required because an engineering rule returned warning/fail.",
    "immediate_intervention_required": "Immediate intervention is required because a safety-priority rule failed.",
}


def risk_level_from_status(status: str) -> str:
    return {
        "pass": "low",
        "warning": "medium",
        "fail": "high",
    }.get(status, "unknown")


def final_answer_text(answer: Dict[str, Any]) -> str:
    indicators = ", ".join(item["variable"] for item in answer.get("top_3_watch_indicators", []))
    key_variables = ", ".join(item["variable"] for item in answer.get("key_observation_variables", []))
    intervention = answer.get("manual_intervention_label") or (
        "required" if answer.get("requires_manual_intervention") else "no_intervention"
    )
    parts = [
        answer.get("most_likely_operating_consequence", ""),
        f"Top watch indicators: {indicators}." if indicators else "",
        f"Manual intervention label: {intervention}.",
        f"Priority audit constraint: {answer.get('priority_audit_constraint')}."
        if answer.get("priority_audit_constraint")
        else "",
        f"Key evidence variables: {key_variables}." if key_variables else "",
        answer.get("scope_note", ""),
    ]
    return " ".join(part for part in parts if part)


def row_labels(rows: List[Any]) -> List[str]:
    return [row.label for row in rows]


def build_teacher_answer(
    parsed_task: Dict[str, Any],
    verification: Dict[str, Any],
    evidence_variables: List[Dict[str, Any]],
) -> Dict[str, Any]:
    non_pass = [check for check in verification.get("checks", []) if check.get("status") != "pass"]
    non_pass.sort(key=lambda check: check.get("priority", 999))
    first_non_pass = non_pass[0] if non_pass else None
    intervention_label = verification.get("human_intervention_label", "no_intervention")
    requires_intervention = intervention_label in {"operator_attention_required", "immediate_intervention_required"}

    if verification.get("overall_status") == "pass":
        consequence = "PipeFormer forecast does not trigger the configured engineering constraint library."
    else:
        consequence = (
            f"PipeFormer forecast triggers {verification.get('overall_status')} engineering review "
            f"under the PDF-style constraint library."
        )
    if first_non_pass:
        consequence += f" The first priority audit trigger is {first_non_pass['name']}."

    return {
        "most_likely_operating_consequence": consequence,
        "top_3_watch_indicators": evidence_variables[:3],
        "requires_manual_intervention": requires_intervention,
        "manual_intervention_label": intervention_label,
        "manual_intervention_reason": INTERVENTION_REASONS.get(intervention_label, "Review the constraint output."),
        "priority_audit_constraint": first_non_pass["name"] if first_non_pass else None,
        "key_observation_variables": evidence_variables[:2],
        "scope_note": (
            "Engineering checks now follow the PDF categories. The current mock run uses normalized proxy "
            "thresholds until real pressure, flow, linepack, compressor, and equipment limits are configured."
        ),
    }
