from __future__ import annotations

from typing import Any, Dict, List


def build_teacher_answer(
    parsed_task: Dict[str, Any],
    verification: Dict[str, Any],
    evidence_variables: List[Dict[str, Any]],
) -> Dict[str, Any]:
    first_non_pass = next((check for check in verification["checks"] if check["status"] != "pass"), None)
    requires_intervention = verification["overall_status"] in {"warning", "fail"}
    consequence = (
        "Mock PipeFormer forecast is numerically finite and does not show a hard pressure or flow proxy violation."
        if verification["overall_status"] != "fail"
        else "Mock PipeFormer forecast triggers at least one failed engineering proxy check."
    )
    if first_non_pass:
        consequence += f" The first audit trigger is {first_non_pass['name']}."

    return {
        "most_likely_operating_consequence": consequence,
        "top_3_watch_indicators": evidence_variables[:3],
        "requires_manual_intervention": requires_intervention,
        "manual_intervention_reason": (
            "Review is recommended because at least one proxy check returned warning/fail."
            if requires_intervention
            else "No manual intervention is required for this mock smoke test."
        ),
        "priority_audit_constraint": first_non_pass["name"] if first_non_pass else None,
        "key_observation_variables": evidence_variables[:2],
        "scope_note": "This is a mock smoke-test trace. It validates pipeline wiring, not real pipeline engineering performance.",
    }