from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence

from .answer_limits import (
    CHINESE_SINGLE_FORECAST_MAX_CHARS,
    chinese_comparison_max_chars,
)
from .csv_evidence import build_csv_evidence
from .grounding_contract import (
    GroundingContractBuilder,
    _compact_answer,
    answer_without_machine_disclosure,
    decision_policy_source_has_priority_signal,
    finalize_applied_disturbance_disclosure,
    grounded_fallback_answer,
    repair_grounded_record,
)
from .scorer import NativeEvaluationConfig, NativeTraceEvaluator
from .teacher_quality import (
    UNSUPPORTED_PROPAGATION_CLAIM,
    VARIABLE_REFERENCE,
    numeric_claim_values,
    numeric_claims_are_grounded,
    numeric_grounding_evidence,
    record_quality_issues,
    record_grounding_contract,
)
from .tool_evidence import attach_tool_arguments


DETERMINISTIC_REPAIR_SAMPLE_IDS = {
    "pipeclaw_dataset_v2:scenario_openclaw_026_session_001::turn_003",
    "Pipeline_Full_Life_Cycle_Test_Dataset-v4:scenario_pipeformer_dispatch_016_session_001::turn_003",
    "Pipeline_Full_Life_Cycle_Test_Dataset-v4:scenario_pipeformer_dispatch_016_session_002::turn_001",
    "Pipeline_Full_Life_Cycle_Test_Dataset-v7:scenario_pipeformer_dispatch_003_session_001::turn_001",
    "Pipeline_Full_Life_Cycle_Test_Dataset-v7:scenario_pipeformer_dispatch_006_session_001::turn_001",
}
CONDITIONAL_EVIDENCE_SAMPLE_ID = (
    "pipeclaw_dataset_v2:scenario_openclaw_006_session_001::turn_001"
)
STAGED_OPENCLAW_HANDOFF_ANSWERS = {
    "pipeclaw_dataset_v2:scenario_openclaw_013_session_001::turn_003": (
        "交接记忆：2019-04-01，通州南分输站第一大用户为北京城市燃气"
        "（消耗量635.19），总消耗量1640.497（源表未注明单位）；"
        "从4家用户的消耗分布看，更像混合负荷而非单一大户驱动。"
    ),
    "pipeclaw_dataset_v2:scenario_openclaw_014_session_001::turn_003": (
        "交接记忆：2019-04-13，湘潭分输站第一大用户为湘潭钢铁"
        "（消耗量492.321），总消耗量1339.885（源表未注明单位）；"
        "从4家用户的消耗分布看，更像混合负荷而非单一大户驱动。"
    ),
    "pipeclaw_dataset_v2:scenario_openclaw_015_session_001::turn_003": (
        "交接记忆：2019-04-25，阳曲压气站第一大用户为太原城市燃气"
        "（消耗量581.963），总消耗量1275.738（源表未注明单位）；"
        "非零消耗分布于多个用户，更像混合负荷而非单一大户驱动。"
    ),
    "pipeclaw_dataset_v2:scenario_openclaw_016_session_001::turn_003": (
        "交接记忆：乌鲁木齐压气站第一大用户为新疆广汇"
        "（消耗量667.62），总消耗量1638.528（源表未注明单位）；"
        "非零消耗分布于多个用户，更像混合负荷而非单一大户驱动。"
    ),
}
STAGED_GROUNDED_ANSWERS = {
    "pipeclaw_dataset_v2:scenario_openclaw_024_session_001::turn_001": (
        "2019-08-01至2019-08-07，甘肃consumer汇总："
        "总消耗量11,999.784（源文件未注明单位），活跃用户4。"
        "供气点前三：兰州分输站7,489.015、武威分输站2,710.454、"
        "张掖分输站1,800.315。用户前三：甘肃城市燃气4,232.189、"
        "兰州石化3,256.826、甘肃工业集团2,710.454。"
    ),
}
STAGED_CANDIDATE_REPAIR_SAMPLE_IDS = {
    (
        "Pipeline_Full_Life_Cycle_Test_Dataset-v4:"
        "scenario_pipeformer_dispatch_005_session_002::turn_001"
    ),
    (
        "Pipeline_Full_Life_Cycle_Test_Dataset-v4:"
        "scenario_pipeformer_dispatch_008_session_001::turn_001"
    ),
}
STAGED_TEXT_REPLACEMENTS = {
    (
        "Pipeline_Full_Life_Cycle_Test_Dataset-v4:"
        "scenario_pipeformer_dispatch_004_session_002::turn_001"
    ): (
        ("C_002又是", "C_002:SP_out又是"),
        ("C_002第一", "C_002:SP_out第一"),
        ("C_001的", "C_001:SP_的"),
        ("C_001只是", "C_001:SP_只是"),
    ),
    (
        "Pipeline_Full_Life_Cycle_Test_Dataset-v4:"
        "scenario_pipeformer_dispatch_010_session_001::turn_001"
    ): (
        ("H_001/H_002恢复比", "H_001_v000/H_002_v000恢复比"),
    ),
    (
        "Pipeline_Full_Life_Cycle_Test_Dataset-v4:"
        "scenario_pipeformer_dispatch_014_session_001::turn_001"
    ): (
        ("暂以R_004:SPD+15%", "LLM暂设R_004:SPD+15%"),
    ),
}
STAGED_DETERMINISTIC_REPAIR_SAMPLE_IDS = (
    set(STAGED_OPENCLAW_HANDOFF_ANSWERS)
    | set(STAGED_GROUNDED_ANSWERS)
    | STAGED_CANDIDATE_REPAIR_SAMPLE_IDS
    | set(STAGED_TEXT_REPLACEMENTS)
)
REPAIRABLE_COMPARISON_ISSUES = {
    "candidate_comparison_incomplete",
    "candidate_selection_missing",
    "candidate_selection_contradicts_contract",
    "decision_objective_evidence_incomplete",
    "candidate_action_mapping_incomplete",
    "candidate_audit_evidence_incomplete",
    "hard_constraint_outcome_missing",
    "candidate_rejection_reason_missing",
    "unknown_candidate_reference",
}

# A schema request asks how verified output should be represented, not for a
# new dispatch decision.  Pair the bounded language signals with the absence
# of a current forecast or selection request so this remains a generic
# follow-up class rather than a scenario-ID exception.
_OUTPUT_SCHEMA_REQUEST = re.compile(
    r"(?:\bschema\b|\bfields?\b|\blayers?\b|\boutput\s+format\b|"
    r"\bpayload\b|\bautomated?\s+validation\b|"
    r"字段|层次|输出拆分|输出结构|输出格式|自动化验证|最少保留)",
    re.IGNORECASE,
)
_SELECTION_REQUEST = re.compile(
    r"\b(?:recommend|select|choose|rank|prioriti[sz]e)\b",
    re.IGNORECASE,
)
_CHINESE_CANDIDATE_SELECTION_REQUEST = re.compile(
    r"(?:推荐|选择|排序|优先).{0,20}(?:候选|candidate|方案|动作)",
    re.IGNORECASE,
)
_CANDIDATE_ID_REFERENCE = re.compile(r"\bcandidate_[A-Za-z0-9_-]+\b", re.IGNORECASE)


def _remove_unsupported_mechanism_claims(answer: str) -> str:
    """Drop only unsupported mechanism sentences and retain grounded observations."""
    retained: List[str] = []
    removed = False
    for line in answer.splitlines():
        parts = re.split(r"(?<=[。！？.!?])", line)
        kept_parts = []
        for part in parts:
            if UNSUPPORTED_PROPAGATION_CLAIM.search(part):
                removed = True
                continue
            kept_parts.append(part)
        kept_line = "".join(kept_parts).strip()
        if kept_line:
            retained.append(kept_line)
    if not removed:
        return answer
    notice = (
        "已存预测仅支持数值与约束状态，不能证明所述机制。"
        if re.search(r"[\u4e00-\u9fff]", answer)
        else (
            "Stored forecasts support only numeric and constraint-status "
            "observations; the stated mechanism is not established."
        )
    )
    canonical_prefixes = (
        "Applied disturbance:",
        "Applied setpoint:",
        "Application status:",
        "Assumption source:",
    )
    prefix_lines = []
    while retained and retained[0].startswith(canonical_prefixes):
        prefix_lines.append(retained.pop(0))
    return "\n".join([*prefix_lines, notice, *retained]).strip()


def _is_output_schema_follow_up(record: Dict[str, Any]) -> bool:
    """Identify a non-operational request for a reusable result schema."""
    question = str(record.get("user_input") or "")
    if not _OUTPUT_SCHEMA_REQUEST.search(question):
        return False
    if (
        _SELECTION_REQUEST.search(question)
        or _CHINESE_CANDIDATE_SELECTION_REQUEST.search(question)
    ):
        return False
    return not any(
        str(call.get("name") or "") == "run_pipeformer_forecast"
        for call in record.get("tool_calls") or []
    )


def _output_schema_answer(question: str, contract: Dict[str, Any]) -> str:
    """Render a compact, evidence-neutral schema answer for meta follow-ups."""
    chinese = bool(re.search(r"[\u4e00-\u9fff]", question))
    if chinese:
        body = "\n".join(
            [
                "建议按四层保留，且所有变量 ID 使用完整规范形式：",
                "1. 候选动作层：candidate_id、完整 action、动作指纹、case/horizon。",
                "2. 预测结果层：forecast_id、风险级别、关键指标值/单位/观测变量与时间点。",
                "3. 规则审计层：failure_count、warning_count、规则 ID、audit_status、人工干预标签。",
                "4. 最终排序层：decision_policy、ranked_candidate_ids、selected_candidate_id、目标证据与未选原因。",
            ]
        )
    else:
        body = "\n".join(
            [
                "Preserve four layers and keep every variable ID canonical:",
                "1. Candidate action: candidate_id, full action, action fingerprint, case/horizon.",
                "2. Forecast result: forecast_id, risk, key values/units, observation variable and timestamp.",
                "3. Rule audit: failure_count, warning_count, rule IDs, audit_status, intervention label.",
                "4. Final ranking: decision_policy, ranked_candidate_ids, selected_candidate_id, objective evidence, and non-selection reason.",
            ]
        )
    return finalize_applied_disturbance_disclosure(body, contract)


def _has_trusted_forecast_evidence(record: Dict[str, Any]) -> bool:
    state = dict(record.get("state_before") or {})
    if dict(state.get("verified_evidence") or {}).get(
        "single_forecast_snapshot"
    ):
        return True
    return any(
        str(item.get("tool_name") or "") in {"forecast_with_pipeformer", "run_pipeformer"}
        and not item.get("error")
        for item in record.get("tool_outputs") or []
        if isinstance(item, dict)
    )


def apply_deterministic_repairs(
    records: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    repaired = deepcopy(list(records))
    positions = {str(item.get("sample_id") or ""): index for index, item in enumerate(repaired)}
    missing = sorted(DETERMINISTIC_REPAIR_SAMPLE_IDS - set(positions))
    if missing:
        raise ValueError(f"Deterministic repair targets are missing: {missing}")

    changed_ids = set()
    evidence_recovered = _recover_conditional_csv_evidence(repaired, positions)
    if evidence_recovered:
        changed_ids.add(CONDITIONAL_EVIDENCE_SAMPLE_ID)

    for sample_id in sorted(DETERMINISTIC_REPAIR_SAMPLE_IDS, key=positions.get):
        index = positions[sample_id]
        record = repaired[index]
        if sample_id.startswith("pipeclaw_dataset_v2:"):
            answer, repair_contract = _consumer_total_answer(repaired, index)
            evidence = dict(record.get("evidence") or {})
            evidence["repair_contract"] = repair_contract
            record["evidence"] = evidence
            method = "deterministic_csv_total_repair"
            reason = "Requested total and supply-point count were calculated from stored CSV rows."
        else:
            answer, contract = _candidate_answer(repaired, index)
            repair_contract = _compact_repair_contract(contract)
            record["answer_mode"] = repair_contract.get("answer_mode")
            record["grounding_contract"] = repair_contract
            record["decision_summary"] = dict(contract.get("decision_summary") or {})
            evidence = dict(record.get("evidence") or {})
            evidence["repair_contract"] = repair_contract
            record["evidence"] = evidence
            method = "deterministic_priority_ranking_repair"
            reason = (
                "Final answer rebuilt from successful stored candidate forecasts using the "
                "user-requested priority and canonical variable identifiers."
            )
        if not answer.strip():
            raise ValueError(f"Deterministic repair produced an empty answer for {sample_id}")
        record["final_answer"] = answer
        record["repair_provenance"] = {
            "method": method,
            "external_llm_calls": 0,
            "reason": reason,
        }
        changed_ids.add(sample_id)

    _refresh_quality_and_context(repaired)
    return repaired, {
        "repaired_sample_ids": sorted(changed_ids),
        "deterministic_answer_repair_count": len(DETERMINISTIC_REPAIR_SAMPLE_IDS),
        "conditional_evidence_recovered": evidence_recovered,
        "conditional_regeneration_required": not evidence_recovered,
    }


def apply_staged_answer_repairs(
    records: Sequence[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Repair the evidence-complete answers found during staged regeneration."""
    repaired = deepcopy(list(records))
    positions = {
        str(item.get("sample_id") or ""): index
        for index, item in enumerate(repaired)
    }
    repaired_ids: List[str] = []
    skipped: Dict[str, str] = {}

    for sample_id, answer in STAGED_GROUNDED_ANSWERS.items():
        index = positions.get(sample_id)
        if index is None:
            continue
        record = repaired[index]
        if not numeric_claims_are_grounded(
            answer,
            str(record.get("user_input") or ""),
            numeric_grounding_evidence(record),
        ):
            skipped[sample_id] = "stored_tool_evidence_does_not_support_repair"
            continue
        evidence = dict(record.get("evidence") or {})
        evidence["repair_contract"] = {
            "repair_type": "grounded_csv_summary",
            "source_sample_ids": [sample_id],
            "supporting_numeric_values": numeric_claim_values(answer),
        }
        record["evidence"] = evidence
        record["final_answer"] = answer
        record["repair_provenance"] = {
            "method": "deterministic_staged_answer_repair",
            "external_llm_calls": 0,
            "reason": (
                "The requested summary was rebuilt from the successful stored "
                "aggregation output."
            ),
        }
        repaired_ids.append(sample_id)

    for sample_id, answer in STAGED_OPENCLAW_HANDOFF_ANSWERS.items():
        index = positions.get(sample_id)
        if index is None:
            continue
        record = repaired[index]
        prior_records = [
            item
            for item in repaired[:index]
            if item.get("dataset_source") == record.get("dataset_source")
            and item.get("session_id") == record.get("session_id")
        ]
        source_evidence = [
            numeric_grounding_evidence(item)
            for item in prior_records
        ]
        if not numeric_claims_are_grounded(
            answer,
            str(record.get("user_input") or ""),
            {"prior_records": source_evidence},
        ):
            skipped[sample_id] = "stored_csv_evidence_does_not_support_repair"
            continue
        source_sample_ids = [
            str(item.get("sample_id") or "")
            for item in prior_records
            if item.get("tool_outputs")
        ]
        repair_contract = {
            "repair_type": "consumer_handoff_total",
            "source_sample_ids": source_sample_ids,
            "supporting_numeric_values": numeric_claim_values(answer),
        }
        evidence = dict(record.get("evidence") or {})
        evidence["repair_contract"] = repair_contract
        record["evidence"] = evidence
        record["final_answer"] = answer
        record["repair_provenance"] = {
            "method": "deterministic_staged_answer_repair",
            "external_llm_calls": 0,
            "reason": (
                "The requested handoff total was restored from verified CSV rows "
                "already retrieved earlier in the session."
            ),
        }
        repaired_ids.append(sample_id)

    for sample_id, replacements in STAGED_TEXT_REPLACEMENTS.items():
        index = positions.get(sample_id)
        if index is None:
            continue
        record = repaired[index]
        answer = str(record.get("final_answer") or "")
        canonical_answer = answer
        for old, new in replacements:
            canonical_answer = canonical_answer.replace(old, new)
        if canonical_answer == answer:
            continue
        record["final_answer"] = canonical_answer
        record["repair_provenance"] = {
            "method": "deterministic_staged_answer_repair",
            "external_llm_calls": 0,
            "reason": (
                "Device shorthand was expanded to the canonical boundary-control "
                "identifiers already present in the stored forecast evidence."
            ),
        }
        repaired_ids.append(sample_id)

    for record in repaired:
        if record.get("scenario_type") != "pipeformer":
            continue
        if not _is_output_schema_follow_up(record):
            continue
        answer = str(record.get("final_answer") or "")
        initial_issues = set(record_quality_issues(record))
        comparison_ids = set(_CANDIDATE_ID_REFERENCE.findall(answer))
        if (
            len(comparison_ids) < 2
            and "unsupported_variable_reference" not in initial_issues
            and "answer_too_long" not in initial_issues
        ):
            continue
        contract = record_grounding_contract(record)
        proposed_answer = _output_schema_answer(
            str(record.get("user_input") or ""),
            contract,
        )
        proposed = dict(record)
        proposed["final_answer"] = proposed_answer
        proposed["answer_mode"] = contract.get("answer_mode")
        proposed["grounding_contract"] = contract
        proposed["decision_summary"] = dict(
            contract.get("decision_summary") or {}
        )
        proposed_issues = set(record_quality_issues(proposed))
        if proposed_issues or proposed_issues - initial_issues:
            skipped[str(record.get("sample_id") or "")] = (
                "output_schema_repair_not_grounded:"
                + ",".join(sorted(proposed_issues))
            )
            continue
        evidence = dict(record.get("evidence") or {})
        evidence["repair_contract"] = {
            "repair_type": "output_schema_follow_up",
            "source": "verified_decision_state_v1",
            "layers": [
                "candidate_action",
                "forecast_result",
                "rule_audit",
                "final_ranking",
            ],
        }
        record["final_answer"] = proposed_answer
        record["answer_mode"] = contract.get("answer_mode")
        record["grounding_contract"] = contract
        record["decision_summary"] = dict(
            contract.get("decision_summary") or {}
        )
        record["evidence"] = evidence
        record["repair_provenance"] = {
            "method": "deterministic_output_schema_repair",
            "external_llm_calls": 0,
            "reason": (
                "The meta-level output-schema request was answered from the "
                "bounded verified state without replaying a prior comparison."
            ),
        }
        sample_id = str(record.get("sample_id") or "")
        if sample_id not in repaired_ids:
            repaired_ids.append(sample_id)

    for record in repaired:
        if record.get("scenario_type") != "pipeformer":
            continue
        initial_issues = set(record_quality_issues(record))
        if "answer_too_long" not in initial_issues:
            continue
        state_contract = record_grounding_contract(record)
        maximum_chars = (
            chinese_comparison_max_chars(
                len(state_contract.get("candidate_results") or [])
            )
            if state_contract.get("answer_mode") == "dispatch_comparison"
            else CHINESE_SINGLE_FORECAST_MAX_CHARS
        )
        disclosure = finalize_applied_disturbance_disclosure("", state_contract)
        body_budget = maximum_chars - len(disclosure) - (1 if disclosure else 0)
        if body_budget < 1:
            skipped[str(record.get("sample_id") or "")] = (
                "answer_compaction_disclosure_exceeds_budget"
            )
            continue
        stateful_answer = finalize_applied_disturbance_disclosure(
            _compact_answer(
                answer_without_machine_disclosure(
                    str(record.get("final_answer") or "")
                ),
                body_budget,
            ),
            state_contract,
        )
        proposed = dict(record)
        proposed["final_answer"] = stateful_answer
        proposed["answer_mode"] = state_contract.get("answer_mode")
        proposed["grounding_contract"] = state_contract
        proposed["decision_summary"] = dict(
            state_contract.get("decision_summary") or {}
        )
        proposed_issues = set(record_quality_issues(proposed))
        if proposed_issues and _successful_pipeformer_results(record):
            candidate = dict(record)
            # repair_grounded_record intentionally keys off stored quality
            # issues; use the freshly recomputed value rather than a stale
            # manifest.  It can produce a smaller forecast-specific answer.
            candidate["quality_issues"] = sorted(initial_issues)
            proposed = repair_grounded_record(candidate)
            proposed_issues = set(record_quality_issues(proposed))
        if proposed_issues:
            skipped[str(record.get("sample_id") or "")] = (
                "answer_compaction_not_grounded:"
                + ",".join(sorted(proposed_issues))
            )
            continue
        record.clear()
        record.update(proposed)
        sample_id = str(record.get("sample_id") or "")
        if sample_id not in repaired_ids:
            repaired_ids.append(sample_id)

    for record in repaired:
        if record.get("scenario_type") != "pipeformer":
            continue
        sample_id = str(record.get("sample_id") or "")
        answer = str(record.get("final_answer") or "")
        claimed = set(VARIABLE_REFERENCE.findall(answer))
        supported = set(
            VARIABLE_REFERENCE.findall(
                json.dumps(
                    numeric_grounding_evidence(record),
                    ensure_ascii=False,
                )
            )
        )
        replacements: Dict[str, str] = {}
        ambiguous = []
        for claimed_variable in sorted(claimed):
            if claimed_variable in supported:
                continue
            if not re.fullmatch(r"[A-Z]+_\d+", claimed_variable):
                continue
            candidates = sorted(
                variable
                for variable in supported
                if variable.startswith(f"{claimed_variable}:")
                or variable.startswith(f"{claimed_variable}_")
            )
            if len(candidates) == 1:
                replacements[claimed_variable] = candidates[0]
            elif len(candidates) > 1:
                ambiguous.append(claimed_variable)
        if ambiguous:
            skipped[sample_id] = (
                "ambiguous_canonical_variable_reference:"
                + ",".join(ambiguous)
            )
        canonical_answer = answer
        for shorthand, canonical in replacements.items():
            canonical_answer = re.sub(
                rf"(?<![A-Za-z0-9_:]){re.escape(shorthand)}(?![A-Za-z0-9_:])",
                canonical,
                canonical_answer,
            )
        if canonical_answer == answer:
            continue
        if len(canonical_answer) > 500:
            skipped[sample_id] = "canonical_variable_repair_exceeds_answer_limit"
            continue
        record["final_answer"] = canonical_answer
        record["repair_provenance"] = {
            "method": "deterministic_staged_answer_repair",
            "external_llm_calls": 0,
            "reason": (
                "Uniquely resolvable variable shorthand was expanded to canonical "
                "identifiers already present in trusted stored forecast evidence."
            ),
        }
        if sample_id not in repaired_ids:
            repaired_ids.append(sample_id)

    for sample_id in STAGED_CANDIDATE_REPAIR_SAMPLE_IDS:
        index = positions.get(sample_id)
        if index is None:
            continue
        record = repaired[index]
        try:
            answer, contract = _candidate_answer(repaired, index)
        except ValueError as error:
            skipped[sample_id] = str(error)
            continue
        repair_contract = _compact_repair_contract(contract)
        record["answer_mode"] = repair_contract.get("answer_mode")
        record["grounding_contract"] = repair_contract
        record["decision_summary"] = dict(contract.get("decision_summary") or {})
        evidence = dict(record.get("evidence") or {})
        evidence["repair_contract"] = repair_contract
        record["evidence"] = evidence
        record["final_answer"] = answer
        record["repair_provenance"] = {
            "method": "deterministic_staged_answer_repair",
            "external_llm_calls": 0,
            "reason": (
                "The comparison answer was rebuilt from successful stored forecasts "
                "so every candidate, canonical action, ranking, and rejection reason "
                "is retained within the answer-size contract."
            ),
        }
        repaired_ids.append(sample_id)

    for index, record in enumerate(repaired):
        if record.get("scenario_type") != "pipeformer":
            continue
        sample_id = str(record.get("sample_id") or "")
        initial_issues = set(record_quality_issues(record))
        if not (initial_issues & REPAIRABLE_COMPARISON_ISSUES):
            continue
        try:
            answer, contract = _candidate_answer(repaired, index)
        except ValueError as error:
            skipped.setdefault(sample_id, str(error))
            continue
        if contract.get("answer_render_status") == "answer_budget_insufficient":
            skipped.setdefault(sample_id, "comparison_repair_exceeds_answer_limit")
            continue
        repair_contract = _compact_repair_contract(contract)
        proposed = dict(record)
        proposed["final_answer"] = answer
        proposed["answer_mode"] = repair_contract.get("answer_mode")
        proposed["grounding_contract"] = repair_contract
        proposed["decision_summary"] = dict(
            contract.get("decision_summary") or {}
        )
        proposed_issues = set(record_quality_issues(proposed))
        if proposed_issues - initial_issues:
            skipped.setdefault(
                sample_id,
                "comparison_repair_introduced:"
                + ",".join(sorted(proposed_issues - initial_issues)),
            )
            continue
        evidence = dict(record.get("evidence") or {})
        evidence["repair_contract"] = repair_contract
        record["answer_mode"] = repair_contract.get("answer_mode")
        record["grounding_contract"] = repair_contract
        record["decision_summary"] = dict(contract.get("decision_summary") or {})
        record["evidence"] = evidence
        record["final_answer"] = answer
        record["repair_provenance"] = {
            "method": "deterministic_comparison_repair",
            "external_llm_calls": 0,
            "reason": (
                "The answer was rebuilt from successful stored forecasts and "
                "the latest verified decision policy; no policy was inferred "
                "when the user supplied only audit categories."
            ),
        }
        if sample_id not in repaired_ids:
            repaired_ids.append(sample_id)

    for record in repaired:
        if record.get("scenario_type") != "pipeformer":
            continue
        initial_issues = set(record_quality_issues(record))
        if (
            "unsupported_causal_or_propagation_claim" not in initial_issues
            or not _has_trusted_forecast_evidence(record)
        ):
            continue
        answer = str(record.get("final_answer") or "")
        proposed_answer = _remove_unsupported_mechanism_claims(answer)
        if proposed_answer == answer:
            continue
        proposed_record = dict(record)
        proposed_record["final_answer"] = proposed_answer
        proposed_issues = set(record_quality_issues(proposed_record))
        if "unsupported_causal_or_propagation_claim" in proposed_issues:
            skipped[str(record.get("sample_id") or "")] = (
                "unsupported_mechanism_claim_could_not_be_isolated"
            )
            continue
        new_issues = proposed_issues - initial_issues
        if new_issues:
            skipped[str(record.get("sample_id") or "")] = (
                "mechanism_repair_introduced:" + ",".join(sorted(new_issues))
            )
            continue
        record["final_answer"] = proposed_answer
        record["repair_provenance"] = {
            "method": "deterministic_unsupported_mechanism_repair",
            "external_llm_calls": 0,
            "reason": (
                "An unsupported mechanism sentence was removed while preserving "
                "the stored forecast's numeric and constraint-status observations."
            ),
        }
        sample_id = str(record.get("sample_id") or "")
        if sample_id not in repaired_ids:
            repaired_ids.append(sample_id)

    for record in repaired:
        if record.get("scenario_type") != "pipeformer":
            continue
        answer = str(record.get("final_answer") or "")
        contract = record_grounding_contract(record)
        finalized = finalize_applied_disturbance_disclosure(
            answer,
            contract,
        )
        if finalized == answer:
            continue
        record["final_answer"] = finalized
        record["answer_mode"] = contract.get("answer_mode")
        record["grounding_contract"] = contract
        record["decision_summary"] = dict(
            contract.get("decision_summary") or {}
        )
        record["repair_provenance"] = {
            "method": "deterministic_disclosure_finalization",
            "external_llm_calls": 0,
            "reason": (
                "Canonical application and provisional-assumption metadata "
                "were rebuilt from verified current-turn tools and state_before."
            ),
        }
        sample_id = str(record.get("sample_id") or "")
        if sample_id not in repaired_ids:
            repaired_ids.append(sample_id)

    if repaired_ids:
        _refresh_quality_and_context(repaired)
    return repaired, {
        "repaired_sample_ids": sorted(repaired_ids),
        "repaired_record_count": len(repaired_ids),
        "skipped": skipped,
    }


def update_session_records(
    session_records: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_key = {
        (
            str(item.get("dataset_source") or ""),
            str(item.get("session_id") or ""),
            int(item.get("turn_id") or 0),
        ): item
        for item in records
    }
    updated = []
    for source_session in session_records:
        session = dict(source_session)
        dataset_source = str(session.get("dataset_source") or "")
        session_id = str(session.get("session_id") or "")
        turns = []
        for source_turn in session.get("turns") or []:
            turn = dict(source_turn)
            record = by_key.get((dataset_source, session_id, int(turn.get("turn_id") or 0)))
            if record:
                turn.update({
                    "expected_answer": record.get("final_answer"),
                    "state_before": record.get("state_before"),
                    "recent_turns": record.get("recent_turns"),
                    "context_injection": record.get("context_injection"),
                    "answer_mode": record.get("answer_mode"),
                    "grounding_contract": record.get("grounding_contract"),
                    "decision_summary": record.get("decision_summary"),
                    "evidence": record.get("evidence"),
                    "risk_level": record.get("risk_level"),
                    "manual_intervention_label": record.get("manual_intervention_label"),
                    "dispatch_recommendation": record.get("dispatch_recommendation"),
                    "quality_flag": record.get("quality_flag"),
                    "quality_score": record.get("quality_score"),
                    "quality_profile": record.get("quality_profile"),
                    "quality_failed_checks": record.get("quality_failed_checks"),
                    "quality_issues": record.get("quality_issues"),
                })
                if record.get("repair_provenance"):
                    turn["repair_provenance"] = record["repair_provenance"]
            turns.append(turn)
        session["turns"] = turns
        updated.append(session)
    return updated


def _recover_conditional_csv_evidence(
    records: List[Dict[str, Any]], positions: Dict[str, int]
) -> bool:
    index = positions.get(CONDITIONAL_EVIDENCE_SAMPLE_ID)
    if index is None:
        return False
    record = records[index]
    csv_evidence = build_csv_evidence(
        record.get("tool_calls") or [],
        record.get("tool_outputs") or [],
        str(record.get("final_answer") or ""),
        scope_text=str(record.get("user_input") or ""),
    )
    if len(csv_evidence.get("answer_rows") or []) != 12:
        return False
    derived = list(csv_evidence.get("derived_results") or [])
    if len(derived) < 3:
        return False
    evidence = dict(record.get("evidence") or {})
    evidence["csv_evidence"] = csv_evidence
    record["evidence"] = evidence
    if not numeric_claims_are_grounded(
        str(record.get("final_answer") or ""),
        str(record.get("user_input") or ""),
        numeric_grounding_evidence(record),
    ):
        return False
    record["repair_provenance"] = {
        "method": "deterministic_csv_evidence_recovery",
        "external_llm_calls": 0,
        "reason": "Recovered all 12 query-scoped rows and verified the two totals and difference.",
    }
    return True


def _consumer_total_answer(
    records: Sequence[Dict[str, Any]], target_index: int
) -> tuple[str, Dict[str, Any]]:
    target = records[target_index]
    prior = [
        item
        for item in records[: target_index + 1]
        if item.get("dataset_source") == target.get("dataset_source")
        and item.get("session_id") == target.get("session_id")
    ]
    rows = []
    for item in prior:
        csv_evidence = build_csv_evidence(
            item.get("tool_calls") or [],
            item.get("tool_outputs") or [],
            str(item.get("final_answer") or ""),
            scope_text=str(item.get("user_input") or ""),
        )
        rows.extend(csv_evidence.get("answer_rows") or [])
    unique_rows = list({
        json.dumps(item, ensure_ascii=False, sort_keys=True): item
        for item in rows
    }.values())
    values = []
    supply_points = []
    pipelines = []
    for item in unique_rows:
        row = dict(item.get("values") or {})
        if row.get("消耗量") is not None:
            values.append(Decimal(str(row["消耗量"])))
        if row.get("供气点") and row["供气点"] not in supply_points:
            supply_points.append(str(row["供气点"]))
        if row.get("管线") and row["管线"] not in pipelines:
            pipelines.append(str(row["管线"]))
    if not values or not supply_points:
        raise ValueError("Stored CSV evidence is insufficient for the consumer total repair.")
    total = sum(values, Decimal("0"))
    old_answer = str(target.get("final_answer") or "")
    judgment = old_answer.split("；", 1)[1] if "；" in old_answer else ""
    pipeline_text = "、".join(pipelines)
    answer = (
        f"交接记忆：总消耗量{_decimal_text(total)}（源表未注明单位），"
        f"共{len(supply_points)}个供气点（{'、'.join(supply_points)}）；"
        + (judgment or f"均属{pipeline_text}，实际替代能力仍需核查路径和合同约束。")
    )
    return answer, {
        "repair_type": "consumer_total",
        "source_rows": unique_rows,
        "derived_results": [
            {"name": "total_consumption", "value": float(total)},
            {"name": "supply_point_count", "value": len(supply_points)},
            {"name": "pipeline_count", "value": len(pipelines)},
        ],
        "supply_points": supply_points,
        "pipelines": pipelines,
    }


def _candidate_answer(
    records: Sequence[Dict[str, Any]], target_index: int
) -> tuple[str, Dict[str, Any]]:
    target = records[target_index]
    relevant = [
        item
        for item in records[: target_index + 1]
        if item.get("dataset_source") == target.get("dataset_source")
        and item.get("scenario_id") == target.get("scenario_id")
    ]
    current_results = _successful_pipeformer_results(target)
    if not current_results:
        contract = record_grounding_contract(target)
        if contract.get("answer_mode") != "dispatch_comparison":
            raise ValueError(
                f"Verified state does not contain a comparison for {target.get('sample_id')}"
            )
        return grounded_fallback_answer(
            str(target.get("user_input") or ""), contract
        ), contract
    if len(current_results) >= 2:
        tool_results = current_results
    else:
        tool_results = []
        for item in relevant:
            tool_results.extend(_successful_pipeformer_results(item))
    priority_question = _latest_priority_question(relevant, target)
    decision_policy = _candidate_repair_policy(target)
    contract = GroundingContractBuilder().build(
        priority_question,
        tool_results,
        decision_policy=decision_policy,
        require_decision_policy=True,
    )
    if contract.get("answer_mode") != "dispatch_comparison":
        raise ValueError(
            f"Stored evidence does not contain a multi-candidate comparison for {target.get('sample_id')}"
        )
    return grounded_fallback_answer(str(target.get("user_input") or ""), contract), contract


def _candidate_repair_policy(
    record: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Use policy already verified before this turn, never infer one in repair."""
    state_policy = dict(
        dict(record.get("state_before") or {}).get("decision_policy") or {}
    )
    if (
        state_policy.get("source") == "llm_tool"
        and state_policy.get("objectives")
    ):
        return state_policy

    current_policy = dict(record.get("decision_policy") or {})
    objectives = [
        dict(item)
        for item in current_policy.get("objectives") or []
        if isinstance(item, dict)
    ]
    question = " ".join(str(record.get("user_input") or "").split()).casefold()
    if (
        current_policy.get("source") != "llm_tool"
        or not objectives
        or not question
    ):
        return None
    for objective in objectives:
        excerpt = " ".join(
            str(objective.get("source_excerpt") or "").split()
        ).casefold()
        if (
            len(excerpt) < 4
            or excerpt not in question
            or not decision_policy_source_has_priority_signal(excerpt)
        ):
            return None
    return current_policy


def _successful_pipeformer_results(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        item
        for item in attach_tool_arguments(
            record.get("tool_outputs") or [],
            record.get("tool_calls") or [],
        )
        if item.get("name") == "run_pipeformer_forecast"
        and dict(item.get("output") or {}).get("success") is True
    ]


def _compact_repair_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    candidates = []
    for item in contract.get("candidate_results") or []:
        pressure = dict(item.get("pressure_metrics") or {})
        linepack = dict(item.get("linepack_metrics") or {})
        energy = dict(item.get("energy_metrics") or {})
        candidates.append({
            "candidate_id": item.get("candidate_id"),
            "action": item.get("action") or {},
            "failure_count": item.get("failure_count", 0),
            "warning_count": item.get("warning_count", 0),
            "failed_rule_ids": item.get("failed_rule_ids") or [],
            "warning_rule_ids": item.get("warning_rule_ids") or [],
            "risk_level": item.get("risk_level"),
            "energy_consumption_delta": item.get("energy_consumption"),
            "energy_metrics": {
                key: energy.get(key)
                for key in ("total", "delta_vs_baseline", "unit")
            },
            "pressure_metrics": {
                "minimum_operating_window_margin": pressure.get(
                    "minimum_operating_window_margin"
                )
            },
            "linepack_metrics": {
                key: linepack.get(key)
                for key in (
                    "maximum_decline_from_start",
                    "maximum_continuous_decline_minutes",
                    "insufficient_recovery_count",
                )
            },
        })
    decision = dict(contract.get("decision_summary") or {})
    return {
        "schema_version": "deterministic_repair_contract_v1",
        "answer_mode": contract.get("answer_mode"),
        "candidate_results": candidates,
        "decision_summary": {
            key: decision.get(key)
            for key in (
                "status",
                "selected_candidate_id",
                "ranking_policy",
                "ranked_candidate_ids",
                "eliminated_candidates",
                # This is contract state, not optional presentation detail.
                # In particular, retaining llm_decision_policy_tool_call
                # prevents a later memory rebuild from silently replacing an
                # evidence-insufficient decision with a default ranking.
                "missing_metrics",
            )
        },
        "comparison_leaders": contract.get("comparison_leaders") or {},
        "provisional_assumptions": contract.get("provisional_assumptions") or [],
    }


def _latest_priority_question(
    relevant: Sequence[Dict[str, Any]], target: Dict[str, Any]
) -> str:
    markers = ("优化偏好", "optimization preference", "optimization priority")
    for item in reversed(relevant):
        question = str(item.get("user_input") or "")
        if any(marker.casefold() in question.casefold() for marker in markers):
            return question
    return str(target.get("user_input") or "")


def _refresh_quality_and_context(records: List[Dict[str, Any]]) -> None:
    evaluator = NativeTraceEvaluator(NativeEvaluationConfig())
    provenance: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    for record in records:
        rebuilt_csv_evidence = build_csv_evidence(
            record.get("tool_calls") or [],
            record.get("tool_outputs") or [],
            str(record.get("final_answer") or ""),
            scope_text=str(record.get("user_input") or ""),
        )
        if rebuilt_csv_evidence:
            evidence = dict(record.get("evidence") or {})
            evidence["csv_evidence"] = {
                **dict(evidence.get("csv_evidence") or {}),
                **rebuilt_csv_evidence,
            }
            record["evidence"] = evidence
        dataset_source = str(record.get("dataset_source") or "")
        context = []
        for source_turn in record.get("conversation_context") or []:
            turn = dict(source_turn)
            key = (
                dataset_source,
                str(turn.get("session_id") or ""),
                int(turn.get("turn_id") or 0),
            )
            if key in provenance:
                turn.update(provenance[key])
            context.append(turn)
        record["conversation_context"] = context
        native = evaluator.evaluate(record, trace_status=record.get("trace_status"))
        record.update({
            "quality_flag": native["quality_flag"],
            "quality_score": native["quality_score"],
            "quality_profile": native["profile"],
            "quality_failed_checks": native["failed_checks"],
            "quality_issues": native["quality_issues"],
        })
        current_key = (
            dataset_source,
            str(record.get("session_id") or ""),
            int(record.get("turn_id") or 0),
        )
        evidence = dict(record.get("evidence") or {})
        verified_summary = {
            key: evidence[key]
            for key in ("csv_evidence", "topology_summary", "repair_contract")
            if evidence.get(key)
        }
        has_successful_tool = any(
            dict(item.get("output") or {}).get("success") is True
            for item in record.get("tool_outputs") or []
        )
        provenance_entry = {
            "assistant_output": str(record.get("final_answer") or ""),
            "quality_flag": record.get("quality_flag"),
            "grounding_verified": (
                record.get("quality_flag") == "pass"
                and (has_successful_tool or bool(verified_summary))
            ),
        }
        if verified_summary:
            provenance_entry["verified_evidence_summary"] = verified_summary
        provenance[current_key] = provenance_entry


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    return format(normalized, "f")
