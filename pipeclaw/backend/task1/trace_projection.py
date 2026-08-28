from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

from pipeclaw.backend.evaluator.quality_references import (
    numeric_claim_values,
    sft_file_references,
    variable_references,
)
from pipeclaw.backend.grounding.evidence.tool import (
    classify_tool_evidence,
    command_python_scripts,
    normalized_tool_path,
    requested_artifacts,
    tool_output_failed,
)
from pipeclaw.backend.pipeline.forecast_registry_contract import authorize_forecast_registry
from pipeclaw.backend.pipeline.forecast_result import (
    COMPACT_COMPARABLE_METRIC_KEYS,
    ForecastResult,
)
from pipeclaw.backend.task1.trace_history import compact_tool_call_arguments


SFT_MAX_TOOL_TEXT_CHARS = 4_000
SFT_MAX_GENERIC_TOOL_PAIRS = 6
SFT_MAX_GENERIC_OUTPUT_CHARS = 2_500
SFT_MAX_PIPEFORMER_VARIABLES = 3
SFT_OMITTED_TOOL_KEYS = {
    "abs_path",
    "cmd",
    "cwd",
    "duration_s",
    "output_dir",
    "run_dir",
    "session_id",
    "timestamp",
    "workspace",
}
def _evidence_blob(output: Optional[Dict[str, Any]]) -> str:
    return json.dumps((output or {}).get("output") or {}, ensure_ascii=False).casefold()


def parse_tool_output(tool_call: Dict[str, Any]) -> Any:
    if "result" in tool_call:
        return tool_call["result"]
    raw = tool_call.get("result_summary")
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _select_fields(value: Mapping[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    return {key: value[key] for key in keys if key in value}


def _compact_watch_variables(
    evidence: Mapping[str, Any],
    *,
    include_auxiliary_variables: bool = True,
) -> Dict[str, Any]:
    keys = ("top_watch_variables", "key_observation_variables")
    item_keys = (
        "variable",
        "role",
        "metric",
        "value",
        "status",
        "mean_prediction",
        "mean_abs_delta_vs_observed",
    )
    return {
        key: [
            _select_fields(item, item_keys)
            for item in list(evidence.get(key) or [])[:3]
        ]
        for key in keys
        if include_auxiliary_variables and evidence.get(key)
    }


def tool_call_id(tool_call: Dict[str, Any], index: int) -> str:
    return str(tool_call.get("tool_call_id") or f"tool_{index:03d}")


def _legacy_projection(
    forecast: ForecastResult,
    arguments: Mapping[str, Any],
) -> Dict[str, Any]:
    """Map the public contract to legacy record field names without reshaping it."""
    prediction = forecast.prediction
    resolution = forecast.task_resolution
    parsed_task = forecast.parsed_task or {
        **_select_fields(
            prediction,
            (
                "case_id",
                "current_operating_condition_number",
                "disturbance_variable",
                "disturbance_direction",
                "disturbance_magnitude_percent",
                "disturbance_assumption",
                "disturbance_source",
                "forecast_horizon_minutes",
            ),
        ),
        **_select_fields(
            arguments,
            (
                "attention_targets",
                "output_state_variables",
                "constraint_verification_types",
                "boundary_conditions",
            ),
        ),
        **{
            key: resolution.get(key, 0 if key.endswith("_count") else [])
            for key in (
                "unresolved_attention_targets",
                "unresolved_output_state_variables",
                "variable_normalizations",
                "vocabulary_normalizations",
                "invalid_normalized_variables",
                "resolved_attention_variable_count",
                "resolved_output_variable_count",
            )
        },
        "forecast_time_step_minutes": dict(
            prediction.get("forecast_window") or {}
        ).get("time_step_minutes"),
    }
    parsed_task = {key: item for key, item in parsed_task.items() if item is not None}
    return {
        "parsed_task": parsed_task,
        "prediction_summary": dict(forecast.prediction),
        "constraint_check": dict(forecast.verification),
        "evidence": dict(forecast.evidence),
        "risk_level": forecast.risk_level,
        "manual_intervention_label": forecast.manual_intervention_label,
        "dispatch_recommendation": forecast.dispatch_recommendation,
        "task_resolution": dict(forecast.task_resolution),
        "provenance": dict(forecast.provenance),
    }


def export_trace_tools(
    trace: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    calls: List[Dict[str, Any]] = []
    outputs: List[Dict[str, Any]] = []
    pipeformer_results: List[Dict[str, Any]] = []
    for index, item in enumerate(trace.get("tool_calls", []), start=1):
        call_id = tool_call_id(item, index)
        tool_name = str(item.get("tool_name") or "")
        calls.append(
            {
                "tool_call_id": call_id,
                "name": tool_name,
                "arguments": item.get("args", {}),
            }
        )
        raw_output = parse_tool_output(item)
        if (
            tool_name == "run_pipeformer_forecast"
            and isinstance(raw_output, dict)
            and raw_output.get("success")
        ):
            arguments = dict(item.get("args") or {})
            candidate_id = arguments.get("candidate_id") or raw_output.get("candidate_id")
            candidate_role = arguments.get("candidate_role") or raw_output.get("candidate_role")
            forecast = ForecastResult.from_payload(raw_output)
            output = forecast.model_dump()
            pipeformer_results.append(
                {
                    "tool_call_id": call_id,
                    "output": output,
                    "projection": _legacy_projection(forecast, arguments),
                }
            )
            if candidate_id:
                output["candidate_id"] = candidate_id
            if candidate_role:
                output["candidate_role"] = candidate_role
        else:
            output = deepcopy(raw_output)
        outputs.append(
            {
                "tool_call_id": call_id,
                "name": tool_name,
                "output": output,
            }
        )
    return calls, outputs, pipeformer_results


def final_answer(trace: Dict[str, Any]) -> str:
    for message in reversed(trace.get("messages", [])):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


class TeacherTraceProjector:
    """Convert raw agent/tool traces into compact, stable training fields."""

    def __init__(
        self,
        *,
        max_tool_text_chars: int = SFT_MAX_TOOL_TEXT_CHARS,
        omitted_tool_keys: Optional[set[str]] = None,
    ) -> None:
        self.max_tool_text_chars = max_tool_text_chars
        self.omitted_tool_keys = frozenset(omitted_tool_keys or SFT_OMITTED_TOOL_KEYS)

    def compact_sft_output(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self.compact_sft_output(item)
                for key, item in value.items()
                if key not in self.omitted_tool_keys
            }
        if isinstance(value, list):
            return [self.compact_sft_output(item) for item in value]
        if isinstance(value, str) and len(value) > self.max_tool_text_chars:
            return value[: self.max_tool_text_chars] + "... [truncated for SFT]"
        return value

    def select_sft_trajectory(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_outputs: List[Dict[str, Any]],
        answer: str,
        *,
        max_pipeformer_variables: Optional[int] = None,
        minimal_generic_computation_path: bool = False,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Keep the smallest evidence-bearing tool trajectory for SFT."""
        outputs_by_id = {
            str(item.get("tool_call_id") or ""): item for item in tool_outputs
        }
        pairs = [
            (index, call, outputs_by_id.get(str(call.get("tool_call_id") or "")))
            for index, call in enumerate(tool_calls)
        ]
        successful = [
            pair
            for pair in pairs
            if pair[2] is not None and not tool_output_failed(pair[2])
        ]
        successful_pipeformer = [
            pair for pair in successful if pair[1].get("name") == "run_pipeformer_forecast"
        ]
        successful_decision_policies = [
            pair for pair in successful if pair[1].get("name") == "set_decision_policy"
        ]
        if successful_pipeformer:
            referenced_variables = self._reference_tokens(answer)
            action_forecasts = [
                pair
                for pair in successful_pipeformer
                if self._forecast_action_variables(pair[1]) & referenced_variables
            ]
            # A current comparison needs every cited action forecast, but an
            # actionless baseline duplicate is already represented by the
            # candidate forecasts' applied-disturbance evidence.
            if action_forecasts:
                successful_pipeformer = action_forecasts
            multiple_pipeformer = len(successful_pipeformer) > 1
            required_registry_call_ids = self._registry_calls_required_for_forecasts(
                successful,
                successful_pipeformer,
            )
            selected_registry_searches = [
                pair
                for pair in successful
                if (
                    pair[1].get("name") == "search_pipeformer_registry"
                    and str(pair[1].get("tool_call_id") or "")
                    in required_registry_call_ids
                )
            ]
            unique_registry_searches = []
            seen_registry_searches = set()
            for pair in selected_registry_searches:
                raw_output = (pair[2] or {}).get("output")
                fingerprint = json.dumps(
                    {
                        "arguments": pair[1].get("arguments") or {},
                        "output": raw_output,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                if fingerprint in seen_registry_searches:
                    continue
                seen_registry_searches.add(fingerprint)
                unique_registry_searches.append(pair)
            selected = sorted(
                [
                    *unique_registry_searches,
                    *(successful_decision_policies[-1:] if multiple_pipeformer else []),
                    *successful_pipeformer,
                ],
                key=lambda pair: pair[0],
            )
        elif successful:
            multiple_pipeformer = False
            successful = self._deduplicate_identical_writes(successful)
            selected = (
                self._select_minimal_generic_computation_path(successful, answer)
                if minimal_generic_computation_path
                else self._select_generic_evidence_pairs(successful, answer)
            )
            selected = self._include_script_producers(successful, selected)
        else:
            multiple_pipeformer = False
            selected = pairs[-1:]

        compact_calls = []
        compact_outputs = []
        for _, call, output in selected:
            arguments = dict(call.get("arguments") or {})
            compact_arguments = compact_tool_call_arguments(arguments)
            if (
                call.get("name") == "write_file"
                and normalized_tool_path(arguments.get("path")).endswith(".py")
                and isinstance(arguments.get("content"), str)
            ):
                compact_arguments["content"] = arguments["content"]
            compact_calls.append(
                {
                    "tool_call_id": call.get("tool_call_id"),
                    "name": call.get("name"),
                    "arguments": compact_arguments,
                }
            )
            if output is None:
                continue
            raw_output = output.get("output")
            if call.get("name") == "run_pipeformer_forecast" and isinstance(raw_output, dict):
                raw_output = self._select_forecast_evidence_for_sft(
                    raw_output,
                    answer,
                    include_auxiliary_variables=not multiple_pipeformer,
                    max_variables=(
                        SFT_MAX_PIPEFORMER_VARIABLES
                        if max_pipeformer_variables is None
                        else max_pipeformer_variables
                    ),
                )
            elif (
                call.get("name") == "search_pipeformer_registry"
                and isinstance(raw_output, dict)
            ):
                # Application-boundary contracts are already bounded; recurse
                # only to apply the shared text sanitization rules.
                raw_output = self.compact_sft_output(raw_output)
            else:
                raw_output = self._compact_generic_sft_output(
                    raw_output,
                    answer,
                    tool_name=str(call.get("name") or ""),
                    arguments=dict(call.get("arguments") or {}),
                )
            compact_outputs.append(
                {
                    "tool_call_id": output.get("tool_call_id"),
                    "name": output.get("name"),
                    "output": raw_output,
                }
            )
        return compact_calls, compact_outputs

    @staticmethod
    def _registry_calls_required_for_forecasts(
        successful: List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
        forecasts: List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
    ) -> set[str]:
        """Select only searches that actually authorize a retained forecast."""
        required: set[str] = set()
        for forecast_index, forecast_call, _ in forecasts:
            preceding = []
            for index, call, output_record in successful:
                if index >= forecast_index:
                    continue
                preceding.append(
                    {
                        "tool_call_id": call.get("tool_call_id"),
                        "name": call.get("name"),
                        "arguments": dict(call.get("arguments") or {}),
                        "output": (output_record or {}).get("output"),
                    }
                )
            authorization = authorize_forecast_registry(
                dict(forecast_call.get("arguments") or {}),
                preceding,
            )
            disturbance_call_ids = [
                str(value)
                for value in authorization.get("disturbance_search_call_ids") or []
                if str(value)
            ]
            if disturbance_call_ids:
                required.add(disturbance_call_ids[-1])
            for call_ids in (authorization.get("candidate_search_call_ids") or {}).values():
                matching = [str(value) for value in call_ids if str(value)]
                if matching:
                    required.add(matching[-1])
        return required

    @staticmethod
    def _forecast_action_variables(call: Dict[str, Any]) -> set[str]:
        boundary = dict(dict(call.get("arguments") or {}).get(
            "boundary_conditions"
        ) or {})
        return {
            str(variable)
            for key in ("percentage_changes", "setpoints")
            for variable in dict(boundary.get(key) or {})
            if str(variable)
        }

    def _select_generic_evidence_pairs(
        self,
        pairs: List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
        answer: str,
    ) -> List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]]:
        references = self._reference_tokens(answer)
        remaining = set(references)
        available = list(pairs)
        selected = []
        while available and len(selected) < SFT_MAX_GENERIC_TOOL_PAIRS:
            ranked = []
            for pair in available:
                covered_count, covered = self._evidence_coverage(pair[2], remaining)
                ranked.append((covered_count, self._evidence_score(pair[2], answer), -pair[0], pair, covered))
            _, _, _, best, covered = max(ranked, key=lambda item: item[:3])
            selected.append(best)
            remaining.difference_update(covered)
            available.remove(best)
            if not remaining and selected:
                break
        return sorted(selected, key=lambda pair: pair[0])

    def _select_minimal_generic_computation_path(
        self,
        pairs: List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
        answer: str,
    ) -> List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]]:
        runs = [pair for pair in pairs if pair[1].get("name") == "run_command"]
        saved_runs = [
            pair
            for pair in runs
            if command_python_scripts(dict(pair[1].get("arguments") or {}))
        ]
        if not runs:
            return self._select_generic_evidence_pairs(pairs, answer)
        computation = max(
            saved_runs or runs,
            key=lambda pair: (self._evidence_score(pair[2], answer), pair[0]),
        )
        source_reads = [
            pair
            for pair in pairs
            if pair[0] < computation[0]
            and pair[1].get("name") == "read_file"
            and normalized_tool_path(
                dict(pair[1].get("arguments") or {}).get("path")
            ).startswith("pipeline_data/")
        ]
        return [*source_reads, computation]

    @staticmethod
    def _deduplicate_identical_writes(
        pairs: List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
    ) -> List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]]:
        write_groups: Dict[
            str,
            List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
        ] = {}
        retained_indices = set()
        for pair in pairs:
            call = pair[1]
            if call.get("name") != "write_file":
                retained_indices.add(pair[0])
                continue
            fingerprint = json.dumps(
                call.get("arguments") or {},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            write_groups.setdefault(fingerprint, []).append(pair)
        for group in write_groups.values():
            path = normalized_tool_path(
                dict(group[0][1].get("arguments") or {}).get("path")
            )
            first_run = min(
                (
                    pair[0]
                    for pair in pairs
                    if pair[1].get("name") == "run_command"
                    and path
                    in command_python_scripts(
                        dict(pair[1].get("arguments") or {})
                    )
                ),
                default=None,
            )
            preceding = (
                [pair for pair in group if pair[0] < first_run]
                if first_run is not None
                else []
            )
            retained_indices.add((preceding[-1] if preceding else group[-1])[0])
        return [pair for pair in pairs if pair[0] in retained_indices]

    @staticmethod
    def _include_script_producers(
        pairs: List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
        selected: List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]],
    ) -> List[tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]]:
        selected_by_index = {pair[0]: pair for pair in selected}
        for run_index, call, _ in selected:
            if call.get("name") != "run_command":
                continue
            for script_path in command_python_scripts(
                dict(call.get("arguments") or {})
            ):
                producers = [
                    pair
                    for pair in pairs
                    if pair[0] < run_index
                    and pair[1].get("name") == "write_file"
                    and normalized_tool_path(
                        dict(pair[1].get("arguments") or {}).get("path")
                    )
                    == script_path
                    and isinstance(
                        dict(pair[1].get("arguments") or {}).get("content"),
                        str,
                    )
                    and bool(dict(pair[1].get("arguments") or {}).get("content"))
                ]
                if producers:
                    producer = producers[-1]
                    selected_by_index[producer[0]] = producer
                elif script_path.startswith("temporary_dir/") or "/temporary_dir/" in script_path:
                    raise ValueError(
                        f"Generated script execution requires a preceding write_file: {script_path}"
                    )
        return [selected_by_index[index] for index in sorted(selected_by_index)]

    @staticmethod
    def _reference_tokens(answer: str) -> set[str]:
        values = set(variable_references(answer))
        values.update(sft_file_references(answer))
        values.update(str(value) for value in numeric_claim_values(answer))
        return {value for value in values if value}

    def _compact_generic_sft_output(
        self,
        value: Any,
        answer: str,
        *,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:
        """Retain a small excerpt containing every answer-grounding token possible."""
        if not isinstance(value, (dict, list, str)):
            return value
        if isinstance(value, dict):
            status = {
                key: value[key]
                for key in ("success", "exit_code", "error", "stderr")
                if key in value and value[key] not in (None, "")
            }
        else:
            status = {}
        if isinstance(value, dict):
            assessment = classify_tool_evidence(
                {"name": tool_name, "arguments": arguments, "output": value}
            )
            evidence_kind = {
                "file_content_read": "file_content",
                "command_content_or_computation": "command_content",
            }.get(assessment.reason)
            if evidence_kind:
                status["evidence_kind"] = evidence_kind
                source_payload = {
                    "path": value.get("path"),
                    "abs_path": value.get("abs_path"),
                    "cmd": value.get("cmd"),
                    "arguments": arguments,
                }
                source_artifacts = list(
                    requested_artifacts(json.dumps(source_payload, ensure_ascii=False))
                )
                if source_artifacts:
                    status["source_artifacts"] = source_artifacts
        compact_value = self.compact_sft_output(value)
        text_value = (
            compact_value
            if isinstance(compact_value, str)
            else json.dumps(compact_value, ensure_ascii=False, indent=2)
        )
        lines = text_value.splitlines() or [text_value]
        references = self._reference_tokens(answer)
        selected_indices = {0, len(lines) - 1}
        normalized_lines = [line.casefold().replace(",", "") for line in lines]
        for reference in references:
            token = reference.casefold().replace(",", "")
            for index, line in enumerate(normalized_lines):
                if token in line:
                    selected_indices.update({max(0, index - 1), index, min(len(lines) - 1, index + 1)})
                    break
        excerpt_lines = [lines[index] for index in sorted(selected_indices)]
        excerpt = "\n".join(excerpt_lines)
        if len(selected_indices) <= 2 and len(excerpt) < min(800, len(text_value)):
            excerpt = text_value[: min(len(text_value), SFT_MAX_GENERIC_OUTPUT_CHARS)]
        if len(excerpt) > SFT_MAX_GENERIC_OUTPUT_CHARS:
            excerpt = excerpt[:SFT_MAX_GENERIC_OUTPUT_CHARS].rsplit("\n", 1)[0]
        return {**status, "evidence_excerpt": excerpt}

    def serialize_sft_record_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        compact = {
            key: value
            for key, value in evidence.items()
            if key not in {
                "top_watch_variables",
                "key_observation_variables",
                "verified_numeric_claims",
                "candidate_forecasts",
            }
        }
        compact.update(_compact_watch_variables(evidence))
        return self.compact_sft_output(compact)

    def serialize_sft_decision_summary(self, value: Any) -> Dict[str, Any]:
        """Keep decision labels and ordering while leaving metric detail to tools."""
        summary = dict(value or {})
        raw_policy = summary.get("ranking_policy")
        if isinstance(raw_policy, dict):
            compact_policy = {
                "source": raw_policy.get("source"),
                "hard_constraints": list(
                    raw_policy.get("hard_constraints") or []
                ),
                "objectives": [
                    _select_fields(objective, ("metric", "direction", "tolerance"))
                    for objective in raw_policy.get("objectives") or []
                    if isinstance(objective, dict)
                ],
            }
        elif isinstance(raw_policy, str) and raw_policy.strip():
            compact_policy = {
                "source": "legacy_named_policy",
                "policy_id": raw_policy.strip(),
            }
        else:
            compact_policy = {}
        compact_summary = _select_fields(
            summary,
            (
                "status",
                "selected_candidate_id",
                "ranked_candidate_ids",
                "ranked_candidate_groups",
                "eliminated_candidates",
                "missing_metrics",
            ),
        )
        if compact_policy:
            compact_summary["ranking_policy"] = compact_policy
        return compact_summary

    @staticmethod
    def _evidence_score(output: Optional[Dict[str, Any]], answer: str) -> int:
        blob = _evidence_blob(output)
        score = sum(
            3
            for value in TeacherTraceProjector._reference_tokens(answer)
            if value.casefold() in blob
        )
        if '"stdout"' in blob or '"content"' in blob:
            score += 1
        return score

    @staticmethod
    def _evidence_coverage(
        output: Optional[Dict[str, Any]],
        remaining: set[str],
    ) -> tuple[int, set[str]]:
        normalized = _evidence_blob(output).replace(",", "")
        covered = {
            value
            for value in remaining
            if value.casefold().replace(",", "") in normalized
        }
        return len(covered), covered

    def _select_forecast_evidence_for_sft(
        self,
        output: Dict[str, Any],
        answer: str,
        *,
        include_auxiliary_variables: bool,
        max_variables: int,
    ) -> Dict[str, Any]:
        compact = ForecastResult.from_payload(output).model_dump()
        prediction = compact["prediction"]
        verification = compact["verification"]
        evidence = compact["evidence"]
        referenced = list(dict.fromkeys(variable_references(answer)))
        if include_auxiliary_variables:
            for key in ("top_watch_variables", "key_observation_variables"):
                referenced.extend(
                    str(item.get("variable"))
                    for item in evidence.get(key) or []
                    if item.get("variable")
                )
            for finding in verification.get("priority_findings") or []:
                referenced.extend(str(value) for value in finding.get("affected_variables") or [])
        referenced = list(dict.fromkeys(referenced))[:max_variables]
        summary = dict(prediction.get("output_forecast_summary") or {})
        metric_keys = (
            "mean_prediction",
            "minimum_prediction",
            "maximum_prediction",
            "max_abs_prediction",
            "prediction_change",
            "max_abs_step_change",
            "max_step_decline",
            "max_decline_from_start",
            "recovery_from_minimum",
        )
        prediction["output_forecast_summary"] = {
            variable: _select_fields(dict(summary[variable] or {}), metric_keys)
            for variable in referenced
            if variable in summary
        }
        if not include_auxiliary_variables and "counterfactual_comparison" in prediction:
            prediction["counterfactual_comparison"] = _select_fields(
                dict(prediction["counterfactual_comparison"] or {}),
                (
                    "mode",
                    "compared_step_count",
                    "compared_output_variable_count",
                    "nonzero_impacted_variable_count",
                    "baseline_reference",
                    "disturbance_variable",
                    "applied_disturbance",
                ),
            )
        if "comparable_metrics" in verification:
            metrics = dict(verification.get("comparable_metrics") or {})
            verification["comparable_metrics"] = _select_fields(
                metrics, COMPACT_COMPARABLE_METRIC_KEYS
            )
            if "energy_evaluation_status" in metrics:
                verification["comparable_metrics"]["evaluation_status"] = metrics[
                    "energy_evaluation_status"
                ]
        verification["priority_findings"] = [
            _select_fields(
                finding,
                (
                    "name",
                    "category",
                    "status",
                    "evaluation_status",
                    "flag",
                    "priority",
                    "affected_variables",
                ),
            )
            for finding in (verification.get("priority_findings") or [])[:5]
        ]
        compact["evidence"] = _compact_watch_variables(
            evidence,
            include_auxiliary_variables=include_auxiliary_variables,
        )
        if evidence.get("boundary_application_evidence"):
            compact["evidence"]["boundary_application_evidence"] = [
                _select_fields(
                    item,
                    (
                        "variable",
                        "mode",
                        "requested_value",
                        "input_values_applied",
                        "verified",
                    ),
                )
                for item in evidence.get("boundary_application_evidence") or []
            ]
        compact["task_resolution"] = _select_fields(
            compact["task_resolution"],
            (
                "resolved_attention_variable_count",
                "resolved_output_variable_count",
                "unresolved_attention_targets",
                "unresolved_output_state_variables",
                "applied_boundary_conditions",
            ),
        )
        compact["provenance"] = _select_fields(
            compact["provenance"], ("checkpoint_id", "forecast_mode", "device")
        )
        compact.pop("risk_level", None)
        compact.pop("manual_intervention_label", None)
        compact.pop("dispatch_recommendation", None)
        return compact

DEFAULT_PROJECTOR = TeacherTraceProjector()

__all__ = [
    "DEFAULT_PROJECTOR",
    "TeacherTraceProjector",
    "export_trace_tools",
    "final_answer",
]
