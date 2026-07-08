from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .condition_parser import parse_condition
from .evidence_extractor import summarize_variables, top_variables
from .pipeformer_inference import load_pipeformer_forecast_context
from .rule_verifier import run_constraint_checks
from .scenario_loader import first_user_input
from .teacher_answer import build_teacher_answer
from .trace_formatter import build_teacher_trace_record


def build_trace_record(
    scenario: Dict[str, Any],
    scenario_path: Path,
    forecast_csv: Path,
    mapping_path: Path,
    *,
    checkpoint_dir: Optional[Path] = None,
    pipeformer_root: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    static_dir: Optional[Path] = None,
    device: str = "cpu",
    use_sample_csv: bool = False,
) -> Dict[str, Any]:
    question = first_user_input(scenario)
    parsed_task = parse_condition(question)
    forecast_context = load_pipeformer_forecast_context(
        parsed_task=parsed_task,
        forecast_csv=forecast_csv,
        mapping_path=mapping_path,
        checkpoint_dir=checkpoint_dir,
        pipeformer_root=pipeformer_root,
        data_dir=data_dir,
        static_dir=static_dir,
        device=device,
        use_sample_csv=use_sample_csv,
    )
    variable_summaries = summarize_variables(forecast_context["real_rows"], forecast_context["predict_rows"])
    verification = run_constraint_checks(variable_summaries)
    evidence_variables = top_variables(variable_summaries, limit=3)
    answer = build_teacher_answer(parsed_task, verification, evidence_variables)

    return build_teacher_trace_record(
        scenario=scenario,
        scenario_path=scenario_path,
        question=question,
        parsed_task=parsed_task,
        mapping_path=mapping_path,
        forecast_context=forecast_context,
        verification=verification,
        evidence_variables=evidence_variables,
        answer=answer,
    )