from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.io_utils import write_json, write_jsonl
from pipeline.scenario_loader import find_scenario, load_scenarios


logger = logging.getLogger("teacher_trace")


def backend_root() -> Path:
    return Path(__file__).resolve().parent


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def short_text(value: Any, limit: int = 700) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def load_backend_env() -> None:
    env_path = backend_root() / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            os.environ.setdefault(key, value.strip().strip("'\""))
    else:
        load_dotenv(env_path, override=False)


def build_parser() -> argparse.ArgumentParser:
    root = backend_root()
    parser = argparse.ArgumentParser(description="Run PipeClaw and export teacher_trace records.")
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=root / "pipeclaw_data" / "mock_pipeformer_tiny_scenarios.json",
    )
    parser.add_argument("--scenario-id", default=None, help="Generate one scenario. Omit to generate all scenarios.")
    parser.add_argument("--agent-id", default="teacher_trace")
    parser.add_argument("--session-id", default=None, help="Only valid with --scenario-id.")
    parser.add_argument("--device", default=os.getenv("PIPEFORMER_DEVICE", "cpu"))
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=root / "generated_teacher_traces" / "teacher_trace.jsonl",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=root / "generated_teacher_traces" / "teacher_trace.pretty.json",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level",
        default=os.getenv("TEACHER_TRACE_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Terminal logging verbosity for teacher trace generation.",
    )
    return parser


def first_user_input(scenario: Dict[str, Any]) -> str:
    for session in scenario.get("sessions", []):
        for turn in session.get("dialogue", []):
            text = str(turn.get("user_input") or "").strip()
            if text:
                return text
    raise ValueError(f"Scenario {scenario.get('scenario_id')} has no non-empty user_input.")


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


def tool_call_id(tool_call: Dict[str, Any], index: int) -> str:
    return str(tool_call.get("tool_call_id") or f"tool_{index:03d}")


def trace_tool_calls(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "call_id": tool_call_id(item, index),
            "tool_name": item.get("tool_name"),
            "arguments": item.get("args", {}),
            "timestamp": item.get("timestamp"),
        }
        for index, item in enumerate(trace.get("tool_calls", []), start=1)
    ]


def trace_tool_outputs(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "call_id": tool_call_id(item, index),
            "output": parse_tool_output(item),
            "timestamp": item.get("timestamp"),
        }
        for index, item in enumerate(trace.get("tool_calls", []), start=1)
    ]


def final_answer(trace: Dict[str, Any]) -> str:
    for message in reversed(trace.get("messages", [])):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return ""


def successful_pipeformer_output(trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for item in trace.get("tool_calls", []):
        if item.get("tool_name") != "run_pipeformer_forecast":
            continue
        output = parse_tool_output(item)
        if isinstance(output, dict) and output.get("success"):
            return output
    return None


def generic_evidence(trace: Dict[str, Any]) -> Dict[str, Any]:
    calls = trace.get("tool_calls", [])
    return {
        "tool_names": [item.get("tool_name") for item in calls],
        "tool_call_count": len(calls),
        "artifacts": trace.get("artifacts", []),
    }


def build_teacher_record(scenario: Dict[str, Any], question: str, trace: Dict[str, Any]) -> Dict[str, Any]:
    pipeformer = successful_pipeformer_output(trace)
    answer = final_answer(trace)
    quality_flag = "pass" if trace.get("status") == "completed" else "needs_review"
    if pipeformer:
        parsed_task = pipeformer.get("parsed_task")
        prediction_summary = pipeformer.get("prediction_summary")
        constraint_check = pipeformer.get("constraint_check")
        evidence = pipeformer.get("evidence")
        risk_level = pipeformer.get("risk_level")
        manual_intervention_label = pipeformer.get("manual_intervention_label")
        dispatch_recommendation = pipeformer.get("dispatch_recommendation")
        answer = answer or pipeformer.get("final_answer", "")
        quality_flag = pipeformer.get("quality_flag", quality_flag)
    else:
        parsed_task = {"task_type": "data_query", "question": question}
        prediction_summary = None
        constraint_check = None
        evidence = generic_evidence(trace)
        risk_level = "low"
        manual_intervention_label = "not_required"
        dispatch_recommendation = "N/A - not a dispatch or PipeFormer prediction task."

    return {
        "sample_id": f"sample_{scenario.get('scenario_id')}",
        "scenario_id": scenario.get("scenario_id"),
        "scenario_type": scenario.get("scenario_type"),
        "user_input": question,
        "parsed_task": parsed_task,
        "tool_calls": trace_tool_calls(trace),
        "tool_outputs": trace_tool_outputs(trace),
        "prediction_summary": prediction_summary,
        "constraint_check": constraint_check,
        "evidence": evidence,
        "risk_level": risk_level,
        "manual_intervention_label": manual_intervention_label,
        "dispatch_recommendation": dispatch_recommendation,
        "final_answer": answer,
        "quality_flag": quality_flag,
    }


def run_backend_trace(scenario: Dict[str, Any], args: argparse.Namespace, index: int, run_stamp: str) -> tuple[str, Dict[str, Any]]:
    from agent.orchestrator import AgentOrchestrator
    from agent.schemas import AgentChatRequest

    question = first_user_input(scenario)
    scenario_id = str(scenario.get("scenario_id") or f"scenario_{index:06d}")
    session_id = args.session_id or f"teacher_{index:06d}_{scenario_id}_{run_stamp}"
    os.environ["PIPEFORMER_DEVICE"] = str(args.device)

    logger.info("Scenario %s started (%d): session_id=%s", scenario_id, index, session_id)
    logger.info("User question: %s", short_text(question, limit=500))

    orchestrator = AgentOrchestrator(
        data_loader=None,
        agent_id=args.agent_id,
        session_id=session_id,
        enable_skills=False,
        workspace_root_base=backend_root() / ".openclaw",
    )
    result = orchestrator.run_agent(AgentChatRequest(agent_id=args.agent_id, session_id=session_id, message=question))
    trace_path = Path(result.trace_summary.trace_path)
    logger.info("Scenario %s finished: status=%s trace=%s", scenario_id, result.trace_summary.status, trace_path)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    logger.info("Loaded trace: messages=%d tool_calls=%d", len(trace.get("messages", [])), len(trace.get("tool_calls", [])))
    trace["_trace_path"] = trace_path.as_posix()
    return question, trace


def selected_scenarios(args: argparse.Namespace) -> List[Dict[str, Any]]:
    scenarios = load_scenarios(args.scenario_file)
    if args.scenario_id:
        return [find_scenario(scenarios, args.scenario_id)]
    return scenarios


def main() -> int:
    load_backend_env()
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    scenarios = selected_scenarios(args)
    if args.session_id and len(scenarios) != 1:
        raise ValueError("--session-id can only be used with --scenario-id.")

    logger.info(
        "Teacher trace generation started: scenarios=%d device=%s scenario_file=%s",
        len(scenarios),
        args.device,
        args.scenario_file,
    )
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    records = []
    for index, scenario in enumerate(scenarios, start=1):
        question, trace = run_backend_trace(scenario, args, index, run_stamp)
        logger.info("Building teacher record for scenario=%s", scenario.get("scenario_id"))
        records.append(build_teacher_record(scenario, question, trace))

    logger.info("Writing JSONL output: %s", args.output_jsonl)
    write_jsonl(args.output_jsonl, records, force=args.force)
    logger.info("Writing pretty JSON output: %s", args.output_json)
    write_json(args.output_json, records[0] if len(records) == 1 else records, force=args.force)
    logger.info("Teacher trace generation complete: records=%d", len(records))
    print(
        json.dumps(
            {
                "status": "ok",
                "records": len(records),
                "output_jsonl": args.output_jsonl.as_posix(),
                "output_json": args.output_json.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
