"""GRPO reward-accuracy simulation (AGENTS rule 3 release gate).

Feeds REAL flawed student session traces plus a positive control (the released
teacher replay on the best-matched reference) through the exact GRPO reward
stack: backend evaluator -> episode_stats -> composite_reward.

Gate: on one and the same teacher reference, the student episode must never
outscore the reference-quality replay. A tie is legitimate (a well-behaved
episode matches the control); anything above is a mis-shaped reward and GRPO
must not launch.

Run: python pipeclaw/task2_student/scripts/simulate_reward.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipeclaw.backend.evaluator import EvaluationProfile, evaluate
from pipeclaw.task2_student.scripts.pass_at_k import composite_reward, episode_stats
from pipeclaw.task2_student.rollout.suite import read_jsonl

DOWNLOADS = Path.home() / "Downloads"
TRACES = {
    # good behaviors
    "good-read-json (ld9q75)": DOWNLOADS / "20260815_022445_ld9q75.json",
    "good-list-segments (w831cq)": DOWNLOADS / "20260816_141156_w831cq.json",
    # flawed behaviors
    "thrash+substitution (rkt0g6)": DOWNLOADS / "20260816_143743_rkt0g6.json",
    "schema-slip+empty-answer (5co5zc)": DOWNLOADS / "20260816_155553_5co5zc.json",
    "fabricated-actions (4ub2ta)": DOWNLOADS / "20260816_141718_4ub2ta.json",
    "thrash+timeout (we3unq)": DOWNLOADS / "20260816_142854_we3unq.json",
    "not-found-loop (cf58gx)": DOWNLOADS / "20260816_141243_cf58gx.json",
}
TEACHER_SPLITS = [
    "pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_train.jsonl",
    "pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_valid.jsonl",
    "pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl",
]
MIN_SIMILARITY = 0.20
EMPTY_MARK = "Model did not return any displayable content."


def _gram_set(text: str, width: int = 4) -> set[str]:
    return {text[index : index + width] for index in range(max(len(text) - width + 1, 1))}


def _similarity(a: str, b: str) -> float:
    ga, gb = _gram_set(a), _gram_set(b)
    return len(ga & gb) / len(ga | gb) if ga | gb else 0.0


def _load_teachers() -> list[dict]:
    return [r for source in TEACHER_SPLITS for r in read_jsonl(Path(source))]


def _student_user_text(session: dict) -> str:
    return " ".join(
        str(message.get("content") or "")
        for message in session.get("messages") or []
        if message.get("role") == "user"
    )


def _student_rollout(session: dict) -> dict:
    """Map one live backend session JSON onto the evaluator's rollout shape."""
    calls: list[dict] = []
    outputs: list[dict] = []
    for group in ("tool_calls", "audit_tool_calls"):
        for item in session.get(group) or []:
            tool_call_id = str(item.get("tool_call_id") or f"{group}-{len(calls)}")
            calls.append(
                {
                    "tool_call_id": tool_call_id,
                    "name": item.get("tool_name"),
                    "arguments": item.get("args") if item.get("args") is not None else {},
                }
            )
            outputs.append(
                {
                    "tool_call_id": tool_call_id,
                    "name": item.get("tool_name"),
                    "output": item.get("result") if item.get("result") is not None else {},
                }
            )
    final = ""
    for message in reversed(session.get("messages") or []):
        if message.get("role") == "assistant":
            content = str(message.get("content") or "")
            final = "" if content == EMPTY_MARK else content
            break
    return {
        "sample_id": str(session.get("session_id") or "student"),
        "tool_calls": calls,
        "tool_outputs": outputs,
        "final_answer": final,
        "trace_status": {
            "completed": "completed",
            "timeout": "max_turns_exceeded",
        }.get(session.get("status"), "generation_error"),
        "recent_turns": [],
        "json_errors": [],
        "turns": len(calls),
    }


def _teacher_replay(record: dict) -> dict:
    return {
        "tool_calls": record.get("tool_calls") or [],
        "tool_outputs": record.get("tool_outputs") or [],
        "final_answer": record.get("final_answer") or "",
        "trace_status": record.get("trace_status") or "success",
        "recent_turns": record.get("recent_turns") or [],
    }


def _reward(rollout: dict, reference: dict) -> tuple[float, dict, dict]:
    report = evaluate(
        rollout, profile=EvaluationProfile.AUTONOMOUS_ROLLOUT, reference=reference
    )
    report_fields = report.to_dict()
    stats = episode_stats(rollout)
    return (
        composite_reward(stats, report_fields),
        report_fields,
        stats,
    )


def _all_rows() -> list[dict]:
    teachers = _load_teachers()
    rows = []
    for label, path in TRACES.items():
        session = json.loads(path.read_text(encoding="utf-8"))
        question = _student_user_text(session)
        reference, similarity = max(
            (
                (record, _similarity(question, str(record.get("user_input") or "")))
                for record in teachers
            ),
            key=lambda pair: pair[1],
        )
        rollout = _student_rollout(session)
        student_reward, report, stats = _reward(rollout, reference)
        control_reward, control_report, _ = _reward(_teacher_replay(reference), reference)
        rows.append(
            {
                "label": label,
                "similarity": round(similarity, 3),
                "matched_reference": reference["sample_id"],
                "reward": student_reward,
                "control": control_reward,
                "passed": report.get("passed"),
                "score": report.get("overall_score"),
                "control_passed": control_report.get("passed"),
                "thrash": stats["thrash_count"],
                "dup_success": stats["duplicate_success_count"],
                "failed_calls": stats["failed_call_count"],
            }
        )
    return rows


def gate(rows: list[dict]) -> list[str]:
    return [
        f"{row['label']}: student reward {row['reward']} > control {row['control']}"
        for row in rows
        if row["similarity"] >= MIN_SIMILARITY and row["reward"] > row["control"]
    ]


def main() -> int:
    rows = _all_rows()
    print(f"{'trace':<34}{'sim':>6}{'reward':>9}{'control':>9}{'pass':>6}{'score':>8}")
    for row in sorted(rows, key=lambda item: item["reward"], reverse=True):
        print(
            f"{row['label']:<34}{row['similarity']:>6}{row['reward']:>9}"
            f"{row['control']:>9}{str(row['passed']):>6}{str(row['score']):>8}"
        )
    matched = [row for row in rows if row["similarity"] >= MIN_SIMILARITY]
    print(f"\nmatched={len(matched)}/{len(rows)} (similarity >= {MIN_SIMILARITY})")
    violations = gate(rows)
    for violation in violations:
        print(f"GATE VIOLATION: {violation}")
    print("gate:", "FAIL" if violations else "PASS")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
