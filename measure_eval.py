"""Measure the autonomous evaluator on the frozen test split."""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, ".")

from pipeclaw.backend.evaluator import evaluate, EvaluationProfile

PATH = os.getenv(
    "EVAL_SPLIT",
    "pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl",
)
recs = [
    json.loads(line)
    for line in open(PATH, encoding="utf-8-sig")
    if line.strip()
]
oc = [r for r in recs if "openclaw" in str(r.get("scenario_type", "")).casefold()]
pf = [r for r in recs if "openclaw" not in str(r.get("scenario_type", "")).casefold()]
print(f"records: total={len(recs)} openclaw={len(oc)} pipeformer={len(pf)}")


def replay_payload(rec):
    return {
        "tool_calls": rec.get("tool_calls") or [],
        "tool_outputs": rec.get("tool_outputs") or [],
        "final_answer": rec.get("final_answer") or "",
        "trace_status": rec.get("trace_status") or "success",
        "recent_turns": rec.get("recent_turns") or [],
    }


vac = 0
for r in oc:
    payload = {
        "tool_calls": [],
        "tool_outputs": [],
        "final_answer": "x",
        "trace_status": "success",
    }
    if evaluate(
        payload, profile=EvaluationProfile.AUTONOMOUS_ROLLOUT, reference=r
    ).passed:
        vac += 1
print(f"\n[probe] one-char answer passes: {vac}/{len(oc)}  (must stay 0)")

for label, group in (("openclaw", oc), ("pipeformer", pf)):
    ok = 0
    stats: dict[str, Counter] = {}
    first_fail = Counter()
    for r in group:
        rep = evaluate(
            replay_payload(r),
            profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
            reference=r,
        )
        ok += rep.passed
        for name, m in rep.metrics.items():
            c = stats.setdefault(name, Counter())
            if not m.applicable:
                c["inapplicable"] += 1
            elif m.passed:
                c["pass"] += 1
            else:
                c["fail"] += 1
        first_fail[next((
            metric.name
            for metric in rep.metrics.values()
            if metric.applicable and metric.included_in_score and not metric.passed
        ), None)] += 1

    print(f"[replay] {label}: {ok}/{len(group)}")
    print(f"\n[{label} metric distribution, n={len(group)}]")
    print(f"{'metric':<32}{'pass':>6}{'fail':>6}{'n/a':>6}")
    for name in sorted(stats):
        c = stats[name]
        print(f"{name:<32}{c['pass']:>6}{c['fail']:>6}{c['inapplicable']:>6}")
    print(f"[{label} first_failing_metric]")
    for name, count in first_fail.most_common():
        print(f"  {name}: {count}")
