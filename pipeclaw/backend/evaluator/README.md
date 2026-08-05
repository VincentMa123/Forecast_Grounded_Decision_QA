# PipeClaw evaluation

`pipeclaw/backend/evaluator/` is the **only** evaluation package in this
repository. Both the Task 1 teacher trace and the Task 2 autonomous student
rollout are scored here, by the same engine, against the same metric
definitions. The former `pipeclaw/task2_student/evaluator/` package has been
removed; `pipeclaw/tests/evaluation/test_layout.py` fails if it reappears.

## Execution versus evaluation

Two responsibilities that used to be interleaved are now separated, and the
boundary is enforced by imports:

| Concern | Owner |
| --- | --- |
| Running a model and its tools (prompt building, bounded turn loop, tool dispatch, workspaces, MS-SWIFT generation) | `pipeclaw/task2_student/rollout/` |
| Runtime grounding used *during* generation (decision policy, trace state, answer limits, CSV/tool/topology evidence) | `pipeclaw/backend/grounding/` |
| Scoring a finished record (metrics, weights, gate, aggregation) | `pipeclaw/backend/evaluator/` |

`rollout/models.py`, `prompting.py`, `runner.py`, and `tools.py` contain no
evaluator import at all — a rollout can be produced with no scoring code
loaded. `rollout/suite.py` is the single seam where this package enters the
rollout side, and it only calls `evaluate()` and `summarize()` after a rollout
record already exists.

## Layout

- `engine.py` — the one `evaluate()` entry point: the score formula and the
  critical gate.
- `models.py` — `EvaluationProfile`, `EvaluationContext`, `MetricResult`,
  `EvaluationReport`, `EVALUATION_SCHEMA_VERSION`, `EvaluationInputError`.
- `profiles.py` — per-profile metric weights, critical sets, and thresholds.
- `adapters.py` — `TeacherTraceAdapter` and `AutonomousRolloutAdapter`
  normalize their inputs into one `EvaluationContext`.
- `oracle.py` — `build_teacher_oracle()` extracts expected targets from a
  held-out teacher record. Target extraction only; no metric or score logic.
- `checks/` — the canonical metric implementations, written once and shared by
  both profiles.
- `aggregation.py` — `summarize()`, the denominator-aware dataset roll-up.
- `scorer.py` — teacher-side compatibility facade (see below).
- `teacher_quality.py` — validates unsupported answer claims while a trace is
  being generated, i.e. before there is a record to score.

## Schema v2 scoring and the critical gate

`EVALUATION_SCHEMA_VERSION` is `pipeclaw_evaluation_v2`. Every report carries
it, and there is exactly one score formula in the repository:

```python
included    = [m for m in metrics if m.applicable and m.included_in_score]
denominator = sum(m.weight for m in included)
overall_score = (
    round(100.0 * sum(m.weight for m in included if m.passed) / denominator, 6)
    if denominator else None
)
hard_gate_passed = not hard_issues and all(m.passed for m in included if m.critical)
```

Three consequences are worth stating explicitly:

- **Inapplicable metrics never enter the denominator.** A record that could not
  raise a given metric is not penalised for it; it is reported with
  `applicable: false` and `status: "not_applicable"` instead of being scored as
  a failure.
- **Diagnostics never enter the denominator either.** `tool_recovery`,
  `portability`, `raw_capture_metadata`, `model_loading_metadata`, and
  `hallucination` carry `included_in_score: false`. They describe a run; they
  do not grade it.
- **A high weighted score cannot buy a pass.** Any failing critical metric, or
  any hard grounding issue, sets `hard_gate_passed` to false on its own.

`passed` is then profile-specific:

- **Teacher** (`EvaluationProfile.TEACHER_TRACE`) — `hard_gate_passed` *and*
  `overall_score >= 85.0` (the profile threshold, `minimum_score`). Weights are
  hand-tuned per check.
- **Autonomous** (`EvaluationProfile.AUTONOMOUS_ROLLOUT`) — `hard_gate_passed`
  and a non-null score. Every applicable deliverable metric carries weight
  `1.0`, and there is no minimum-score threshold: a student rollout is judged
  on whether it got the critical work right, not on clearing a curve.

## Derived hallucination rate

Hallucination is **not** a separately computed check. `EvaluationReport.to_dict()`
copies the `evidence_consistency` result into a `hallucination` entry marked
`derived_from: "evidence_consistency"` and `included_in_score: false`, and only
for the autonomous profile. At dataset level, `summary["hallucination_rate"]`
is exactly `evidence_consistency.failure_rate`. The grounding checker runs
once per record; the second name is a compatibility view of the first result,
so the two can never disagree.

## Teacher compatibility surface

`scorer.py` keeps the released teacher-side API callable, but everything it
returns is copied from a schema-v2 `EvaluationReport` rather than recomputed:

- `apply_quality_aliases()` writes the `quality_*` fields listed below from one
  report. **The `quality_*` names are deprecated.** New code should read
  `overall_score`, `passed`, `profile`, and `metrics` from the report directly;
  the aliases exist so the released master records, the quality workbook, and
  the Task 1 deliverables keep their historical field names.
- `NativeTraceEvaluator` is a thin facade — `evaluate()`, `summarize()`, and
  `load()` delegate straight to `evaluate()`/`summarize()`. **Deprecated**; call
  `pipeclaw.backend.evaluator.evaluate()` instead.

Each generated master record keeps these deprecated alias fields:

- `quality_flag`: `pass` or `needs_review` — from `report.passed`
- `quality_score`: numeric score from 0 to 100 — from `report.overall_score`
- `quality_profile`: native evaluator profile — from `report.profile`
- `quality_failed_checks`: applicable metrics that did not pass
- `quality_issues`: hard answer-grounding issues — from
  `report.diagnostics["hard_issues"]`

## Teacher-trace commands

Evaluate every record in the current master trace with:

```powershell
python pipeclaw/backend/evaluate_teacher_trace.py
```

The full run also creates `generated_teacher_traces/task1_deliverables/` with:

- schema-compliant audit copies of `teacher_trace_train.jsonl`, `teacher_trace_valid.jsonl`, and `teacher_trace_test.jsonl`;
- `teacher_trace_schema.json`;
- `teacher_trace_quality_report.xlsx`;
- `teacher_trace_statistics.xlsx` with the Task 1.9 data-statistics tables;
- per-record `quality_evaluation.jsonl` and its JSON summary.

It also rebuilds the compact, quality-approved SFT projections in
`generated_teacher_traces/splits/`. A filtered `--scenario-id` or `--sample-id`
run intentionally leaves all split files unchanged.

The quality workbook applies the Section 1.8 schema, numerical-grounding,
rule-consistency, and safety-first dispatch checks to every record. It includes
per-check totals, category and rule outcomes, source/split coverage, detailed
needs-review rows, and a deterministic, stratified 25% manual-review queue with
blank human-signoff fields.

The separate statistics workbook provides source, task-type, constraint-type,
constraint-outcome, rule-outcome, risk, intervention, cross-tab, split, evidence,
and quality tables plus compact charts. Both workbooks are fixed audit snapshots
without formulas or external links.

The audit split copies retain every Task 1.7 field. The regenerated
`generated_teacher_traces/splits/` files remain the smaller SFT projection, so
quality auditing does not make the 7B/14B training input unnecessarily verbose.

Evaluate a different file or one scenario with:

```powershell
python pipeclaw/backend/evaluate_teacher_trace.py `
  --teacher-trace path/to/teacher_trace.jsonl `
  --scenario-id scenario_pipeformer_prediction_003
```

The default teacher pass threshold is 85. Critical execution or grounding
checks still block a pass regardless of the weighted score; see "Schema v2
scoring and the critical gate" above.

## Autonomous student evaluation

The same engine scores Task 2 rollouts under
`EvaluationProfile.AUTONOMOUS_ROLLOUT`. The rollout is produced first, by
`pipeclaw/task2_student/rollout/`, and only then scored:

```python
from pipeclaw.backend.evaluator import EvaluationProfile, evaluate, summarize

report = evaluate(rollout, profile=EvaluationProfile.AUTONOMOUS_ROLLOUT, reference=source)
summary = summarize(reports)
```

`reference` is the held-out teacher record; `AutonomousRolloutAdapter` extracts
its expectations through `build_teacher_oracle()`. Calling the autonomous
profile without a reference raises `EvaluationInputError`, which the dataset
suite catches per record — any other evaluator exception aborts the run rather
than silently scoring a broken record. The user-facing command is documented in
`pipeclaw/task2_student/scripts/README.md`.

For causal or disturbance-impact questions, `run_pipeformer_forecast` accepts `include_baseline_comparison=true`. It runs one unchanged baseline in addition to the disturbed forecast and returns only a compact `counterfactual_comparison`.

## Resumable OpenClaw regeneration

Regenerate the selected OpenClaw sources without replacing PipeFormer or other
records by running the checkpointed driver from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  pipeclaw/backend/regenerate_openclaw_teacher_traces.ps1
```

The driver processes the 30 v4 scenarios first and the 40 v2 scenarios second.
It invokes `--replace-selected-scenario` once per scenario, records each success
in `generated_teacher_traces/openclaw_regeneration.completed.txt`, skips those
entries when resumed, and prints all failures after continuing through the run.
Use `-DryRun` to verify the 70-scenario selection without making LLM calls.

After all 70 replacements succeed, the driver automatically rebuilds the
repaired master, quality report, statistics workbook, and compact
train/validation/test projections. The equivalent standalone command is:

```powershell
python pipeclaw/backend/evaluate_teacher_trace.py `
  --teacher-trace pipeclaw/backend/generated_teacher_traces/teacher_trace.json `
  --repair-grounded-records `
  --repair-output pipeclaw/backend/generated_teacher_traces/teacher_trace_repaired.json
```
