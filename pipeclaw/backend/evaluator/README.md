# PipeClaw evaluator

This is the repository's single evaluation package. It scores finalized
teacher traces and student-distillation autonomous rollouts with the same
metric definitions, profiles, and report schema.

## Execution and evaluation are separate

| Responsibility | Package |
| --- | --- |
| Prompt construction, model calls, tool dispatch, workspaces, and bounded turns | `pipeclaw/student_distillation/rollout/` |
| Runtime grounding used while a trace is generated | `pipeclaw/backend/grounding/` |
| Metrics, weights, gates, and aggregation for a finished record | `pipeclaw/backend/evaluator/` |

`rollout/suite.py` is the only student-distillation module that imports this package. A
rollout is generated first and scored afterward.

## Python API

```python
from pipeclaw.backend.evaluator import EvaluationProfile, evaluate, summarize

report = evaluate(
    rollout_record,
    profile=EvaluationProfile.AUTONOMOUS_ROLLOUT,
    reference=teacher_record,
)
summary = summarize([report])
```

The autonomous profile requires the held-out teacher record as `reference`.
Teacher scoring uses `EvaluationProfile.TEACHER_TRACE`.

## Schema-v3 result semantics

Reports use `EVALUATION_SCHEMA_VERSION == "pipeclaw_evaluation_v3"`.

- `overall_score` is the weighted percentage of applicable metrics included in
  the score denominator.
- Inapplicable metrics are reported as `not_applicable`, not failures.
- Diagnostics such as `tool_recovery`, `portability`, `hallucination`, and
  model-loading metadata are outside the score denominator.
- `passed` also requires the critical gate: a failing critical metric or hard
  grounding issue fails the record regardless of its weighted score.
- Teacher reports additionally require the configured minimum score (currently
  85.0).

The main modules are:

- `engine.py` — `evaluate()` and the critical gate.
- `models.py` — report, metric, profile, and input types.
- `profiles.py` — metric weights and critical sets.
- `adapters.py` and `oracle.py` — normalize records and extract teacher targets.
- `checks/` — shared metric implementations.
- `aggregation.py` — denominator-aware dataset summaries.
- `scorer.py` — deprecated teacher compatibility facade.

The `quality_*` fields and `NativeTraceEvaluator` in `scorer.py` remain only for
released teacher-trace artifacts. New code should use `evaluate()` and the v3
report fields directly.

## Evaluate teacher traces

Run from the repository root:

```powershell
python -m pipeclaw.backend.teacher_traces.evaluate_teacher_trace
```

To inspect one record or another input file:

```powershell
python -m pipeclaw.backend.teacher_traces.evaluate_teacher_trace `
  --teacher-trace path/to/teacher_trace.jsonl `
  --scenario-id scenario_pipeformer_prediction_003
```

The full run writes audit copies, quality/statistics workbooks, per-record
reports, and refreshed compact SFT projections under
`pipeclaw/backend/generated_teacher_traces/`. A filtered run leaves the split
files unchanged.

## Evaluate autonomous student rollouts

The user-facing command is documented in
[student-distillation scripts](../../student_distillation/scripts/README.md). It writes
`rollouts.jsonl` and `summary.json`, including metric denominators and
scenario-type breakdowns. Use the `pipeformer` filter for PipeFormer cases and
`openclaw` for PipeClaw agent cases.

## Repair and regenerate

List repairable teacher records without calling a model:

```powershell
python -m pipeclaw.backend.scripts.repair_teacher_trace `
  --list-regeneration-targets
```

After an approved repair, regenerate the evaluation deliverables:

```powershell
python -m pipeclaw.backend.teacher_traces.evaluate_teacher_trace `
  --teacher-trace pipeclaw/backend/generated_teacher_traces/teacher_trace.json `
  --repair-grounded-records `
  --repair-output pipeclaw/backend/generated_teacher_traces/teacher_trace_repaired.json
```
