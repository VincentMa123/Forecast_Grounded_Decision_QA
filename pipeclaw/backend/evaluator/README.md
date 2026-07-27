# Native PipeClaw teacher-trace evaluation

`teacher_quality.py` validates unsupported answer claims while a trace is being generated. `scorer.py` scores the compact record that is actually eligible for SFT.

Each generated master record keeps separate fields:

- `quality_flag`: `pass` or `needs_review`
- `quality_score`: numeric score from 0 to 100
- `quality_profile`: native evaluator profile
- `quality_failed_checks`: native checks that did not pass
- `quality_issues`: hard answer-grounding issues

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

The default pass threshold is 85. Critical execution or grounding checks still block a pass regardless of the weighted score.

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
