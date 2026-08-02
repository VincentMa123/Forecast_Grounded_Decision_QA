# Task 2 Programs

This directory is reserved for the focused dataset-preparation, fail-closed
validation, Qwen3.5 token-profiling, student-evaluation, and
experiment-comparison programs specified in the approved design. Programs
must not mutate Task 1 inputs.

- `prepare_dataset.py` creates the three deterministic MS-SWIFT projections
  and checksum manifest from the frozen Task 1 splits.
- `validate_dataset.py` independently verifies the sources, derived records,
  split isolation, tool/loss contracts, and manifest checksums.
- `profile_tokens.py` verifies the manifest-selected train/valid files and
  measures exact untruncated Qwen3.5/MS-SWIFT template lengths without loading
  model weights. It writes a reviewable summary JSON and an ignored per-record
  JSONL audit.
- `evaluate_autonomous.py` evaluates a student with a bounded prompt/tool loop;
  it never places teacher tool calls or the teacher answer in the prompt. The
  loop uses the production `PromptBuilder`, validates an allow-listed tool set,
  records partial traces, and scores semantic task/tool/constraint/evidence
  metrics against the held-out teacher oracle.

## Autonomous rollout evaluation

Use the authoritative teacher split for both the prompt request and oracle, and
the trace projection only to provide the exact eight tool schemas:

```powershell
python -m pipeclaw.task2_student.scripts.evaluate_autonomous `
  --source pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl `
  --tool-schema-source pipeclaw/task2_student/data/trace_level/test.jsonl `
  --scenario-type pipeformer `
  --adapters pipeclaw/task2_student/outputs/qwen35_9b_trace_level/checkpoint-20 `
  --output-dir pipeclaw/task2_student/outputs/evaluation/autonomous
```

`--model` may be omitted when the adapter's `adapter_config.json` contains
`base_model_name_or_path`. Add `--dry-run` (and omit `--adapters`) to write the
exact prompt messages and tool schemas without loading model weights. The dry
run should show only `system,user` messages; the teacher's tool calls,
responses, and answer are retained only for oracle scoring.

During a tool turn the loop appends MS-SWIFT's native `tool_call` and
`tool_response` roles, matching the trace-level training projection; OpenAI-style
SDK responses are normalized at the parser boundary.

The evaluator writes `rollouts.jsonl` and `summary.json`. A rollout can end as
`completed`, `empty_response`, or `max_turns_exceeded`; malformed tool JSON and
tool failures remain in the record instead of aborting the suite. Tool calls are
schema-checked and only read-only/topology/registry/forecast operations are
allow-listed; write, edit, and shell tools are never dispatched. Forecast calls
also enforce the registry-grounding precondition used by PipeClaw.

The aggregate is denominator-aware: a metric with no applicable oracle is marked
`not_applicable`, not counted as a zero. Use the `pipeformer` filter for the gas
pipeline deliverable; the same harness can be run without the filter for the
open-domain cases. `summary.json` includes both `pass_rate` and `failure_rate`;
use `failure_rate` for the requested hallucination rate.
