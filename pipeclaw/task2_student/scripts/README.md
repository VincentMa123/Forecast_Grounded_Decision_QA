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
- `evaluate_autonomous.py` is a thin CLI. It parses arguments and hands off to
  `pipeclaw.task2_student.rollout.suite`, which runs the bounded prompt/tool
  loop and then scores the finished rollouts with the canonical evaluator. It
  never places teacher tool calls or the teacher answer in the prompt.

## Execution and evaluation are separate

The rollout and its score are produced by two different packages, in that
order:

- **Execution** — `pipeclaw/task2_student/rollout/`: `prompting.py` builds the
  request through the production `PromptBuilder`, `tools.py` parses and
  dispatches tool calls, `runner.py` drives the bounded turn loop, and
  `scenarios.py` owns allow lists, workspaces, and result compaction.
  `swift_generator.py` loads MS-SWIFT. None of these modules import the
  evaluator, so a rollout can be generated with no scoring code loaded.
- **Evaluation** — `pipeclaw/backend/evaluator/`, the only evaluation package
  in the repository. It sees a finished rollout record and the held-out teacher
  record, and nothing else.

`rollout/suite.py` is the single module that touches both sides: it calls
`evaluate(rollout, profile=EvaluationProfile.AUTONOMOUS_ROLLOUT, reference=source)`
per record and `summarize(reports)` at the end. Keep it that way — adding an
evaluator import to any other rollout module puts scoring back inside
execution.

`rollout/__init__.py` re-exports only the hardware-free core (`models`,
`prompting`, `runner`, `tools`). Import `scenarios`, `swift_generator`, and
`suite` from their modules directly, so that importing the package does not
pull in torch or MS-SWIFT.

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

The command, its flags, and the dry-run workflow are unchanged by the
evaluation refactor; only the internals moved.

`--model` may be omitted when the adapter's `adapter_config.json` contains
`base_model_name_or_path`. Add `--dry-run` (and omit `--adapters`) to write the
exact prompt messages and tool schemas without loading model weights. The dry
run should show only `system,user` messages; the teacher's tool calls,
responses, and answer are retained only for oracle scoring.

For the OpenClaw (PipeClaw agent) portion of the held-out split, use the
dataset label `openclaw` (the `pipeclaw` spelling is accepted as an alias):

```powershell
python -m pipeclaw.task2_student.scripts.evaluate_autonomous `
  --source pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl `
  --tool-schema-source pipeclaw/task2_student/data/trace_level/test.jsonl `
  --scenario-type openclaw `
  --adapters pipeclaw/task2_student/outputs/qwen35_9b_trace_level/checkpoint-55 `
  --output-dir pipeclaw/task2_student/outputs/evaluation/openclaw
```

Omit `--scenario-type` to evaluate both families in one run. The combined
`summary.json` also contains `by_scenario_type` entries, each with its own
record count and denominator-aware metric rates.

During a tool turn the loop appends MS-SWIFT's native `tool_call` and
`tool_response` roles, matching the trace-level training projection; OpenAI-style
SDK responses are normalized at the parser boundary. Qwen3.5's
`<function>/<parameter>` text representation is also parsed; when it appears
alongside a duplicate native call, the typed text representation is preferred.

The evaluator writes `rollouts.jsonl` and `summary.json`, both stamped
`"schema_version": "pipeclaw_evaluation_v2"`. A rollout can end as
`completed`, `empty_response`, or `max_turns_exceeded`; malformed tool JSON and
tool failures remain in the record instead of aborting the suite. Tool calls are
schema-checked and only read-only/topology/registry/forecast operations are
allow-listed for PipeFormer cases; write, edit, and shell tools are never
dispatched there. Forecast calls also enforce the registry-grounding
precondition used by PipeClaw. OpenClaw cases use an isolated scenario
workspace: `read_file`, `write_file`, and `edit_file` are workspace-bounded,
logical `pipeline_data/...` reads remain read-only, and `run_command` is limited
to a Python script located inside that workspace with a 1--60 second timeout.

### Schema-v2 score and critical gate

Each record in `rollouts.jsonl` carries an `overall_score`, a `passed` flag, and
a `metrics` mapping in which every entry reports `applicable`, `passed`,
`weight`, `critical`, `included_in_score`, and a nested `details`.

Under the autonomous profile every applicable deliverable metric carries weight
`1.0`, and `overall_score` is the percentage of that weight which passed. Only
metrics that are both `applicable` and `included_in_score` enter the
denominator, so `overall_score` is `null` when a record raised no scorable
metric at all.

`passed` requires the critical gate, not a score threshold: **any** failing
critical metric, or any hard grounding issue, fails the record however high the
weighted score is. Critical metrics cover applicable task parsing, tool
execution, assumption consistency, PipeFormer authenticity, disturbance and
horizon correctness, constraint execution and judgment, verification, registry
ordering, decision labels, grounding, answer completeness, JSON validity, and
requested artifact evidence. Unlike the teacher profile there is no
minimum-score bar; a student is judged on getting the critical work right.

`tool_recovery`, `portability`, `raw_capture_metadata`, `model_loading_metadata`,
and `hallucination` are diagnostics: `included_in_score: false` and never
critical. They describe the run without grading it.

The evaluator reuses the base-model loading settings saved in the adapter's
`args.json`. Therefore a 4-bit QLoRA checkpoint is loaded with bitsandbytes
4-bit NF4 settings instead of silently loading the 9B base model in BF16. The
loader prints the resolved mode at startup. Use `--quant-bits 4` or
`--quant-bits 8` to override the checkpoint metadata, or `--no-quantization`
to intentionally load the unquantized base model.

Successful PipeFormer forecast responses use the same canonical projection as
`backend/task1/generate_teacher_trace.py` before they are sent back to the
student and written to `rollouts.jsonl`. Registry and other tool responses keep
their registered shape. The dispatcher still uses the full result for
authorization, and `--save-raw-tool-outputs` can retain a separate raw
diagnostic copy when needed.

The aggregate is denominator-aware: a metric with no applicable oracle is marked
`not_applicable`, not counted as a zero. Use the `pipeformer` filter for the gas
pipeline deliverable and `openclaw` for the PipeClaw agent cases. Each metric in
`summary.json` reports `numerator`, `denominator`, `pass_rate`, and
`failure_rate`. OpenClaw records additionally expose the generic
`artifact_evidence` metric, which checks that files named in the current
request were actually read or used as evidence rather than merely mentioned.

### Hallucination rate is derived, not measured separately

`summary["hallucination_rate"]` is exactly `evidence_consistency.failure_rate`,
and each autonomous record's `hallucination` entry is a copy of its
`evidence_consistency` result tagged `derived_from: "evidence_consistency"` and
`included_in_score: false`. The grounding checker runs once per record; there is
no second pass, so the two figures cannot drift apart. Report
`hallucination_rate` (or equivalently the `evidence_consistency` failure rate)
as the requested hallucination rate.

PipeFormer records that carry a non-empty `disturbance_assumption` marker are
scored assumption-aware. When the request does not state the disturbance
direction or magnitude, the student may assume one, and **its assumption does
not have to equal the teacher's sampled value.** Concretely, for every field the
teacher marked assumed:

- `task_parsing` excludes that field from comparison entirely — a different
  value is not a parsing failure.
- `disturbance_application` and `assumption_consistency` resolve the expected
  boundary change from the student's own executed prediction rather than the
  teacher's value.

The student must still emit a valid `up`/`down` direction and finite numeric
values, keep its forecast prediction consistent with its own tool call, and
apply the corresponding boundary change — it is held to internal consistency,
not to the teacher's arbitrary choice. Explicit (non-assumed) disturbance fields
remain strict. Inherited provisional disturbances in a multi-turn
`state_before.scope` are marked the same way when the preceding teacher turn
discloses an LLM assumption. Numeric evidence accepts values from
successful student tool outputs as well as the teacher oracle, while candidate
indexes and ordered-list markers are ignored. `tool_call` remains strict (any
failed call is visible), and `tool_recovery` separately reports whether a
required tool eventually succeeded after a failed retry.

### Raw-response diagnostics

The normal rollout stores normalized calls and messages. To preserve the
generator response before parsing, enable the opt-in diagnostic flag and keep
the run small:

```powershell
python -m pipeclaw.task2_student.scripts.evaluate_autonomous `
  --source pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl `
  --tool-schema-source pipeclaw/task2_student/data/trace_level/test.jsonl `
  --scenario-type pipeformer `
  --adapters pipeclaw/task2_student/outputs/qwen35_9b_trace_level/checkpoint-55 `
  --output-dir pipeclaw/task2_student/outputs/evaluation/diagnostic `
  --limit 1 `
  --max-turns 1 `
  --max-new-tokens 512 `
  --save-raw-responses
```

The resulting record contains `raw_responses`, alongside the normalized
`messages`, `tool_calls`, and `json_errors`. Raw capture is disabled by default
because SDK response objects can be large.
