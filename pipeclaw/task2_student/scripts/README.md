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
  loop uses the production `PromptBuilder`, validates a scenario-specific
  allow-listed tool set, records partial traces, and scores semantic
  task/tool/constraint/evidence metrics against the held-out teacher oracle.

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

The evaluator writes `rollouts.jsonl` and `summary.json`. A rollout can end as
`completed`, `empty_response`, or `max_turns_exceeded`; malformed tool JSON and
tool failures remain in the record instead of aborting the suite. Tool calls are
schema-checked and only read-only/topology/registry/forecast operations are
allow-listed for PipeFormer cases; write, edit, and shell tools are never
dispatched there. Forecast calls also enforce the registry-grounding
precondition used by PipeClaw. OpenClaw cases use an isolated scenario
workspace: `read_file`, `write_file`, and `edit_file` are workspace-bounded,
logical `pipeline_data/...` reads remain read-only, and `run_command` is limited
to a Python script located inside that workspace with a 1--60 second timeout.

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
pipeline deliverable and `openclaw` for the PipeClaw agent cases. `summary.json`
includes both `pass_rate` and `failure_rate`; use `failure_rate` for the
requested hallucination rate. OpenClaw records additionally expose the generic
`artifact_evidence` metric, which checks that files named in the current
request were actually read or used as evidence rather than merely mentioned.

PipeFormer records that carry a non-empty `disturbance_assumption` marker are
scored assumption-aware: the teacher's sampled disturbance direction and
magnitude are not exact-match targets. The student must still emit valid
`up`/`down` and finite numeric values, keep its forecast prediction consistent
with its tool call, and apply the corresponding boundary change. These records
also expose an `assumption_consistency` metric; explicit (non-assumed)
disturbance fields remain strict. Inherited provisional disturbances in a
multi-turn `state_before.scope` are marked the same way when the preceding
teacher turn discloses an LLM assumption. Numeric evidence accepts values from
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
