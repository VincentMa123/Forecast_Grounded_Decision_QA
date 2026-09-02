# Student-distillation scripts

These command-line programs prepare student-distillation data, validate release
invariants, measure token lengths, and evaluate student rollouts. Run them from
the repository root.

## Dataset preparation

`prepare_dataset.py` reads the immutable teacher-trace splits and writes the
`answer_only`, `trace_level`, and `constraint_multitask` projections plus a
checksum manifest:

```bash
python -m pipeclaw.student_distillation.scripts.prepare_dataset
```

Validate an existing release independently before training or copying it to a
GPU host:

```bash
python -m pipeclaw.student_distillation.scripts.validate_dataset
```

Validation covers source identities, split isolation, tool schemas and
call/response pairs, loss flags, structured targets, checksums, and required
answer-generation coverage. The generators do not modify the teacher-trace source
files.

`validate_dataset.py` owns the release paths, split counts, projection names, and
registered tool-schema lookup. `prepare_dataset.py` imports that contract when it
writes projections, avoiding a second source of truth.

## Token profiling

`profile_tokens.py` keeps encoding, measurement, summaries, provenance checks,
and report publication together in execution order. It renders train and valid
records through the MS-SWIFT/Qwen3.5 training template without loading model
weights or silently truncating input:

```bash
python -m pipeclaw.student_distillation.scripts.profile_tokens \
  --projections answer_only trace_level constraint_multitask
```

Only `train` and `valid` are accepted for profiling. The script writes a
reviewable summary JSON and, optionally, a per-record JSONL audit. Choose a
configuration `max_length` only after checking the reported maximum; the full
release configuration uses `max_length: 18432`.

## Autonomous evaluation

The evaluator builds prompts from the production `PromptBuilder`, hides teacher
future actions, runs a bounded model/tool loop, and scores the finished record
with `pipeclaw/backend/evaluator/`:

```powershell
python -m pipeclaw.student_distillation.scripts.evaluate_autonomous `
  --source pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl `
  --tool-schema-source pipeclaw/student_distillation/data/trace_level/test.jsonl `
  --scenario-type pipeformer `
  --adapters pipeclaw/student_distillation/outputs/qwen35_9b_trace_level/checkpoint-20 `
  --output-dir pipeclaw/student_distillation/outputs/evaluation/autonomous
```

Use `--scenario-type openclaw` for PipeClaw agent records, or omit the option
to evaluate both families. Use `--dry-run` without `--adapters` to inspect the
exact messages and tool schemas without loading a model. The output directory
receives `rollouts.jsonl` and `summary.json` with schema version
`pipeclaw_evaluation_v3`.

`--execution-mode production-agent` evaluates through the deployed backend
`AgentOrchestrator`; the default `raw-student` mode loads a checkpoint directly.

## Pass@k and GRPO support

`pass_at_k.py` runs repeated episodes for a checkpoint or a deployed
OpenAI-compatible student:

```bash
python -m pipeclaw.student_distillation.scripts.pass_at_k \
  --source pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_valid.jsonl \
  --tool-schema-source pipeclaw/student_distillation/data/trace_level/valid.jsonl \
  --adapters path/to/checkpoint \
  --episodes 1 --temps 0.0 \
  --output-dir pipeclaw/student_distillation/outputs/evaluation/pass_at_k
```

Use `--execution-mode production-agent` when the deployed model must be tested
through the same backend orchestration and guards as the application. GRPO
uses `grpo_plugin.py`, `data/grpo/rl_train.jsonl`, and the reviewed
`qwen35_9b_grpo.yaml`; it needs a trainable MS-SWIFT/vLLM setup rather than a
plain inference endpoint.

## Execution/evaluation boundary

The modules under `pipeclaw/student_distillation/rollout/` only construct prompts,
dispatch allow-listed tools, manage isolated workspaces, and record episodes.
They do not import the evaluator. `rollout/suite.py` is the single seam that
passes a completed rollout to `evaluate()` and `summarize()`.

For PipeFormer scenarios, only read-only/topology/registry/forecast tools are
allowed. OpenClaw workspaces bound `read_file`, `write_file`, `edit_file`, and
Python `run_command` to the scenario workspace. Failed calls and malformed
JSON remain in the output record so a run can be diagnosed instead of silently
discarding evidence.
