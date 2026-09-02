# Student distillation: MS-SWIFT training

Student distillation turns finalized PipeClaw teacher traces into training data,
fine-tunes a student model with MS-SWIFT, and evaluates autonomous tool use. It
does not retrain PipeFormer and it never edits the teacher-trace source splits.

## Workflow

```text
Teacher-trace source splits
        │
        ├─ prepare_dataset.py ──► answer_only / trace_level / constraint_multitask
        │                              │
        │                              └─ profile_tokens.py
        │
        └─ validate_dataset.py
                                       │
                               MS-SWIFT SFT/GRPO
                                       │
                         autonomous rollout + evaluator
```

The frozen source splits currently contain 1,073 train, 147 validation, and
139 test records. The test split is reserved for final evaluation.

## Environments

Dataset conversion uses an environment that can import the repository's
backend dependencies. Training uses the separate Python 3.12 CUDA environment
declared by `environment.yml`:

```bash
conda env create -f pipeclaw/student_distillation/environment.yml
conda activate task2-ms-swift
python -m pip check
```

The training environment contains MS-SWIFT, Transformers, PyTorch, bitsandbytes,
and the optional multi-GPU dependencies. Check the host driver and available GPU
memory before starting a remote run.

## Prepare and validate data

Run from the repository root:

```bash
python -m pipeclaw.student_distillation.scripts.prepare_dataset
python -m pipeclaw.student_distillation.scripts.validate_dataset
```

The converter writes deterministic JSONL files and
`data/manifests/task2_dataset_manifest.json`. The validator checks source
identity, split isolation, tool schemas, call/response pairs, loss flags,
structured targets, and SHA-256 checksums. See
[data/README.md](data/README.md) for projection details.

## Profile token lengths

Before selecting `max_length`, measure the exact MS-SWIFT/Qwen3.5 rendering for
train and validation records:

```bash
python -m pipeclaw.student_distillation.scripts.profile_tokens \
  --projections answer_only trace_level constraint_multitask
```

The checked-in profile covers 5,166 train/validation records and reports a
maximum of 18,127 rendered tokens. The remote SFT configuration currently uses
`max_length: 18432` with `truncation_strategy: delete`; re-profile any changed
dataset and confirm that this behavior is acceptable before a training run.

## Train a student

Configuration files and their intended hardware are documented in
[configs/README.md](configs/README.md).

For a local wiring check, run the 0.8B smoke pair in order:

```bash
swift sft pipeclaw/student_distillation/configs/qwen35_08b_smoke_step10.yaml
swift sft pipeclaw/student_distillation/configs/qwen35_08b_smoke_resume_step20.yaml
```

For the 9B trace-level experiment, benchmark first and then launch the full
configuration:

```bash
swift sft pipeclaw/student_distillation/configs/qwen35_9b_remote_benchmark_step20.yaml
swift sft pipeclaw/student_distillation/configs/qwen35_9b.yaml
```

The remote files default to four processes with DeepSpeed ZeRO-2. On one GPU,
remove `NPROC_PER_NODE` and `deepspeed`, then increase gradient accumulation to
preserve the effective batch size. Do not upload checkpoints or generated
outputs to the repository.

GRPO is a separate stage using the reviewed scheduler and reward plugin:

```bash
swift rlhf pipeclaw/student_distillation/configs/qwen35_9b_grpo.yaml --rlhf_type grpo
```

Run it only after validating the SFT checkpoint and `data/grpo/rl_train.jsonl`.

## Evaluate autonomous behavior

Evaluation generates a fresh bounded model/tool episode and then scores it with
the shared backend evaluator. It does not compare the final text alone and it
does not place teacher future actions in the prompt.

```powershell
python -m pipeclaw.student_distillation.scripts.evaluate_autonomous `
  --source pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl `
  --tool-schema-source pipeclaw/student_distillation/data/trace_level/test.jsonl `
  --scenario-type pipeformer `
  --adapters pipeclaw/student_distillation/outputs/qwen35_9b_trace_level/checkpoint-20 `
  --output-dir pipeclaw/student_distillation/outputs/evaluation/autonomous
```

Use `--scenario-type openclaw` for PipeClaw agent cases, or omit it for both
families. Add `--dry-run` and omit `--adapters` to inspect prompts and tool
schemas without loading model weights. The output contains `rollouts.jsonl`
and `summary.json` with schema version `pipeclaw_evaluation_v3`.

For repeated episodes or a deployed OpenAI-compatible student, use
`pass_at_k.py`; its options and production-agent mode are documented in
[scripts/README.md](scripts/README.md).

## Directory guide

- `configs/` — reviewed MS-SWIFT SFT and GRPO YAML files.
- `data/` — derived projections, manifests, token profiles, and GRPO prompts.
- `rollout/` — prompt construction, allow-listed tool dispatch, bounded loops,
  isolated workspaces, and model generation.
- `scripts/` — preparation, validation, profiling, rollout evaluation, and
  pass@k commands.
- `tests/` — student-distillation contract tests.
- `outputs/` — ignored checkpoints, predictions, and logs.

## Boundaries and safety

`scripts/validate_dataset.py` owns the dataset release contract and registered
tool schemas; `prepare_dataset.py` owns projections, while `profile_tokens.py`
keeps its sequential profiling workflow together in one sectioned command.
Rollout modules do not import the evaluator; `rollout/suite.py` is the single
execution-to-scoring seam. PipeFormer scenarios allow only read-only topology,
registry, and forecast tools. OpenClaw file operations and Python commands are
workspace-bounded. Failed calls and malformed JSON remain in the episode record
for diagnosis instead of being silently discarded.
