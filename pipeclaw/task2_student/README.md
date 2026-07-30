# Task 2: MS-SWIFT Student Distillation

This directory contains the reproducible Task 2 pipeline for distilling the
finalized teacher traces into an external lightweight decision model. It does
not retrain PipeFormer and does not modify PipeClaw's Task 1 dataset.

## Authoritative inputs

The source splits are read-only:

- `pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_train.jsonl`
- `pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_valid.jsonl`
- `pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl`

The frozen source counts are 902 / 124 / 114. Derived records must preserve
sample IDs and split assignments. The test split is reserved for final
evaluation.

## Framework and models

- Framework: MS-SWIFT 4.x
- Local smoke test: `Qwen/Qwen3.5-0.8B`
- Main remote student: `Qwen/Qwen3.5-9B`
- Adaptation: 4-bit QLoRA

Task 2 uses a dedicated WSL2/Ubuntu Python environment. It does not install
MS-SWIFT into the existing PipeClaw backend environment.

## Layout

- `configs/`: reviewed local and remote experiment commands.
- `data/`: deterministic answer-only, trace-level, and constraint-aware
  projections plus checksum manifests.
- `scripts/`: dataset preparation, validation, token profiling, evaluation,
  and comparison programs.
- `tests/`: Task 2 unit and contract tests.
- `outputs/`: ignored local checkpoints, predictions, and logs.

Generated JSONL, model weights, checkpoints, and caches are not committed.
Small manifests and summary metrics may be committed after review.

## Workflow

1. Convert the immutable Task 1 splits into framework-neutral agent records.
2. Validate identities, split isolation, tool-call pairs, and loss masking.
3. Profile exact Qwen3.5 token lengths without silent truncation.
4. Run the local 0.8B smoke test.
5. Compare answer-only, trace-level, and constraint-aware SFT.
6. Reuse the same artifacts for the remote 9B experiment.

## Dataset conversion

Run the converter from the repository root in the existing `pipeclaw` Conda
environment:

```powershell
conda run -n pipeclaw python pipeclaw/task2_student/scripts/prepare_dataset.py
```

It reads the three authoritative splits, requires the frozen 902 / 124 / 114
counts, loads all eight schemas from PipeClaw's actual tool registry, and
writes:

```text
data/answer_only/{train,valid,test}.jsonl
data/trace_level/{train,valid,test}.jsonl
data/constraint_multitask/{train,valid,test}.jsonl
data/manifests/task2_dataset_manifest.json
```

Every record keeps `source_sample_id`, scenario/session/turn identity, and
`split`. Answer-only and trace-level `example_id` values equal the source
sample ID. Constraint-aware rows use
`<source_sample_id>::<task_type>` so the five task targets remain unique and
auditable.

Trace records use MS-SWIFT's standard `tools` JSON string and `system`,
`user`, `tool_call`, `tool_response`, and `assistant` roles. Teacher tool
calls and final answers have `loss: true`; verified tool responses have
`loss: false` and are input evidence only. Failed, unmatched, or unregistered
tool calls stop conversion.

The trace-level system message starts with the same shared static policy used
by the production PipeClaw prompt builder. It then adds only the example's
bounded verified decision state and recent dialogue. Changing runtime state,
workspace paths, assets, control files, skills, evidence memory, and trace
metadata are not copied into SFT examples. The shorter answer-only and
constraint-multitask prompts remain task-focused.

Validate an existing release independently with:

```powershell
conda run -n pipeclaw python pipeclaw/task2_student/scripts/validate_dataset.py
```

The validator checks source and output checksums, identities, split isolation,
tool schemas and call/response pairs, loss flags, structured targets, and
answer-generation coverage. Generated JSONL files remain ignored by Git; the
small checksum manifest is trackable.

## Exact Qwen3.5 token profile

Create a dedicated Python 3.12 environment in WSL. For a CPU-only profiling
environment, install the matching PyTorch pair from the CPU wheel index before
the remaining requirements:

```bash
cd /mnt/c/path/to/Forecast_Grounded_Decision_QA
python3.12 -m venv ~/.venvs/task2-ms-swift
~/.venvs/task2-ms-swift/bin/python -m pip install \
  torch==2.13.0+cpu torchvision==0.28.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
~/.venvs/task2-ms-swift/bin/python -m pip install \
  -r pipeclaw/task2_student/requirements.txt
```

The CPU wheel selection is only for validation and token profiling. Before
remote GPU training, install the PyTorch build matching that server's CUDA
runtime instead.

Run token profiling after dataset validation:

```bash
~/.venvs/task2-ms-swift/bin/python \
  pipeclaw/task2_student/scripts/profile_tokens.py
```

The command downloads the `Qwen/Qwen3.5-0.8B` processor/tokenizer and uses
MS-SWIFT's training template; it does not load model weights. It verifies the
six train/valid projection files against the dataset manifest and deliberately
does not accept the test split. Outputs are:

```text
data/token_profiles/qwen35_08b_token_profile.json
data/token_profiles/qwen35_08b_token_records.jsonl
```

The summary reports exact rendered minimum, median, p95, p99, maximum, and
coverage at 1,024 through 16,384 tokens. It also groups by projection, split,
scenario type, and multitask type. Field totals use the same tokenizer on raw
system, user, tool-schema, tool-call, tool-response, and assistant content;
they intentionally exclude chat-template overhead and therefore do not sum to
the exact rendered totals.

No `max_length` or truncation strategy is passed while profiling. Any later
training limit must fit a complete record or have a reviewed, explicit
disposition; tool evidence and its grounded final answer must not be separated.

### Measured release result

The 2026-07-30 profile covers all 4,199 train/valid projection records:

| Scope | Count | Median | p95 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| All projections | 4,199 | 1,445 | 8,828 | 10,758 | 12,397 |
| Answer only | 1,026 | 107 | 362 | 464 | 609 |
| Trace level | 1,026 | 5,023 | 9,616 | 10,712 | 12,136 |
| Constraint multitask | 2,147 | 1,445 | 9,072 | 11,178 | 12,397 |

Complete-record coverage is 43.01% at 1,024, 56.11% at 2,048, 61.92% at
4,096, 93.00% at 8,192, and 100% at 16,384 tokens. Therefore the lossless
full-data training configuration should use `max_length=16384`. Shorter local
smoke tests must select complete records that fit their limit; they must not
silently truncate the released examples.

Across raw content fields, the system prompt contributes 39.15% of tokens,
tool schemas 25.78%, and user messages 24.33%. These are the first candidates
for a future reviewed compact-prompt experiment, but the released Phase 5
profile preserves them exactly.
