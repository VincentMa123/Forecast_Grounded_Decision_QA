# Task 2: MS-SWIFT Student Distillation

This directory contains the reproducible Task 2 pipeline for distilling the
finalized teacher traces into an external lightweight decision model. It does
not retrain PipeFormer and does not modify PipeClaw's Task 1 dataset.

## Authoritative inputs

The source splits are read-only:

- `pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_train.jsonl`
- `pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_valid.jsonl`
- `pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl`

The frozen source counts are 1073 / 147 / 139. Derived records must preserve
sample IDs and split assignments. The test split is reserved for final
evaluation.

## Framework and models

- Framework: MS-SWIFT 4.x
- Local smoke test: `Qwen/Qwen3.5-0.8B`
- Main remote student: `Qwen/Qwen3.5-9B`
- Adaptation: 4-bit QLoRA

Task 2 uses a dedicated Linux (WSL2/Ubuntu locally, rented Linux server
remotely) Python 3.12 environment. It does not install MS-SWIFT into the
existing PipeClaw backend environment. See "Environment setup" below.

## Layout

- `configs/`: reviewed local and remote experiment commands.
- `data/`: deterministic answer-only, trace-level, and constraint-aware
  projections plus checksum manifests.
- `rollout/`: autonomous model/tool **execution** — prompt construction, tool
  parsing and dispatch, the bounded turn loop, scenario allow lists and
  workspaces, MS-SWIFT generation, and the dataset suite.
- `scripts/`: dataset preparation, validation, token profiling, evaluation,
  and comparison programs.
- `tests/`: Task 2 unit and contract tests.
- `outputs/`: ignored local checkpoints, predictions, and logs.

Generated JSONL, model weights, checkpoints, and caches are not committed.
Small manifests and summary metrics may be committed after review.

There is **no** evaluator package under `task2_student/`. All scoring lives in
`pipeclaw/backend/evaluator/`, the single evaluation package for both tasks;
`pipeclaw/tests/evaluation/test_layout.py` fails if a second one reappears.
`rollout/suite.py` is the only rollout module that imports it, so execution and
evaluation stay separable.

## Workflow

1. Convert the immutable Task 1 splits into framework-neutral agent records.
2. Validate identities, split isolation, tool-call pairs, and loss masking.
3. Profile exact Qwen3.5 token lengths without silent truncation.
4. Run the local 0.8B smoke test.
5. Compare answer-only, trace-level, and constraint-aware SFT.
6. Reuse the same artifacts for the remote 9B experiment.

## Held-out autonomous evaluation

After SFT, evaluate the model with an autonomous rollout rather than comparing
its text to the teacher answer verbatim. The harness builds each request through
PipeClaw's production `PromptBuilder`, hides all teacher future actions, runs a
bounded model/tool loop, and compares canonical task fields, tool success,
constraint labels, risk/intervention labels, evidence, answer presence, and JSON
validity against the held-out teacher record.

Execution and scoring are two steps in two packages: `rollout/` produces the
record, then `pipeclaw/backend/evaluator/` scores it under
`EvaluationProfile.AUTONOMOUS_ROLLOUT`. The command below is unchanged by that
split.

For the gas-pipeline deliverable, use the 24 PipeFormer records in the frozen
teacher test split and the trace projection only for tool schemas:

```powershell
python -m pipeclaw.task2_student.scripts.evaluate_autonomous `
  --source pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl `
  --tool-schema-source pipeclaw/task2_student/data/trace_level/test.jsonl `
  --scenario-type pipeformer `
  --adapters pipeclaw/task2_student/outputs/qwen35_9b_trace_level/checkpoint-20 `
  --output-dir pipeclaw/task2_student/outputs/evaluation/autonomous
```

The same evaluator also supports the 90 OpenClaw (PipeClaw agent) records:

```powershell
python -m pipeclaw.task2_student.scripts.evaluate_autonomous `
  --source pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl `
  --tool-schema-source pipeclaw/task2_student/data/trace_level/test.jsonl `
  --scenario-type openclaw `
  --adapters pipeclaw/task2_student/outputs/qwen35_9b_trace_level/checkpoint-55 `
  --output-dir pipeclaw/task2_student/outputs/evaluation/openclaw
```

Omit `--scenario-type` to evaluate both scenario families. The combined
summary includes separate `by_scenario_type` counts and metric denominators;
OpenClaw workspaces persist across simulated sessions within each scenario and
restrict file operations and Python scripts to the evaluation workspace.

Run the same command with `--dry-run` and without `--adapters` to inspect the
prompt/tool inputs first. Results are written to `rollouts.jsonl` and
`summary.json`, both stamped `"schema_version": "pipeclaw_evaluation_v2"`.

### Production-agent pass@k

`pass_at_k.py` has two execution modes. `raw-student` remains the default and
loads a checkpoint directly. `production-agent` calls the already deployed
OpenAI-compatible student through the same `AgentOrchestrator` used by the app,
including verified state, recent turns, production tool guards, loop closure,
and answer finalization. It defaults to the production limit of 30 turns and
writes complete traces to `trajectories.jsonl` beside compact `episodes.jsonl`.

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY="$STUDENT_API_KEY"
export OPENAI_MODEL=pipeclaw-student

python -m pipeclaw.task2_student.scripts.pass_at_k \
  --source pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_valid.jsonl \
  --tool-schema-source pipeclaw/task2_student/data/trace_level/valid.jsonl \
  --execution-mode production-agent \
  --all-scenarios \
  --episodes 1 --temps 0.0 --max-new-tokens 2048 \
  --output-dir pipeclaw/task2_student/outputs/evaluation/production_agent
```

Do not point GRPO at an ordinary `swift deploy` app endpoint. GRPO needs the
current trainable policy's token log-probabilities and synchronized weights.
For a remote inference process, use MS-SWIFT's `swift rollout` plus
`vllm_mode: server`; its API key may secure the rollout server, but weight
synchronization—not the key—is what makes that server valid for training. The
PipeClaw multi-turn scheduler remains responsible for production tool execution
and deterministic reward inside GRPO.

Under schema v2, every applicable deliverable metric carries weight `1.0` and
`overall_score` is the percentage of that weight which passed; inapplicable
metrics are reported explicitly rather than scored as failures, and diagnostics
(`tool_recovery`, `portability`, capture and model-loading metadata,
`hallucination`) stay out of the denominator. A record's `passed` flag is a
critical gate, not a threshold: one failing critical metric or hard grounding
issue fails it regardless of score. `summary["hallucination_rate"]` is derived
from `evidence_consistency.failure_rate` rather than measured separately. See
`scripts/README.md` for the safety allowlist, failure-state details, and the
full metric list.

## Dataset conversion

Run the converter from the repository root in the existing `pipeclaw` Conda
environment:

```powershell
conda run -n pipeclaw python -m pipeclaw.task2_student.scripts.prepare_dataset
```

It reads the three authoritative splits, requires the frozen 1073 / 147 / 139
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

Every projection starts with the same shared static policy used by the
production PipeClaw prompt builder so comparisons hold the global instructions
constant. Answer-only then supplies only the original request and answer.
Trace-level adds bounded verified decision state, recent dialogue, and the
teacher tool trajectory. Constraint-multitask retains its task-specific
instruction and bounded context after the shared policy. Changing runtime
state, workspace paths, assets, control files, skills, evidence memory, and
trace metadata are not copied into SFT examples.

Validate an existing release independently with:

```powershell
conda run -n pipeclaw python -m pipeclaw.task2_student.scripts.validate_dataset
```

The validator checks source and output checksums, identities, split isolation,
tool schemas and call/response pairs, loss flags, structured targets, and
answer-generation coverage. Generated JSONL files remain ignored by Git; the
small checksum manifest is trackable.

## Environment setup

Task 2 requires Python 3.12 (MS-SWIFT's recommended version, and required by
flash-linear-attention) and a CUDA-enabled PyTorch build.

`environment.yml` is the only install file; every pinned dependency lives in it.
One Conda command builds the complete training environment, locally or on a
fresh rented Linux server:

```bash
conda env create -f pipeclaw/task2_student/environment.yml
conda activate task2-ms-swift
```

That single file installs the GPU PyTorch pair *and* the pinned MS-SWIFT stack
in one pip pass, so nothing else is needed for single-GPU LoRA/QLoRA training.
Update an existing environment with
`conda env update -f pipeclaw/task2_student/environment.yml --prune`.

Verify before training; a clean `pip check` is the acceptance criterion:

```bash
python -m pip check      # must print: No broken requirements found
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### CUDA build

The pip block sets `--extra-index-url https://download.pytorch.org/whl/cu130`
and pins `torch==2.13.0+cu130` / `torchvision==0.28.0+cu130` explicitly. The
`+cu130` local version guarantees pip takes torch from the PyTorch index, while
PyPI stays reachable for the rest of the stack — a global `--index-url` would
hide the packages that are published only on PyPI.

Check the host driver first; cu130 wheels need a 580-series or newer driver:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
nvidia-smi | head -3
```

If the reported CUDA version is 12.x, change `cu130` to `cu129` on the three
torch lines in `environment.yml`. That index publishes the same verified pair,
`torch 2.13.0` / `torchvision 0.28.0`, so nothing else in the stack moves. cu128
is not equivalent — it stops at torch 2.11.0 / torchvision 0.26.0, so a cu128
host must re-run the smoke test.

## Qwen3.5 token profile status

Run token profiling after dataset validation:

```bash
~/.venvs/task2-ms-swift/bin/python \
  -m pipeclaw.task2_student.scripts.profile_tokens
```

The command downloads the `Qwen/Qwen3.5-0.8B` processor/tokenizer and uses
MS-SWIFT's training template; it does not load model weights. It verifies the
six train/valid projection files against the dataset manifest and deliberately
does not accept the test split. Outputs are:

```text
data/token_profiles/qwen35_08b_token_profile.json
data/token_profiles/qwen35_08b_token_records.jsonl
```

The checked-in profile was regenerated on 2026-08-22 with MS-SWIFT 4.4.2 and
the Qwen3.5 training template. It covers 5,166 train/validation rows and matches
the current dataset manifest. The trace-level maximum is 18,127 tokens (p95
10,523; p99 12,890); the constraint-multitask maximum is 17,622 and the
answer-only maximum is 2,532. Use 18,432 for lossless trace-level training.

No `max_length` or truncation strategy is passed while profiling. Any later
training limit must fit a complete record or have a reviewed, explicit
disposition; tool evidence and its grounded final answer must not be separated.

The profile writer reports exact rendered lengths, coverage, and field totals
when a current tokenizer is available. Keep any resulting figures with the
profile provenance; do not copy them into this README without a successful
refresh.

## Phase 7: local CUDA smoke test

The local hardware and package snapshot is not a refreshed profile result.
Check the environment again before training:

```bash
conda activate task2-ms-swift   # or: source ~/.venvs/task2-ms-swift/bin/activate
python -m pip check
python -m pip show ms-swift torch torchvision bitsandbytes
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Do not use `swift --version`; MS-SWIFT 4.4.2 treats `--version` as a command
name and raises `KeyError`. Use `python -m pip show ms-swift` instead.

### Known MS-SWIFT / Transformers CLI limitation

`include_num_input_tokens_seen` is typed `str | bool` in Transformers 5.12 and
accepts `"no"`, `"all"`, or `"non_padding"` in Python. On the command line,
however, `HfArgumentParser` strips `str` from that union and parses the flag as
a boolean, so MS-SWIFT — which forwards YAML keys as CLI flags — fails with:

```text
sft.py: error: argument --include_num_input_tokens_seen: Truthy value expected:
got non_padding but expected one of yes/no, true/false, t/f, y/n, 1/0
```

Both smoke configurations therefore pass `include_num_input_tokens_seen: true`,
which Transformers maps to `"all"`. With `per_device_train_batch_size=1` and
packing disabled there is no intra-batch padding, so `"all"` and
`"non_padding"` report identical supervised token counts. This is a CLI parsing
limitation, not a YAML-versus-flags problem: the same value fails when passed
directly on the command line.

The smoke configurations retain their reviewed `max_length=2048` and
fail-closed `delete` strategy. The selected 32 training rows have a maximum of
1,949 tokens and the selected 8 validation rows have a maximum of 1,931.

Run steps 1 through 10 from the repository root:

```bash
swift sft \
  pipeclaw/task2_student/configs/qwen35_08b_smoke_step10.yaml
```

Confirm that `checkpoint-10` contains the adapter and resumable trainer state,
then continue from step 10 through step 20:

```bash
swift sft \
  pipeclaw/task2_student/configs/qwen35_08b_smoke_resume_step20.yaml
```

The second configuration deliberately uses the same output directory and
restores model, optimizer, scheduler, random state, and data position. Success
requires finite nonzero training loss, nonzero supervised tokens,
`checkpoint-10` and `checkpoint-20`, a final global step of 20, and no restart
from step zero. Model download and the two training commands are left for the
user to run on the CUDA environment.

## Remote server run (Qwen3.5-9B)

Two configurations cover the rented Linux server. Both use the full trace-level
projection with `max_length=18432`, 4-bit NF4 QLoRA with LoRA rank 32 / alpha
64, `flash_attn`, gradient checkpointing, and DeepSpeed ZeRO-2 across four
ranks (effective batch size 32). The current exact trace-level maximum is
18,127 tokens, so this limit is lossless.

```bash
# 1. Build the environment, then add the compiled kernels (needs nvcc)
conda env create -f pipeclaw/task2_student/environment.yml
conda activate task2-ms-swift

# 2. Measure peak VRAM and tokens/sec on 20 steps of the real configuration
swift sft pipeclaw/task2_student/configs/qwen35_9b_remote_benchmark_step20.yaml

# 3. Only then start the full run
swift sft pipeclaw/task2_student/configs/qwen35_9b.yaml
```

The benchmark config is byte-for-byte identical to the full run on every setting
that drives memory and throughput, so its measurement is what the rental cost
estimate should be based on.

DeepSpeed is only needed for the multi-GPU path. `NPROC_PER_NODE` in the config's
`ENV` block is what makes `swift sft` launch `torch.distributed.run`, and ZeRO-2
then shards optimizer state and gradients across those ranks; it must equal the
number of visible GPUs, because MS-SWIFT rejects DeepSpeed when a single process
holds a `device_map`-sharded model. On a single 80 GB GPU, delete
`NPROC_PER_NODE` and `deepspeed` and raise `gradient_accumulation_steps` to 32 —
that path needs no DeepSpeed at all, so the `--no-build-isolation` install is
then only about flash-attn and causal-conv1d.

Stay on `zero2`. ZeRO-3 shards the base weights, which does not work with
bitsandbytes 4-bit quantized parameters; `zero2_offload` is the fallback if
optimizer state does not fit.

Switching projection is a two-line change (`dataset`, `val_dataset`, plus
`output_dir`) to `answer_only` or `constraint_multitask`; every other value must
stay identical so the three-way comparison holds the recipe constant.

## Correct Python tool-call continuation

The release validator rejects truncated or syntactically invalid Python and any
`run_command` execution of a saved script without a preceding complete
`write_file`. The derived correction set is intentionally small: 129 training
records (43 Python-writing traces plus 86 deterministic replay traces) and 18
validation records (6 plus 12).

After copying the corrected data to the training host, regenerate the dedicated
token profile and require its reported maximum to be at most 24,576 tokens:

```bash
python -m pipeclaw.task2_student.scripts.profile_tokens \
  --projections python_correction \
  --summary-output pipeclaw/task2_student/data/token_profiles/qwen35_08b_python_correction_profile.json \
  --records-output pipeclaw/task2_student/data/token_profiles/qwen35_08b_python_correction_records.jsonl
```

Then continue from the best rank-32 adapter with fresh optimizer state:

```bash
CKPT=pipeclaw/task2_student/outputs/qwen35_9b_32l_trace_level_improved/checkpoint-60
swift sft pipeclaw/task2_student/configs/qwen35_9b_python_correction.yaml \
  --resume_from_checkpoint "$CKPT"
```

Serve the corrected adapter deterministically with a 2,048-token response
budget. Do not pass `--repetition_penalty`; penalizing repeated JSON keys and
code tokens can damage tool-call syntax.

```bash
CKPT=pipeclaw/task2_student/outputs/qwen35_9b_python_correction/checkpoint-200
CUDA_VISIBLE_DEVICES=0 swift deploy \
  --model Qwen/Qwen3.5-9B \
  --adapters "$CKPT" \
  --infer_backend transformers \
  --quant_method bnb \
  --quant_bits 4 \
  --bnb_4bit_compute_dtype bfloat16 \
  --torch_dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --api_key "$STUDENT_API_KEY" \
  --served_model_name pipeclaw-student \
  --temperature 0 \
  --max_new_tokens 2048
```

Set the PipeClaw client-side ceiling to the same value with
`ZAI_MAX_TOKENS=2048`.
