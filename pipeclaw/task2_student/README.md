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

Task 2 uses a dedicated Linux (WSL2/Ubuntu locally, rented Linux server
remotely) Python 3.12 environment. It does not install MS-SWIFT into the
existing PipeClaw backend environment. See "Environment setup" below.

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
conda run -n pipeclaw python pipeclaw/task2_student/scripts/validate_dataset.py
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

### The three compiled kernels that cannot be in the file

`causal-conv1d`, `flash-attn`, and `deepspeed` are the only training
dependencies not in `environment.yml`. PyPI ships them source-only and their
`setup.py` imports torch, so they must be built with build isolation disabled —
and `--no-build-isolation` is a pip *command* option that a requirements file
(which is what Conda feeds the pip block) cannot carry. They also need `nvcc`,
which the local WSL host does not have. On an nvcc host, after activating the
environment:

```bash
python -m pip install -U --no-build-isolation \
  "causal-conv1d>=1.6.2" "flash-attn==2.8.3" "deepspeed>=0.18.9"
```

DeepSpeed is required by `configs/qwen35_9b_remote_*.yaml` (`deepspeed: zero2`)
and flash-attn by `attn_impl: flash_attn`. causal-conv1d is the fast kernel for
the Qwen3.5 gated-DeltaNet layers, and `flash-linear-attention` — which *is* in
the environment file, wheel-only and needing no compiler — is the Triton
fallback for it. A single-GPU `sdpa` run therefore needs none of the three.

### Without Conda

The local WSL host has no Conda installed, so training currently runs from the
Python 3.12 virtual environment `~/.venvs/task2-ms-swift`. To refresh or rebuild
that path, install the pip block of `environment.yml` by hand — that file stays
the source of truth for every version:

```bash
source ~/.venvs/task2-ms-swift/bin/activate
python -m pip install -U --extra-index-url https://download.pytorch.org/whl/cu130 \
  torch==2.13.0+cu130 torchvision==0.28.0+cu130
python -m pip install -U ms-swift==4.4.2 transformers==5.12.1 peft==0.19.1 \
  trl==0.29.1 "accelerate>=1.14,<2.0" datasets==4.8.4 "modelscope>=1.39,<2.0" \
  "tokenizers>=0.22,<0.23" "safetensors>=0.8,<0.9" "qwen-vl-utils>=0.0.14" \
  "liger-kernel>=0.8.1" "flash-linear-attention>=0.4.2" bitsandbytes==0.50.0 \
  "fsspec[http]>=2023.1.0,<=2026.2.0" "pillow>=8,<12"
python -m pip check
```

Order matters here: the CUDA wheels come from the PyTorch index first, then the
pinned stack from PyPI. For a CPU-only validation or profiling host, replace the
first command with the CPU wheels:

```bash
python -m pip install torch==2.13.0+cpu torchvision==0.28.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

### Transitive upper bounds

The last two pins in the file constrain `fsspec` to the range `datasets 4.8.4`
accepts and Pillow to the range Gradio (an ms-swift dependency) accepts. Without
them pip resolves the newest transitive versions and reports:

```text
datasets requires fsspec[http]<=2026.2.0
gradio requires pillow<12.0
```

Those warnings represent a genuinely inconsistent environment and must be
resolved before renting a GPU or starting a checkpointed run. With the pins the
local WSL environment reports `No broken requirements found` (fsspec 2026.2.0,
Pillow 11.3.0).

### Serving environment

vLLM and SGLang stay out of the training environment on purpose: they pin their
own torch build and would silently replace the verified `torch 2.13.0+cu130`.
Phase 7 adapter inference uses `swift infer` with the Transformers backend, so
no serving stack is required to accept Phase 7. If one is needed later, build a
throwaway environment beside the training one:

```bash
conda create -n task2-ms-swift-serve python=3.12 -y
conda run -n task2-ms-swift-serve python -m pip install \
  "vllm>=0.17.0" "evalscope>=1.0"
```

## Exact Qwen3.5 token profile

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
| All projections | 4,199 | 2,840 | 9,675 | 12,029 | 13,792 |
| Answer only | 1,026 | 1,506 | 1,761 | 1,863 | 2,008 |
| Trace level | 1,026 | 5,023 | 9,616 | 10,712 | 12,136 |
| Constraint multitask | 2,147 | 2,840 | 10,467 | 12,573 | 13,792 |

Complete-record coverage is 0% at 1,024, 25.91% at 2,048, 58.89% at 4,096,
90.33% at 8,192, and 100% at 16,384 tokens. Therefore the lossless
full-data training configuration should use `max_length=16384`. Shorter local
smoke tests must select complete records that fit their limit; they must not
silently truncate the released examples.

Across raw content fields, the system prompt contributes 56.26% of tokens,
tool schemas 18.54%, and user messages 17.49%. These are the first candidates
for a future reviewed compact-prompt experiment, but the released Phase 5
profile preserves them exactly.

## Phase 7: local CUDA smoke test

The local CUDA installation has been observed with:

```text
GPU:          NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB
PyTorch:      2.13.0+cu130
torchvision:  0.28.0+cu130
CUDA runtime: 13.0
MS-SWIFT:     4.4.2
Transformers: 5.12.1
bitsandbytes: 0.50.0
```

`torch.cuda.is_available()` returned true and `python -m pip check` now reports
`No broken requirements found`. Check the environment again before training:

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

The original smoke-test proposal used a 1,024-token maximum, but the exact
profile shows that the shortest current record is 1,430 tokens. Phase 7
therefore uses 32 answer-only training records and 8 answer-only validation
records with `max_length=2048`. The answer-only maximum is 2,008, so all
selected records fit without truncation. The configuration uses the
fail-closed `delete` strategy as a guard against an unexpectedly oversized
record.

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
projection at the lossless `max_length=16384`, 4-bit NF4 QLoRA with LoRA
rank 32 / alpha 64, `flash_attn`, gradient checkpointing, and DeepSpeed ZeRO-2
across four ranks (effective batch size 32).

```bash
# 1. Build the environment, then add the compiled kernels (needs nvcc)
conda env create -f pipeclaw/task2_student/environment.yml
conda activate task2-ms-swift
python -m pip install -U --no-build-isolation \
  "causal-conv1d>=1.6.2" "flash-attn==2.8.3" "deepspeed>=0.18.9"

# 2. Measure peak VRAM and tokens/sec on 20 steps of the real configuration
swift sft pipeclaw/task2_student/configs/qwen35_9b_remote_benchmark_step20.yaml

# 3. Only then start the full run
swift sft pipeclaw/task2_student/configs/qwen35_9b_remote_trace_level.yaml
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
