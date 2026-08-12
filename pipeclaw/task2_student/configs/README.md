# Experiment Configurations

The checked-in YAML files are intended to be launched from the repository root
with MS-SWIFT 4.x.

## Phase 7 local smoke test

The smoke test uses the answer-only projection. The checked-in token profile is
stale, so its historical 1,430--2,008-token range is only background and must
be rechecked with a current tokenizer before treating the 2,048-token limit as
safe. The original proposed 1,024-token limit is not a supported default.

- `qwen35_08b_smoke_step10.yaml` trains
  `Qwen/Qwen3.5-0.8B` with 4-bit QLoRA for 10 optimizer steps and writes
  `checkpoint-10`.
- `qwen35_08b_smoke_resume_step20.yaml` keeps the same model, data, seeds,
  QLoRA parameters, and output directory, restores the full training state
  from `checkpoint-10`, and continues to optimizer step 20.

Both configurations use 32 deterministic training records, 8 validation
records, batch size 1, rank 8 / alpha 32 LoRA, gradient checkpointing, and
`truncation_strategy=delete`. They retain optimizer, scheduler, and random
state so the resume test is genuine; `resume_only_model` must remain false.

Both configurations set `include_num_input_tokens_seen: true`. Transformers 5.12
types that field `str | bool` and accepts `"no"`, `"all"`, or `"non_padding"` in
Python, but `HfArgumentParser` drops `str` from the union when building the
command line, so MS-SWIFT — which forwards YAML keys as CLI flags — rejects the
string form with `Truthy value expected: got non_padding`. `true` maps to
`"all"`, and with batch size 1 and packing disabled there is no intra-batch
padding, so `"all"` and `"non_padding"` count the same supervised tokens.

Run the first half:

```bash
swift sft \
  pipeclaw/task2_student/configs/qwen35_08b_smoke_step10.yaml
```

After confirming that
`pipeclaw/task2_student/outputs/qwen35_08b_answer_only_smoke/checkpoint-10`
exists, run:

```bash
swift sft \
  pipeclaw/task2_student/configs/qwen35_08b_smoke_resume_step20.yaml
```

Do not change the dataset, seeds, batch settings, or LoRA settings between the
two commands.

## Remote server configurations

- `qwen35_9b_remote_benchmark_step20.yaml` runs 20 optimizer steps of the real
  9B configuration to measure peak VRAM and tokens/sec before renting hours.
- `qwen35_9b.yaml` is the full trace-level run, five epochs over the 902
  training records.

Both use `Qwen/Qwen3.5-9B` with `max_length=16384`, 4-bit NF4 QLoRA with rank
32 / alpha 64, `attn_impl: flash_attn`,
gradient checkpointing, and `deepspeed: zero2` over `NPROC_PER_NODE: 4` — four
ranks × batch 1 × 8 accumulation steps, an effective batch size of 32. The
checked-in profile is stale; confirm current record coverage before describing
this limit as lossless. Every memory- and throughput-relevant value is the same
in both files so the benchmark transfers to the full run.

`NPROC_PER_NODE` belongs in the config's `ENV:` block: MS-SWIFT applies that
block before deciding whether to launch `torch.distributed.run`, so no wrapper
script or exported variable is needed. It must match the number of visible GPUs,
because DeepSpeed is rejected when one process holds a `device_map`-sharded
model. For a single GPU, remove `NPROC_PER_NODE` and `deepspeed` and set
`gradient_accumulation_steps: 32`; DeepSpeed is then not required at all.

Keep the stage at `zero2`. ZeRO-3 shards base weights and is incompatible with
bitsandbytes 4-bit quantized parameters; `zero2_offload` is the fallback when
optimizer state does not fit.

To run the other two projections, change `dataset`, `val_dataset`, and
`output_dir` to `answer_only` or `constraint_multitask` and leave everything else
untouched, so the comparison isolates the projection.
