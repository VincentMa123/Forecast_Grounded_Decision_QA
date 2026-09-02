# Student-distillation experiment configurations

The YAML files in this directory are MS-SWIFT 4.x launch configurations. Run
them from the repository root after creating the
`pipeclaw/student_distillation/environment.yml` environment.

## Local smoke test

The two smoke files use `Qwen/Qwen3.5-0.8B`, 4-bit LoRA, the answer-only
projection, 32 training rows, and 8 validation rows:

1. `qwen35_08b_smoke_step10.yaml` trains for 10 optimizer steps and saves
   `checkpoint-10`.
2. `qwen35_08b_smoke_resume_step20.yaml` restores that checkpoint and continues
   to step 20 with the optimizer, scheduler, random state, and data position.

Run both in order:

```bash
swift sft pipeclaw/student_distillation/configs/qwen35_08b_smoke_step10.yaml
swift sft pipeclaw/student_distillation/configs/qwen35_08b_smoke_resume_step20.yaml
```

The smoke configurations intentionally use `max_length: 2048`, no packing, and
`resume_only_model: false` so the resume path exercises the complete trainer
state.

## Remote 9B SFT

- `qwen35_9b_remote_benchmark_step20.yaml` — 20-step benchmark for measuring
  memory and throughput.
- `qwen35_9b.yaml` — full trace-level SFT run.

Both use the trace-level projection, `Qwen/Qwen3.5-9B`, 4-bit NF4 QLoRA (rank 32,
alpha 64), Flash Attention, and DeepSpeed ZeRO-2 across four processes by
default. The checked-in `max_length: 18432` is the full-release ceiling and the configs use
`truncation_strategy: delete`; run the token profiler before a new run and
confirm that the selected records fit that limit.

```bash
conda env create -f pipeclaw/student_distillation/environment.yml
conda activate task2-ms-swift
swift sft pipeclaw/student_distillation/configs/qwen35_9b_remote_benchmark_step20.yaml
swift sft pipeclaw/student_distillation/configs/qwen35_9b.yaml
```

For a single GPU, remove `NPROC_PER_NODE` and `deepspeed`, then set
`gradient_accumulation_steps` to 32. Keep `zero2`; ZeRO-3 is not compatible
with the 4-bit quantized model path.

To compare another projection, change `dataset`, `val_dataset`, and
`output_dir` together and leave the other training settings unchanged.

## GRPO

`qwen35_9b_grpo.yaml` uses the generated `data/grpo/rl_train.jsonl`, the
`python_scenario_scheduler`, and the deterministic `python_episode_reward`
plugin. Launch it only after the SFT checkpoint and GRPO data have been
validated:

```bash
swift rlhf pipeclaw/student_distillation/configs/qwen35_9b_grpo.yaml --rlhf_type grpo
```

Do not treat a configuration as a dataset generator. Prepare and validate the
JSONL projections with the scripts documented in
[`../scripts/README.md`](../scripts/README.md).
