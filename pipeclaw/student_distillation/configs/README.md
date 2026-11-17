# Student-distillation experiment configurations

The YAML files in this directory are MS-SWIFT 4.x launch configurations. Run
them from the repository root after creating the
`pipeclaw/student_distillation/environment.yml` environment.

## Local smoke test

The smoke file uses `Qwen/Qwen3.5-0.8B`, 4-bit LoRA, the answer-only
projection, 32 training rows, and 8 validation rows:

- `qwen38_27b_smoke_step10.yaml` trains for 10 optimizer steps and saves
  `checkpoint-10`.

Run:

```bash
swift sft pipeclaw/student_distillation/configs/qwen38_27b_smoke_step10.yaml
```

The smoke configuration intentionally uses `max_length: 2048`, no packing, and
`resume_only_model: false` so the setup exercises the complete trainer
state.

## Remote 27B SFT

- `qwen38_27b.yaml` — trace-level / ablation SFT run for five epochs.

Uses `Qwen/Qwen3.8-27B`, 4-bit NF4 QLoRA (rank 32,
alpha 64), Flash Attention, and DeepSpeed ZeRO-2 across three visible processes by
default. The checked-in setting uses `max_length: 24576` with
`truncation_strategy: delete`.

```bash
conda env create -f pipeclaw/student_distillation/environment.yml
conda activate ms-swift
swift sft pipeclaw/student_distillation/configs/qwen38_27b.yaml
```

For a single GPU, remove `NPROC_PER_NODE` and `deepspeed`, then set
`gradient_accumulation_steps` to 32. Keep `zero2`; ZeRO-3 is not compatible
with the 4-bit quantized model path.

To compare another projection, change `dataset`, `val_dataset`, and
`output_dir` together and leave the other training settings unchanged.

Do not treat a configuration as a dataset generator. Prepare and validate the
JSONL projections with the scripts documented in
[`../scripts/README.md`](../scripts/README.md).
