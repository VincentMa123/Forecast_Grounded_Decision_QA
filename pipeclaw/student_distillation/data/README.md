# Student-distillation datasets

This directory contains deterministic training projections derived from the
immutable teacher-trace splits. Generators preserve source IDs, scenario
and session metadata, and train/valid/test assignments.

## Projections

| Directory | Purpose | Current records (train / valid / test) |
| --- | --- | --- |
| `answer_only/` | Original request → grounded final answer. | 1615 / 331 / 326 |
| `trace_level/` | Bounded context, tool schemas, successful tool calls, and answer. | 1615 / 331 / 326 |
| `constraint_multitask/` | Condition parsing, tool planning, constraint judgment, evidence extraction, and answer generation. | 3909 / 987 / 952 |
| `grpo/` | Historical prompts from retired GRPO training; retained as artifacts. | 146 archived training rows |

The three active projections each have `train.jsonl`, `valid.jsonl`, and `test.jsonl`. The test
split is reserved for final evaluation.

Trace-level records supervise successful assistant tool calls and answers;
tool responses are input evidence (`loss: false`). Runtime workspace paths,
secrets, and mutable session state are not copied into training examples.

## Trace ablation

`ablation/` contains matched trace-level projections. Every condition includes
the same ordered messages and supervises the final answer.

| Condition | Directory | Training tool definitions | Successful call loss |
| --- | --- | --- | --- |
| A | `ablation/A/` | Omitted | Masked |
| B | `ablation/B/` | Omitted | Enabled |
| C | `ablation/C/` | Included | Masked |
| D | `trace_level/` | Included | Enabled |

Generate a new version with `python -m pipeclaw.student_distillation.scripts.prepare_ablation --output-root <new-directory>`.
Use `--skip-invalid` to exclude rejected source rows consistently and record their
IDs and reasons in the ablation manifest. Keep evaluation tools and settings fixed
across conditions. The exported loss flags control which tokens receive loss.
Verify separate answer and call loss normalization in the trainer before running the ablation.

## Manifests and token profiles

- `manifests/task2_dataset_manifest.json` — converter version, source/output
  counts, tool-schema hash, and SHA-256 checksums.
- `token_profiles/` — exact MS-SWIFT/Qwen3.5 length measurements for train and
  valid records; generated profiles do not load model weights.
- `examples/` — small loading examples and contract fixtures.

Generated JSONL and per-record profiling files are local artifacts. Validate a
release before training or transferring it:

```bash
python -m pipeclaw.student_distillation.scripts.validate_dataset
```

Regenerate the baselines using the source selection frozen for A–D:

```bash
python -m pipeclaw.student_distillation.scripts.prepare_dataset --ablation-manifest pipeclaw/student_distillation/data/ablation/manifest.json
```

See [the student-distillation guide](../README.md) and
[script reference](../scripts/README.md) for environment and training steps.
