# Student-distillation datasets

This directory contains deterministic training projections derived from the
immutable teacher-trace splits. Generators preserve source IDs, scenario
and session metadata, and train/valid/test assignments.

## Projections

| Directory | Purpose | Current records (train / valid / test) |
| --- | --- | --- |
| `answer_only/` | Original request → grounded final answer. | 1073 / 147 / 139 |
| `trace_level/` | Bounded context, tool schemas, successful tool calls, and answer. | 1073 / 147 / 139 |
| `constraint_multitask/` | Condition parsing, tool planning, constraint judgment, evidence extraction, and answer generation. | 2388 / 338 / 314 |
| `grpo/` | Prompt data for the GRPO scheduler/reward plugin. | generated separately |

Every projection has `train.jsonl`, `valid.jsonl`, and `test.jsonl`. The test
split is reserved for final evaluation.

Trace-level records supervise successful assistant tool calls and answers;
tool responses are input evidence (`loss: false`). Runtime workspace paths,
secrets, and mutable session state are not copied into training examples.

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

Regenerate all projections with:

```bash
python -m pipeclaw.student_distillation.scripts.prepare_dataset
```

See [the student-distillation guide](../README.md) and
[script reference](../scripts/README.md) for environment and training steps.
