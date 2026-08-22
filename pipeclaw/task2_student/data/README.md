# Derived Task 2 Data

All data in this directory is derived from the immutable Task 1 compact SFT
splits. Generators must preserve source sample IDs, scenario/session/turn
metadata, and split assignments.

- `answer_only/`: user-to-final-answer baseline.
- `trace_level/`: MS-SWIFT agent trajectories with supervised tool calls and
  assistant answers; tool responses are input-only.
- `constraint_multitask/`: explicit condition-parsing, tool-planning,
  constraint-judgment, evidence-extraction, and answer-generation examples.
- `manifests/`: counts, checksums, converter version, and exclusions.
- `token_profiles/`: reviewed tokenizer summaries plus ignored per-record
  profiling details.

Generated JSONL files are ignored. Manifests remain reviewable and may be
committed.

Each projection contains `train.jsonl`, `valid.jsonl`, and `test.jsonl`:

- Answer-only rows contain the shared production static policy, original user
  request, and supervised final answer.
- Trace-level rows contain the shared production static policy, bounded
  verified state and recent dialogue, actual PipeClaw tool schemas,
  supervised successful tool calls, masked tool responses, and the supervised
  grounded answer. Runtime-only workspace context is excluded.
- Constraint-aware rows contain the shared production static policy followed
  by their focused instruction and bounded context. They use the task types
  `condition_parsing`, `tool_planning`, `constraint_judgment`,
  `evidence_extraction`, and `answer_generation`. Empty source targets are
  omitted rather than invented.

`manifests/task2_dataset_manifest.json` records converter version, generation
time, tool-schema hash, per-file counts, SHA-256 checksums, and auxiliary task
counts. Run `../scripts/validate_dataset.py` before training or transferring a
release to a remote GPU.

`token_profiles/qwen35_08b_token_profile.json` is created from train and valid
only and records the exact MS-SWIFT-rendered length distribution used for
sequence-length and hardware selection. Its sibling per-record JSONL remains
local because it is a generated audit artifact.

The checked-in 2026-08-22 profile matches the current manifest and covers 5,166
train/validation records. The trace-level maximum is 18,127 tokens, the
constraint-multitask maximum is 17,622, and the answer-only maximum is 2,532.
A context limit of 18,432 therefore preserves every current trace-level record.
