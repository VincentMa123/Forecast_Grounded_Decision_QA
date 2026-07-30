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

- Answer-only rows contain the original user request and supervised final
  answer.
- Trace-level rows contain the shared production static policy, bounded
  verified state and recent dialogue, actual PipeClaw tool schemas,
  supervised successful tool calls, masked tool responses, and the supervised
  grounded answer. Runtime-only workspace context is excluded.
- Constraint-aware rows use the task types `condition_parsing`,
  `tool_planning`, `constraint_judgment`, `evidence_extraction`, and
  `answer_generation`. Empty source targets are omitted rather than invented.

`manifests/task2_dataset_manifest.json` records converter version, generation
time, tool-schema hash, per-file counts, SHA-256 checksums, and auxiliary task
counts. Run `../scripts/validate_dataset.py` before training or transferring a
release to a remote GPU.

`token_profiles/qwen35_08b_token_profile.json` is created from train and valid
only and records the exact MS-SWIFT-rendered length distribution used for
sequence-length and hardware selection. Its sibling per-record JSONL remains
local because it is a generated audit artifact.

The 2026-07-30 release profile contains 4,199 records with median 1,445, p95
8,828, p99 10,758, and maximum 12,397 tokens. All records fit a 16,384-token
context without truncation; 93.00% fit 8,192.
