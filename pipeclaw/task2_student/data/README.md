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

Generated JSONL files are ignored. Manifests remain reviewable and may be
committed.

Each projection contains `train.jsonl`, `valid.jsonl`, and `test.jsonl`:

- Answer-only rows contain the original user request and supervised final
  answer.
- Trace-level rows contain bounded verified state, actual PipeClaw tool
  schemas, supervised successful tool calls, masked tool responses, and the
  supervised grounded answer.
- Constraint-aware rows use the task types `condition_parsing`,
  `tool_planning`, `constraint_judgment`, `evidence_extraction`, and
  `answer_generation`. Empty source targets are omitted rather than invented.

`manifests/task2_dataset_manifest.json` records converter version, generation
time, tool-schema hash, per-file counts, SHA-256 checksums, and auxiliary task
counts. Run `../scripts/validate_dataset.py` before training or transferring a
release to a remote GPU.
