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
