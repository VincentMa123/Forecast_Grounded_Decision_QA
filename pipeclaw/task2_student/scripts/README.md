# Task 2 Programs

This directory is reserved for the focused dataset-preparation, fail-closed
validation, Qwen3.5 token-profiling, student-evaluation, and
experiment-comparison programs specified in the approved design. Programs
must not mutate Task 1 inputs.

- `prepare_dataset.py` creates the three deterministic MS-SWIFT projections
  and checksum manifest from the frozen Task 1 splits.
- `validate_dataset.py` independently verifies the sources, derived records,
  split isolation, tool/loss contracts, and manifest checksums.
- `profile_tokens.py` verifies the manifest-selected train/valid files and
  measures exact untruncated Qwen3.5/MS-SWIFT template lengths without loading
  model weights. It writes a reviewable summary JSON and an ignored per-record
  JSONL audit.
