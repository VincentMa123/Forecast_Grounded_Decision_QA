# Task 2 MS-SWIFT Student Distillation Design

## Purpose

Task 2 will distill the finalized Task 1 teacher traces into an external
lightweight decision model for use inside PipeClaw. The student will learn
condition parsing, tool-call planning, verified tool-result interpretation,
pipeline-constraint judgment, evidence extraction, and grounded answer
generation. This work does not retrain PipeFormer and does not distill
PipeClaw itself.

## Approved Technology Choices

- Training framework: MS-SWIFT.
- Local smoke-test model: `Qwen/Qwen3.5-0.8B`.
- Main remote model: `Qwen/Qwen3.5-9B`.
- Adaptation method: 4-bit QLoRA.
- Local execution environment: WSL2/Ubuntu with a dedicated Python
  environment, separate from the existing `pipeclaw` Conda environment.
- Main dataset split: the frozen Task 1 split of 902 train, 124 validation,
  and 114 test records.
- Model-source preference: ModelScope by default, with Hugging Face available
  through MS-SWIFT when required.

LLaMA-Factory remains a fallback framework. It is not part of the primary
experimental path.

## Dataset Authority and Immutability

The authoritative Task 1 inputs are:

```text
pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_train.jsonl
pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_valid.jsonl
pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl
```

Task 2 must not modify these files, the master teacher-trace dataset, sample
IDs, or split assignments. All Task 2 datasets are deterministic derived
artifacts. Every derived record must retain its source `sample_id`,
`scenario_id`, `session_id`, `turn_id`, and split name.

The test split must not be used for training, early stopping, prompt
selection, sequence-length selection, or hyperparameter selection.

## Training Projections

### Answer-only baseline

The answer-only projection uses `user_input` as the input and
`final_answer` as the supervised assistant output. It measures how much the
student gains by learning only answer style and content.

### Trace-level SFT

The trace-level projection represents each verified trajectory using
MS-SWIFT's standard agent format:

```json
{
  "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"tool_name\",\"description\":\"...\",\"parameters\":{}}}]",
  "messages": [
    {"role": "system", "content": "Forecast-grounded pipeline decision instructions and bounded verified state."},
    {"role": "user", "content": "The current user request."},
    {"role": "tool_call", "content": "{\"name\":\"tool_name\",\"arguments\":{}}"},
    {"role": "tool_response", "content": "{\"success\":true}"},
    {"role": "assistant", "content": "The grounded final answer."}
  ]
}
```

MS-SWIFT will compute loss for valid teacher `tool_call` messages and
assistant answers. User, system, and `tool_response` content supplies context
and does not receive training loss.

The projection must preserve the chronological relationship between tool
calls and their matching outputs. A failed or superseded tool call may remain
available as audit provenance, but it must not be trained as a desired action.
The initial trace-level experiment will therefore include only successful,
relevant teacher tool calls as supervised targets.

`state_before` and `recent_turns` are bounded verified context. They must use
the existing compact Task 1 representation rather than restoring full raw
conversation transcripts or replaying historical tool payloads.

### Constraint-aware multitask SFT

The constraint-aware projection creates auditable auxiliary examples for the
five tasks specified in the project statement:

1. Condition parsing: `user_input` to `parsed_task`.
2. Tool planning: current context and available tools to `tool_calls`.
3. Constraint judgment: verified forecast and constraint results to typed
   risk, status, and intervention outputs.
4. Evidence extraction: verified forecast and constraint results to
   structured evidence.
5. Answer generation: user request plus verified evidence and decision state
   to `final_answer`.

These are explicit structured targets, not free-form hidden
chain-of-thought. Empty source fields do not produce synthetic targets.

## Tool and Evidence Rules

- Tool schemas must come from PipeClaw's actual registered tools rather than
  handwritten approximations.
- Registry searches required by the teacher trajectory remain before
  PipeFormer forecast calls.
- Tool-call arguments preserve exact canonical variable identifiers.
- Tool responses are evidence inputs and are never language-model targets.
- Failed tool outputs do not enter verified decision state.
- Binary status setpoints remain limited to `0` and `1`.
- Canonical application-disclosure lines remain unchanged in final answers.
- The converter must preserve Chinese text, signed values, scientific
  notation, and canonical suffixes such as `:SNQ`, `:ST`, `:FR`, `:SP_`, and
  `:SP_out`.

## Repository Structure

Phase 2 will create the following source-controlled scaffold:

```text
pipeclaw/task2_student/
  README.md
  requirements.txt
  data/
    README.md
    answer_only/
    trace_level/
    constraint_multitask/
    manifests/
  configs/
    qwen35_08b_smoke.sh
    qwen35_08b_answer_only.sh
    qwen35_08b_trace.sh
    qwen35_9b_trace.sh
  scripts/
    prepare_dataset.py
    validate_dataset.py
    profile_tokens.py
    evaluate_student.py
    compare_experiments.py
  tests/
    test_prepare_dataset.py
    test_validate_dataset.py
    test_profile_tokens.py
  outputs/
    .gitkeep
```

Generated datasets, model caches, checkpoints, merged weights, predictions,
and large logs will be excluded from Git. Small manifests, configurations,
summary metrics, and documentation may be versioned.

Each module has one responsibility:

- `prepare_dataset.py` builds deterministic derived projections.
- `validate_dataset.py` enforces split, role, tool-call, loss, and identity
  invariants.
- `profile_tokens.py` measures lengths with the selected Qwen3.5 tokenizer.
- `evaluate_student.py` runs Task 2 capability metrics on saved predictions.
- `compare_experiments.py` produces aligned comparisons across baselines.

## Dataset Validation

Dataset generation fails closed if any of the following occurs:

- The authoritative counts are not 902/124/114.
- A source or derived sample ID is missing or duplicated within a projection.
- A record changes split.
- A tool call lacks its matching output.
- A supervised tool call is failed, irrelevant, or superseded.
- A tool name is absent from the PipeClaw tool registry.
- A tool response is configured to receive loss.
- A required final answer is missing.
- A record contains invalid JSON.
- Canonical identifiers or application-disclosure lines change.
- Test records appear in a training or validation artifact.

The generation manifest records source and output checksums, record counts,
projection names, converter version, and creation time.

## Token-Length Policy

Before model training, all derived datasets will be tokenized with the exact
Qwen3.5 processor used by MS-SWIFT. The profile will report minimum, median,
95th percentile, 99th percentile, and maximum lengths, plus coverage at
1,024, 2,048, 4,096, 8,192, and 16,384 tokens.

No record may be silently truncated. A configured training limit must either
fit a record or produce an explicit exclusion/splitting decision recorded in
the manifest. Tool evidence and final answers may not be separated in a way
that removes their grounding relationship.

## Local Smoke Test

The first GPU run uses `Qwen/Qwen3.5-0.8B`, 4-bit QLoRA, 32 training records,
8 validation records, a 1,024-token maximum length, batch size 1, gradient
checkpointing, and 20 optimizer steps.

The smoke test succeeds only if:

- CUDA is visible from WSL2.
- The selected MS-SWIFT and dependency versions are recorded.
- Dataset preprocessing reports nonzero supervised tokens.
- Training loss is finite and nonzero.
- A checkpoint and LoRA adapter are saved.
- Training resumes from the checkpoint.
- Adapter inference produces parseable tool calls or a final answer.

Smoke-test quality is not used as evidence for the final 9B model's
engineering accuracy.

## Main Experiments

The main comparison consists of:

1. Untuned Qwen3.5 student baseline.
2. Answer-only SFT.
3. Trace-level SFT.
4. Constraint-aware trace-level SFT.
5. Teacher-model upper bound.

Qwen3.5-0.8B provides a local engineering baseline. Qwen3.5-9B is the main
remote student model. The same converter, manifests, metric definitions, and
split assignments apply to both.

## Evaluation

Task 2 reports:

- Condition-parsing accuracy.
- Tool-selection and tool-argument accuracy.
- Registry-before-forecast compliance.
- Binary-setpoint validity.
- Constraint-judgment and risk-classification accuracy.
- Human-intervention accuracy.
- Evidence consistency and numerical-reference consistency.
- Dispatch-recommendation feasibility and priority consistency.
- Canonical variable preservation.
- Structured-output and JSON validity.
- Unsupported numerical or causal claims.
- Hallucination rate.
- Final-answer completeness.
- Latency, GPU memory, training time, and inference cost.

Validation data selects configurations. The frozen test split is evaluated
after configuration selection.

## Remote Reproducibility and Security

The remote package contains only Task 2 source, derived training and
validation data, configurations, environment locks, and manifests. It must
not contain `backend/.env`, API keys, private operational data, historical
repair staging, or unrelated repository artifacts.

Model caches, checkpoints, adapters, and logs live on persistent remote
storage. Checkpoints are saved frequently and all final artifacts are copied
off the GPU provider before its storage is terminated.

## Phase 2 Acceptance Criteria

Phase 2 is complete when the directory scaffold exists, generated artifacts
are safely ignored, each planned module has a documented responsibility, and
the scaffold does not modify Task 1 data or unrelated PipeFormer work.
Dataset conversion, validation logic, training, and evaluation are later
implementation phases and must be independently tested before use.
