# Teacher-Trace Repair Handoff — 2026-08-04

> **Latest-status rule:** Section 0 is the final Task 1 state, Section 9 is the
> Task 2 local preparation state, Section 10 is the remote GPU benchmark and
> cost model, and **Section 11 is the current state of the codebase**
> (unified evaluation and rollout refactor). Later sections supersede earlier
> ones wherever they conflict on counts, commands, module paths, or pending
> work; the earlier sections are retained only as an audit trail. Where
> Section 11 contradicts an earlier section on where code lives or how a score
> is computed, Section 11 wins. Section 10 remains authoritative for GPU cost,
> memory, and training-configuration decisions.

## 0. Final continuation update — 2026-07-29

### Final dataset outcome

The regeneration, human review, approved merge, deterministic migration, and
final evaluation cycle is complete.

Current authoritative invariants:

```text
Master records:                    1,140
Unique sample IDs:                 1,140
Native evaluation pass:            1,140
Task 1 evaluation pass:            1,140
Needs review:                           0
Registry-contract regeneration targets: 0
Compact SFT records:               1,140
Train / valid / test:          902 / 124 / 114
Largest compact SFT record:       34,422 characters
SFT character cap:                35,000 characters
```

Every master record contains `state_before` and `recent_turns`. The quality
and statistics workbooks both verify with zero formula/error cells. No further
scenario regeneration or merge is required for the current master.

The deterministic 285-record release spot-check has been completed by the
user. It does not need to be repeated unless the source master, split
assignments, or derived release changes.

### What was completed

#### Regeneration and merge

- All PipeFormer scenarios selected by the repair workflow were regenerated or
  deterministically repaired from successful stored evidence.
- Generated attempts were reviewed before approval.
- Approved scenario attempts were merged without changing the 1,140-record
  sample-ID set or original split assignments.
- The final master, sessions, compact SFT splits, evaluation JSONL, annotations,
  schema, quality report, and statistics report were synchronized.
- Dynamic `registry-contract` discovery now reports zero trajectory defects.
- Existing good scenarios were not regenerated merely because evaluator or
  memory code changed.

#### Canonical application disclosure

- Successful forecast execution now produces exact machine-generated clauses:

  ```text
  Applied disturbance: T_001:SNQ=-8%
  Applied setpoint: R_006:ST=0
  Application status: R_006:ST=no-op; prior=0; applied=0
  ```

- The finalizer removes incorrect/duplicate canonical lines and prepends the
  verified block without rewriting the remaining natural-language answer.
- Percentage values, binary setpoints, signed zero, decimals, scientific
  notation, and verified no-op applications are normalized deterministically.
- Natural Chinese/English synonyms no longer establish critical application
  disclosure; the validator checks the exact canonical block.

#### Verified bounded memory

- Runtime and teacher generation now use `verified_decision_state_v1`.
- State contains bounded verified CSV/topology evidence, exact registry IDs,
  candidate actions and scalar metrics, policy, applied disturbances,
  unresolved inputs, and source provenance.
- Failed tool calls remain in the audit trace but do not enter verified state.
- Equivalent renamed actions are deduplicated by normalized action fingerprint.
- Case, disturbance, horizon, or action-scope changes invalidate stale
  candidates and policy.
- The active prompt/SFT projection contains one `state_before` snapshot plus at
  most two `recent_turns`; previous raw tool payloads are not replayed.
- `conversation_context` remains audit-only. Full runtime transcripts remain on
  disk for audit/resume rather than being injected wholesale.
- State migration preserves typed decision fields such as `missing_metrics`, so
  an evidence-insufficient policy cannot silently become a default ranking.

#### Grounding, ranking, and answer compaction

- Forecast execution requires successful relevant registry searches for the
  disturbance and all candidate action variables.
- Binary status variables are validated by registry semantics and accept only
  setpoints `0` or `1`.
- `set_decision_policy` converts LLM-extracted user priorities into a validated
  typed objective list; the deterministic ranker evaluates verified metrics.
- Multi-turn candidate comparisons use verified state rather than relying on
  long conversational recall.
- Complete candidate comparison evidence uses a cardinality-aware Chinese
  answer budget: 750 characters through three candidates, then 100 additional
  characters per extra candidate, capped at 1,200. Single-forecast Chinese
  answers remain limited to 500 characters.
- Canonical variable suffixes such as `:SNQ`, `:ST`, `:FR`, `:SP_`, and
  `:SP_out` are preserved.

#### Evaluator and deterministic repair

- Prior verified CSV source files in typed history now satisfy later evidence
  requests.
- Explicit disclaimers such as `不代表其是瓶颈` are not misclassified as
  unsupported operational claims.
- Tool execution evidence is separated from answer-format quality, preventing
  failed-format cascades.
- Derived counts, ratios, signed values, and scientific notation are supported
  deterministically while ranking/list ordinals are ignored.
- Comparison validation uses every verified current/prior candidate and applies
  the correct answer budget.
- Added fail-closed command:

  ```powershell
  python -m scripts.repair_teacher_trace --repair-current-quality
  ```

  It repairs evidence-complete quality defects, rebuilds verified memory,
  preflights all compact SFT records, and commits master/sessions/splits
  transactionally only when every expected record is exportable.
- `--list-regeneration-targets` now prints a separate quality/SFT projection, so
  `target_count=0` cannot be confused with the compact split count.

Before the final deterministic pass:

```text
Direct quality exclusions:       16
Dependent context exclusions:    33
Compact SFT records:           1,091
```

After the final pass:

```text
Direct quality exclusions:        0
Dependent context exclusions:     0
Compact SFT records:           1,140
```

#### Reporting code organization

The historical `evaluator/task1.py` mixed evaluation and workbook generation.
It was split by responsibility:

```text
pipeclaw/backend/evaluator/teacher_trace_audit.py
  record checks, schema, sampling, evidence counting, and statistics

pipeclaw/backend/reporting/teacher_trace_quality_report.py
  schema-file output, audit splits, quality workbook, workbook verification

pipeclaw/backend/reporting/statistics_report.py
pipeclaw/backend/reporting/pipeformer_audit_report.py
pipeclaw/backend/reporting/workbook_style.py
```

`evaluate_teacher_trace.py` now composes
`TeacherTraceQualityAuditor` with `TeacherTraceQualityReportWriter`. The
relocated repair utilities remain intentionally under
`pipeclaw/backend/scripts/`.

### Final verification performed

The final full evaluation reported:

```text
Native pass:                       1,140 / 1,140
Task 1 pass:                       1,140 / 1,140
Average/minimum/maximum score:     100 / 100 / 100
Quality issue counts:              {}
Quality workbook errors:           0
Statistics workbook errors:        0
Compact split counts:              902 / 124 / 114
```

Additional verification:

```text
Reporting split tests:             3 passed
Reporting/repair module imports:   passed
Modified Python module compilation: passed
Stale evaluator.task1 references:  0
```

### Current commands

From:

```powershell
cd C:\Users\NIGGABALLS\Documents\Forecast_Grounded_Decision_QA\pipeclaw\backend
conda activate pipeclaw
```

Evaluate the current master after any future dataset/evaluator change:

```powershell
python -X utf8 evaluate_teacher_trace.py
```

Inspect future trajectory defects:

```powershell
python -X utf8 -m scripts.repair_teacher_trace `
  --list-regeneration-targets `
  --target-profile registry-contract
```

Run the fail-closed deterministic quality/memory migration when needed:

```powershell
python -X utf8 -m scripts.repair_teacher_trace --repair-current-quality
```

Do not run regeneration, merge, or evaluation again merely for reassurance.
Rerun them only after relevant source data, master records, evaluator logic, or
generation logic changes.

### Next project steps

1. **Optional release signoff**
   - Complete or record the disposition of the 285 deterministic manual
     spot-check rows if the deliverable requires formal human signoff.
   - No dataset repair is required to perform this administrative review.

2. **Create a clean project checkpoint**
   - Review the dirty worktree carefully.
   - Do not use `git add .`; staging and generated directories contain extensive
     historical artifacts.
   - Stage source refactors and intended final deliverables explicitly.
   - Preserve unrelated PipeFormer work.

3. **Freeze the dataset release**
   - Record the final master checksum and dataset version.
   - Preserve the 1,140 sample IDs and the 902/124/114 split assignment.
   - Archive the final quality summary and workbooks with the dataset release.
   - Keep repair staging and full traces only as audit evidence; they are no
     longer inputs to the released compact SFT dataset.

4. **Begin student-model distillation**
   - Train first on the finalized compact SFT splits.
   - Evaluate tool planning, registry-before-forecast behavior, binary setpoint
     handling, evidence extraction, constraint judgment, candidate ranking, and
     canonical disclosure separately.
   - Compare 7B and 14B behavior using the unchanged validation/test splits.
   - Track exact-variable preservation and unsupported numerical/causal claims
     as dedicated error categories.

5. **Future maintenance workflow**
   - For answer-format or memory-schema changes, use deterministic migration;
     do not regenerate good tool trajectories.
   - Regenerate only newly discovered trajectory defects such as unauthorized
     registry use, failed forecasts, invalid arguments, unverified application,
     or incomplete verification.
   - After any approved merge, rerun the evaluator once and require all master,
     session, split, annotation, and report invariants to remain synchronized.

## Historical continuation update — 2026-07-27

### Current outcome

The latest staging summary contains:

- 21 staged scenarios;
- 15 scenarios with `automatic_pass=true`, all still pending human review;
- 6 unresolved scenarios;
- no newly approved or merged scenarios from this continuation;
- no live API calls made while implementing the latest code fixes.

Current unresolved scenarios:

```text
Pipeline_Full_Life_Cycle_Test_Dataset-v4:
  scenario_pipeformer_dispatch_010   real failed calls; regenerate after current fixes
  scenario_pipeformer_dispatch_012   HTTP 402 / insufficient balance

Pipeline_Full_Life_Cycle_Test_Dataset-v7:
  scenario_pipeformer_dispatch_014   HTTP 402 / insufficient balance
  scenario_pipeformer_dispatch_015   HTTP 402 / insufficient balance
  scenario_pipeformer_prediction_012 HTTP 402 / insufficient balance
  scenario_pipeformer_prediction_015 HTTP 402 / insufficient balance
```

The other 15 staged scenarios are automatic-pass candidates, not approved
records. They still require turn-by-turn human review before merge.

### Work completed since the 2026-07-26 handoff

#### Registry and forecast authorization

- Every forecast now requires relevant successful registry evidence for the
  exact disturbance and every candidate action variable.
- Exact-ID and meaningful normalized searches can authorize the disturbance;
  zero-match and broad role-only searches cannot invent a regional mapping.
- Candidate actions must be returned as controllable inputs.
- Registry and forecast tools remain visible; no regex routing or wording-based
  tool hiding was added.
- Registry search now supports deterministic `offset` pagination and returns
  `next_offset` when another page exists.

#### OpenClaw forecast applicability

- Future-looking wording alone no longer forces PipeFormer.
- A forecast requires a canonical disturbance, valid operating case, and
  relevant registry evidence.
- When those inputs are unavailable, the agent must give a bounded qualitative
  answer from verified CSV/topology evidence and name the missing forecast
  inputs.
- Unverified history summaries are excluded; verified summaries may be reused.
- Decimal aggregation guidance now requires safe string serialization.

#### Binary-state correctness

- Registry semantics, not only the variable suffix, determine variable type.
- Binary `:ST` disturbances and actions require setpoint `0` or `1`.
- Binary percentages and non-binary setpoints are rejected.
- A binary disturbance requires top-level `disturbance_setpoint`.
- Deterministic answers render `R_006:ST=0` or `R_006:ST=1`, never `None%`.

#### Decision policy and scalable ranking

- Added `set_decision_policy` so the LLM converts current user wording into a
  typed, source-grounded objective list.
- Every objective requires its own exact contiguous `source_excerpt`.
- `METRIC_CATALOG` is the central extensible registry for ranking metrics,
  direction, label, unit, and extraction path.
- Ranking applies hard constraints first, then lexicographic ordered
  objectives; no scenario-specific keyword rule is used by the deterministic
  ranker.
- Equivalent candidate actions are deduplicated by canonical action signature
  even if the LLM renames the candidate.
- Exact objective ties are retained as tie groups. Stable candidate ID is only
  a deterministic presentation tie-break, not fabricated engineering
  preference.

#### Multi-turn grounding and answer generation

- `DecisionTraceState` carries verified candidate results, disturbances, and
  decision policy across turns.
- `GroundingContractBuilder` can rank prior verified candidates when the user
  supplies priorities in a later turn.
- The next LLM turn now receives a compact verified candidate-state projection
  containing every candidate ID, canonical action, constraint evidence, and
  comparison metrics while excluding unused audit payloads.
- Policy-only follow-ups are instructed to reuse unchanged verified forecasts.
  A new forecast is needed only when case, disturbance, horizon, or action
  changes.
- Canonical IDs must retain suffixes such as `:SNQ`, `:ST`, `:FR`, `:SP_`, and
  `:SP_out`.
- Ordinary Chinese/mixed forecast answers use a 500-character target;
  multi-candidate comparisons use 650 characters; English uses 160 words.
- Deterministic compaction preserves candidates, actions, rankings, essential
  audit evidence, and assumption disclosure.
- Scientific notation such as `4.3e-05` is accepted and rendered deterministically.

#### Regeneration usability

- `repair_teacher_trace.py` supports `--workers N` using bounded threads for
  independent scenario subprocesses.
- Default `--workers 1` remains the safest option for a single scenario or
  constrained GPU/API resources.
- Progress logs identify `[current/total] START`, `DONE`, or `FAILED`, scenario
  ID, attempt, elapsed time, automatic-pass state, and manifest path.
- Failed attempts clean stale attempt artifacts before writing new output.

### Latest `v4 dispatch_010` audit and fix

The newly generated five-record trace is **not mergeable**. Its final
turn-3 ranking is grounded and correct, but the trajectory contains real tool
failures:

1. Turn 1 called `search_pipeformer_registry(offset=30)` before pagination was
   supported.
2. Turn 3 passed policy metrics
   `flow.max_abs_supply_demand_gap` and `energy.delta_vs_baseline` as
   PipeFormer `output_state_variables`, causing four
   `unresolved_task_vocabulary` failures before successful retries.

It also exposed two evaluator bugs:

1. Native scoring counted the explicit baseline as another candidate, so
   `2 candidates + 1 baseline` and `3 candidates + 1 baseline` failed candidate
   count, checkpoint, disturbance, and horizon checks.
2. Staging used “does this output contain grounding evidence?” as “did this
   tool fail?”, so successful zero-result or locator-only calls could be
   reported as failed.

Long-term fixes implemented:

- registry pagination is now a supported validated interface;
- decision-policy metric IDs are blocked before forecast execution using
  `METRIC_CATALOG`, with the internal correction omitted from SFT traces;
- native scoring matches parsed candidates to forecast outputs by
  `tool_call_id`, excluding baselines and superseded retries;
- staging reports only `ToolEvidenceState.EXECUTION_FAILED` as a failed tool;
- missing/locator evidence remains unusable for grounding;
- verified prior candidates are compactly exposed and reused on policy-only
  turns.

The existing staged trace still contains recorded failures and must not be
deterministically upgraded. It requires one fresh regeneration.

### Latest local verification

No live scenario generation was run. Focused verification:

```powershell
python -m unittest tests.test_pipeformer_registry_tool agent.test_forecast_execution_contract tests.test_pipeline_data_contract tests.test_parallel_regeneration tests.test_repair_progress_logging
python -m unittest pipeclaw.backend.tests.test_native pipeclaw.backend.tests.test_v4_teacher_trace_export
python -m compileall -q pipeclaw/backend/agent pipeclaw/backend/evaluator pipeclaw/backend/pipeline pipeclaw/backend/generate_teacher_trace.py pipeclaw/backend/repair_teacher_trace.py
```

Results:

```text
31 tests passed
71 tests passed
102 focused tests total
Python compilation: exit code 0
git diff --check: no whitespace errors; only existing LF/CRLF warnings
```

Design and implementation records:

```text
docs/superpowers/specs/2026-07-27-dispatch-trace-reliability-design.md
docs/superpowers/plans/2026-07-27-dispatch-trace-reliability.md
```

### Exact next steps

#### 1. Regenerate only `v4 dispatch_010`

In `repair_teacher_trace.py`, leave only this target uncommented:

```python
(
    "Pipeline_Full_Life_Cycle_Test_Dataset-v4",
    "scenario_pipeformer_dispatch_010",
),
```

Then run without `--resume`:

```powershell
conda activate pipeclaw
cd C:\Users\NIGGABALLS\Documents\Forecast_Grounded_Decision_QA\pipeclaw\backend
python -X utf8 repair_teacher_trace.py --stage-regeneration --attempts 1 --workers 1
```

For several independent scenarios, `--workers 2` is supported, but a larger
value can increase API throttling and memory/GPU pressure.

#### 2. Review the regenerated scenario

Require all of the following:

- `command_exit_code=0`;
- `automatic_pass=true`;
- all five expected sample IDs present;
- no recorded failed tools or unresolved vocabulary;
- baseline excluded from candidate count but retained in the trace;
- turn 2 contains all three candidates and audit categories;
- turn 3 calls `set_decision_policy`, reuses verified candidate forecasts when
  unchanged, and ranks `candidate_1 > candidate_3 > candidate_2` from the
  stored balance-then-energy policy;
- every action keeps its complete canonical variable suffix;
- no failed trajectory is approved.

#### 3. Restore provider balance and regenerate the five API-blocked scenarios

The current quota-blocked list is:

```text
v4 scenario_pipeformer_dispatch_012
v7 scenario_pipeformer_dispatch_014
v7 scenario_pipeformer_dispatch_015
v7 scenario_pipeformer_prediction_012
v7 scenario_pipeformer_prediction_015
```

Regenerate them one at a time without `--resume`, or use `--workers 2` only if
the provider and local resources can sustain parallel requests.

#### 4. Human review, approval, and merge

- Review every turn in each changed scenario, including turns that passed
  before scenario-level regeneration.
- Create an approval entry only for an automatic-pass, human-reviewed attempt.
- Ensure each approved scenario is present in `REGENERATION_TARGETS`.
- Run `--merge-approved`; never approve the staging directory wholesale.
- No unchanged PipeFormer scenario needs regeneration solely because grounding
  and evaluator code changed. Regenerate only unresolved or substantively
  affected scenarios.

#### 5. Preserve annotations and regenerate deliverables

Before report regeneration:

```powershell
python -X utf8 repair_teacher_trace.py --export-annotations
```

After approved merges:

```powershell
python -X utf8 evaluate_teacher_trace.py
```

Final invariant checklist:

- exactly 1,140 master records;
- exactly 1,140 unique sample IDs;
- original split assignment for every sample ID;
- no failed tool trajectory marked pass;
- repaired records reset to pending human review;
- untouched reviewer decisions preserved;
- no unexpected SFT compaction-grounding or oversized-record warnings;
- master JSON, JSONL, sessions, splits, annotations, and reports synchronized.

## 1. Project goal and current status

Repository:

```text
C:\Users\NIGGABALLS\Documents\Forecast_Grounded_Decision_QA
```

Primary working directory:

```text
C:\Users\NIGGABALLS\Documents\Forecast_Grounded_Decision_QA\pipeclaw\backend
```

Goal: produce a grounded, human-reviewed teacher-trace dataset suitable for 7B/14B distillation, with valid train/valid/test splits, reliable native and Task 1 evaluation, polished Excel reports, and safe repair/regeneration tooling.

Current master dataset:

- 1,140 records.
- 1,140 unique sample IDs.
- Splits preserved:
  - train: 902
  - valid: 124
  - test: 114
- Stored quality flags:
  - pass: 1,078
  - needs_review: 62
- Current compact SFT split counts:
  - train: 614
  - valid: 103
  - test: 87

The master dataset has already received the original deterministic repairs, including OpenClaw 006 evidence recovery and five answer repairs. The later eight-scenario repairs described below are still staged and have not been merged into the master.

Regeneration staging status:

- 33 scenario manifests exist.
- 13 scenarios have `automatic_pass=true`.
- 20 scenarios still fail automatic checks and must not be merged.
- The eight recently repaired scenarios contain 80 records, all synchronized and passing native and Task 1 evaluation.

Reviewer annotations:

- 420 workbook annotation rows exported to JSONL.
- 379 marked pass.
- 41 failed rows representing 37 unique failed sample IDs.
- Sheets:
  - Manual Spot Check: 285 rows
  - Needs Review: 135 rows

No Git commit was created. The worktree is heavily dirty and contains both task changes and unrelated PipeFormer work. Do not reset or discard existing changes.

## 2. Important decisions and assumptions

### Tool routing

Do not add regex-based routing between OpenClaw and PipeFormer.

- Keep the existing tool set available.
- Let the LLM choose tools.
- Judge the selected tool by relevance and grounding.
- A successful but irrelevant PipeFormer forecast is still unacceptable.
- OpenClaw scenarios 021–024 must not be rejected solely because they are OpenClaw scenarios.

### Repair strategy

Use a hybrid approach:

- Deterministic wording/answer repair only when stored tool execution and evidence are correct.
- Regenerate scenarios with failed tools, wrong forecast inputs, parsing failures, incomplete verification, irrelevant tools, or incorrect answers.
- Stage regeneration before touching the master.
- Human-review every changed turn before merging.
- Never force an unresolved record to `pass`.

### Regeneration controls

`REGENERATION_TARGETS` in `repair_teacher_trace.py` acts as the active scenario subset.

- It is acceptable to comment out scenarios.
- Commented targets produce an `unassigned_warning`, not a `ValueError`.
- Uncommented scenarios are executed.
- The validator still confirms that the workbook contains 37 unique failed records.

Preferred command:

```powershell
python -X utf8 repair_teacher_trace.py --stage-regeneration --attempts 1
```

The user prefers manually rerunning failed scenarios rather than automatically performing three attempts.

`--resume` semantics:

- It skips a scenario if these three staged files exist and are non-empty:
  - `teacher_trace.json`
  - `teacher_trace.jsonl`
  - `teacher_trace_sessions.jsonl`
- It does not inspect `automatic_pass`.
- Do not use `--resume` when intentionally overwriting a failed staged scenario.
- Commented scenarios must be uncommented again before merging because `_merge_approved` rejects approvals for scenarios absent from `REGENERATION_TARGETS`.

### Evidence and SFT eligibility

Task 1 quality pass and SFT eligibility are different:

- Task 1 pass means the record passes schema, numerical, rule, dispatch, and native checks.
- SFT eligibility additionally requires:
  - no SFT exclusion reason;
  - all prior context turns passing;
  - grounding surviving compaction;
  - compact record size within the cap.

The SFT cap is currently 35,000 characters. Increasing it only affects oversized records. It does not fix:

```text
Skipping SFT record with evidence removed by compaction
```

That warning means the compacted SFT record no longer contains enough evidence for the answer’s claims.

### Dataset invariants

Preserve:

- exactly 1,140 master records;
- the complete sample-ID set;
- existing split assignments;
- tool trajectories for deterministic answer-only repairs;
- reviewer annotations for untouched records.

Reset repaired records to pending human review when regenerating reports.

## 3. Files created or modified

The following inventory covers teacher-trace/evaluator work. Unrelated dirty `pipeFormer/` changes were not part of this repair work and must be preserved.

### Core repair and evaluation files

Created:

```text
pipeclaw/backend/repair_teacher_trace.py
pipeclaw/backend/agent/llm_provider.py
pipeclaw/backend/evaluator/csv_evidence.py
pipeclaw/backend/evaluator/deterministic_repairs.py
pipeclaw/backend/evaluator/grounding_contract.py
pipeclaw/backend/evaluator/pipeformer_audit_report.py
pipeclaw/backend/evaluator/pipeline_scope.py
pipeclaw/backend/evaluator/reviewer_annotations.py
pipeclaw/backend/evaluator/statistics_report.py
pipeclaw/backend/evaluator/task1.py
pipeclaw/backend/evaluator/tool_evidence.py
pipeclaw/backend/evaluator/topology_evidence.py
pipeclaw/backend/evaluator/workbook_style.py
```

Created tests:

```text
pipeclaw/backend/evaluator/test_csv_evidence.py
pipeclaw/backend/evaluator/test_deterministic_repairs.py
pipeclaw/backend/evaluator/test_grounding_contract.py
pipeclaw/backend/evaluator/test_regeneration_targets.py
pipeclaw/backend/evaluator/test_reviewer_annotations.py
pipeclaw/backend/evaluator/test_sft_grounding.py
pipeclaw/backend/pipeline/test_repair_prerequisites.py
```

Modified:

```text
pipeclaw/backend/evaluate_teacher_trace.py
pipeclaw/backend/generate_teacher_trace.py
pipeclaw/backend/evaluator/README.md
pipeclaw/backend/evaluator/scorer.py
pipeclaw/backend/evaluator/teacher_quality.py
pipeclaw/backend/pipeline/teacher_trace_store.py
pipeclaw/backend/requirements.txt
pipeclaw/how_to_run.md
```

### Agent and tool runtime

Modified:

```text
pipeclaw/backend/agent/orchestrator.py
pipeclaw/backend/agent/prompt_builder.py
pipeclaw/backend/agent/tools/README.md
pipeclaw/backend/agent/tools/pipeformer_tools.py
pipeclaw/backend/agent/tools/workspace_tools.py
pipeclaw/backend/executor/runner.py
pipeclaw/backend/main.py
```

### PipeFormer execution and constraints

Modified:

```text
pipeclaw/backend/pipeline/constraints/common.py
pipeclaw/backend/pipeline/engineering_constraints.py
pipeclaw/backend/pipeline/pipeformer_inference.py
pipeclaw/backend/pipeline/pipeformer_tool_runtime.py
pipeclaw/backend/pipeline/scenario_preflight.py
pipeclaw/backend/pipeline/variable_registry.py
pipeclaw/backend/pipeclaw_data/Pipeline_Full_Life_Cycle_Test_Dataset-v4.json
```

### Topology fixtures and mappings

Modified:

```text
pipeclaw/backend/pipeline_data/consumer_station.csv
pipeclaw/backend/pipeline_data/synthetic_fixture_manifest.json
```

Created node fixtures:

```text
pipeclaw/backend/pipeline_data/node_flow/20190401_node.csv
pipeclaw/backend/pipeline_data/node_flow/20190413_node.csv
pipeclaw/backend/pipeline_data/node_flow/20190425_node.csv
pipeclaw/backend/pipeline_data/node_flow/20190510_node.csv
```

Created pipeline fixtures:

```text
pipeclaw/backend/pipeline_data/pipeline_flow/20190401_pipeline.csv
pipeclaw/backend/pipeline_data/pipeline_flow/20190413_pipeline.csv
pipeclaw/backend/pipeline_data/pipeline_flow/20190425_pipeline.csv
pipeclaw/backend/pipeline_data/pipeline_flow/20190510_pipeline.csv
```

Modified node fixtures for synthetic-data consistency:

```text
20190114, 20190211, 20190223, 20190307, 20190319,
20191001, 20191013, 20191025, 20191029,
20191101–20191107, 20191118, 20191128–20191130,
20191201–20191207
```

All are under:

```text
pipeclaw/backend/pipeline_data/node_flow/
```

Modified pipeline fixtures:

```text
20190102, 20190105, 20190108, 20190111,
20190114–20190116, 20190119–20190121,
20190124–20190130,
20190202–20190208, 20190211, 20190223,
20190307, 20190319,
20191201–20191207
```

All are under:

```text
pipeclaw/backend/pipeline_data/pipeline_flow/
```

### Generated master and splits

Generated or updated:

```text
pipeclaw/backend/generated_teacher_traces/teacher_trace.json
pipeclaw/backend/generated_teacher_traces/teacher_trace.jsonl
pipeclaw/backend/generated_teacher_traces/teacher_trace_sessions.jsonl
pipeclaw/backend/generated_teacher_traces/scenario_preflight.json
pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_train.jsonl
pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_valid.jsonl
pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl
```

### Reports and annotations

Created or updated:

```text
pipeclaw/backend/generated_teacher_traces/task1_deliverables/manual_quality_decisions.jsonl
pipeclaw/backend/generated_teacher_traces/task1_deliverables/pipeformer_additional_audit.xlsx
pipeclaw/backend/generated_teacher_traces/task1_deliverables/teacher_trace_quality_report.xlsx
pipeclaw/backend/generated_teacher_traces/task1_deliverables/teacher_trace_statistics.xlsx
```

Additional audit outputs:

```text
pipeclaw/backend/generated_teacher_traces/pipeformer_audit_eval/quality_evaluation.jsonl
pipeclaw/backend/generated_teacher_traces/pipeformer_audit_eval/quality_evaluation_summary.json
pipeclaw/backend/generated_teacher_traces/pipeformer_audit_eval/teacher_trace_quality_report.xlsx
pipeclaw/backend/generated_teacher_traces/pipeformer_audit_eval/teacher_trace_schema.json
pipeclaw/backend/generated_teacher_traces/pipeformer_audit_eval/teacher_trace_statistics.xlsx
```

### Regeneration staging

Created:

```text
pipeclaw/backend/generated_teacher_traces/repair_staging/staging_summary.json
```

There are 33 scenario directories under:

```text
repair_staging/pipeclaw_dataset_v2/
repair_staging/Pipeline_Full_Life_Cycle_Test_Dataset-v4/
repair_staging/Pipeline_Full_Life_Cycle_Test_Dataset-v7/
```

Each generated `attempt_01` directory contains, where applicable:

```text
teacher_trace.json
teacher_trace.jsonl
teacher_trace_sessions.jsonl
scenario_preflight.json
quality_evaluation.jsonl
attempt_manifest.json
splits/teacher_trace_train.jsonl
splits/teacher_trace_valid.jsonl
splits/teacher_trace_test.jsonl
```

Raw runtime traces also exist under:

```text
pipeclaw/backend/.openclaw/tt_runs/
```

## 4. Important code changes

### Repair orchestration

`repair_teacher_trace.py` now provides:

- staged scenario regeneration;
- up to three attempts, though `--attempts 1` is preferred;
- `--resume` based only on non-empty generated files;
- staged native and Task 1 evaluation;
- tool-failure, parsing, verification, session, and sample-ID checks;
- safe scenario-level merge with `TeacherTraceStore.replace_scenario`;
- annotation export and repaired-ID reset handling;
- non-blocking assignment validation when targets are commented;
- automatic staged deterministic repairs during `_evaluate_attempt`.

### Deterministic staged repairs

Eight scenarios were classified as repairable without another LLM call:

```text
pipeclaw_dataset_v2:
  scenario_openclaw_013
  scenario_openclaw_014
  scenario_openclaw_015
  scenario_openclaw_016

Pipeline_Full_Life_Cycle_Test_Dataset-v4:
  scenario_pipeformer_dispatch_004
  scenario_pipeformer_dispatch_005
  scenario_pipeformer_dispatch_008
  scenario_pipeformer_dispatch_013
```

Repairs applied:

- OpenClaw 013 handoff total: `1640.497`
- OpenClaw 014 handoff total: `1339.885`
- OpenClaw 015 handoff total: `1275.738`
- OpenClaw 016 handoff total: `1638.528`
- Dispatch 004:
  - canonical action identifiers;
  - `LLM暂设` accepted as an explicit assumption disclosure.
- Dispatch 005:
  - includes all three real candidates;
  - correct ranking: `C_002:SP_out +5% > C_001:SP_ +5% > C_014:SP_out +8%`.
- Dispatch 008:
  - first answer shortened below the PipeFormer answer limit;
  - energy totals and deltas retained in compact repair evidence.
- Dispatch 013:
  - `LLM暂设` no longer triggers a false positive.

These eight scenarios now have:

- `automatic_pass=true`;
- all 80 records passing native and Task 1 evaluation;
- synchronized JSON, JSONL, session, split, evaluation, manifest, and staging-summary artifacts.

They remain staged and require human approval before merge.

### CSV and numerical grounding

The evaluator now:

- rebuilds CSV evidence from successful stored `read_file` calls;
- keeps up to 12 relevant evidence rows;
- supports single-group totals such as consumer-station total consumption;
- records zero and nonzero row counts;
- preserves verified derived counts and totals;
- ignores digits in dates, compact dates, filenames, years, list numbering, and candidate identifiers;
- handles signed percentages;
- carries trusted conversation evidence forward;
- preserves supporting numerical values through SFT compaction.

### Topology evidence

Added:

- missing fixtures for 20190401, 20190413, 20190425, and 20190510;
- canonical mappings:
  - `通州南分输站 → 通州南站`
  - `湘潭分输站 → 湘潭站`
  - `乌鲁木齐压气站 → 乌鲁木齐站`
- synthetic, non-disruptive topology support for `阳曲压气站`;
- clearer topology failure messages;
- preflight failure when required topology files or mappings are missing.

### Binary status variables

Binary `:ST` handling now requires:

- only setpoints `0` or `1`;
- no percentage change for status variables;
- explicit setpoint verification in boundary evidence;
- invalid percentage or non-binary values rejected before forecast execution.

### Reports

Excel generation now includes:

- consistent workbook styling;
- formatted status cells;
- adjusted widths, filters, freeze panes, and charts;
- separate Task 1 quality and statistics workbooks;
- durable annotation export/import;
- reviewer decisions preserved for untouched records.

### SFT size

`SFT_MAX_RECORD_CHARS` is now:

```python
35_000
```

This is a character cap, not a tokenizer context guarantee.

## 5. Commands run and results

Environment:

```powershell
conda activate pipeclaw
cd C:\Users\NIGGABALLS\Documents\Forecast_Grounded_Decision_QA\pipeclaw\backend
```

Main evaluation/report command:

```powershell
python -X utf8 evaluate_teacher_trace.py
```

This command generates evaluation JSONL, summary/schema outputs, quality/statistics workbooks, and compact splits using default paths.

Regeneration command:

```powershell
python -X utf8 repair_teacher_trace.py --stage-regeneration --attempts 1
```

Previous three-attempt command:

```powershell
python -X utf8 repair_teacher_trace.py --stage-regeneration --attempts 3
```

The three-attempt run was slow and difficult to observe because subprocess output is captured until a scenario completes.

Resume command:

```powershell
python -X utf8 repair_teacher_trace.py --stage-regeneration --attempts 1 --resume
```

It successfully skips any scenario with non-empty staged JSON, JSONL, and session output, regardless of `automatic_pass`.

Deterministic master repair was previously applied. Current master confirms:

- five answer repairs contain `repair_provenance`;
- OpenClaw 006 contains deterministic CSV evidence recovery;
- all six affected master records currently pass.

Recent verification:

```powershell
python -X utf8 -m unittest evaluator.test_csv_evidence evaluator.test_sft_grounding evaluator.test_deterministic_repairs
```

Result:

```text
Ran 16 tests in 18.930s
OK
```

Compilation:

```powershell
python -X utf8 -m py_compile evaluator\csv_evidence.py evaluator\teacher_quality.py evaluator\deterministic_repairs.py repair_teacher_trace.py
```

Result: exit code 0.

Custom staged consistency verification:

- 8 scenarios verified.
- 80 records verified.
- JSON and JSONL identical.
- Session answers and quality fields match records.
- All sessions complete.
- Split sample-ID sets match scenario record sets.
- All native and Task 1 evaluations pass.
- All eight manifests report `automatic_pass=true`.

Commented-target validator simulation:

- 37 unique failed reviewer records found.
- A one-target subset assigned 7 records to active repair paths.
- 30 records reported as unassigned warnings.
- No `ValueError` occurred.

## 6. Errors encountered and fixes

### Commenting regeneration targets raised `ValueError`

Original error:

```text
ValueError: Failed reviewer records are not assigned to a repair path
```

Cause: assignment validation treated every failed workbook record as requiring an active target.

Fix: commented scenarios are now reported under:

```text
unassigned_record_count
unassigned_sample_ids
unassigned_warning
```

They no longer block staging.

### “Evidence removed by compaction” warnings persisted after raising the cap

Cause: this is a grounding failure, not a size failure.

Fixes:

- compacted tool calls and arguments retained;
- CSV evidence rebuilt;
- dates and filenames excluded from numeric claims;
- verified derived totals/counts preserved;
- up to 12 query-relevant rows retained.

### Oversized SFT record

One record was approximately 31,012 characters under the old cap.

Fix: cap increased to 35,000 characters. This does not guarantee compatibility with every 7B/14B tokenizer context.

### Topology execution failures

Error:

```json
{
  "error": "Topology evidence could not be built from the requested files and target."
}
```

Causes:

- missing same-day node/pipeline fixtures;
- consumer supply-point names differed from topology node names;
- `阳曲压气站` had no reachable topology.

Fix: added fixtures, aliases, synthetic connection, preflight checks, and more specific errors.

### False-positive numerical claims

Examples included:

- sums derived from CSV rows;
- dates such as `2019-04-01`;
- filenames such as `20190401_consumer.csv`;
- nonzero counts;
- signed percentages;
- numbers repeated from trusted earlier turns.

Fix: expanded numerical normalization, derived evidence, and trusted-context handling.

### `LLM暂设` incorrectly treated as undisclosed

Cause: the assumption-disclosure matcher recognized `假设`, `暂定`, and similar phrases but not `暂设`.

Fix: `LLM暂设` and `暂设` are now accepted.

### Canonical variable false positives

Some answers used device shorthand such as `C_001` after first naming `C_001:SP_`.

Fix: deterministic repair expands affected references to canonical action identifiers instead of weakening the evaluator broadly.

### Silent or apparently stalled regeneration

Cause:

```python
subprocess.run(..., capture_output=True)
```

in `repair_teacher_trace.py` captures child logs until the scenario finishes.

The inner orchestrator now logs request start, response, tool calls, and failures, but the repair parent still hides those logs while the subprocess is running.

This remains a usability issue. Workarounds:

- run the individual generation command directly;
- inspect `.openclaw/tt_runs`;
- or modify the repair subprocess to stream output.

### API `402 Usage limit reached`

Several v7 scenarios failed with:

```text
HTTP 402 Payment Required
Usage limit reached
```

These are not dataset-quality failures. Retry after the provider quota resets or use a valid provider account/configuration.

## 7. Remaining tasks and recommended next steps

### A. Human-review the 13 automatic-pass scenarios

Currently passing:

```text
pipeclaw_dataset_v2:
  scenario_openclaw_013
  scenario_openclaw_014
  scenario_openclaw_015
  scenario_openclaw_016
  scenario_openclaw_023

Pipeline_Full_Life_Cycle_Test_Dataset-v4:
  scenario_pipeformer_dispatch_003
  scenario_pipeformer_dispatch_004
  scenario_pipeformer_dispatch_005
  scenario_pipeformer_dispatch_008
  scenario_pipeformer_dispatch_011
  scenario_pipeformer_dispatch_013
  scenario_pipeformer_prediction_015

Pipeline_Full_Life_Cycle_Test_Dataset-v7:
  scenario_pipeformer_dispatch_008
```

Review every turn, including previously passing turns changed by scenario-level regeneration.

### B. Regenerate the 20 failing scenarios

Do not merge these:

```text
pipeclaw_dataset_v2:
  scenario_openclaw_021
  scenario_openclaw_022
  scenario_openclaw_024

Pipeline_Full_Life_Cycle_Test_Dataset-v4:
  scenario_pipeformer_dispatch_007
  scenario_pipeformer_dispatch_009
  scenario_pipeformer_dispatch_010
  scenario_pipeformer_dispatch_012
  scenario_pipeformer_dispatch_014
  scenario_pipeformer_dispatch_015
  scenario_pipeformer_dispatch_019
  scenario_pipeformer_prediction_012

Pipeline_Full_Life_Cycle_Test_Dataset-v7:
  scenario_pipeformer_dispatch_004
  scenario_pipeformer_dispatch_011
  scenario_pipeformer_dispatch_012
  scenario_pipeformer_dispatch_014
  scenario_pipeformer_dispatch_015
  scenario_pipeformer_dispatch_020
  scenario_pipeformer_prediction_012
  scenario_pipeformer_prediction_015
  scenario_pipeformer_prediction_016
```

Most fail because of failed forecast calls. Seven v7 manifests contain `generation_command_failed`, largely associated with provider/API failures.

For a subset:

1. Comment all unwanted entries in `REGENERATION_TARGETS`.
2. Do not use `--resume`.
3. Run:

```powershell
python -X utf8 repair_teacher_trace.py --stage-regeneration --attempts 1
```

### C. Create an approval file after human review

Example:

```json
{
  "approved": [
    {
      "dataset_source": "pipeclaw_dataset_v2",
      "scenario_id": "scenario_openclaw_013",
      "attempt": 1
    }
  ]
}
```

Before merging, ensure every approved scenario is uncommented in `REGENERATION_TARGETS`.

Merge:

```powershell
python -X utf8 repair_teacher_trace.py --merge-approved approved_repair_attempts.json
```

Never approve the entire staging directory automatically.

### D. Preserve annotations before regenerating reports

Run:

```powershell
python -X utf8 repair_teacher_trace.py --export-annotations
```

Then regenerate evaluation, workbooks, and splits:

```powershell
python -X utf8 evaluate_teacher_trace.py
```

Confirm afterward:

- 1,140 master records;
- 1,140 unique IDs;
- unchanged split assignments;
- repaired records reset to pending human review;
- untouched reviewer decisions preserved;
- no unexpected SFT compaction or oversized warnings.

### E. Consider live subprocess logging

The repair script still uses `capture_output=True`. A future improvement could stream stdout/stderr while keeping a bounded tail in the manifest. This should not change repair logic.

## 8. User instructions and preferences

- Use the `pipeclaw` Conda environment.
- Do not implement regex-based PipeFormer/OpenClaw tool routing.
- Let the LLM choose from the existing tools.
- Do not hide PipeFormer tools based on wording.
- Use deterministic edits for minor, evidence-complete wording problems.
- Regenerate wrong-tool, failed-tool, parsing, verification, or materially wrong-answer scenarios.
- Stage before merging.
- Preserve all sample IDs and split assignments.
- Human-review regenerated turns before acceptance.
- Do not force persistent LLM mistakes to `pass`.
- Prefer `--attempts 1` and manual reruns over automatic three-attempt retry.
- Allow commenting entries in `REGENERATION_TARGETS` to choose a subset.
- `--resume` should skip generated outputs regardless of `automatic_pass`.
- Do not use `automatic_pass` as a substitute for human review.
- Keep the 35,000-character SFT cap for now.
- Preserve Chinese text and canonical variable identifiers.
- Improve workbook readability and styling.
- Avoid unrelated or unnecessarily broad changes.
- The user is comfortable running regeneration personally and wants clear commands and visible failure information.

## 9. Task 2 student-model distillation — current continuation state

### Task transition

Task 1 and its 285-record release signoff are complete. Do not reopen or repeat
that signoff unless the Task 1 release data changes. The active work is Task 2:
distilling the verified teacher data into a smaller MS-SWIFT student model.

The authoritative released split remains:

```text
Train / valid / test: 902 / 124 / 114
```

The original split assignments are preserved across every Task 2 projection;
test records are not included in training.

### Completed Task 2 preparation

- Phase 2 created the `pipeclaw/task2_student/` scaffold.
- Phase 3 created three comparison projections:
  - `answer_only`
  - `trace_level`
  - `constraint_multitask`
- Phase 4 added fail-closed validation for IDs, split counts and leakage, tool
  call/response pairing, registered tools, loss masking, final answers, JSON,
  and canonical identifiers. A manifest records counts and checksums.
- The production static policy is shared by answer-only, trace-level, and
  constraint-multitask examples. Dynamic workspace state and trace metadata are
  not embedded into every system prompt.
- Phase 5 profiled all 4,199 train/valid projected records with the
  Qwen3.5-0.8B tokenizer.
- Phase 6 prepared the MS-SWIFT environment.
- Phase 7 now has a corrected local smoke-test configuration. **Superseded by
  Section 10:** the 10-step run and the resume run have both since been executed
  and passed, and a 20-step Qwen3.5-9B benchmark has run on a rented GPU.

Current exact token-profile facts:

```text
All projections:      4,199 records; median 2,840; p95 9,675;
                      p99 12,029; maximum 13,792
Answer only:          1,026 records; minimum 1,430; maximum 2,008
Trace level:          1,026 records; maximum 12,136
Constraint multitask: 2,147 records; maximum 13,792
Coverage:             0% at 1,024; 25.91% at 2,048;
                      58.89% at 4,096; 90.33% at 8,192;
                      100% at 16,384
```

The full lossless experiments therefore require `max_length=16384`. The local
answer-only smoke test uses `max_length=2048`, because every answer-only record
fits intact and no current record fits the obsolete 1,024-token proposal.

### Local environment observed on 2026-07-30

```text
GPU:          NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB
PyTorch:      2.13.0+cu130
torchvision:  0.28.0+cu130
CUDA runtime: 13.0
MS-SWIFT:     4.4.2
Transformers: 5.12.1
datasets:     4.8.4
bitsandbytes: 0.50.0
```

`torch.cuda.is_available()` returned true. `swift --version` is not a valid
MS-SWIFT 4.4.2 command; use `python -m pip show ms-swift`.

The forced CUDA wheel installation had left two real dependency conflicts:

```text
datasets 4.8.4 requires fsspec[http]<=2026.2.0
gradio 5.50.0 requires pillow<12.0
```

Both are now pinned away. `pipeclaw/task2_student/environment.yml` is the single
install file for Task 2: it carries the cu130 PyTorch index and explicit
`+cu130` pins, the pinned MS-SWIFT stack, and those two transitive upper bounds.

```bash
conda env create -f pipeclaw/task2_student/environment.yml
conda activate task2-ms-swift
python -m pip check
```

WSL has no Conda installed, so the local environment is still the venv
`~/.venvs/task2-ms-swift`; the "Without Conda" section of
`pipeclaw/task2_student/README.md` lists the same pins as plain pip commands.
Only `causal-conv1d`, `flash-attn`, and `deepspeed` stay outside the file: PyPI
ships them source-only, so they need `--no-build-isolation` and nvcc, and they
are required only by the multi-GPU remote configs.

A clean `python -m pip check` is required before starting the smoke test. The
local environment now reports `No broken requirements found` (fsspec 2026.2.0,
Pillow 11.3.0).

### Phase 7 files and commands

The two configurations are:

```text
pipeclaw/task2_student/configs/qwen35_08b_smoke_step10.yaml
pipeclaw/task2_student/configs/qwen35_08b_smoke_resume_step20.yaml
```

They use Qwen3.5-0.8B, 4-bit NF4 QLoRA, 32 answer-only training records, 8
validation records, batch size 1, gradient checkpointing, LoRA rank 8 / alpha
32, deterministic seeds, and fail-closed deletion of any unexpectedly
oversized record. The second configuration restores the complete trainer state
from the predictable `checkpoint-10` path; it is not an adapter-only restart.

Run steps 1 through 10 from the repository root:

```bash
swift sft \
  pipeclaw/task2_student/configs/qwen35_08b_smoke_step10.yaml
```

Inspect `checkpoint-10`, then resume through step 20:

```bash
swift sft \
  pipeclaw/task2_student/configs/qwen35_08b_smoke_resume_step20.yaml
```

Phase 7 is accepted only when the logs show finite nonzero loss and nonzero
supervised tokens, `checkpoint-10` contains adapter plus trainer state, the
second run resumes at step 10 rather than restarting, and `checkpoint-20`
finishes at global step 20. Run one adapter inference afterward and verify that
its answer is parseable.

### Model and remote-GPU disposition

Keep Qwen3.5 for the current experiment. Current MS-SWIFT documentation lists
DeepSeek-V4, but the official DeepSeek-V4 checkpoints are not small student
models: even V4-Flash is a very large mixture-of-experts model. It is not a
replacement for Qwen3.5-0.8B locally or Qwen3.5-9B remotely.

For the lossless Qwen3.5-9B `max_length=16384` experiment, use a single 80 GB
GPU such as A800 80 GB as the safe default on AutoDL. A 48 GB card is a
reasonable benchmark candidate when substantially cheaper. A 32 GB RTX 5090
may run a rank-64 configuration with batch size 1 and gradient checkpointing,
but must be tested with the exact 16K configuration before a full rental. Avoid
planning the rank-64/rank-128 16K experiment around a 24 GB 4090/3090.

Increasing `lora_rank` increases adapter, gradient, and optimizer memory.
Increasing `lora_alpha` alone changes the LoRA scaling and does not materially
increase parameter count or VRAM. Rank 64 has eight times the LoRA parameters
of rank 8; rank 128 has sixteen times as many. Base weights and long-context
activations still dominate the total.

> **Superseded by Section 10 on GPU sizing.** The 80 GB recommendation was a
> pre-measurement safety margin. The 16K rank-32 QLoRA configuration has now
> been measured at **18.47 GiB peak**, so a 32 GB card is sufficient with room
> for rank 64 (~19.9 GiB projected). 80 GB is only needed if quantization is
> dropped *and* batch size is raised.

### Immediate next steps

> **Superseded by Section 10.** Steps 1–5 below are done. Retained for the
> acceptance criteria they record. See Section 10 for the live next steps.

1. Reconcile the two pip conflicts and obtain a clean `python -m pip check`.
2. Run the 10-step smoke configuration and inspect `checkpoint-10`.
3. Run the resume configuration and confirm it continues to step 20.
4. Preserve both logs and record peak VRAM, elapsed time, processed tokens, and
   measured tokens per second.
5. Run one inference from the adapter and validate the structured answer.
6. Only after Phase 7 passes, add full answer-only, trace-level, and
   constraint-multitask experiment configurations.
7. Benchmark the exact 16K Qwen3.5-9B rank planned for the final run for 20
   steps on the candidate AutoDL GPU. Use measured tokens per second to estimate
   rental time and cost before launching full SFT.

## 10. Remote-GPU benchmark and cost model — current state (2026-08-01)

### Where the project stands

Task 1 is closed. Task 2 local preparation (Sections 9) is complete: Phase 7's
10-step and resume-to-20 smoke runs both passed on the RTX 3050, and a real
20-step Qwen3.5-9B benchmark has now been executed on a rented AutoDL GPU. The
project is no longer blocked on infrastructure. It is blocked on two decisions:
whether to keep 4-bit quantization, and which projections to spend GPU hours on.

### Phase 7 local smoke test — passed

`pipeclaw/task2_student/outputs/qwen35_08b_answer_only_smoke/` contains
`checkpoint-10` and `checkpoint-20` plus `logging.jsonl`. Measured on the
RTX 3050 4 GB with Qwen3.5-0.8B, answer-only, 4-bit NF4 QLoRA, LoRA rank 8:

```text
Trainable:      5.4113M of 609.5898M params (0.8877%)
Peak memory:    2.2 GiB
train_runtime:  24.18 s for 20 steps
train_loss:     1.685    best eval_loss: 3.510
```

Loss is finite and nonzero, supervised tokens are nonzero, and the resume run
continued from step 10 rather than restarting. The absolute loss values are
meaningless at this scale — the run exists only to prove the plumbing.

### Remote 20-step benchmark on Qwen3.5-9B — completed

Artifacts are committed under
`pipeclaw/task2_student/outputs/qwen35_08b_benchmark_step20/`
(directory name says `08b`; the run is the **9B** model — the name is a
leftover and is worth renaming). Contents: `args.json`, `logging.jsonl`,
`adapter_model.safetensors`, `additional_config.json`, and the exact yaml the
server ran.

**Critical provenance fact.** The server copy of the config was edited before
launch and diverges from the local
`configs/qwen35_9b_remote_benchmark_step20.yaml`. As actually run:

```text
packing:                        false      (local config says true)
gradient_accumulation_steps:    32         (local config says 8)
NPROC_PER_NODE / deepspeed:     commented out
global_world_size:              1
```

So every measured number below is **unpacked, single-GPU** throughput at an
effective batch of 1 x 32 x 1 = 32. Do not describe these numbers as reflecting
the packed 4-GPU config.

Measured configuration and results:

```text
model:            Qwen/Qwen3.5-9B      template: qwen3_5
quant:            bnb 4-bit nf4, double quant, compute dtype bfloat16
torch_dtype:      bfloat16             attn_impl: flash_attn
max_length:       16384                truncation_strategy: delete
LoRA:             rank 32, alpha 64, dropout 0.05, all-linear
loss_scale:       default+ignore_empty_think
add_non_thinking_prefix: true          lazy_tokenize: true
gradient_checkpointing: true           use_liger_kernel: false
dataset:          data/trace_level/train.jsonl (+ valid.jsonl)

Trainable:        86.5567M of 6037.1182M params (1.4337%)
Peak memory:      18.47 GiB   (trace: 17.37 -> 17.82 -> 18.47)
train_runtime:    3448.5 s for 20 steps  = 172.4 s/step
num_input_tokens_seen: 3,755,956 over 20 steps = ~187,798 tokens/step
Throughput:       ~1,089 tokens/s
train_loss:       0.9721      token_acc rising 0.774 -> 0.832
eval_loss:        1.3576      eval_token_acc: 0.8166 (171.4 s for 124 recs)
total_flos:       1.91e17
```

Two things worth carrying forward. First, 6,037 M reported params for a 9 B
model is consistent with 4-bit quantized linears plus bf16 embeddings and
`lm_head` over the 248,320-token vocabulary. Second, 187,798 tokens/step over an
effective batch of 32 is ~5,869 tokens per record, which matches trace-level's
median of 5,023 — confirming records are **not** padded to `max_length` (that
would be 524,288 tokens/step). This matters for the packing argument below.

### Cost model

AutoDL pricing confirmed on the rented instance: **¥1.58 per GPU-hour**, billed
only while the instance is running, plus a **20 GB data-disk expansion at ¥0.17
per calendar day**, billed continuously **including while shut down**. The
expansion requires shutdown to apply, is data-disk only, and **cannot be
shrunk**. Releasing the instance stops the disk charge but erases the disk;
15 consecutive shut-down days triggers automatic release and irrecoverable
erase. Verify the exact release-day proration on the 费用明细 page — this could
not be confirmed against official docs (the pricing doc URL 404s).

Projected full-run costs from the measured 172.4 s/step, at 902 train records:

| Projection | Train recs | Steps | Median tokens | Time | Cost |
|---|---|---|---|---|---|
| answer_only | 902 | 140 | 1,506 | ~7h09m | ~¥11.30 |
| trace_level | 902 | 140 | 5,023 | ~7h09m | ~¥11.30 |
| constraint_multitask | 1,878 | 290 | 2,840 | ~14h50m | ~¥23.43 |

Programme total including setup (3.5 h), the benchmark itself (1.0 h), the full
ablation grid (87.4 h), DPO (7.0 h), and 14 days of disk: **~¥158.6 for the full
grid, ~¥88.5 for a staged ablation**, realistically **¥118–150**.

**Known inflation:** the `answer_only` 7h09m figure is derived from step count,
not from token throughput. Answer-only records have a median of 1,506 tokens
versus trace-level's 5,023, so its true wall-clock is far lower. Re-deriving it
on tokens/s would pull the programme total below ¥120. Not yet done.

**Staged ablation** means running the cheap, most-informative arms first
(answer_only and trace_level at one rank) and only funding the remaining grid
cells if the first results are ambiguous — as opposed to launching all arms
up front.

### Technical findings established from installed MS-SWIFT source

Read under `~/miniconda3/envs/task2-ms-swift/lib/python3.12/site-packages/swift/`.

**Packing.** `dataset/packing.py:40-42` uses best-fit-decreasing bin packing via
`binpacking` 2.0.1 (confirmed installed); `packing_strategy: sequential`
(`:19-39`) is order-preserving next-fit. `packing_length` defaults to
`max_length` (`arguments/base_args/base_args.py:198-199`). `packing: true`
forces `padding_free = True` and requires `attn_impl: flash_attn`.
`template/base.py:675-696` concatenates records and **resets `position_ids` per
record**, and `cu_seq_lens` keeps flash-attention block-diagonal, so records do
not attend across each other. Packing is **incompatible with `lazy_tokenize`**
(`base_args.py:137-138` raises) — and `lazy_tokenize` is auto-set to true for
Qwen3.5 because it registers as multimodal (`:130-132`).

**Packing's real benefit is fewer forward/backward passes, not less padding.**
At `per_device_train_batch_size: 1` there is no padding to eliminate at all, as
the token count above proves. The gain is pass-count reduction: roughly 10.7x
for answer_only, 2.8x for trace_level, 4.1x for constraint_multitask, plus more
deterministic memory. **The danger:** step counts collapse proportionally
(trace_level 140 -> ~50, answer_only 140 -> ~10), so if packing is enabled
`gradient_accumulation_steps` must be cut proportionally (~8 for trace_level,
~4 for answer_only) or the run ends after a handful of optimizer steps.

**Effective batch.** `B = per_device_batch x grad_accum x world_size`. Only
`world_size` gives real wall-clock speedup (it is also the all-reduce divisor);
`grad_accum` is sequential on one device; raising per-device batch needs VRAM and
forfeits the `logits_to_keep` optimization, which gates on `labels.shape[0] == 1`
(`trainers/mixin.py:1184`).

**LoRA rank and memory.** The rank-32 adapter is 346,302,176 bytes for
86.5567 M trainable params = exactly 4 bytes each, i.e. **fp32 LoRA weights**.
Weights + gradients + two Adam moments is therefore ~1.4 GB at rank 32 and
~2.8 GB at rank 64, putting projected peak at ~19.9 GiB against the measured
18.47 GiB. **32 GB is comfortably enough for rank 64.** Rank appears only in the
parameter term; activations scale with batch x seq_len x hidden x layers, not
with rank. (An earlier estimate of "+0.5 GiB" for rank 64 was wrong by ~3x.)

**Quantization.** 4-bit NF4 is block-wise scaling with 16 levels placed at
normal-distribution quantiles; double quant compresses the scales (~0.4 GB);
`bnb_4bit_compute_dtype: bfloat16` dequantizes on the fly for every matmul,
which costs time. `arguments/base_args/quant_args.py:45-47` returns `None` the
moment `quant_method` is `None`, so the `bnb_4bit_*` keys become dead config if
`quant_method` is removed. Freezing the base weights prevents error from
*compounding*, but does **not** remove the distortion: it is present in every
forward pass, gradients are computed through the degraded model, and the adapter
**co-adapts to the quantization** — which is a portability trap when merging the
adapter back into bf16 weights. 4-bit is also what forces `zero2`: ZeRO-3 cannot
shard bnb-quantized parameters.

**bf16 feasibility without quantization.** Quantized linears ~3.5 GB + bf16
embeddings and `lm_head` ~4 GB = ~7.5 GB measured; full bf16 weights are ~18 GB,
a delta of ~10 GB, giving **~29 GiB of a 32 GB card — marginal but not
impossible** at 16,384 tokens, batch 1, gradient checkpointing. It must be
benchmarked before committing to a full rental.

### Open decision: drop quantization?

The MS-SWIFT Qwen3.5 best-practice page
(`https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html`)
uses **no quantization anywhere**. That is a real signal, but its dense SFT
recipe is not comparable to ours: 4B model, `max_length` 2048, 4 GPUs at
"4 * 20GiB", batch 4 / accum 1, LoRA r8 alpha32, zero2, 1 epoch,
`group_by_length: true` (with `--packing true` offered as the alternative),
`add_non_thinking_prefix: true`, `loss_scale ignore_empty_think`. Its only
reduced-precision path is FP8 under Megatron. Our run is a 9B model at 16,384
tokens, which is a different memory regime entirely.

Dropping quantization is viable and would remove the co-adaptation and merge
risks, and would unlock ZeRO-3 on the 4-GPU path. It is marginal on a single
32 GB card. To remove it, delete these five lines and keep `torch_dtype:
bfloat16` and `attn_impl: flash_attn`:

```yaml
quant_method: bnb
quant_bits: 4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
```

They are at `configs/qwen35_9b.yaml:55-59` and
`configs/qwen35_9b_remote_benchmark_step20.yaml:27-31`. **Not yet applied** —
awaiting a decision.

### Remote server operational notes

- **Outbound TCP:443 is blocked by default** on the AutoDL instance while ICMP
  is not, so `ping github.com` succeeds at ~102 ms / 0% loss while `git clone`
  hangs for 130 s and then reports "Connection timed out". These filters are
  independent; a successful ping proves nothing about HTTPS. Fix with
  `source /etc/network_turbo`, which must be re-run in every new shell — it sets
  only `http_proxy`/`https_proxy`, **does not cover SSH**, and does not carry
  into a pre-existing screen/tmux session.
- Private-repo auth is solved (SSH). GitHub port 22 is typically blocked from
  mainland hosts, so route SSH over `ssh.github.com:443` via `~/.ssh/config`.
  A deploy key authenticates **only** the `git@github.com:` transport; an
  `https://` remote ignores it and will always prompt for a username.
  **Delete the deploy key from GitHub when the instance is released** — the
  private key lives on a rented machine.
- Never embed a PAT in a clone URL; it persists in `.git/config`. If HTTPS is
  needed, prefer `credential.helper 'cache --timeout=3600'` over `store`.
- **Training data must be `scp`'d to the server.** `.gitignore:30-32` excludes
  `*.jsonl`, so a fresh clone has no `data/*/train.jsonl`.
- **The system disk filled up.** Redirect `output_dir` to
  `/root/autodl-tmp/outputs/...` before any long run; a run that fills the
  system disk mid-training loses the checkpoint.
- A stale clone may still occupy `/root/Forecast_Grounded_Decision_QA`.
- The `flash-attn` wheel referenced in `environment.yml` is a third-party build
  (mjun0812), not an official release. Noted as a supply-chain consideration.

### Next steps

**A. Settle the precision decision with one ¥1.58 benchmark.** A single 20-step
run answers three open questions at once: whether bf16 fits in 32 GB, what
rank 64 actually costs in memory, and what packing does to throughput. This is
the highest-value hour of GPU time available and should come before any full run.

**B. Fix the config divergence before launching anything.** The local configs and
the server's edited copy disagree on `packing` and `gradient_accumulation_steps`.
Decide one way, commit it, and push — otherwise a fresh clone on the server gets
configs that were never the ones benchmarked. Specifically:

1. Decide `packing`. If enabled, cut `gradient_accumulation_steps` to ~8
   (trace_level) or ~4 (answer_only), and note that packing cannot coexist with
   `lazy_tokenize`.
2. Decide quantization (the five lines above).
3. If going bf16 on 4 GPUs, change `deepspeed: zero2` to `zero3`.
4. Redirect `output_dir` to `/root/autodl-tmp/outputs/...`.
5. Consider `use_liger_kernel: true` — currently false, and it is a
   free throughput/memory win on Qwen-family models.
6. Rename `outputs/qwen35_08b_benchmark_step20/` to `..._9b_...`; the model in
   it is the 9B.
7. Fix stale comments: `configs/qwen35_9b.yaml:70-71` claims "3 epochs is
   roughly 84 optimizer steps" while `num_train_epochs: 5` gives ~140, and
   line 4 still references the old filename
   `qwen35_9b_remote_trace_level.yaml`.

**C. Re-derive the `answer_only` cost estimate on token throughput** rather than
step count. Expected to pull the programme total below ¥120.

**D. Then run the actual experiments.** Full answer-only, trace-level, and
constraint-multitask SFT at the chosen precision and rank, staged rather than as
a full grid, preserving logs and recording peak VRAM, elapsed time, tokens seen,
and tokens per second for each arm.

**E. Evaluation and preference optimization** on the held-out 114 test records
come after SFT. Test records must stay out of training, as they have throughout.

### Standing constraints (unchanged)

- Task 1's frozen 1,140-record release (902 / 124 / 114) must **not** be
  regenerated.
- Do not use `git add .`; stage explicit paths. Generated directories contain
  extensive historical artifacts.
- Do not reset or discard existing changes. Preserve unrelated PipeFormer work.
- Do not implement regex-based PipeFormer/OpenClaw tool routing, and do not hide
  PipeFormer tools based on wording.
- Preserve Chinese text and canonical variable identifiers.
- Keep the 35,000-character SFT cap for now.
- Nothing is staged or committed unless explicitly requested.

---

## 11. Unified evaluation and rollout refactor — current state (2026-08-04)

Plan: `docs/superpowers/plans/2026-08-04-unified-evaluation-rollout-refactor.md`.
This is a **structural refactor only**. No training data, adapter, checkpoint,
model configuration, or teacher-trace content changed, and no difficulty-scoring
mechanism was added — that idea was evaluated and deliberately dropped as
unnecessary for this project.

### What the codebase looked like before

Two evaluators existed side by side: `pipeclaw/backend/evaluator/` for teacher
traces and `pipeclaw/task2_student/evaluator/` for student rollouts. They had
their own score formulas, their own metric implementations, and their own
notion of what "passed" meant. The backend evaluator additionally owned runtime
grounding code used *during* generation, and the Task 2 evaluator interleaved
model/tool execution with scoring, so a rollout could not be produced without
loading scoring code and vice versa.

### Completed migration

**One evaluation package.** `pipeclaw/backend/evaluator/` is now the sole
evaluation package. `pipeclaw/task2_student/evaluator/` is deleted.
`pipeclaw/tests/evaluation/test_layout.py` fails if it reappears.

**Runtime grounding moved out of the evaluator** into
`pipeclaw/backend/grounding/` (contract, decision policy, decision-trace state,
answer limits, and CSV/pipeline-scope/tool/topology evidence).
`teacher_trace_audit.py` and `reviewer_annotations.py` moved to
`pipeclaw/backend/reporting/`. The canonical PipeFormer projection
(`project_pipeformer_output` / `compact_pipeformer_output`) moved out of
`backend/task1/generate_teacher_trace.py` into
`grounding/pipeformer_projection.py`; teacher generation and rollout scenarios
now import the same functions instead of keeping fallback copies, and the
dynamic import of `task1.generate_teacher_trace` is gone.

**Execution separated from evaluation.** Task 2 model/tool execution lives in
`pipeclaw/task2_student/rollout/`: `models.py`, `prompting.py`, `tools.py`,
`runner.py`, `scenarios.py`, `swift_generator.py`, `suite.py`. The first four
contain no evaluator import, so a rollout can be generated with no scoring code
loaded. `suite.py` is the single seam where the evaluator enters the rollout
package — keep it that way. `rollout/__init__.py` re-exports only the
hardware-free core; `scenarios`, `swift_generator`, and `suite` must be imported
directly, and torch/MS-SWIFT imports inside `suite.py` are deferred into
`_build_runner` so importing the package does not pull in CUDA.

**API changes** (all call sites updated):

```text
run_case(...)                    -> RolloutRunner(gen, dispatcher, policy=...).run(case, RolloutConfig(...)).to_dict()
evaluate_rollout(source, roll)   -> evaluate(roll, profile=EvaluationProfile.AUTONOMOUS_ROLLOUT, reference=source)
aggregate_results(results)       -> summarize(reports)
```

`scripts/evaluate_autonomous.py` is now a thin CLI over `rollout.suite`.

### Schema-v2 output contract

`EVALUATION_SCHEMA_VERSION = "pipeclaw_evaluation_v2"`, stamped on every report
and on `summary.json`. One score formula exists in the repository, in
`evaluator/engine.py`:

```python
included    = [m for m in metrics if m.applicable and m.included_in_score]
denominator = sum(m.weight for m in included)
overall_score = (
    round(100.0 * sum(m.weight for m in included if m.passed) / denominator, 6)
    if denominator else None
)
hard_gate_passed = not hard_issues and all(m.passed for m in included if m.critical)
```

- **Teacher profile** — hand-tuned per-check weights, `passed` requires
  `hard_gate_passed` *and* `overall_score >= 85.0`.
- **Autonomous profile** — every applicable deliverable metric weighs `1.0`,
  no minimum-score threshold, `passed` = `hard_gate_passed` and a non-null
  score.
- Inapplicable metrics and diagnostics (`tool_recovery`, `portability`,
  `raw_capture_metadata`, `model_loading_metadata`, `hallucination`) never
  enter the denominator.
- **Hallucination is derived, not measured twice.** The per-record
  `hallucination` entry is a copy of `evidence_consistency` tagged
  `derived_from: "evidence_consistency"`, `included_in_score: false`, emitted
  only for the autonomous profile; `summary["hallucination_rate"]` equals
  `evidence_consistency.failure_rate`. The grounding checker runs once.

**Deprecated but still callable** (teacher side, in `evaluator/scorer.py`):
`apply_quality_aliases()` writes the released `quality_flag`, `quality_score`,
`quality_profile`, `quality_failed_checks`, and `quality_issues` fields, and
`NativeTraceEvaluator` forwards `evaluate`/`summarize`/`load`. Both **copy from**
an `EvaluationReport` rather than recomputing anything. New code should use
`evaluate()` and read the report fields directly.

### Preserved commands

Unchanged by this refactor, flags and all:

```powershell
python -m pipeclaw.task2_student.scripts.evaluate_autonomous `
  --source pipeclaw/backend/generated_teacher_traces/splits/teacher_trace_test.jsonl `
  --tool-schema-source pipeclaw/task2_student/data/trace_level/test.jsonl `
  --scenario-type pipeformer `
  --adapters pipeclaw/task2_student/outputs/qwen35_9b_trace_level/checkpoint-20 `
  --output-dir pipeclaw/task2_student/outputs/evaluation/autonomous
```

The `--dry-run` workflow (omit `--adapters`) still writes prompt messages and
tool schemas without loading model weights, and still shows only `system,user`
messages. `python pipeclaw/backend/evaluate_teacher_trace.py` and the
resumable OpenClaw regeneration driver are also unchanged.

### Verification evidence (2026-08-04)

```powershell
python -m unittest discover -s pipeclaw/tests -p "test_*.py"     # exit 0 — 121/121 OK
python -m compileall -q pipeclaw/backend pipeclaw/task2_student  # exit 0
python -m pipeclaw.task2_student.scripts.evaluate_autonomous --help  # exit 0
```

Stale-import and duplicate-score searches over `pipeclaw/**/*.py`:

- `task2_student.evaluator`, `evaluator.grounding_contract`,
  `evaluator.decision_policy` — **no matches.**
- `def evaluate_rollout` / `def aggregate_results` — only in
  `pipeclaw/tests/evaluation/legacy_shape.py` (test support, see below).
- `overall_score =` — only in `evaluator/engine.py`. One score formula.

### Limitations and open items

**1. `pipeclaw/tests/evaluation/legacy_shape.py` is a judgment call.** The
deleted `task2_student/evaluator/oracle_metrics.py` was already a thin facade
translating canonical results into the released legacy metric shape
(`record_pass`/`rate`, details flattened), but roughly twenty tests were written
against that shape and cover behaviour the schema-v2 contract tests do not:
assumption consistency, tool recovery, artifact evidence, and numeric-evidence
rules. Rather than delete the coverage or keep a production module the refactor
exists to remove, the adapter was moved **verbatim** into
`pipeclaw/tests/evaluation/legacy_shape.py` as test support. Nothing in
`pipeclaw/backend` or `pipeclaw/task2_student` imports it. The cleaner endpoint
is to rewrite those tests against `evaluate()`/`summarize()` directly and delete
the shim.

**2. Assumption-aware disturbance scoring was completed, not just moved.** When
a request does not state the disturbance variable's direction or magnitude, the
student is allowed to assume one, and **that assumption does not have to match
the teacher's sampled value.** Three of the four code paths already implemented
this: `task_field_comparison` (`checks/common.py`) excludes assumed fields from
task comparison entirely, `disturbance_was_applied` (`checks/common.py:204`)
prefers the student's predicted value for assumed fields, and
`assumption_consistency` checks the student against *itself* — that its stated
direction/magnitude is valid and matches what its own forecast call executed.

The fourth, `expected_applied_disturbance` (`checks/assumptions.py`), did not:
it was a verbatim extraction of older inline logic that read the teacher's
values unconditionally, so a legitimately-divergent assumption failed
`applied_boundary_conditions` and, through it, the critical
`assumption_consistency` gate. It now takes an optional `assumed_fields`
argument and applies the same precedence rule as `disturbance_was_applied`:
explicit task values win, assumed fields fall back to the student's executed
prediction. `assumption_consistency` passes the teacher-derived assumed-field
set through rather than re-deriving it. The docstring that described this
behaviour was correct; the code now matches it.

Verified by `test_native_scorer_uses_student_values_for_inferred_disturbance`,
which was the one failing test here and now passes:
`pipeclaw/task2_student/tests/test_oracle_metrics.py` is 19/19.

**3. `pipeclaw/task2_student/tests/` is gitignored** (`.gitignore:21`), so those
tests are *not* part of the committed suite and are not covered by the
`discover -s pipeclaw/tests` verification above. Run separately:
`python -m unittest discover -s pipeclaw/task2_student/tests` → 98/101. The
three failures are all `test_scaffold_contract`: missing
`pipeclaw/task2_student/requirements.txt`, missing CUDA requirements file, and
`include_num_input_tokens_seen` asserting `'non_padding'` against the `True`
that the Transformers 5.12 CLI limitation in Section 9 forces. All three are
pre-existing environment/scaffold drift, unrelated to this refactor.

**4. Nothing is staged or committed.** The working tree carries this refactor
alongside unrelated dirty dataset and prompt changes. Stage explicit paths only;
`pipeclaw/backend/evaluator/scorer.py` is missing from the plan's own Step 7
stage list and must be added explicitly, or `apply_quality_aliases` will be left
out of the commit.

**5. `pipeclaw/task2_student/evaluator/` was untracked**, so git could not have
restored it. A copy was taken to
`%TEMP%\pipeclaw_task2_evaluator_backup` before deletion. Delete that backup
once the refactor is committed and reviewed.





