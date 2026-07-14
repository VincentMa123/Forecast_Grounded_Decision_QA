# PipeFormer v2 Dataset and Evaluation Refactoring TAD

## 1. Overall Objective

This TAD corresponds to the v7 dataset and the v2 evaluator. Its objective is to convert the first 40 PipeFormer scenarios into single-response tasks and implement an automated evaluation loop that fully aligns with the description in the paper.

## 2. Dataset Loading Layer

### 2.1 Input File

The input file is `Pipeline_Full_Life_Cycle_Test_Dataset-v7.json`. Its top level is a list containing 40 scenario dictionaries.

### 2.2 Scenario Object

After loading, each scenario must be converted into an internal `ScenarioRecord` containing at least the following fields:

- `scenario_id`
- `scenario_type`
- `scenario_class_label`
- `scenario_description`
- `session_id`
- `user_input`

### 2.3 Scenario-Type Detection

Do not rely on `scenario_type` to distinguish prediction tasks from dispatch tasks. Determine the task type by parsing `scenario_id`.

- `scenario_pipeformer_prediction_*` is classified as prediction
- `scenario_pipeformer_dispatch_*` is classified as dispatch

### 2.4 Scenario-Metadata Extraction

Because the v7 dataset is intentionally kept simple, the evaluator must extract and cache the following fields from `user_input` during loading and store them in an internal registry.

- `mock_case_id`
- `target_boundary_var`
- `perturbation`
- `forecast_horizon`
- `scenario_kind`
- `scenario_class_label`
- `preference`, for dispatch tasks only
- `field_constraint`, for dispatch tasks only

Implementing a `PipeformerPromptParser` that uses regular expressions and lightweight rules to extract these fields is recommended. The extracted results must be persisted as `scenario_registry_v2.json`, but this file does not need to be included in the public dataset.

## 3. Parameter Registry

### 3.1 Input

The parameter registry is built from `附件2：管道设备参数.zip`. It is recommended that the attachment be parsed and cached once before evaluation begins.

### 3.2 Parsed Output

The internal registry must include at least the following structures.

- `compressor_meta`
- `compressor_curve_envelopes`
- `ball_valve_meta`
- `regulator_meta`
- `pipe_meta`
- `segment_meta`

### 3.3 Compressor Envelopes

Parse the compressor file using the `总体说明` sheet and the individual sheets from `C_001` through `C_023`. Store the following data for each compressor.

- Equipment ID
- Upstream and downstream nodes
- Drive type
- Minimum flow in the characteristic curve
- Maximum flow in the characteristic curve
- The characteristic-curve table itself

The current attachment does not provide a uniform hard pressure limit for all equipment. Therefore, do not construct a fabricated table of pressure limits. Compressor verification must be limited to envelope checks genuinely supported by the attachment.

### 3.4 Valves and Regulators

For ball valves and regulators, store at least the equipment ID, upstream and downstream nodes, and regulator type. The evaluator uses these data to verify equipment existence and boundary-variable validity.

### 3.5 Pipelines and Pipeline Segments

Pipeline and pipeline-segment parameters must initially be used as identity and topology registries. If later code requires additional diagnostic indicators, velocity or pressure-drop proxy quantities may be calculated from diameter and length, but these proxies must not contribute to hard scoring by default.

## 4. Execution-Trace Collection

### 4.1 TraceCollector

`TraceCollector` is responsible for consistently collecting command-execution, file-access, file-modification, and output-directory access events. It is recommended that at least the following files be persisted.

- `trace/run_command.jsonl`
- `trace/file_events.jsonl`
- `trace/output_reads.jsonl`

### 4.2 `run_command` Events

Each event must contain at least:

- `ts`
- `cmd`
- `cwd`
- `exit_code`
- `stdout_path`
- `stderr_path`

### 4.3 `file_events` Events

Each event must contain at least:

- `ts`
- `op`
- `path`
- `sha256_before`
- `sha256_after`
- `size_after`

## 5. Reference-Rerun Layer

### 5.1 ReferenceRunner

`ReferenceRunner` independently reruns the original and modified inputs produced by the model. All reruns must use `python -m real_predict.main`.

### 5.2 Input Reconstruction

Input reconstruction consists of two steps.

First, read the `Boundary.csv`, `batch_jobs_for_skill_1.json`, and `batch_jobs_for_skill_2.json` files actually written by the model.

Second, reconstruct separate run directories for the original and modified versions.

### 5.3 Output Collection

After each rerun, collect the split CSV files under a consistent structure, for example:

- `reference_runs/original/...`
- `reference_runs/modified/...`
- `reference_runs/candidate_A1/...`
- `reference_runs/candidate_A2/...`

### 5.4 CSV Alignment and Comparison

Implement `SplitCsvComparator`. The comparison procedure is as follows.

1. Align by filename
2. Align by `TIME` and column name
3. Perform numerical comparisons on non-time columns
4. Output `max_abs_diff`, `mean_abs_diff`, and `relative_l1_diff`

The predictor is deterministic, so strict equality should be preferred. If CSV text formatting introduces minor floating-point differences, a very small numerical tolerance may be allowed.

## 6. G1 Call Authenticity Implementation

Implement `CallAuthenticityScorer`. The raw score directly counts how many of the seven mandatory checks pass and is normalized using `passed / 7 * 20`. The seven checks are:

- `run_command` appears
- `Boundary.csv` is read or written
- `batch_jobs_for_skill_1.json` is read or written
- `batch_jobs_for_skill_2.json` is read or written
- `python -m real_predict.main` is executed at least twice
- At least one type of split CSV is read
- The final answer explicitly compares modified and original results

## 7. G2 Execution Authenticity Implementation

### 7.1 Dual-Run Completeness

Implement `RunCompletenessChecker`. Prediction tasks require complete original and modified outputs. Dispatch tasks require an original output and at least one candidate action that has been actually rerun. If the model claims to have compared multiple actions, output directories must exist for all corresponding actions.

### 7.2 Reference-Rerun Consistency

Implement `ReferenceMatchScorer`. It reads the model-saved outputs and independently produced reference outputs, then compares all split CSV files at both the file and numerical levels. The recommended division of the 10 points is:

- 5 points for file completeness and schema consistency
- 5 points for numerical consistency

## 8. G3 Verification Authenticity Implementation

### 8.1 Parameter-Bound Auditor

Implement `ParameterBoundAuditor`. It must check at least the following:

- Whether the target boundary variable exists in the actual Boundary schema
- Whether the target equipment exists in the parameter registry
- Whether `FR` is within `[0, 1]`
- Whether `ST` is in the discrete state domain
- Whether `SNQ` and the various pressure setpoints are valid numerical values
- Whether compressor `q_in` or `q_out` falls within the flow range of the corresponding compressor characteristic curve

The current attachment cannot directly support a uniform upper pressure limit for pipelines or nodes. Therefore, do not fabricate such scoring rules.

### 8.2 CSV Evidence Auditor

Implement `EvidenceGroundingAuditor`. It must construct a consolidated summary table from the original and modified outputs. The recommended summary fields for each variable are:

- `delta_last`
- `delta_mean`
- `delta_max_abs`
- `family`, such as pressure, flow, linepack, or power
- `source_file`

Then generate a reference conclusion according to the scenario class.

### 8.3 Reference-Conclusion Generation for Prediction Tasks

Implement `PredictionOutcomeSynthesizer`. It automatically generates the following from the summary table and rule engine:

- `main_consequence`
- `watch_indicators`
- `manual_intervention`
- `constraint_priority`
- `evidence_vars`

This replaces manually written gold-standard text and ensures that conclusions are derived from actual outputs.

### 8.4 Reference-Conclusion Generation for Dispatch Tasks

Implement `DispatchReplayAuditor`. The procedure is as follows.

1. Extract candidate actions from the model answer
2. Actually rerun each candidate action
3. Audit every action against the rules
4. Rerank the actions according to the objective function of the scenario class
5. Use the best action, ranking result, and primary elimination reasons as the reference conclusion

## 9. Dispatch-Action Parsing and Reranking

### 9.1 Action Extraction

Implement `DispatchActionExtractor`. Structured JSON should be supported first, followed by Markdown lists. Normalize the extracted result into the following structure.

```json
{
  "candidate_actions": [
    {
      "action_id": "A1",
      "ops": [
        {"var": "T_002:SNQ", "change": "-5%"}
      ],
      "reported_rank": 1,
      "reported_reason": "...",
      "reported_status": "accept"
    }
  ]
}
```

### 9.2 Objective Functions by Scenario Class

Implement `DispatchUtilityEngine`. Different classes use different ranking objectives, but field constraints must always be processed first.

- D1: hard constraints first, followed by pressure-violation rate and critical-node pressure margin, then energy consumption
- D2: hard constraints first, followed by total violation rate, then energy consumption
- D3: hard constraints first, followed by linepack or system margin, then energy consumption
- D4: hard constraints first, followed by pressure-fluctuation amplitude and signs of single-point instability, then energy consumption
- D5: hard constraints first, followed by elimination of high velocity or potential limit violations, then energy consumption

### 9.3 Interpretation of Field Constraints

Convert field constraints in the prompt into machine-executable rules.

- "Temporarily do not allow compressors to be stopped" assigns a hard penalty to any new `C_xxx:ST -> off` operation.
- "Avoid large changes to the main gas-source setpoint" assigns a hard penalty to large adjustments of the main gas-source variable `T_xxx`.
- "Minimize valve operations" assigns a soft penalty to actions involving too many `B_xxx` or `R_xxx` variables.
- "Avoid sacrificing terminal pressure" assigns a hard penalty to reductions in terminal-pressure margin.
- "Prioritize reducing additional energy consumption" treats energy consumption as the primary ranking criterion among feasible solutions.

Thresholds must be configurable rather than hard-coded.

## 10. G4 Diagnostic Correctness Implementation

### 10.1 Prediction Tasks

Implement `PredictionAnswerNormalizer` and `PredictionDiagnosticScorer`. Use the following 35-point raw scoring structure.

- Main conclusion: 10 points
- Top-3 indicators: 10 points
- Human intervention: 5 points
- Priority constraint: 5 points
- Evidence variables: 5 points

Finally, calculate `raw / 35 * 20`.

### 10.2 Dispatch Tasks

Implement `DispatchAnswerNormalizer` and `DispatchDiagnosticScorer`. Use the same 35-point raw scoring structure.

- Preferred action: 10 points
- Ranking consistency: 10 points
- Elimination or rejection reasons: 5 points
- Priority constraint: 5 points
- Evidence variables: 5 points

Finally, calculate `raw / 35 * 20`.

## 11. G5 Asset Authenticity Implementation

Implement `AssetAuthenticityScorer`. The following five asset types must exist and be mutually traceable.

- `trace_manifest.json`
- `boundary_diff.json`
- `normalized_answer.json`
- `report.md`
- `report.pdf`

The recommended score is 4 points per item. If a file exists but cannot be traced to the corresponding run or scenario, that item must not receive full credit.

## 12. Constraints for Paper Text Generation

`eval_pipeformer_ability_paper_paragraph-v2.tex` and `design_pipeformer_dataset_paper_paragraph-v2.tex` must be written in English. The writing style must follow academic-paper conventions, use short sentences and consistent terminology, and avoid colloquial language and unnecessary abbreviations. The content must support the central narrative in `paper_blueprint.md`: long-term operations, transient prediction, closed-loop dispatch, and auditable assets.
