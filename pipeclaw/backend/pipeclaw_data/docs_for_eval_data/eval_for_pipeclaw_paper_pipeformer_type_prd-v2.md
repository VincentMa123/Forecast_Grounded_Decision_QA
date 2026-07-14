# PipeFormer v2 Dataset and Evaluation Refactoring PRD

## 1. Objective

This document is intended for engineering implementation. It has two objectives. First, rewrite the first 40 PipeFormer scenarios so that they are suitable for single-response evaluation. Second, rewrite the evaluation code so that G1 through G5 fully align with the descriptions in the paper, eliminating inconsistencies between the paper and the implementation.

## 2. Deliverables

The following files must be produced after the engineering implementation is complete.

- `Pipeline_Full_Life_Cycle_Test_Dataset-v7.json`
- `eval_for_pipeclaw_paper_pipeformer_type_prd-v2.md`
- `eval_for_pipeclaw_paper_pipeformer_type_tad-v2.md`
- `eval_pipeformer_ability_paper_paragraph-v2.tex`
- `design_pipeformer_dataset_paper_paragraph-v2.tex`

## 3. Dataset Refactoring Requirements

### 3.1 Data File Structure

The top level of `Pipeline_Full_Life_Cycle_Test_Dataset-v7.json` must be a list. The list must contain only 40 PipeFormer scenario dictionaries. Do not add an overall description, statistical fields, group summaries, or top-level metadata. Each scenario must retain the basic structure of the previous version, with only one additional classification field.

Each scenario must contain the following keys.

- `scenario_id`
- `scenario_type`
- `scenario_class_label`
- `scenario_description`
- `sessions`

The value of `scenario_type` must always be `pipeformer`. The value of `scenario_class_label` must be the Chinese class name rather than a P1-P5 or D1-D5 code.

### 3.2 Single-Turn Requirements

Each scenario must contain exactly one session and one turn.

- `sessions` must contain exactly one element.
- `dialogue` must contain exactly one element.
- `offset_hours` must always be 0.

The single-turn prompt must preserve the business objective of the original task. Important constraints must not be removed mechanically. A prediction-task prompt must ask the model to provide the operational consequences, Top-3 indicators, human-intervention decision, priority constraint, and two key evidence variables. A dispatch-task prompt must ask the model to provide a ranking of candidate actions, the reason for the preferred action, reasons for eliminating or rejecting other actions, the priority constraint, and two key evidence variables. It must also preserve the optimization preferences and field constraints from the original task.

### 3.3 Variable Selection Requirements

The 40 scenarios in v7 must use 40 distinct boundary variables. Variable names must come from actual columns in `Boundary.csv` and must follow the naming semantics defined in the competition PDF. The following boundary-variable types are allowed.

- `T_xxx:SNQ`: gas-source flow setpoint
- `T_005:SP`: gas-source pressure setpoint
- `E_xxx:SNQ`: user-side demand-flow setpoint
- `B_xxx:FR`: ball-valve flow ratio
- `C_xxx:SP_out`: compressor outlet-pressure setpoint
- `C_xxx:ST`: compressor on/off state
- `R_xxx:SPD`: regulator downstream-pressure setpoint
- `R_xxx:ST`: regulator on/off state

Pseudo-variable names from previous drafts, such as `C_xxx:SP_`, must not be reused. Variables should be distributed across assets with middle and higher index numbers rather than concentrated among a small number of variables with the `001` prefix.

### 3.4 Ten Problem Classes

Version 7 still contains two task groups and 40 tasks in total. Each group contains five classes, with four instances per class.

| Group | Class Code | Chinese Class Name | Primary Focus |
|---|---|---|---|
| Prediction | P1 | 富余抬升与高压能耗 | High pressure on the surplus side, local overpressure, and increased energy consumption |
| Prediction | P2 | 紧平衡与低压消耗 | Low-pressure risk, declining linepack, and reduced supply-security margin |
| Prediction | P3 | 压缩机工况边界 | Compressor load boundaries, compressor switching, and outlet-pressure setpoint changes |
| Prediction | P4 | 节流瓶颈与通道受限 | Bottleneck exposure and bypass crowding after tightening a valve or regulator |
| Prediction | P5 | 重分配与稳态重构 | Path switching and a new steady state after opening a channel or increasing pressure |
| Dispatch | D1 | 末端保压与关键供气 | Priority protection of terminal or critical-node pressure |
| Dispatch | D2 | 约束减违与安全回边界 | Reduce the violation rate before considering energy consumption |
| Dispatch | D3 | 管存与系统余度保持 | Linepack decline, supply-demand margin, and system flexibility |
| Dispatch | D4 | 波动抑制与单点失稳防控 | Pressure-fluctuation control and prevention of single-point anomaly propagation |
| Dispatch | D5 | 低负荷优化与越限消除 | Low-load energy optimization and elimination of high velocity or potential limit violations |

## 4. Overall Evaluation Requirements

The total score for the PipeFormer track remains 100 points. Each of the five capability dimensions is worth 20 points. This overall structure must not be changed.

- G1 Call Authenticity
- G2 Execution Authenticity
- G3 Verification Authenticity
- G4 Diagnostic Correctness
- G5 Asset Authenticity

## 5. G1 Call Authenticity

G1 checks only whether the model actually completes the skill execution loop. Scoring is based on seven mandatory checks. The raw score is calculated from the number of passed checks and then linearly mapped to 20 points.

The mandatory checks are as follows.

1. Whether `run_command` appears
2. Whether `Boundary.csv` is read or modified
3. Whether `batch_jobs_for_skill_1.json` is read or modified
4. Whether `batch_jobs_for_skill_2.json` is read or modified
5. Whether `python -m real_predict.main` is executed at least twice
6. Whether at least one type of split CSV in the output directory is read
7. Whether the final answer explicitly compares modified and original results

## 6. G2 Execution Authenticity

G2 must be divided into two subcomponents.

### 6.1 Dual-Run Completeness

Verify that both original and modified outputs were actually obtained. Prediction tasks require complete baseline and single-perturbation outputs. Dispatch tasks require complete baseline and candidate-action outputs, with at least one candidate action actually rerun.

### 6.2 Reference-Rerun Consistency

The evaluator must independently rerun the original and modified inputs produced by the model. The rerun command remains `python -m real_predict.main`. The rerun results must be compared file by file with the split CSV outputs saved by the model. The comparison must cover all generated files among `B.csv`, `C.csv`, `H.csv`, `N.csv`, `P.csv`, `R.csv`, and `T&E.csv`. The code must align files by time column and column name before performing numerical comparisons.

The recommended implementation for the 20 G2 points is 10 points for dual-run completeness and 10 points for reference-rerun consistency.

## 7. G3 Verification Authenticity

G3 must also be divided into two subcomponents.

### 7.1 Parameter-Bound Verification

The evaluator must read `附件2：管道设备参数.zip` and construct a parameter registry. The registry must parse at least the following files.

- Compressor basic-parameter tables and characteristic curves for each compressor
- Ball-valve basic-parameter table
- Regulator basic-parameter table
- Pipeline basic-parameter table
- Pipeline-segment basic-parameter table

Parameter-bound verification must not invent hard thresholds that do not exist. The checks reliably supported by the current attachment include:

- Whether the target equipment actually exists
- Whether `FR` falls within the legal ratio domain
- Whether `ST` falls within the legal state domain
- Whether `SNQ` and the various pressure setpoints fall within their basic numerical domains
- Whether compressor output flow falls within the flow envelope supported by the corresponding compressor characteristic curve

For data without an explicit upper bound in the attachment, do not fabricate hard thresholds such as a "maximum pressure." Such cases must instead be covered by reference reruns and evidence verification.

### 7.2 CSV Evidence Grounding

The model's risk assessment must be supported by evidence in the actual outputs. The evaluator must calculate summaries of key differences from the original and modified split CSV files and use them to determine:

- Whether the primary risk category agrees with the direction of the actual output changes
- Whether the Top-3 indicators are actual high-impact variables
- Whether the priority constraint was triggered by an actual conflict
- Whether both evidence variables exist in the output and genuinely support the conclusion

The recommended implementation for the 20 G3 points is 10 points for parameter-bound verification and 10 points for CSV evidence grounding.

## 8. G4 Diagnostic Correctness

G4 must not use full-text string matching. It must first extract the minimum conclusion fields and then score them.

### 8.1 Minimum Fields for Prediction Tasks

- `main_consequence`
- `watch_indicators`
- `manual_intervention`
- `constraint_priority`
- `evidence_vars`

### 8.2 Raw Scoring for Prediction Tasks

- Main conclusion: 10 points
- Top-3 indicators: 10 points
- Whether human intervention is required: 5 points
- Priority constraint: 5 points
- Evidence variables: 5 points

The raw maximum for a prediction task is 35 points, which is then linearly normalized to 20 points.

### 8.3 Minimum Fields for Dispatch Tasks

- `top_action`
- `action_ranking`
- `rejection_reasons`
- `constraint_priority`
- `evidence_vars`

### 8.4 Scoring Principles for Dispatch Tasks

Dispatch tasks must not use fixed text as the gold standard. The evaluator must first extract the candidate actions proposed by the model, rerun each action, and rank the actions according to the objective function and field constraints of the scenario class. The core of G4 is to verify whether the ranking, preferred action, and elimination reasons reported by the model agree with the audit results from the reruns. The following 35-point raw structure is recommended, followed by linear normalization to 20 points.

- Preferred action: 10 points
- Ranking consistency: 10 points
- Elimination or rejection reasons: 5 points
- Priority constraint: 5 points
- Evidence variables: 5 points

## 9. G5 Asset Authenticity

G5 verifies whether intermediate results are actually persisted as traceable assets. The recommended implementation consists of five equally weighted subitems worth 4 points each.

1. `trace_manifest.json`
2. `boundary_diff.json`
3. `normalized_answer.json`
4. `report.md`
5. `report.pdf`

The assets must be mutually traceable. At a minimum, the report must link to the normalized conclusions, which must in turn link to the execution trace and the record of input modifications.

## 10. Engineering Constraints

- Do not add explanatory fields at the top level of the v7 data file.
- Do not include explanatory paper text in the dataset.
- Do not impose scores based on physical upper bounds that are absent from the attachment.
- The paper-paragraph files must use academic English and follow the narrative direction of `paper_blueprint.md`.
- The paper paragraphs should favor pipeline operations and dispatch terminology and minimize purely computational terminology.
