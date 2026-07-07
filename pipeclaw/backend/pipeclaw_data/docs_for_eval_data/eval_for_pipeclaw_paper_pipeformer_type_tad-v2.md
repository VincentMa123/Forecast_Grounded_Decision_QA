# PipeFormer v2 数据集与评测改造 TAD

## 1. 总体目标
本 TAD 对应 v7 数据集和 v2 评测器。目标是把前 40 个 PipeFormer 场景改造成单次回答任务，并实现一个与论文叙述完全一致的自动评测闭环。

## 2. 数据集读取层

### 2.1 输入文件
输入文件为 `Pipeline_Full_Life_Cycle_Test_Dataset-v7.json`。文件最外层是 list。list 内共有 40 个 scenario dict。

### 2.2 场景对象
加载后统一转成内部 `ScenarioRecord`，字段至少包含：

- `scenario_id`
- `scenario_type`
- `scenario_class_label`
- `scenario_description`
- `session_id`
- `user_input`

### 2.3 场景类型判定
不要再依赖 `scenario_type` 区分 prediction 与 dispatch。统一通过 `scenario_id` 解析。

- `scenario_pipeformer_prediction_*` 归为 prediction
- `scenario_pipeformer_dispatch_*` 归为 dispatch

### 2.4 场景元信息抽取
由于 v7 数据集刻意保持简单，评测器需要在加载阶段从 `user_input` 中抽取并缓存以下字段，写入内部 registry。

- `mock_case_id`
- `target_boundary_var`
- `perturbation`
- `forecast_horizon`
- `scenario_kind`
- `scenario_class_label`
- `preference`，仅 dispatch
- `field_constraint`，仅 dispatch

建议实现 `PipeformerPromptParser`，通过正则和轻量规则抽取这些字段。抽取结果落盘为 `scenario_registry_v2.json`，但该文件不需要作为公开数据集的一部分。

## 3. 参数注册表

### 3.1 输入
参数注册表从 `附件2：管道设备参数.zip` 构建。建议在评测启动前完成一次解析并缓存。

### 3.2 解析结果
内部注册表至少包括以下结构。

- `compressor_meta`
- `compressor_curve_envelopes`
- `ball_valve_meta`
- `regulator_meta`
- `pipe_meta`
- `segment_meta`

### 3.3 压缩机包络
压缩机文件按 `总体说明` 和 `C_001` 到 `C_023` 各 sheet 解析。对每台压缩机保存如下数据。

- 设备编号
- 上下游节点
- 驱动方式
- 特性曲线最小流量
- 特性曲线最大流量
- 特性曲线表本身

当前附件没有给出所有设备统一的硬压力上限，因此不要构造伪造的压力上限表。压缩机部分只做附件真实支持的包络检查。

### 3.4 阀门与调节阀
球阀与调节阀文件至少保存设备编号、上下游节点和调节阀类型。评测使用这些数据做设备存在性校验和边界变量合法性校验。

### 3.5 管道与管段
管道和管段参数先作为身份与拓扑注册表使用。若后续代码需要附加诊断指标，可以基于直径和长度计算速度代理量或压降代理量，但这些代理量默认不参与硬评分。

## 4. 执行痕迹采集

### 4.1 TraceCollector
TraceCollector 负责统一收集运行命令、文件读写和输出目录访问事件。建议至少落盘以下文件。

- `trace/run_command.jsonl`
- `trace/file_events.jsonl`
- `trace/output_reads.jsonl`

### 4.2 run_command 事件
每条事件至少包含：

- `ts`
- `cmd`
- `cwd`
- `exit_code`
- `stdout_path`
- `stderr_path`

### 4.3 file_events 事件
每条事件至少包含：

- `ts`
- `op`
- `path`
- `sha256_before`
- `sha256_after`
- `size_after`

## 5. 参考复跑层

### 5.1 ReferenceRunner
ReferenceRunner 负责对模型产出的 original 与 modified 输入做独立复跑。所有复跑统一使用 `python -m real_predict.main`。

### 5.2 输入恢复
输入恢复分两步。

第一步，读取模型实际写入的 `Boundary.csv`、`batch_jobs_for_skill_1.json`、`batch_jobs_for_skill_2.json`。  
第二步，按 original 和 modified 两个版本重建独立运行目录。

### 5.3 输出采集
每次复跑后都要把 split CSV 收集到统一结构下，例如：

- `reference_runs/original/...`
- `reference_runs/modified/...`
- `reference_runs/candidate_A1/...`
- `reference_runs/candidate_A2/...`

### 5.4 CSV 对齐比较
实现 `SplitCsvComparator`。比较步骤如下。

1. 先按文件名对齐
2. 再按 `TIME` 和列名对齐
3. 对非时间列做数值比较
4. 输出 `max_abs_diff`、`mean_abs_diff`、`relative_l1_diff`

预测器是确定性的，因此优先使用严格一致。若 CSV 文本格式导致轻微浮点差异，可允许极小数值容差。

## 6. G1 调用真实性实现
实现 `CallAuthenticityScorer`。原始打分直接统计 7 个必查项的通过数量，再做 `passed / 7 * 20` 归一化。7 个检查项如下。

- 出现 `run_command`
- 读或写 `Boundary.csv`
- 读或写 `batch_jobs_for_skill_1.json`
- 读或写 `batch_jobs_for_skill_2.json`
- 至少执行两次 `python -m real_predict.main`
- 读取至少一种 split CSV
- 最终回答明确包含 modified 与 original 的对照

## 7. G2 运行真实性实现

### 7.1 双运行完整性
实现 `RunCompletenessChecker`。预测题要求 original 与 modified 都有完整输出。调度题要求 original 存在，且至少 1 个候选动作完成真实复跑。若模型声称比较了多个动作，则相应动作都应有输出目录。

### 7.2 参考复跑一致性
实现 `ReferenceMatchScorer`。它读取模型保存输出和独立 reference 输出，对全部 split CSV 做文件级与数值级一致性比较。建议把 10 分拆成两层。

- 5 分给文件齐备和 schema 一致
- 5 分给数值一致性

## 8. G3 校核真实性实现

### 8.1 参数边界校核器
实现 `ParameterBoundAuditor`。它至少检查以下内容。

- 目标边界变量是否存在于真实 Boundary schema
- 目标设备是否存在于参数注册表
- `FR` 是否在 `[0, 1]`
- `ST` 是否在离散状态域
- `SNQ` 和各类设定压力是否为合法数值
- 压缩机 `q_in` 或 `q_out` 是否落在对应机组特性曲线流量范围内

当前附件不能直接支撑管道或节点统一压力上限，因此不要编造这类评分规则。

### 8.2 CSV 证据校核器
实现 `EvidenceGroundingAuditor`。它需要基于 original 与 modified 输出构造一个统一摘要表。摘要表建议包含每个变量的：

- `delta_last`
- `delta_mean`
- `delta_max_abs`
- `family`，例如 pressure、flow、linepack、power
- `source_file`

然后按场景类别生成参考结论。

### 8.3 预测题参考结论生成
实现 `PredictionOutcomeSynthesizer`。它根据摘要表和规则引擎自动生成：

- `main_consequence`
- `watch_indicators`
- `manual_intervention`
- `constraint_priority`
- `evidence_vars`

这一步取代人工金标准文案，保证结论来自真实输出。

### 8.4 调度题参考结论生成
实现 `DispatchReplayAuditor`。流程如下。

1. 从模型答案中抽取候选动作
2. 对每个候选动作真实复跑
3. 对每个动作做规则审计
4. 根据场景类别的目标函数重新排序
5. 把最优动作、排序结果和主要淘汰理由作为参考结论

## 9. 调度题动作解析与重排

### 9.1 动作抽取
实现 `DispatchActionExtractor`。建议优先支持结构化 JSON，其次支持 markdown 列表。统一归一成如下结构。

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

### 9.2 场景类别目标函数
实现 `DispatchUtilityEngine`。不同类别使用不同排序目标，但都必须先处理现场约束。

- D1：先看硬约束，再看压力违反率和关键节点压力余度，再看能耗
- D2：先看硬约束，再看总违反率，再看能耗
- D3：先看硬约束，再看管存或系统余度，再看能耗
- D4：先看硬约束，再看压力波动幅度和单点失稳迹象，再看能耗
- D5：先看硬约束，再看高流速或潜在超限消除情况，再看能耗

### 9.3 现场约束解释
把 prompt 中的现场约束转成机器规则。

- “暂时不允许停压缩机” 对任何新增 `C_xxx:ST -> off` 给出硬惩罚
- “不希望大幅改动主气源设定” 对主气源 `T_xxx` 的大幅调节给出硬惩罚
- “尽量少动阀门” 对涉及过多 `B_xxx` 或 `R_xxx` 的动作给出软惩罚
- “不希望牺牲末端压力” 对末端压力余度下降给出硬惩罚
- “优先减少额外能耗” 在可行方案中把能耗作为主要排序项

阈值要写成可配置参数，不要写死在代码里。

## 10. G4 诊断正确性实现

### 10.1 预测题
实现 `PredictionAnswerNormalizer` 和 `PredictionDiagnosticScorer`。原始评分使用 35 分结构。

- 主结论 10 分
- Top-3 指标 10 分
- 人工干预 5 分
- 优先约束 5 分
- 证据变量 5 分

最后做 `raw / 35 * 20`。

### 10.2 调度题
实现 `DispatchAnswerNormalizer` 和 `DispatchDiagnosticScorer`。原始评分也使用 35 分结构。

- 首选动作 10 分
- 排序一致性 10 分
- 淘汰或拒绝理由 5 分
- 优先约束 5 分
- 证据变量 5 分

最后做 `raw / 35 * 20`。

## 11. G5 资产真实性实现
实现 `AssetAuthenticityScorer`。要求以下五类资产存在且互相可追踪。

- `trace_manifest.json`
- `boundary_diff.json`
- `normalized_answer.json`
- `report.md`
- `report.pdf`

建议每项 4 分。若文件存在但无法追踪到对应 run 或对应 scenario，则该项不得满分。

## 12. 论文文本生成约束
`eval_pipeformer_ability_paper_paragraph-v2.tex` 和 `design_pipeformer_dataset_paper_paragraph-v2.tex` 必须用英文撰写。文本风格要符合学术论文习惯，句子要短，术语要稳，不要使用口语和不必要的缩写。内容需要服务于 `paper_blueprint.md` 的主线，即长期运维、瞬态预测、调度闭环和可审计资产。