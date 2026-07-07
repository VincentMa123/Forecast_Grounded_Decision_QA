# PipeClaw 中文说明

[English README](README.md) | [运行说明](how_to_run.md)

PipeClaw 是一个面向天然气管网运行分析的 trace-first 生命周期评测与智能体运行时项目。它面向这类问题：同一个任务同时涉及 GIS 资产信息、管网拓扑结构和按时间索引的运行数据，而这类问题最重要的不是文字是否流畅，而是结果是否正确、是否可追溯。

这个开源仓库延续了项目后期的 PipeClaw 命名，而论文中对应的研究系统名称是 Pipeline-Agent。论文链接如下：
[Pipeline-Agent: A Comprehensive Decision Support System for Natural Gas Pipelines via Verifiable Computational Reasoning](https://www.sciencedirect.com/science/article/pii/S2667143326000429)

这个项目遵循论文中的一个核心原则：先计算，再解释。系统不会直接自由生成答案，而是先在受限工作区中编写并执行确定性脚本，保存中间产物，再返回一个可复核、可重放的结果包，供工程人员检查。

## 为什么是 PipeClaw

天然气管网决策通常依赖多个来源的数据，而且这些系统之间经常存在命名不一致、标识不统一、表关联脆弱等问题。一次看似正常的 join，就可能悄悄丢失记录、扭曲供需平衡，或者选错资产。PipeClaw 针对的正是这个问题：用显式计划、确定性执行和证据留存，把分析流程变成可验证计算。

## 项目结构

- `backend/`：FastAPI 服务、agent runtime、执行循环以及公开 API 层。
- `backend/pipeclaw_data/`：随仓库公开的 PipeClaw 子构件与场景资源。
- `frontend/`：基于 Vite + React 的交互界面，用于浏览流量、日期、节点详情和 agent 对话结果。
- `docs/images/`：整理后的论文图片资源，可直接在 GitHub 展示。
- `huggingface_dataset/`：已经整理好的 Hugging Face Dataset 仓库目录，包含数据、说明和上传脚本。

## 前后端如何加载

这个仓库应该被理解成一个完整项目，而不只是论文附属代码。

- 后端入口是 `python backend/main.py`，会启动监听在 `http://localhost:8003` 的 FastAPI 服务。
- 前端入口是在 `frontend/` 下执行 `npm run dev`，启动 Vite 开发服务器。
- `frontend/src/api/client.ts` 以 `/api` 为基础路径，`frontend/vite.config.ts` 会在开发环境把 `/api` 和 `/assets` 代理到 `http://localhost:8003`。
- 后端提供 `/api/flow/nodes`、`/api/flow/pipelines`、`/api/flow/consumers`、`/api/dates` 等流量与日期接口，同时提供 `/api/agent/*` 下的智能体接口。
- 实际运行时，前端通过这些接口加载管网流量数据和 agent 响应，因此 GitHub 用户接触到的是一个可以跑起来的系统，而不是单纯的论文配套仓库。

## 数据集开源方式

本次开源包额外提供了 `huggingface_dataset/` 目录，里面已经把公开数据整理成适合 Hugging Face Dataset Hub 的结构：原始 JSON 保存在 `raw/`，供 Dataset Viewer 与 `datasets.load_dataset(...)` 使用的 JSONL 保存在 `data/`，同时附带校验脚本和一键上传脚本。主应用默认不附带原始业务流量底库。

- 支持的数据集配置名：`llm-question-all`、`llm-question-template-80`、`llm-question-template-all`、`pipeclaw-v2`、`pipeformer-v4`、`pipeformer-v7`
- 一键上传命令：`python huggingface_dataset/scripts/upload_to_hf.py --repo-id zly7/pipeclaw-open-datasets`
- 如果环境里没有 `HF_TOKEN`，脚本会安全地提示输入，不要求你先手动写到文件里。
- 上传完成后，外部用户可以通过 `snapshot_download(...)` 下载整个数据仓库，或者通过 `load_dataset("zly7/pipeclaw-open-datasets", "pipeclaw-v2", split="train")` 直接加载。

## 界面展示

### 前端工作界面

![Frontend workspace](docs/images/frontend_schematic_picture.png)

这张图展示了产品的主交互界面，包括 agent 控制区、当前对话画布、最终回答区域，以及右侧可回看历史会话的记录面板。

### Word 解析与报告案例

![Support Word parsing case](docs/images/Support%20Word%20parsing%20case.png)

这个案例图展示了面向文档的交互流程：用户在问题中直接要求输出分析结果并生成 `.docx` 报告，界面会把结构化最终回答直接呈现在聊天工作区中。

## 核心思想

- 统一时空数据模式：把资产位置、网络连接和日尺度运行数据整理成稳定的表结构。
- 显式计划产物：每次运行都维护一个持久 plan 文件，让推理过程可以被查看和复核。
- 工具落地执行：关键计算通过脚本真实执行，而不是依赖纯文本推断。
- 可验证结果包：最终输出会附带脚本、日志和中间文件，方便审计和复现。

## 工作流程

1. 把用户请求转换成显式计划。
2. 载入节点、管段和用户三类核心数据表。
3. 在工作区内生成一个小型分析脚本。
4. 对表结构、标识映射和任务约束执行确定性校验。
5. 返回结构化结果以及可复现的证据文件。

## 论文图片

### 系统总览

![Pipeline-Agent architecture](docs/images/system_preview.png)

这张图更具体地说明了系统总览的三层结构：最上层是用户请求、任务上下文和显式计划；中间层是在受限工作区中对管网数据表执行确定性脚本与校验；最下层是最终返回的可验证结果包，其中包含计算输出、中间产物以及执行日志，便于复核与审计。

### 综合性能图

![Aggregate performance](docs/images/fig_combined_4x1.png)

这张图展示了整个 benchmark 上的综合评测结果，从整体分数、稳定性和成本相关指标几个方面说明系统级表现，而不是只看单一任务样例。

### 类别热力图

![Category heatmap](docs/images/fig_02_score_heatmap.png)

这张热力图对比了不同任务类别上的表现，能直观看出这套流程在哪些问题上更强，尤其是需要多表聚合、结构化推理和严格数值一致性的场景。

### 相对最佳基线的差值图

![Margin vs best baseline](docs/images/fig_07_margin_vs_best_baseline.png)

这张图突出展示了各类别相对最佳基线的领先幅度，更适合用来观察显式计划与工具落地执行在不同任务上的真实增益分布。

### 案例展示

![Pipeline-Agent case study](docs/images/case_study.png)

这个案例图展示了一条具体分析链路，包括生成代码、中间结果和最终检查后的输出，用来说明 PipeClaw 如何把自然语言请求转成可审计的工程分析流程。

## 仓库内容

- `backend/agent` 与 `backend/executor`：trace-first runtime 与执行循环
- `backend/pipeclaw_data`：本次随仓库开放的 PipeClaw 子构件文件
- `frontend/`：可直接运行的交互前端
- `huggingface_dataset/`：可直接发布到 Hugging Face 的数据集仓库目录与上传脚本
- `docs/images/`：适合 GitHub 直接展示的论文图片
- `how_to_run.md`：中英双语环境配置与运行说明

## 快速开始

1. 安装后端依赖：`pip install -r backend/requirements.txt`
2. 安装前端依赖：`cd frontend && npm install`
3. 启动后端：`python backend/main.py`
4. 启动前端：`cd frontend && npm run dev`

详细运行方式请看 [how_to_run.md](how_to_run.md)。对应英文说明见 [README.md](README.md)。仓库默认附带的是可跑通界面的 mock `backend/pipeline_data/`，不包含原始业务流量底库。
