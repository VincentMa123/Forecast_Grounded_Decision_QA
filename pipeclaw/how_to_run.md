# How To Run

[中文](#中文) | [English](#english)

## 中文

### 1. 环境要求

- Python 3.10+
- Node.js 18+
- npm 9+

### 2. 这份公开包里有什么

- `backend/pipeclaw_data/` 与 `backend/llm_data/`：公开的数据与任务资源
- `huggingface_dataset/`：可直接发布到 Hugging Face 的 dataset repo
- `backend/pipeline_data/`：结构兼容的 mock/demo 运行底库，用于本地把前后端完整跑通
- 不包含原始业务流量底库；如果你有内部数据，需要自行替换 `backend/pipeline_data/`

### 3. 后端启动

在仓库根目录执行：

```bash
pip install -r backend/requirements.txt
```

复制环境变量模板：

```bash
cp backend/.env.example backend/.env
```

至少补齐这些变量：

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `OPENAI_MODEL`

启动后端：

```bash
python backend/main.py
```

默认监听：`http://localhost:8003`

### 4. 前端启动

```bash
cd frontend
npm install
npm run dev
```

默认前端地址：`http://localhost:3000`

`frontend/vite.config.ts` 已把 `/api` 和 `/assets` 代理到 `http://localhost:8003`。

### 5. Demo 数据说明

- 仓库自带的 `backend/pipeline_data/` 是 mock/demo 数据，不是真实业务底库
- 这些 CSV 文件遵循 `backend/data_loader.py` 当前读取的字段结构
- 因此用户克隆公开仓库后可以直接看到日期、节点、管段和用户流量接口正常返回
- 如果你有内部私有数据，直接替换同名目录结构即可

### 6. 文档双语模式

这个导出包当前实现的是“文档双语模式”：

- 根目录 `README.md` 与 `README_zh.md` 对照呈现
- 根目录 `how_to_run.md` 同时提供中文和 English 两部分
- 前端界面本身保留现有中英混合文案，不额外引入 i18n 框架

## English

### 1. Requirements

- Python 3.10+
- Node.js 18+
- npm 9+

### 2. What this public package contains

- `backend/pipeclaw_data/` and `backend/llm_data/`: released public task resources
- `huggingface_dataset/`: push-ready dataset repo for Hugging Face
- `backend/pipeline_data/`: structure-compatible mock/demo runtime data so the app can run end-to-end locally
- The original private operational flow base is not included; replace `backend/pipeline_data/` if you have internal data

### 3. Start the backend

From the repository root:

```bash
pip install -r backend/requirements.txt
```

Copy the environment template:

```bash
cp backend/.env.example backend/.env
```

At minimum, configure:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `OPENAI_MODEL`

Start the backend:

```bash
python backend/main.py
```

Default backend URL: `http://localhost:8003`

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Default frontend URL: `http://localhost:3000`

`frontend/vite.config.ts` already proxies `/api` and `/assets` to `http://localhost:8003`.

### 5. Demo data note

- The bundled `backend/pipeline_data/` is synthetic mock/demo data, not real operational data
- The CSV layout matches what `backend/data_loader.py` expects
- This lets public users run the frontend and backend without access to private business tables
- If you have private internal data, replace the same directory structure with your own files

### 6. Bilingual mode

This export currently implements bilingual documentation mode:

- `README.md` and `README_zh.md` form the aligned English and Chinese landing pages
- `how_to_run.md` contains both Chinese and English sections
- The frontend UI itself keeps the existing mixed CN/EN copy and does not introduce a full i18n framework yet
