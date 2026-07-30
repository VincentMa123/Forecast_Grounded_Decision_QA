# Repository Guidelines

## Project Structure & Module Organization

This repository contains two related packages. `pipeFormer/` holds the forecasting model, data preprocessing, sparse decoder implementation, configs, mock data, and decoder tests. Important subtrees include `pipeFormer/data/`, `pipeFormer/models/decoder/`, `pipeFormer/training/`, and `pipeFormer/configs/`. `pipeclaw/` is the runnable QA and visualization stack: `backend/` contains the FastAPI service, agent runtime, execution pipeline, mock `pipeline_data/`, and released task data; `frontend/` contains the Vite React app; `huggingface_dataset/` contains dataset publishing helpers; `docs/images/` stores paper and UI figures.

## Build, Test, and Development Commands

- `cd pipeclaw && pip install -r backend/requirements.txt`: install backend dependencies.
- `cd pipeclaw && python backend/main.py`: start the FastAPI backend on `http://localhost:8003`.
- `cd pipeclaw/frontend && npm install`: install frontend dependencies from `package-lock.json`.
- `cd pipeclaw/frontend && npm run dev`: start the Vite dev server on port `3000`; `/api` and `/assets` proxy to the backend.
- `cd pipeclaw/frontend && npm run build`: run TypeScript checking and build the frontend.
- `cd pipeclaw/frontend && npm run lint`: run the declared ESLint check.
- `cd pipeFormer && pip install -r requirements.txt`: install model and preprocessing dependencies.
- `cd pipeFormer && python train.py --config configs/quick_test_decoder.json`: run a small decoder training configuration.

## Coding Style & Naming Conventions

Python code uses 4-space indentation, `snake_case` modules/functions, and `PascalCase` classes. Keep model configs as JSON under `pipeFormer/configs/`. Frontend code is TypeScript + React with `strict`, `noUnusedLocals`, and `noUnusedParameters` enabled in `frontend/tsconfig.json`; use `.tsx` for components and colocated `.css` files for component/page styling.

## Testing Guidelines

The Python test surface currently lives under `pipeFormer/models/tests/` and uses pytest-style `test_*.py` files and `Test*` classes. Run a focused decoder test with `cd pipeFormer && python -m pytest models/tests/test_decoder_functions.py` after installing pytest if it is not already available. No frontend `test` script is declared in `package.json`; use `npm run lint` and `npm run build` for the checked frontend workflow.

## Commit & Pull Request Guidelines

Recent commits are short, imperative summaries such as `Test run the pipeFormer and the pipeclaw` and `Add project source files`. Keep commit subjects concise and task-focused. Pull requests should describe which package changed, list commands run, call out data/config changes, and include screenshots when frontend views or `docs/images/` assets change.

## Security & Configuration Tips

`pipeclaw/backend/main.py` loads `backend/.env` for `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL`. Do not expose private credentials or internal operational data; replace `backend/pipeline_data/` with private data only in local or controlled environments.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
