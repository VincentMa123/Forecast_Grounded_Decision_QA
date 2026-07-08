# Tools directory

## Workspace tools (recommended)
Minimal toolset for the agent:
- `write_file`: write or overwrite files in the session workspace
- `edit_file`: exact string replacement in a workspace file
- `run_command`: run approved commands (python/pip/node, powershell/cmd/bash, FS ops like mkdir/cp/copy/xcopy/robocopy) inside the workspace
- `read_file`: read workspace files (JSON auto-parse supported)
- `run_pipeformer_forecast`: run PipeFormer checkpoint inference for forecast, what-if, risk, dispatch, or transient-operation questions.

### Workflow
1. Create or update `plan.md` first.
2. Write your script (e.g. `task.py`).
3. Run it with `run_command(["python", "task.py"])`.
4. Read results from `runs/run_<timestamp>[_NN]/output/answer.txt` or `.json` (use run_command output_files).

### Environment variables for scripts
- `BACKEND_DIR`
- `NODE_FLOW_DIR`
- `PIPELINE_FLOW_DIR`
- `CONSUMER_FLOW_DIR`
- `OUTPUT_DIR`

Example:
```python
import os
from pathlib import Path
import pandas as pd

node_flow_dir = Path(os.environ["NODE_FLOW_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])

# Read a daily node flow file
# filename pattern: YYYYMMDD_\u8282\u70b9.csv
# (use the actual date string, e.g. 20190101_\u8282\u70b9.csv)

output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "answer.txt").write_text("done", encoding="utf-8")
```

## Deprecated tools
`data_tools.py` and `analytics_tools.py` are no longer registered as tools.
If needed, import them inside your scripts instead of calling them directly.

## PipeFormer tool configuration
`run_pipeformer_forecast` uses the mock-tiny checkpoint by default. Override paths with environment variables when running another checkpoint or a separate PipeFormer environment:

- `PIPEFORMER_ROOT`
- `PIPEFORMER_CHECKPOINT_DIR`
- `PIPEFORMER_DATA_DIR`
- `PIPEFORMER_STATIC_DIR`
- `PIPEFORMER_MAPPING_CSV`
- `PIPEFORMER_DEVICE`

For DeepSeek, keep using the OpenAI-compatible settings consumed by the orchestrator: `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL`.
