# Tools directory

## Workspace tools (recommended)
Minimal toolset for the agent:
- `write_file`: write or overwrite files in the session workspace
- `edit_file`: exact string replacement in a workspace file
- `run_command`: run approved commands (python/pip/node, powershell/cmd/bash, FS ops like mkdir/cp/copy/xcopy/robocopy) inside the workspace
- `read_file`: read workspace files or read-only `pipeline_data/...` files (JSON auto-parse supported)
- `run_pipeformer_forecast`: run PipeFormer checkpoint inference for forecast, what-if, risk, dispatch, or transient-operation questions.
- `search_pipeformer_registry`: find canonical controllable inputs and nearby equipment before proposing dispatch actions.
- `set_decision_policy`: convert the user's natural-language dispatch priorities into a validated ordered metric policy before evaluating multiple candidates.

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
# filename patterns: YYYYMMDD_node.csv, YYYYMMDD_pipeline.csv, YYYYMMDD_consumer.csv
# (use the actual date string, e.g. 20190101_node.csv)

output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "answer.txt").write_text("done", encoding="utf-8")
```

## Deprecated tools
`data_tools.py` and `analytics_tools.py` are no longer registered as tools.
If needed, import them inside your scripts instead of calling them directly.

## PipeFormer tool configuration
For dispatch comparisons, call `search_pipeformer_registry` before the first forecast. Filter with `role=input` and
`controllable=true`; optionally pass `attention_targets` to rank controls by graph distance. The tool returns a small
semantic projection rather than exposing the full registry or local file paths.

Before the second dispatch candidate, call `set_decision_policy`. The agent must infer the objective order from the
user request and preserve that visible tool call in the trace; scenario metadata must not supply a hidden ranking label.

`run_pipeformer_forecast` uses the mock-tiny checkpoint by default. Override paths with environment variables when running another checkpoint or a separate PipeFormer environment:

- `PIPEFORMER_ROOT`
- `PIPEFORMER_CHECKPOINT_DIR`
- `PIPEFORMER_DATA_DIR`
- `PIPEFORMER_STATIC_DIR`
- `PIPEFORMER_MAPPING_CSV`
- `PIPEFORMER_DEVICE`

## LLM provider configuration

DeepSeek and other OpenAI-compatible endpoints use:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
OPENAI_THINKING=true
OPENAI_REASONING_EFFORT=high
```

The thinking settings are optional and are forwarded through `extra_body` for
OpenAI-compatible providers that support them.

Omitting `LLM_PROVIDER` also defaults to `openai`. Native GLM uses the Z.AI SDK:

```env
LLM_PROVIDER=zai
ZAI_API_KEY=your_key
ZAI_MODEL=glm-5.2
ZAI_THINKING=enabled
ZAI_REASONING_EFFORT=max
ZAI_MAX_TOKENS=65536
ZAI_TEMPERATURE=1.0
```

`LLM_PROVIDER=None` is invalid; remove the setting or select `openai` when returning to DeepSeek.
