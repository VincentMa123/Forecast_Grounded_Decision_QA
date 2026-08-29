# Agent tools

These are the tools registered by the PipeClaw agent. They let a run inspect
data, write and execute a small analysis, query PipeFormer's registry, request
a forecast, and preserve the resulting evidence.

## Registered tools

| Tool | Use |
| --- | --- |
| `read_file` | Read a workspace file or a logical `pipeline_data/...` CSV. |
| `write_file` | Create or replace a file in the session workspace. |
| `edit_file` | Apply an exact string replacement in a workspace file. |
| `run_command` | Run an approved command in the session workspace. |
| `search_pipeformer_registry` | Find canonical controllable variables and nearby equipment. |
| `set_decision_policy` | Record the user's ordered objectives before ranking candidates. |
| `run_pipeformer_forecast` | Run checkpoint inference and return grounded forecast evidence. |

## Recommended workflow

1. Create or update `plan.md`.
2. Read the required data with `read_file`.
3. Write a short script with `write_file` and execute it with `run_command`.
4. For PipeFormer dispatch questions, call
   `search_pipeformer_registry` before the first forecast and
   `set_decision_policy` before ranking candidates when the user has stated
   priorities.
5. Read the output artifact from `runs/run_<timestamp>/output/` and answer from
   successful structured results only.

`run_command` executes inside an isolated workspace. It does not expose a
`pipeline_data/` directory; scripts must use the injected environment variables
below to locate CSVs.

## Workspace variables

The executor provides these variables to analysis scripts:

```text
BACKEND_DIR       backend root
WORKSPACE_DIR     current session workspace
NODE_FLOW_DIR     node-flow CSV directory
PIPELINE_FLOW_DIR pipeline-flow CSV directory
CONSUMER_FLOW_DIR consumer-flow CSV directory
OUTPUT_DIR        current run output directory
RUN_DIR           current run directory
```

Example:

```python
import os
from pathlib import Path

nodes = Path(os.environ["NODE_FLOW_DIR"])
output = Path(os.environ["OUTPUT_DIR"])
rows = list(nodes.glob("*_node.csv"))
output.joinpath("answer.txt").write_text(f"files={len(rows)}", encoding="utf-8")
```

Use logical paths such as `pipeline_data/node_flow/20190114_node.csv` only with
`read_file`; do not paste host-specific paths into scripts or tool calls.

## PipeFormer configuration

`run_pipeformer_forecast` uses the mock-tiny checkpoint by default. Override it
with environment variables when running another checkpoint or data root:

```text
PIPEFORMER_ROOT
PIPEFORMER_CHECKPOINT_DIR
PIPEFORMER_DATA_DIR
PIPEFORMER_STATIC_DIR
PIPEFORMER_MAPPING_CSV
PIPEFORMER_DEVICE
```

The forecast tool is read-only. It requires registry grounding for dispatch
recommendations and returns compact evidence suitable for the agent trace.

## Model provider configuration

Create `pipeclaw/backend/.env` for live agent calls. The provider defaults to
OpenAI-compatible mode:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=your_model
OPENAI_API_BASE=https://your-endpoint/v1
OPENAI_REASONING_EFFORT=high
```

For the native Z.AI client, use `LLM_PROVIDER=zai` with `ZAI_API_KEY`,
`ZAI_MODEL`, `ZAI_THINKING`, `ZAI_REASONING_EFFORT`, `ZAI_MAX_TOKENS`, and
`ZAI_TEMPERATURE`. Keep credentials out of source control.

For backend installation and startup, see
[the backend guide](../../README.md).
