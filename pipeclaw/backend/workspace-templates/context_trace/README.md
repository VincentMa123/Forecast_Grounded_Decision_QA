# Context traces

Each session writes `context_trace/<session_id>.json` as its durable trace.
The file stores session metadata, messages, tool calls, context injections,
decision-log entries, and produced artifacts so a run can be replayed and
audited.

`TRACE_TEMPLATE.json` is the minimal starting shape. Runtime code owns the
schema and fills the arrays as the session progresses; do not put secrets or
large generated files directly in the trace.
