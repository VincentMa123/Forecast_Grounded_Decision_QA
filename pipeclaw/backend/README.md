# Backend

This backend folder is the stripped-down open-source export for PipeClaw.

Included here:

- FastAPI entrypoint for the current agent runtime
- Trace-first agent orchestration code
- PipeClaw evaluation package
- Minimal shared utilities reused by PipeClaw
- Structure-compatible mock/demo `pipeline_data/` so the public frontend and backend can run end-to-end

Excluded here:

- The original private business flow base
- Baseline model implementations
- Historical run artifacts
- Unrelated visualization and experimental modules

If you have access to internal data, replace `backend/pipeline_data/` with your private files while keeping the same directory and CSV schema layout.
