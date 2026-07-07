# Mock Pipeline Data

This directory contains a small structure-compatible mock/demo runtime dataset for the public PipeClaw package.

It is included so that the FastAPI backend and the frontend UI can run end-to-end without the original private operational flow base.

Contents:

- `node_flow/`: daily node flow CSV files
- `pipeline_flow/`: daily pipeline flow CSV files
- `consumer_flow/`: daily consumer flow CSV files
- `consumer_station.csv`: mapping from supply point to station name

Notes:

- These files are synthetic demo data only.
- They are not suitable for scientific claims or production analysis.
- Replace this directory with your own internal data if you need the original business semantics.
