# Demo pipeline data

This directory is the small, structure-compatible dataset used by the public
PipeClaw backend and frontend. It lets the flow and date APIs run without the
private operational data used by the original system.

## Layout

- `node_flow/` — daily node-flow CSV files (`YYYYMMDD_node.csv`).
- `pipeline_flow/` — daily pipeline-flow CSV files (`YYYYMMDD_pipeline.csv`).
- `consumer_flow/` — daily consumer-flow CSV files (`YYYYMMDD_consumer.csv`).
- `consumer_station.csv` — supply-point to station-name mapping.
- `synthetic_fixture_manifest.json` — provenance, counts, and available dates.

The files are synthetic demo data. They are not physically validated and are
not suitable for production decisions or scientific claims.

To use private data, replace this directory while preserving the same folders,
filenames, and CSV columns expected by `pipeclaw/backend/data_loader.py`.
