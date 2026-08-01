# Graph Report - .  (2026-08-01)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 6 nodes · 8 edges · 2 communities (1 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `06069a0d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0

## God Nodes (most connected - your core abstractions)
1. `register()` - 3 edges
2. `normalize()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `normalize()` --references--> `register()`  [EXTRACTED]
  plugin.py → registry.py

## Import Cycles
- None detected.

## Communities (2 total, 1 thin omitted)

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Not enough signal to generate questions. This usually means the corpus has no AMBIGUOUS edges, no bridge nodes, no INFERRED relationships, and all communities are tightly cohesive. Add more files or run with --mode deep to extract richer edges._