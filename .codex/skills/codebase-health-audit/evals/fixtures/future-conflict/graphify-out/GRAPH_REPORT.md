# Graph Report - .  (2026-08-01)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 8 nodes · 7 edges · 3 communities (0 shown, 3 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2

## God Nodes (most connected - your core abstractions)
1. `output_path()` - 2 edges
2. `retry_count()` - 2 edges
3. `AGENTS.md portable path standard` - 1 edges
4. `ROADMAP.md cross-platform output requirement` - 1 edges
5. `ROADMAP.md canonical retry requirement` - 1 edges
6. `ROADMAP.md retry source requirement` - 1 edges

## Surprising Connections (you probably didn't know these)
- `ROADMAP.md cross-platform output requirement` --requires_configurable_output_storage--> `output_path()`  [EXTRACTED]
  ROADMAP.md → runner.py
- `ROADMAP.md canonical retry requirement` --requires_canonical_retry_source--> `retry_count()`  [EXTRACTED]
  ROADMAP.md → runner.py

## Import Cycles
- None detected.

## Communities (3 total, 3 thin omitted)

## Knowledge Gaps
- **4 isolated node(s):** `AGENTS.md portable path standard`, `ROADMAP.md cross-platform output requirement`, `ROADMAP.md canonical retry requirement`, `ROADMAP.md retry source requirement`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `retry_count()` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.286) - this node is a cross-community bridge._
- **What connects `AGENTS.md portable path standard`, `ROADMAP.md cross-platform output requirement`, `ROADMAP.md canonical retry requirement` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._