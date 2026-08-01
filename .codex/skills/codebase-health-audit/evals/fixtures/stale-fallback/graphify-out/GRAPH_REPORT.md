# Graph Report - .  (2026-08-01)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 6 nodes · 9 edges · 2 communities (0 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `06069a0d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1

## God Nodes (most connected - your core abstractions)
1. `legacy_timeout()` - 3 edges
2. `resolved_timeout()` - 3 edges
3. `timeout_seconds()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `resolved_timeout()` --calls--> `timeout_seconds()`  [EXTRACTED]
  service.py → settings.py
- `resolved_timeout()` --calls--> `legacy_timeout()`  [EXTRACTED]
  service.py → legacy.py

## Import Cycles
- None detected.

## Communities (2 total, 2 thin omitted)

## Knowledge Gaps
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `resolved_timeout()` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `timeout_seconds()` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._