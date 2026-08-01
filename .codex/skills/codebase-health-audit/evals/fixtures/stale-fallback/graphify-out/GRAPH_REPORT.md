# Graph Report - .  (2026-08-01)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 8 nodes · 11 edges · 2 communities (1 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1

## God Nodes (most connected - your core abstractions)
1. `legacy_timeout()` - 4 edges
2. `timeout_seconds()` - 4 edges
3. `resolved_timeout()` - 3 edges
4. `AGENTS.md configuration contract` - 1 edges
5. `AGENTS.md invalid input requirement` - 1 edges

## Surprising Connections (you probably didn't know these)
- `AGENTS.md invalid input requirement` --prohibits_silent_fallback--> `legacy_timeout()`  [EXTRACTED]
  AGENTS.md → legacy.py
- `AGENTS.md configuration contract` --defines_configuration_contract--> `timeout_seconds()`  [EXTRACTED]
  AGENTS.md → settings.py
- `resolved_timeout()` --calls--> `timeout_seconds()`  [EXTRACTED]
  service.py → settings.py
- `resolved_timeout()` --calls--> `legacy_timeout()`  [EXTRACTED]
  service.py → legacy.py

## Import Cycles
- None detected.

## Communities (2 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.60
Nodes (3): AGENTS.md invalid input requirement, legacy_timeout(), resolved_timeout()

## Knowledge Gaps
- **2 isolated node(s):** `AGENTS.md configuration contract`, `AGENTS.md invalid input requirement`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `timeout_seconds()` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.310) - this node is a cross-community bridge._
- **Why does `resolved_timeout()` connect `Community 0` to `Community 1`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **What connects `AGENTS.md configuration contract`, `AGENTS.md invalid input requirement` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._