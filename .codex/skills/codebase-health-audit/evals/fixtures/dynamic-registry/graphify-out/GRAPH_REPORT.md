# Graph Report - .  (2026-08-01)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 8 nodes · 10 edges · 3 communities (0 shown, 3 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2

## God Nodes (most connected - your core abstractions)
1. `normalize()` - 3 edges
2. `register()` - 3 edges
3. `run()` - 2 edges
4. `AGENTS.md decorator plugin discovery` - 1 edges
5. `AGENTS.md PLUGINS lookup reachability` - 1 edges

## Surprising Connections (you probably didn't know these)
- `AGENTS.md PLUGINS lookup reachability` --documents_dynamic_reachability--> `run()`  [EXTRACTED]
  AGENTS.md → main.py
- `AGENTS.md decorator plugin discovery` --documents_dynamic_reachability--> `normalize()`  [EXTRACTED]
  AGENTS.md → plugin.py
- `normalize()` --references--> `register()`  [EXTRACTED]
  plugin.py → registry.py

## Import Cycles
- None detected.

## Communities (3 total, 3 thin omitted)

## Knowledge Gaps
- **2 isolated node(s):** `AGENTS.md decorator plugin discovery`, `AGENTS.md PLUGINS lookup reachability`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `normalize()` connect `Community 2` to `Community 1`?**
  _High betweenness centrality (0.286) - this node is a cross-community bridge._
- **Why does `register()` connect `Community 1` to `Community 2`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **What connects `AGENTS.md decorator plugin discovery`, `AGENTS.md PLUGINS lookup reachability` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._