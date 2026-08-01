---
name: codebase-health-audit
description: Use when assessing an existing repository for accumulated technical debt, stale or legacy behavior, duplicate paths, hidden fallbacks, hard-coded assumptions, architectural inconsistencies, future-maintenance risks, or conflicts with repository standards and implementation plans.
---

# Codebase Health Audit

## Safety boundary

This audit is strictly read-only and stops after reporting. Inspect repository files, history, status, tests, configuration, and existing generated knowledge only. Do not edit or delete files, write generated artifacts, install dependencies, stage, commit, push, open a pull request, or run destructive or repository-writing commands. A later request for fixes starts a separate implementation workflow.

## Workflow

1. Resolve the audited scope, then discover and read its repository instructions, contributor or architecture guidance, and future-plan documents (`AGENTS.md`, `ROADMAP.md`, or equivalents), plus `git status`. Map every applicable requirement to current source. Exclude generated, vendored, cached, fixture, and test-only paths from production claims unless they are relevant evidence.
2. Before broad browsing, check the scope's `graphify-out/graph.json`; only then check an applicable repository-level graph. Never call Graphify unavailable without checking the scoped path. When not running from the graph's scope, pass its path explicitly with `--graph` to `query`, `path`, or `explain`. Read existing lessons/wiki when present, but never run `reflect`, `save-result`, `update`, rebuild, or another graph-writing operation without separate authorization. Treat graph output as navigation, never proof.
3. Map entry points, public interfaces, configuration resolution, registries/decorators/plugins/reflection, dynamic imports, schemas, persistence, tests, and external contracts. This map guards against false dead-code claims.
4. Load [references/audit-contract.md](references/audit-contract.md) before classifying any candidate. Inspect its categories deliberately.
5. Verify every candidate in current source with exact lines. Trace callers, dynamic usage, entry points, tests, configuration, documentation, and external-contract risk. Assign every investigated item a disposition; use `Keep` or `Defer` when removal is not proven. Separate verified facts from hypotheses.
6. Produce the contract's exact evidence-backed report. Give every distinct documented conflict its own finding with relative priority, confidence, disposition, conditional impact, smallest safe remediation, and specific future verification; never collapse or omit one conflict. Include graph, runtime, scope, and external-consumer limitations. If nothing consequential is verified, say so and list what was examined.

## Red flags

Stop and correct the audit if it:

- names files without exact line evidence;
- omits a disposition or future-fix verification;
- recognizes dynamic reachability but still invents defects or replacement machinery;
- treats unconfirmed growth, consumers, import order, or test behavior as findings;
- labels several findings equally without explaining relative impact;
- treats Graphify output as current-source proof; or
- drifts into style commentary or implementation.
