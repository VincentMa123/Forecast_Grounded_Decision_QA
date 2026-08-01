# Audit Contract

Load this contract before classifying candidates. Report only consequential, evidence-backed risks; preserve useful boundaries and explicit behavior.

## Audit categories and proving evidence

| Category | Evidence that can prove or disprove it |
|---|---|
| Stale, unreachable, superseded, or legacy behavior | Entry points, import/call/reference traces, feature flags, release history, tests, public imports, dynamic registration, external compatibility contracts |
| Redundant execution paths, competing implementations, obsolete aliases | Call graphs, routing/dispatch branches, equivalent inputs and outputs, migration docs, compatibility consumers |
| Duplicate parsing, validation, mapping, projection, error handling, or configuration | Side-by-side source semantics, shared schemas, caller expectations, tests showing equivalent responsibility |
| Multiple sources of truth | Independent constants, state stores, schemas, or configuration resolvers for the same concept; demonstrated divergence or drift path |
| Hard-coded assumptions | Literal paths, identifiers, thresholds, model/environment values, duplicated constants, deployment/configuration requirements |
| Needless abstraction or harmful indirection | One-use wrappers, pass-through helpers, stateless classes, or large helpers traced through callers and stable interface boundaries |
| Hidden failure behavior | Broad catches, swallowed exceptions, silent/default-success fallbacks, recovery branches, bridges, compatibility shims, error-path tests and logs |
| Logical or semantic inconsistency | Contradictory branches, divergent schemas, inconsistent success/failure semantics, invariants, tests, and caller handling |
| Bug-prone edge cases | Partial writes, unbounded growth, stale state, invalidation gaps, concurrency hazards, unsupported input combinations, cleanup paths, boundary tests |
| Standards or future-plan conflicts | Exact repository instructions, architecture decisions, roadmaps, release constraints, implementation plans, and conflicting current lines |
| Missing protection around high-risk behavior | Absence of relevant tests, validation, observability, or ownership after locating the risky behavior and its existing protections |
| Dependency and maintenance hazards | Manifests, lockfiles, imports, compatibility constraints, deprecation notices already present in the repository, ownership boundaries |

Do not count comments, types, tests, explicit validation, clear errors, or useful observability as bloat. Do not prefer denser code merely to reduce line count.

## Dispositions

- **Remove:** verified obsolete or unreachable behavior.
- **Consolidate:** duplicated behavior or multiple sources of truth.
- **Correct:** inconsistent or unsafe behavior.
- **Keep:** necessary boundary, compatibility contract, or meaningful domain separation.
- **Defer:** plausible concern without enough evidence or outside the requested scope.

Every investigated item receives a disposition, including items preserved to prevent false-positive cleanup.

## Priority

- **P0:** imminent data loss, critical security exposure, or repository-wide breakage.
- **P1:** demonstrated correctness risk, major architectural conflict, or dangerous hidden fallback.
- **P2:** real maintainability defect with bounded impact.
- **P3:** lower-value cleanup or a well-supported future risk.

Rank findings relative to one another. If equal priorities are justified, explain why their impact is equivalent.

## Evidence rules

1. Cite exact current file and line locations for every verified source or documentation claim.
2. Use Graphify only to locate relationships; confirm them in current source, configuration, tests, or documentation.
3. Trace callers, consumers, entry points, public imports, configuration, registries, decorators, plugins, reflection, dynamic imports, tests, and external contracts before claiming code is dead or removable.
4. Distinguish production, generated, vendored, fixture, cached, and test-only paths.
5. State the concrete inconsistency, affected behavior, likely impact, smallest safe future remediation, and how that remediation would be verified.
6. Do not infer runtime occurrence from the existence of a branch. Cite execution evidence or make the conditional nature explicit.
7. Treat absent tests as supporting evidence only after the risky behavior itself is proven; absence alone is not a defect.

## Uncertainty rules

- Label speculative concerns as **Hypothesis**; never count them as verified findings.
- Use **Defer** when external consumers or compatibility evidence determine removability.
- If Graphify is missing, stale, incomplete, or fails traversal, state the limitation and continue with feasible source inspection.
- If repository instructions conflict, report the conflict instead of choosing silently.
- If tests cannot be run or would write repository state, leave dynamic verification outstanding.
- If scope prevents full inspection, name the inspected areas and the remaining coverage gap.
- Confidence reflects evidence completeness, not severity. Use High, Medium, or Low and explain material gaps.

## Exact report template

```markdown
# Codebase health audit

## Executive assessment
- Overall risk: Critical | High | Moderate | Low
- Verified findings: N
- Highest-risk area: ...
- Graph coverage: current | stale | incomplete | unavailable

## Prioritized findings
| # | Priority | Disposition | Finding | Evidence | Confidence |

## Finding details
### [P1] Finding title
- Evidence: exact file and line references, callers, tests, or plans
- Why it matters: concrete impact
- Smallest safe remediation: read-only recommendation
- Verification: commands or tests needed after a future fix

## Cross-cutting patterns
- Repeated structural problems that span findings

## Keep or defer
- Investigated item and why removal or change is not currently justified

## Recommended remediation sequence
1. Independently verifiable future change

## Audit limitations
- Graph freshness, uninspected external consumers, unavailable tests, or other confidence limits
```

Only include actionable verified findings in the prioritized table. Put hypotheses and non-actionable investigations under **Keep or defer**. If no consequential issue is verified, state that clearly and list the areas examined.
