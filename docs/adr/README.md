# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Quantum Agent Scheduler project.

## ADR Template

When creating a new ADR, use the following template:

```markdown
# ADR-XXXX: [Short descriptive title]

## Status
[Proposed | Accepted | Rejected | Superseded]

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing or have agreed to implement?

## Consequences
What becomes easier or more difficult to do and any risks introduced by this change?

## Alternatives Considered
What other options were considered and why were they rejected?
```

## Naming Convention
- ADR files should be named: `ADR-NNNN-short-title.md`
- NNNN is a 4-digit sequential number
- Use hyphens to separate words in the title

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0001](ADR-0001-quantum-backend-abstraction.md) | Quantum Backend Abstraction | Accepted | 2025-01-15 |
| [ADR-0002](ADR-0002-qubo-formulation-strategy.md) | QUBO Formulation Strategy | Accepted | 2025-01-15 |
| [ADR-0003](ADR-0003-hybrid-solver-selection.md) | Hybrid Solver Selection | Accepted | 2025-01-15 |