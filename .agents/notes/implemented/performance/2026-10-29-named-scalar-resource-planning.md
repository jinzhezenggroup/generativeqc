# Decision: prepare named scalar resource plans at their native batch boundary

Status: implemented
Date: 2026-10-29

## Problem

Named scalar correction calculations already execute through the shared prepared
batch owner. With a ResourceBudget, the scalar wrapper first generated a complete
plan; preparation then regenerated the electronic/correction requests and
validated that internally generated plan as an external selection. Profiling the
D4 budget regression identified repeated resource planning and model resolution
in the short scalar endpoint.

## Decision

Let named scalar corrections use the prepared owner's existing automatic budget
planning. That owner reserves complete electronic and correction inventories for
both energy and force requests. Keep output-aware plain CPU semilocal planning,
whose requested force scratch changes its capacity, at the scalar boundary.
Resolve scalar KS transport options only for paths that actually consume them;
the prepared branch already owns its effective selection.

## Invariants and validation

External ResourcePlans still undergo full request, budget and selection checks.
Execution still revalidates the active inputs and schedule. No request cache,
mutable-object identity shortcut, native equation, provider allowance or budget
is changed. Scalar D4 tests exercise energy and forces at the minimum declared
capacity and one byte below, with ordinary E/F parity and bounded refusal.
Existing independent numerical and prepared replay tests remain required.

This removes redundant frontend work; it does not qualify the separate CUDA
optional-XC scheduling regression or complete #2227. Performance acceptance still
requires source-bound complete endpoint measurements with the unchanged gate.
