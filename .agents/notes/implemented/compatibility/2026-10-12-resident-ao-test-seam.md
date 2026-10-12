# Resident AO lease tests use the existing production-loop probe

Status: implemented
Date: 2026-10-12

## Problem

Replacing the resident AO AST assertion initially extracted the feature-lease
selector from the stationary CUDA endpoint. Focused provider tests passed, but
required CI exposed two consumers of that endpoint: the large-grid probe executes
its submission loop with explicit globals, and DFT-MP-v1 capacity preflight binds
the frozen endpoint source and its resident AO policy. The extraction invalidated
both consumers even though the selection was mechanically equivalent.

## Decision

Keep the production endpoint bytes unchanged and reuse the existing
`_geometry_loop` probe from `test_stationary_large_grid_work.py`. The new tests
execute the real selection, derivative-order validation, geometry consumption,
and drain with controlled providers and feature leases. They do not require a
particular conditional spelling or an exact private call count.

This batch removes the resident AO AST assertions and overlapping pointer/call
spelling checks. It does not renew a frozen function-body hash or weaken capacity,
budget, snapshot, cache, or provider-admission checks.

## Evidence

The old assertion rejected an equivalent inverted conditional. The executable
tests accept it and reject truthiness-based map selection, missing tile offsets,
wrong map domains, and skipped derivative validation. Allocation, validation, and
consumer failures retain release ordering and do not reach the drain.

## Revisit when

The existing loop probe still uses AST extraction to isolate orchestration from
device execution. A broader callable production seam should migrate its frozen
capacity consumers and retain independent admission/lifetime oracles together;
changing only the feature-selection call leaves those consumers incomplete.

## References

- Issue #2296: independent batches replacing implementation-shape guards.
- PR #2328: resident AO feature-lease execution contracts.
- `tools/dft_mp_v1/qualify_capacity.py`: frozen source-bound preflight, not
  scientific qualification.
