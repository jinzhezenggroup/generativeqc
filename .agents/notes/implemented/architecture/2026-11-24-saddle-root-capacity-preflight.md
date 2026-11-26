# Decision: Preflight the saddle root-panel budget

Status: implemented
Date: 2026-11-24

## Decision

Check the detached orbital export allowance, root-overlap panels and the last accepted
frame's retained numeric payload before starting a new molecular SCF. The expanded basis
and linked spin count determine the detached export shape without a solve. Use the same
checked export sizing function as the native endpoint, and authenticate the prediction
against the solved frame before allocating its coefficient output.

## Invariants

The checked sum preserves the existing conservative root-panel allowance on both the
initial and later frames. The accepted frame survives a trial's preflight refusal; a
successful Hessian already drops its batch after any warm seed has been captured. The new
seed, if requested, still has to fit within the remaining root budget at capture.

## Boundaries

This moves the existing root check earlier; it is not system-wide host admission.
In particular, orbital verification/eigensolver scratch and cross-geometry overlap
provider scratch are not covered by the root panel budget. Do not infer that the root
allowance makes orbitals executable inside the energy/force/Hessian SDK lease.

## Evidence

The native molecular evaluator test exercises restricted and unrestricted frames, reads
the root-capacity receipt with an accepted state present, and checks exact admission,
transactional refusal one byte below it, and unchanged energy and gradient. The short
request must leave both the native work counters and derivative phase records unchanged.
