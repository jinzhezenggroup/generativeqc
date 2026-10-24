# Decision: allow finite nonzero host GMRES triangular pivots

Status: implemented
Date: 2026-10-15

## Problem

The host Python cutover exposed a native absolute-pivot rejection: A=1e-15 I,
b=1e-15 [1,2,-1] failed with a unit relative residual under default controls,
whereas the preserved Python least-squares controller solved it in one iteration.
An absolute triangular pivot floor confuses physical scale with singularity.

## Decision and invariants

Attempt every finite nonzero triangular pivot. Reject nonfinite coefficients and
assembled candidates, and continue to require the actual unpreconditioned physical
residual for acceptance.
The existing Arnoldi/Givens breakdown thresholds, convergence tolerances,
iteration/callback budgets and actual-work accounting are unchanged. Native
failure remains terminal; there is no Python numerical retry.

This bounded repair affects the shared host controller only. It does not change
resident/device arithmetic, direct diagonal-preconditioner admission, callbacks,
ownership, ABI, storage admission or molecular equations.

## Rejected alternatives

A relative floor based on the whole triangular matrix newly rejected an existing
successful solve of diag(1e15,1), b=[1e-15,1]. A per-column relative floor still
rejected [[1e15,1e15],[0,1]], b=[0,1], whose exact solution is [-1,1]. Such floors
add an unintended condition-number cutoff instead of testing the true residual.

A relative Arnoldi terminal-breakdown guard also lost an existing successful
solve of diag(1e18,1), b=[1,1], which needs a residual-correction restart. A broader
relative-breakdown/checkpoint redesign is outside this cutover repair. Changing
the Givens or Arnoldi gates is unnecessary for the demonstrated identity regression.

## Evidence and limits

Public C/C++ and Python tests cover signed tiny/ordinary/large identity operators,
independent nonsymmetric exact solutions with explicitly zero breakdown tolerance,
existing mixed-scale successes, inconsistent singular systems, and unrepresentable
backsolve coefficients and overflowing right-preconditioned combinations. Sparse
actions can ignore an infinite nullspace coordinate, so candidate finiteness must
be checked before its physical callback. Actual callback counts, preserved finite
initial guesses and physical residuals are checked.
Independent mixed-scale probes compare against original-controller successes.

The existing absolute Arnoldi/Givens gates still limit default solves of tiny
nonnormal operators; this also predates the cutover in Python. The repair does not
claim general operator-scale invariance. Callback rounding near the smallest
subnormals can erase algebraic residual information before GMRES sees it.
Prior molecular endpoint timings and work histories remain bound to the source
revision that produced them. Repaired-source RKS/UKS complete-solve,
reconverged-density and recycling correctness checks passed. This bounded repair
makes no new timing claim.

References: #2542; #2227; `docs/reference/native_api.md`.
