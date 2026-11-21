# Decision: Charge the retained r²SCAN-3c correction owners before SCF

Status: implemented
Date: 2026-11-21

## Decision

Budgeted CUDA Hessian preparation uses the exact named r²SCAN-3c graph and
the method-owned basis. Before any device context is created the D4 request
reserves both the D4 and gCP primal lifetimes, while preserving the typed D4
diagnostic for equality checking. The new gCP layout query uses the same
numerical allocation size as `GcpCudaPlan` and the same host allowance as
named preparation. The correction-specific cap reserves the largest per-item
combined allowance for every equal share; the global plan costs the actual
sum of the retained owners. These are distinct bounds.

The Hessian borrows the immutable correction data. Its action and status
panels are charged by the Hessian arena, not by an independently prepared
second-order gCP plan. The optional PBE/D4 fleet reservation does not cover
this composite. Fitted preparations require only the SCF value inventory;
this does not admit fitted force requests or CPU composite budgets.

## Evidence

Host inventory tests verify D4/gCP separation, replicated gCP allocations
for ragged systems, and rejection one byte below the correction-specific
cap. The focused host run passed 89 tests with 13 skips. The requested
build reused the verified sccache 0.16.0 launcher; pre/post statistics,
binary hashes and validation logs are in the local SM120 build under
`r2scan-orbital-f/hessian-r2scan3c-budget-*`.

Slurm 2255 on RTX 5090 passed all four public budgeted tests: RKS and UKS
with Direct and CUDA DF. The matrix and HVP passed independently reconverged
energy differences including D4 and gCP, native parity, symmetry, translation,
the device peak cap and response scratch retirement without host fallback.
