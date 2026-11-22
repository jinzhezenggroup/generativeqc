# Decision: Charge the fitted r²SCAN+rVV10 grid and pair owners separately

Status: implemented
Date: 2026-11-21

## Decision

For canonical FP64, device-fused fitted r²SCAN+rVV10, use the native
variational r²SCAN state/XC layout and the same VV10 workspace contract
that validates provider preparation. A separate rVV10 grid retains its own
quadrature, grid, PBE-family AO/XC bridge, and pageable system/point data.
Sharing follows the normalized grid specifications, not matching point
counts. The second construction bound composes conservatively with the
first grid and Fock owner. This is a logical capacity bound, not RSS or
the exact bytes allocated by an incumbent SCF replay.

The provider's `nonlocal_memory_budget_bytes` remains a separate hard limit.
Test it against the actual rVV10 point count in the native query rather than
against a CPU host estimate based on the semilocal grid. The public request
does not invent an r²SCAN+rVV10 force capability. Count both grids' VV10 pair
work in the C ABI receipt, matching the CLI's reported semantic work.

## Evidence

Slurm `2236` (RTX 5090) passed 15 regressions using hash-checked library and
CLI binaries. Public fitted r²SCAN+rVV10 RKS with a shared grid and UKS with
an independent grid matched unbudgeted dense Hessian and HVP results. Both
exercise the pair receipt, hard native nonlocal cap, global preflight and
response allocation rejection, and release of response bytes. Six host
suites passed 261 tests (11 skipped), including distinct equal-point-count
grid accounting and unchanged publication on a rejected native sub-budget.
Artifacts are in `build/hessian-gpu-sm120/r2scan-orbital-f/` with prefix
`hessian-fitted-rvv10-final`.

## Limits

This qualification does not admit Direct r²SCAN+rVV10, range-separated KS,
or nonlocal response on other CUDA architectures. Range-separated fitted KS
must account for the additional radial source and both automatically selected
fit/exact transactions as described in the [hybrid capacity note](2026-11-21-cuda-global-hybrid-fitted-hessian-budget.md).
