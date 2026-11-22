# Decision: Compose fitted PBE Hessians with the public resource budget

Status: implemented
Date: 2026-11-21

## Decision

Use the prepared DF Fock inventory for the same all-electron orbital and
auxiliary topology passed to CUDA preparation. The KS plan replaces the
Direct provider bytes with the fitted retained and source-construction
capacities and uses the v2 fleet composer. A missing or rejected DF inventory
never falls back to Direct sizing. Retain the native fitted budget and metric
threshold in the request identity; a global budget does not retune the fit.

The host estimate composes the native source packer, both solver allowances,
geometry/derivative basis copies, and the separately bounded optional raw
Hermite, MD Coulomb, and host panel allocations. Charge the whole fitted host
envelope across all phases. This deliberately does not assume an allocation
has retired by the time response starts. It can overreserve substantially on
small systems; the number is a logical capacity allowance, not sampled RSS.

## Evidence

Slurm `2231` on RTX 5090 passed eight endpoint regressions. The restricted and
unrestricted fitted PBE cases published dense matrices and HVPs under the
global ledger, matched unbudgeted fitted results, reported DF response work,
and passed both preflight and native capacity rejection checks. The five host
suites passed 284 tests (11 skipped).

Slurm `2232` repeated all three public budget cases, with the UKS case now
using a distinct 8-AO auxiliary basis and a 32 MiB explicit DF sub-budget.
It verifies that the request retains both basis shapes and that the Fock
capacity is not replaced by a Direct estimate. Source-matched logs are in
`build/hessian-gpu-sm120/r2scan-orbital-f/hessian-fitted-public-regression-n5.log`
and `build/hessian-gpu-sm120/r2scan-orbital-f/hessian-fitted-aux-regression-n5.log`.

## Remaining composition

The [full-range hybrid decision](2026-11-21-cuda-global-hybrid-fitted-hessian-budget.md)
extends this inventory to PBE0 and B3LYP. Fitted long-range owners independently
retain value and metric buffers and may construct after the primary owner is
live. Nonlocal correlation and its independent grid need their own inventory
before admitting those models under `ResourceBudget`.

## References

- [Prepared fitted device and metric workspace capacity](2026-11-21-cuda-prepared-df-fock-capacity.md)
- [Current Hessian contract](../../../../docs/developer/dft_hessian.md)
