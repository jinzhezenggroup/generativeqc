# Decision: Inventory fitted full-range hybrid KS without a Direct provider

Status: implemented
Date: 2026-11-21

## Decision

Extend the [fitted PBE inventory](2026-11-21-cuda-pbe-fitted-hessian-public-budget.md)
to canonical PBE0 and B3LYP. The fitted state/XC leaf counts the K matrices
in the allocator layout and shares the rebuild allowance with native
preparation. It does not run a Direct inventory query. The separate fitted
source query receives the RKS occupied rank (alpha electron count), not a
zero-rank Coulomb-only promotion. UKS does not receive the restricted
projection allowance. The native capacity formula still honors explicit
fitted device caps and retains bounded panel fallbacks. A different MethodIR
under the same descriptive selector does not inherit this qualification.

A full-range K slot and a Direct exchange provider are independent concepts.
Using the Direct stationary estimate for fitted hybrid forces would describe
the wrong integral owner, and treating fitted K as pure J would omit SCF
state. This change does not add a numerical schedule or alter DF thresholds.

## Evidence

Slurm `2233` on an RTX 5090 passed 11 tests: seven public global-budget
cases (including the four new fitted hybrid spin/method combinations) and
four compiled CLI fitted-hybrid matrix/HVP consistency cases. Both hybrid
UKS cases use neutral H3 doublets with two occupied spin channels, a
distinct 12-AO auxiliary basis and an explicit 32 MiB fitted cap. The
public tests compare the full matrix and HVP with the unbudgeted fit and
exercise both preflight rejection and response allocation failure without a
leaked ledger charge. Build, launcher statistics, binary hashes and Slurm
output are under `build/hessian-gpu-sm120/r2scan-orbital-f/` with prefix
`hessian-fitted-global-hybrid`.

Six host suites passed 260 tests (11 skipped). The native leaf ABI tests
verify that fitted K increases state without changing XC in either spin
layout and that invalid flags do not publish outputs. The host Hessian
admission fixture exercises the migrated allocation calls through its
instrumented allocator, keeping the existing failure and cleanup controls.

## Limits

An independently prepared long-range provider must be composed at its actual
construction phase and with its own retained storage. Neither range
separation nor nonlocal correlation is admitted by this inventory.
Automatic long-range fitting can retire a split-budget candidate and retry
exact exchange using the original primary budget. Its plan must bound both
transactions without assuming the split persists or summing two retired
candidate states. The [fitted r²SCAN+rVV10 inventory](2026-11-21-fitted-rvv10-hessian-public-budget.md)
separately accounts for the independent nonlocal grid and XC arenas as well
as the VV10 pair and ingredient workspace.
