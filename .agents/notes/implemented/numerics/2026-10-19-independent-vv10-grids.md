# Decision: independently prescribed XC and VV10 quadratures

Status: implemented
Date: 2026-10-19

## Scope

The user stopped the grid-precision investigation and requested only a PR for
independent grids, retaining the original calculation path when grids coincide.
All precision-scan Slurm allocations were cancelled. The earlier investigation
and its raw receipts remain on `feat/readme-vv10-grid-benchmarks`; this PR starts
from master and neither deletes README methods nor publishes replacement
timings, selected grids, speedups or FP64/Fast convergence claims.

## Ownership and equivalence

`KsOptions.nonlocal_grid` is optional. Omission and complete normalized
prescription equality reuse the incumbent grid and combined derivative pass.
Equal products such as 8x4x8 and 4x8x8 are not equal rules. Every radial/angular,
radius, partition, tolerance and pruning field participates in identity.

The stable 160-byte KS descriptor and 184-byte precision extension are retained.
A discovered, exact-layout 200-byte private extension borrows a second grid
descriptor; preparation validates and copies it. Unknown, incomplete or
unsupported extensions fail closed. A nonlocal grid without a nonlocal term is
rejected. Distinct native model identities include the second prescription;
snapshot data and CUDA leases originate from that exact token's native owner.

## Independent execution

CPU RKS/UKS integrate the nonlocal term on its own prepared molecular grid.
CUDA retains its physical XC owner and uses an independent zero-scaled PBE GGA
transport for total-density features and nonlocal AO assembly. This is not a
second functional contribution: zero scales produce zero semilocal E/V, while
the qualified GGA transport supplies the rho/gradient pullback VV10 requires.
It avoids inventing WB97M-V component-scaling qualification. Energy and AO
potential are added on the owner stream; electron diagnostics retain the XC
quadrature. Errors and generation publication cover both owners, including
ordinary and bounded device-chunk execution.

The stationary derivative consumers use independent points, atomic measures,
owners and partition responses. CUDA shares bounded AO scratch sequentially
between the two traversals, never a task or active-map lease across different
grids. Mandatory secondary storage is charged to actual native allocations;
force planning conservatively includes both grid inventories.

## Explicit limits

Hessian/HVP and coupled-response paths still assume a shared quadrature and
reject distinct prescriptions before numerical work. Explicit resource-ledger
planning also rejects this unmodeled inventory rather than underestimating it.
Mixed force qualification remains limited to its previously measured shared
rule. Uniform atom-major and per-grid point admission limits remain in force.
No limit is removed to admit a benchmark, and no CPU/reference fallback is
introduced for CUDA execution.

## Evidence

Tests cover legacy descriptor preservation, a same-count/different-rule
collision, separate radii/partition transport, missing capabilities, invalid
models, same-grid complete endpoint regression, independent upstream CPU/GPU
E+F comparisons in restricted/open-shell states, and reconverged energy
differences with independent radii and partition iteration counts. Both smaller
and larger secondary point counts are covered. These same-prescription checks
use deliberately small grids, not converged-grid accuracy claims: upstream E
agreement is bounded by 1e-8 hartree and maximum force-component disagreement
by 1e-7 hartree/bohr. Explicit equal prescriptions, including native-normalized
unit radii, retain E within 1e-13 hartree and F within 1e-12 hartree/bohr.

The source-matched Release CPU build passes 448 host regression tests, with 17
explicit CUDA/environment skips. This includes nine complete CPU endpoints,
snapshot/provider authentication, option/ABI checks, resource admission and
benchmark protocol/rendering checks. The CUDA Release sm120 build succeeds;
all nine CUDA endpoint tests pass on n1's RTX 5090 under Slurm job 10407,
requesting one GPU with a finite 20-minute limit. Six compare independent-grid
RKS/UKS E+F against GPU4PySCF; two retain the implicit shared-grid endpoint;
one checks independent radius/partition force response against reconverged
energy differences. GPU4PySCF emits only its upstream SciPy deprecation
warnings. The allocation exits after verification; no accuracy sweep resumes.
