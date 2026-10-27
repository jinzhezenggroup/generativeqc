# Decision: share DF shell executables by resolved schedule

Status: implemented and qualified on NVCC 12.9.1 / SM120.
Date: 2026-10-27

## Problem

Each of the 80 ordered DF shell classes instantiates three public schedule
variants. Their names do not imply different execution: compact reaches 32
lanes for larger component counts, and the shared-memory bound can reduce
packed execution to one group. At baseline `2152ae767`, the 240 class/variant
pairs resolve to 173 distinct complete schedule contracts. Sixty-five classes
have two schedules, fourteen have three, and `(3, 3, 4)` has one.

## Decision

The scientific compiler chooses the first equivalent schedule within each
class and emits its execution variant alongside every public schedule. Compare
the entire `ShellSchedule`, including cooperative ownership and storage, rather
than assuming equal block dimensions suffice. Native launch dispatch maps the
requested public variant to that compile-time representative before it
instantiates the existing launch and device templates.

## Invariants

Public warp/packed/compact selection, production tuning manifests and class
admission do not change. No orbital exchange or cross-class mathematical
alias is introduced. Rys/moment, screening, mixed/strict arithmetic and
panel/packet specializations retain their own template dimensions. The selected
lane/group ownership, shared-memory allocation, reduction order, primitive
work, task clipping and transactional strict retry remain unchanged.

No new allocation, synchronization, runtime compiler or generic fallback is
introduced. Invalid variant values still return the existing CUDA error.

## Rejected alternatives

- Dropping packed or compact as a public choice would discard distinct
  schedules for some classes and alter the tuning interface.
- Comparing only thread count would conflate different subgroup ownership.
- Grouping different angular classes would require separate mathematical,
  weight-layout and derivative-scatter proofs.
- Assuming compiler/linker folding removes named template instantiations does
  not establish reduced front-end work or CUDA entry-point count.

## Evidence

The actual 80 objects contain 932 kernel entries instead of 1,260. Every
original kernel maps to identical instruction bytes and device resources.
DF object bytes fall from 356,273,672 to 343,371,360 (3.62%); the same configured
complete library shrinks by 1.62%. One cached compile phase per arm takes
272.19 versus 188.91 seconds; retained cache statistics include 14 candidate
CUBIN-stage hits, so this is not a cold-cache timing distribution.

The host mapping/dispatch/ledger suite passes 151 cases; independent mathematical
regressions pass 470 with 12 skips. The complete-force GPU matrix passes all
72 cases against PySCF. Its explicit scalar/BLAS panel contract now requests
panel storage: automatic occupied response otherwise caused the same six
counter failures on both arms. This changes the test setup, not production
selection or numerical tolerances.

Seven endpoint configurations, three fresh-process pairs each, cover RHF/UHF,
all selectors, batch four, water8 streamed response and water16. Cold, warm and
changed geometry retain identical energies and semantic work, with maximum
force difference below 8.18e-14 Eh/Bohr and maximum median time ratio 1.0031.
The 1.05 non-regression gate passes. Smaller exploratory memory budgets that
failed or were cancelled are retained as limitations rather than support claims.

See [the source-matched receipts and reproduction scripts](../../../../benchmarks/results/df-schedule-sharing-20261027/README.md).

## Revisit when

If another schedule-dependent property is introduced, include it in the
compiler-owned schedule contract before reusing executables. Requalify actual
device resources and complete cold/warm/displaced endpoints when changing the
mapping, compiler flags or numerical execution templates.
