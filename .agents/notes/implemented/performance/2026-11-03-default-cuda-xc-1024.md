# Decision: apply 1024-point CUDA XC tiles by default

Status: implemented
Date: 2026-11-03

## Problem

The 96-atom PBE0/def2-SVP campaign measured complete resident E/F and complete
fresh-process cold improvements with an explicit 1024-point XC tile, but left
ordinary users on 256 unless they supplied that override. Benchmark receipts
alone did not activate the measured configuration. The user explicitly asked
to apply 1024 directly rather than wait for workload-specific profile packaging.

## Decision

Promote the native automatic default for ordinary CUDA semilocal XC, including
direct, hybrid and unrestricted calculations, to 1024 points. Do not gate this
default on a specific geometry, basis size, GPU FP64 ratio or local profile.
When the force tile is also automatic, inherit 1024 there as well: changing
only SCF would not apply the configuration used by the successful E/F campaign.

Explicit SCF or force tiles still win. CPU, nonlocal CUDA policy and numeric
resource promises retain their existing schedules. Resource inventories quote
256 for an automatic budgeted owner and therefore must not silently grow it
after admission. Ordinary automatic SCF allocation failure retains its one
256-point reconstruction; automatic force failure retains its one-way 256-point
retirement. No numerical, grid, precision or convergence setting changes.

## Rejected alternatives

Do not rewrite the old benchmark receipts or claim a universal speedup. The
choice of a broader execution default is a user-authorized performance policy,
not evidence that every molecule or accelerator improves. Do not set the Python
field to an explicit 1024: that would erase automatic-versus-user provenance,
bypass allocation fallbacks and alter resource-planned defaults.

## Evidence

The source-matched historical campaigns in
`benchmarks/results/dft96-local-autotune-20261103/` and
`benchmarks/results/dft96-cold-autotune-20261103/` measured 10.96% less resident
endpoint time and 6.64% less complete-process cold time, respectively. They
remain evidence for explicit 1024 versus 256, not a new-source benchmark.

Executable native policy tests cover consumer GPUs, small/direct workloads,
explicit overrides, numeric-budget fallback and unchanged nonlocal selection.
Backend diagnostic tests independently compare CPU/CUDA energies to PySCF and
check the actual selected default rather than the unresolved Python option.
Fresh-source Slurm 1476 observes 1024 in ordinary 96-atom calls and verifies
complete cold/warm E/F against the independent reference and explicit 1024.
Cold work is 17 actual Fock/J/K applications and warm work is one, identical
between automatic and explicit owners. The budget-promised automatic owner
remains 256. The application evidence, full vectors and source/binary identities
are retained in `benchmarks/results/default-cuda-xc-1024-20261103/`.

The final CPU selection passes 357 tests; the scheduled diagnostic and
independent-force selections pass 28 and six tests. An unchanged transport
counter assertion fails on both the new owner (Slurm 1474) and original
source/library (1475), so the qualified selection excludes it without weakening
it. Four expanded unrelated post-SCF checks lack their pinned local D3 source
cache. Both limitations and their original logs remain in the evidence rather
than being represented as passing regressions.

## Consequences and revisit conditions

Ordinary CUDA owners can use more memory in return for less launch/batching
overhead. Users can request 256 explicitly; numeric promises remain hard limits.
Keep any future measured regression separate from correctness, and revise the
default with complete endpoint and semantic-work evidence rather than a kernel
microbenchmark.

## References

- [Original resident qualification](2026-11-03-resident-dft96-qualification.md)
- `src/dft/ks_tile_policy.hpp`
- `src/methods/dft_method.cpp`
- `docs/reference/python_contracts.md`
