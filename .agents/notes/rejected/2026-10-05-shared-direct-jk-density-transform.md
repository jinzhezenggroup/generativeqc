# Rejected: one density transform for independent generated J/K

Status: rejected for lack of demonstrated complete-endpoint benefit
Date: 2026-10-05

## Candidate

The full-range complete-streaming value dispatcher can prepare K's existing
Cartesian spin-density source once and lend it to independent J/K traversals.
RKS J reads that same density directly. UKS J forms the total density from the
two transformed spin matrices, using its existing Cartesian scratch. The
scientific basis transform and sum kernels remain shared owners.

The candidate removes J's separate pair of basis-transform launches from a
joint request. It does not combine integral traversals, change J/K screening,
cache density across requests, or add storage. Standalone J, standalone K,
mixed J, range K and bounded higher-l providers retain their existing paths.
All outputs retain public AO order and are produced once on the provider stream.

## Numerical and lifetime boundaries

For UKS, summing after the linear transform changes FP64 rounding relative to
transforming the public total density. Independent signed nonsymmetric RKS/UKS
matrix oracles must cover Cartesian/spherical s/p/d and both batch items.
Cold, warm and moved SCF trajectories and strict final physical audits must be
qualified before claiming an endpoint improvement. There is no inter-call
pointer/epoch cache whose validity depends on an earlier request.

The native regression poisons standalone J preparation scratch and verifies
that the joint path leaves it untouched while producing the same independent
matrices. The previous library must fail this specific new invariant. This
negative control distinguishes an executed work reduction from the
[rejected wrapper-only candidate](../rejected/2026-10-05-unused-j-preparation-wrapper.md).

This is input data-movement reuse, not the complete independent prepared J/K
algorithm work required by #1892. Complete 48/96 endpoint evidence decides
whether this candidate should land; removed launches alone are insufficient.

## Qualification and endpoint decision

207 focused host tests passed. Finite n1 RTX 5090 Slurm job 5784 passed signed
nonsymmetric Cartesian/spherical RKS/UKS matrices against independent CPU ERIs,
mixed/canonical/range/derivative gates, complete native KS, and whole-process
memcheck/synccheck with zero errors. The unchanged #1908 library failed the new
joint-preparation sentinel as required. The first qualification driver stopped
because remote `rg` was absent; that operational failure is retained separately.

Slurm job 5785 then completed fresh matched reference/control/candidate PBE0
energy + analytic-force endpoints at 48/96 atoms, with cold, five warm, moved,
and five moved-warm calls per process. Both arms used the frozen optimized
local-AO/indexed/phased composition; the only candidate implementation change
was this density preparation. Ordered control/candidate at 48 and
candidate/control at 96 are observations, not interleaved causal estimates.
Each arm/size had its own initial stationary artifact cache.

| Atoms | Phase | Control s | Candidate s | Candidate / control |
| ---: | --- | ---: | ---: | ---: |
| 48 | cold | 125.36796 | 125.99919 | 1.0050 |
| 48 | warm median | 11.22592 | 11.30720 | 1.0072 |
| 48 | moved | 55.30640 | 55.51049 | 1.0037 |
| 48 | moved-warm median | 11.16198 | 11.24895 | 1.0078 |
| 96 | cold | 273.22313 | 261.46337 | 0.9570 |
| 96 | warm median | 32.53611 | 33.44083 | 1.0278 |
| 96 | moved | 137.18802 | 138.39334 | 1.0088 |
| 96 | moved-warm median | 32.55717 | 33.06780 | 1.0157 |

96-atom cold trajectories differed: control 28 iterations, candidate 26.
The cold difference cannot establish a preparation speedup. Both arms used
13 moved iterations at 96; 48 used 25 cold and 12 moved; every warm call used
one iteration. Actual public Fock counts remain null in these historical rows.
They must not be filled from those iteration numbers or the separate census.

All 288 native/reference same-geometry E/F comparisons and 144 additional
control/candidate comparisons pass the unchanged 1e-8 Eh / 1e-7 Eh/Bohr gates.
Actual sparse SCF/force maps, phased Becke coverage, and the complete native
integral derivative route were checked in every native record. All failed or
negative samples and five-repeat arrays are retained. The candidate remains
1.90x / 3.30x the fresh reference in 48/96 original warm medians.

No complete-endpoint improvement is established, so this prototype is not
selected for merge or default promotion. It does not solve #1892's independent
J/K algorithm requirement. Revisit only as part of a larger prepared-source
algorithm with measured endpoint benefit.

Control: `7e5343ff567deae29e4feb6232cdb8620e35ec4f`, source
`cb81f4c481e5cadd85999a7ff791f93a95a61f9fcf7bb5917ba3c3bfffa849b8`.
Candidate: `5f957d696a8786d5fb0eb319145289bea9225c34`, source
`35a26a50a7e794b2b21ac9b5210fdc6af4d7b93bbf5dc37d28831b68951bf316`.
Raw records, source/binary receipts, run script, all-pair verifier and checksummed
summary are under `.artifacts/endpoint-5785/` in
`/data/jzzeng/qc-1892-composed-shared-density-20261005`.

Refs #1892, #1895, #1908, #1912, #1920.

Permanent selected observations, reproducible source patches, full endpoint arrays,
and the independently audited shell-task census are retained in
[`benchmarks/results/pbe0-jk-negative-work-20261005/`](../../../benchmarks/results/pbe0-jk-negative-work-20261005/README.md).
The separate census is energy-only and must not fill historical null endpoint counts.
