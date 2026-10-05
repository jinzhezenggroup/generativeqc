# Decision: preserve mapped AO work in the XC density provider

Status: implemented
Date: 2026-10-05

## Problem

The 384/768-AO dense crossover cannot qualify the actual composed PBE0 baseline.
Its local AO maps retain only 14.52%/5.44% of dense summands, and its maximum
active extents are 342/536. Restoring global dense work to reach a library would
undo that reduction. The retained counts and provenance are in the
[KS preparation note](../architecture/2026-10-05-ks-density-provider-reservations.md).

## Decision

Add an indexed candidate to the existing canonical density request. Both dense
and mapped factors lower the same `symmetric_density_element()` scalar graph;
the mapped materialization gathers `0.5*(D[ids,ids] + D[ids,ids].T)` separately
for each nonempty tile. Scientific identity and strict FP64 precision remain
the same, while materialization and algorithm identity differ. Dense endpoint
qualification cannot qualify this candidate.

The generic panel provider admits runtime columns bounded by its prepared
capacity and uses compact matrix, panel and spin strides. Each GEMM operates
on the actual active extent. Preparation retains one handle and one cache;
execution never allocates or searches for a provider. The cache conservatively
reserves `spins * global_nao^2 * sizeof(double)`, even when all maps are smaller.
No maximum-active-sized reservation or per-map cache is claimed.

AO discovery invalidates a previously prepared dense recipe only after map
construction succeeds. It preserves generated execution until explicit
preparation against the new layout. Completely empty maps retain no optional
provider; empty tiles use the generated zero-domain operation. Mixed arithmetic,
signed response, insufficient budget and unavailable providers retain their
existing scientific bindings. Budget availability alone never promotes GEMM.

## Invariants

- Gather only the immutable, validated sorted map; omitted AOs stay omitted.
- Refresh the factor on every nonempty tile and captured replay. Density pointer
  identity is not a content lifetime.
- Batch offsets use the actual compact extent, never the global capacity.
- Preserve finite publication and sticky error reporting, including gathered
  nonsymmetric density inputs.
- Discovery cannot reuse a dense recipe for an indexed view. Failed discovery
  retains the previous binding; successful discovery retires it.

## Evidence and scope

`generativeqc_dft_cuda_tests --indexed-density-provider` checks an independent
long-double nonsymmetric-density oracle for one and two spins, active extents
1/17 within larger global matrices, full/tail point panels and three captured
replays with changed density, AO values and noncontiguous indices. It checks
alias rejection, finite publication and exact cache/provider/host accounting.

Five admission routes cover unqualified/qualified candidates, insufficient
budget, unavailable provider and a completely empty map. Explicit local-map
cases compare independent CPU LDA/PBE/meta-GGA E/V and sparse WB97M-V bilinears.
Discovered maps compare scaled-PBE RKS/UKS with independent full-AO CPU E/V at
`1e-10`, for Cartesian and spherical bases, including empty domains. This also
checks invalidation of a prepared dense recipe before mapped preparation.

The indexed gate, existing dense gate, full native XC suite and indexed
memcheck/initcheck run on n1/node1 through finite Slurm `main/gpu:5090:1`
allocations. C++/CUDA compilation uses verified ccache commands; source hashes,
binary hashes, scripts and cache statistics are retained in ignored local
`.artifacts/1876-indexed-density/`. Host binding/admission checks and repository
ownership checks accompany native qualification.

No speedup or production promotion is asserted here. This PR supplies the
bounded executable needed to measure the actual local domain. Qualification
must next include per-tile gather, complete fixed-density XC and then composed
SCF/force endpoints at the retained geometries and scientific settings.

## Rejected alternatives and revisit conditions

Disabling maps restores discarded work; extrapolating dense timing ignores
materialization cost. Retaining all map matrices duplicates density storage
and requires a different lifetime/resource plan. Revisit cache sizing or
batching only with actual work/traffic counts and complete endpoint evidence.

References: #1876, #1886, #1959, #1960; `docs/developer/xc_native_cuda.md`.
