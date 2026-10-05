# Decision: admit optional density preparation at the native KS resource boundary

Status: implemented
Date: 2026-10-05

## Problem

The [density-panel candidate](../performance/2026-10-05-xc-density-panel-provider.md)
has fixed-density numerical and crossover evidence, but ordinary KS supplied
no optional preparation resources. Qualifying a library implementation through
a method-local test hook that secretly allocates would omit the resource
contract and could bypass the public numeric allocation ledger.

## Decision

Accept a provider-neutral `CudaXcPreparationBudget` at native KS construction.
The enclosing caller reserves these host/device bytes within its resource plan.
Zero remains the default. Host capacity must cover the prepared owner before
any optional preparation; device capacity goes to the existing canonical
lowering boundary. The method does not choose a library or change scientific
precision admission. Mere budget availability still cannot qualify a provider.

After preparation, KS reports exact matrix-cache bytes under XC device storage,
the provider allowance under provider device storage, and metadata under retained
host storage. The generic panel provider uses the shared numeric ledger for
cache allocation/free. The provider-internal allowance stays outside that ledger
and must be withheld by the enclosing planner, as for other opaque providers.

Device allocation/budget failure can discard the optional context and retain
generated execution. Host bookkeeping failure propagates instead of being
misclassified as a smaller device budget. Owner teardown releases the cache
before its stream/Fock provider can disappear. The budget declaration and KS
header remain usable without CUDA SDK headers.

## Invariants and qualification domain

No public resource-planner policy is promoted here: public callsites still
pass zero optional resources. The native test supplies an explicit 128 MiB
device and 16 KiB host reservation before enabling the lowerer's test-only
qualification. Independent CPU PBE0 checks exercise real KS convergence,
energies and returned density; they do not stand in for production performance
evidence at hundreds of AOs.

`generativeqc_ks_cuda_tests --density-provider` covers seven admission routes
for RKS water and UKS H3, at original and moved geometries, with cold/new-owner
and warm solves. Routes are unqualified, qualified, insufficient device budget,
insufficient host budget, unavailable provider, local AO, and actual numeric
ledger exhaustion. It checks exact live numeric bytes and complete cleanup.
The final energy/residual gates are `1e-9` and the independent density gate is
`2e-8`. Changed geometry uses a copied previous-geometry seed for every route.

Reported preparation timing covers native KS construction, while the Fock
provider, basis and grid are already prepared. Cold/warm timings cover complete
`CudaKsPlan::run` including exported results. These small qualification fixtures
make no speedup claim. Source/binary hashes, ccache commands/statistics, finite
Slurm qualification scripts, native regression and sanitizer logs are retained
under `.artifacts/1876-ks-density/`.

## Revisit when

Compose this reservation with potential-assembly candidates and a single
end-to-end resource plan. Qualify large 384/768 AO PBE0 energy/force endpoints
before promoting a shape policy; do not extrapolate the fixed-density crossover
or small convergence gates into full endpoint speedups.

The existing composed PBE0 baseline already uses local AO. Its retained
`benchmarks/results/pbe0-composed-baseline-20261005/components-{48,96}.json.gz`
records report the following original-geometry scientific work:

| atoms / global AOs | tiles | maximum active AOs | mapped summands | dense summands |
| --- | --- | --- | --- | --- |
| 48 / 384 | 4,608 | 342 | 25,257,336,832 | 173,946,175,488 |
| 96 / 768 | 9,216 | 536 | 75,674,112,000 | 1,391,569,403,904 |

Mapped work is only 14.52% / 5.44% of the full dense domain. Neither baseline
has any full-global-AO tile. Disabling local maps to reach the current dense
matrix candidate would restore substantial discarded work. The meaningful next
comparison must include bounded gather/pack costs at the actual local extents,
preserve empty/noncontiguous maps, and compare against the local generated
incumbent. Dense fixed-density qualification alone cannot promote that path.

References: #1876, #1886, #1959; `docs/developer/xc_native_cuda.md`.
