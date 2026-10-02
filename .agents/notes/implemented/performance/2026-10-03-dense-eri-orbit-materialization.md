# Decision: evaluate one representative per dense ERI symmetry orbit

Status: implemented
Date: 2026-10-03

## Problem

The bounded resident RHF reference cache removes repeated integral evaluation
from physical Fock builds, but its dense producer still evaluates all eight
permutations independently. The qualified 56-AO reference takes about 29.5
seconds per warm call even though these permutations represent the same real
Coulomb integral. The following MO provider has its own source lifetime.

## Decision and invariants

The compiler emits a value-only materialization schedule that accepts the
existing flat batch-by-n^4 launch index. Only i>=j, k>=l and the lexicographic
pair order (i,j)>=(k,l) evaluate the unchanged FP64 contracted integral. That
thread writes all eight dense permutations. Every dense slot belongs to exactly
one representative. Coincident indices produce identical repeated stores from
the same thread, so no atomic operation or inter-thread synchronization is
needed. The padded launch tail remains guarded.

The output layout, allocated bytes, launch grid and Fock consumers are unchanged.
With P=n(n+1)/2, the expensive contracted evaluations are P(P+1)/2 per system;
the dense tensor still contains n^4 values. This is not an eightfold reduction
in launched threads, primitive operations, stores, or complete endpoint time.
The completed reference journal reports the contraction census separately from
dense values and resident bytes. Reference export is a single-system owner.

The compiler owns the schedule in a separate generated materialization header.
Both pre-existing Cartesian and contraction headers remain byte-for-byte
identical to the parent revision. Native SCF retains allocation and launch
ownership; no second integral algebra or production reference oracle is added.
No derivative or screened direct-contraction schedule inherits this fold.
Ordinary resident RHF/UHF consumers also use this producer, including real
spherical d/f basis functions.

Orbit mates now share the representative's FP64 rounding, rather than each
reducing their own primitive order. Independent physical energy/force gates
therefore remain required even though the mathematical symmetry is exact.
The optional RHF cache admission, budget fallback, stream lifetime and
256-MiB/s-p reference domain remain unchanged. The MO provider does not borrow
this cache. The public CCSD(T) force boundary remains 28 AOs.

## Evidence

The host test executes the actual emitted helper with a counting contractor
and an independent triangular-pair value oracle. It checks every dense slot,
multiple systems, repeated indices and padded tails for 1/2/3/7/14/28/56 AOs.
The 28-AO case has 614,656 dense values and 82,621 contractions; the 56-AO
case has 9,834,496 values and 1,274,406 contractions.

GPU qualification covers complete cold, twice-warm and changed-geometry
CCSD(T) energy/force calls, physical RHF/UHF batch parity against independent
CPU execution, spherical d/f inputs, allocation fallback/teardown, and complete
force memcheck. Performance evidence must include both the reference phase
and the full endpoint; the later MO source and response work remain substantial.

## Rejected alternatives and revisit conditions

Adding the helper to the common contraction header needlessly invalidates
heavy unrelated CUDA translation units. The separate generated header confines
the consumer dependency while reusing the existing contraction implementation.
Computing all mates retains unnecessary expensive work; atomics add no ownership
benefit for this unique representative mapping.

A packed launch could remove inactive threads but needs an independently
validated quartet mapping and full endpoint evidence. Revisit it only if the
remaining flat-grid overhead matters in measurements. Cross-phase cache reuse
requires explicit geometry/source identity, overlapping lifetime accounting and
stream handoff; permutation symmetry alone does not establish those contracts.

## References

- [RHF reference residency decision](2026-10-03-cuda-rhf-reference-residency.md)
- `tests/python/test_eri_orbit_materialization.py`
- `benchmarks/ccsdt_prepared_endpoint.py`
