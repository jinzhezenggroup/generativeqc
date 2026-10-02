# Decision: share ERI orbit production with complete prepared source tiles

Status: implemented
Date: 2026-10-03

## Problem

After folding the resident RHF tensor's real Coulomb symmetry orbits, the
56-AO CCSD(T) warm endpoint still takes 43.78 seconds. Its MO problem/provider
phase accounts for about 29.5 seconds. The provider reads one complete public-AO
tensor into an already charged device buffer, but the prepared Direct source
still evaluates every symmetry mate independently.

## Decision and invariants

Reuse the compiler-owned orbit materialization schedule when a validated source
request starts at zero on every axis and spans the complete public basis on
every axis. The new single-system entry point selects its input metadata from
the original packed batch and writes a single n^4 output. The resident batched
producer calls that same entry point with the corresponding output offset.
This prevents duplication of either integral algebra or permutation ownership.
The source launch is declared in the existing provider-kernel interface. Its
implementation lives with the shared dense producer; the provider host does
not acquire the resident Fock kernel interface or change its dependency boundary.

Partial rectangular tiles keep the original independent producer. Their symmetry
mates need not belong to the requested output region, so globally canonical
filtering would leave holes or write outside the caller's capacity. The full
tile is selected only after the existing dimension, element-count, item, device
and pointer checks. The caller retains stream and allocation ownership. There
is no additional allocation, cache, reference-to-provider borrowing, budget
change, or host tensor construction.

The single-system launcher retains the source producer's bounded grid and
grid-stride loop. This matters at 56 AOs, where n^4 exceeds 65,535 blocks of
128 threads. The full dense values, launch work and MO transformations remain;
only contracted ERI evaluations are folded to P(P+1)/2, where P=n(n+1)/2.
Public source_values continues to mean dense values delivered, not contracted
or primitive evaluations. A completed full tile and the independently checked
schedule determine its contraction census; partial tiles retain one contraction
per requested dense value. The RHF reference and MO source still build separate
tensors with separate lifetimes.

Representative FP64 rounding can change relative to independently reduced
permutations. Every retained endpoint must pass independent exact-basis energy,
triples and force gates. No derivative schedule, response equation, degenerate
orbital gate or public force size boundary changes.

## Validation

The host execution test checks the emitted schedule's unique work and every
dense slot for 1/2/3/7/14/28/56 AOs. It now also selects a nonzero batch item
into a guarded single-system destination, including padded launch tails.

The native provider test compares both complete and asymmetric rectangular
tiles against independent CPU integrals through f shells, for Cartesian and
real spherical AOs. Two batch members differ in geometry and primitive metadata.
Its --eri-tiles-only entry point permits focused Slurm/memcheck qualification;
the ordinary native suite also includes the new complete-tensor cases.
Complete CCSD(T) cold/warm/changed energy and force calls remain the performance
boundary, with unchanged public work and resource accounting checked explicitly.

## Rejected alternatives and revisit conditions

Using global canonical filtering for arbitrary rectangles is incorrect unless
the producer also defines ownership among the orbit images inside that rectangle.
Allocating a full tensor for every small request would defeat bounded execution.
Sharing the earlier RHF cache would require a different identity/lifetime and
budget contract; this optimization does not establish it.

Revisit partial-tile orbit reuse only if constrained-memory endpoint evidence
shows a worthwhile bottleneck and a complete ownership mapping is validated.
Revisit packed launches or cross-phase borrowing using complete endpoint data,
not the dense source_values count alone.

## References

- [Resident orbit materialization](2026-10-03-dense-eri-orbit-materialization.md)
- `tests/native/test_cuda_fock_provider.cpp --eri-tiles-only`
- `tests/python/test_eri_orbit_materialization.py`
