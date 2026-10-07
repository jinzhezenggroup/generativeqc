# Decision: share cached dddd recurrence in stationary DFT J'/K' sources

Status: implemented
Date: 2026-10-06

## Problem

The stationary DFT full-range derivative endpoint uses the bounded
`FullSources` scheduler, rather than the native dddd value stream. Optimizing
that stream's force helper would leave the actual PBE0 endpoint unchanged.
The retained high-order derivative consumer prepares primitive recurrence for
each component and independent atom.

## Decision

Add a separately frozen, default-off
`GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_DERIVATIVES=1` qualification schedule.
The bounded full-source scheduler and its angular-partition mode share the
existing `Dual3` Hermite/Coulomb algebra across the complete dddd component
domain. J' and K' contract the same derivative with independent retained density
weights. The compiler owns the scientific consumer; native code owns admission,
launch, optional workspace and lifetimes.

Repeated shell centers share one atom seed. N-1 distinct atoms are differentiated
and translation recovers the last. The per-CTA derivative workspace is below
32 KiB. Separate kernel specializations preserve the disabled path's register
and shared-memory footprint. Cache/offset absence, reachable/convolution
derivative schedules, non-dddd classes and other radial operators retain the
bounded fallback. Production allocates no qualification counters or component
derivative tensor.

## Rejected alternatives

- Modifying the native dddd force stream alone: stationary DFT does not call it.
- Another maintained raising/lowering derivative formula: retained forward AD
  already supplies the geometry chain rule and all three Cartesian responses.
- Extending the shared Dual3 simplex through f immediately: its larger storage
  requires a separately measured schedule and resource contract.
- Inferring endpoint improvement from preparation counts: shared AD consumption
  and bounded scheduling still require complete E/F qualification.

## Invariants and evidence

Keep individual primitive coefficient multiplication order, cached primal
geometry, axis Gaussian arithmetic, per-component Schwarz gates, independent
source meanings, repeated-atom response and translation balance. Never recover
raw factors by dividing weighted cache coefficients or Gaussian factors.

Finite n1 Slurm job 6158 passed the retained orders 5--12 value controls and
new dddd derivative controls in 31.02 seconds. The derivative controls compare
against independently traversed raw Dual3 components and host symmetry-orbit
J'/K' contractions, for both spins, combined/J-only/K-only channels, repeated
atoms, same-pair domains, screened shells and inactive claims. Four distinct
atoms, two primitives per shell and 1,296 components require 48 Coulomb
preparations shared by both sources. The scientific consumer remains
default-off pending the full release library, independent two-oxygen Libcint
responses, sanitizers and complete endpoint timing/work qualification.

Evidence is retained under n1
`.artifacts/1892-pair-forces/native-6158/`. The preceding compile failure was a
test-fixture `auto` type mismatch; numerical gates were unchanged.

## References

- Issue #1892; value foundations #2000 and #2002.
- `docs/developer/direct_pair_recurrence.md`.
- `tests/python/test_cuda_hybrid_snapshot.py` for independent Libcint gates.

The later scalar-prefix schedule supersedes the per-atom Coulomb preparation;
see `2026-10-06-direct-pair-scalar-spatial-responses.md`. The original decision
and its measurements above remain the provenance for the first derivative slice.
