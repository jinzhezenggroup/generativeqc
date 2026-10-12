# Decision: preserve symmetry-reduced work for recurrence-only mixed Direct J

Status: implemented
Date: 2026-10-12

## Problem

The bounded DFT warmup screen exposed a provider dispatch bug, not evidence that
FP32 recurrence is intrinsically slower. A mixed-J request excluded generated
J, MD-J and resident FP64 values for legitimate precision-contract reasons, but
also excluded the already-budgeted canonical source. The remaining independent
kernel evaluated dense ordered AO quartets with runtime angular dispatch.
Large PBE0, PBE and r2SCAN warmups could exceed process watchdogs during the first
J build. A blocking diagnostic located the wait inside that independent kernel;
debugger wall times are not benchmark measurements.

This supersedes the provider diagnosis in the
[bounded-warmup note](../numerics/2026-10-12-dft-bounded-mixed-warmup.md), not its
historical measurements or default-promotion requirements.

## Decision

Admit mixed J to the existing optional canonical pair domain and angular
buckets. Reuse compiler-owned public/Cartesian ERI sources instantiated with
`MixedPrecisionFloat` and the incumbent FP64 orbit scatter. Evaluate every
admitted canonical source once, rather than its ordered eightfold orbit.
The source plan, Schwarz ordering, screening and capacity accounting are shared
with strict consumers. A compiler-owned exact-zero density predicate considers
both orientations of both pairs and the FP64 total-spin density. It skips only
orbits whose Coulomb scatter has no contribution.

Canonical components alone still repeat preparation for every component.
The first fix completed caffeine PBE0 in 148.21 s (FP64 50.54 s) and water16
PBE0 in 111.42 s (FP64 55.26 s), so removing the ordered-domain cliff was not
sufficient for acceleration. Retain those source-matched intermediate captures.

When its existing shell metadata/cache fits, prefer the shared compiler-owned
primitive-pair schedule instantiated with the same FP32 evaluation scalar.
One bounded Hermite/Coulomb state serves all component lanes; primitive
contraction sums and Gaussian-cache construction remain FP64. Extend complete
component capacities to orders 0--12, with smaller pair workspaces for low
orders. Shared storage uses aligned trivially-copyable raw storage because CUDA
does not support the evaluation scalar's dynamic shared initialization. Every
field is published through the incumbent preparation/barrier protocol.
Range or exchange consumers of this mixed helper fail closed. The strict
template default and all derivative consumers remain double precision.

The shell row-prefix lease must also cover all seven second-pair buckets.
Its former five-plane allocation/preparation covered only orders 3--9. Merely
extending kernel dispatch caused uninitialized low-order prefixes and an
out-of-bounds order-12 load, caught by mixed census and Slurm memcheck. Charge
all seven planes to the same optional budget and prepare orders 0--12 before
publishing the lease. Never expand consumers without expanding their work
metadata's complete domain. Retain the failed prefix prototype captures as
unqualified, not as performance measurements.

Keep J and K independent: mixed sources never supply exchange or populate/replay
the strict resident-value cache. Generated/materialized/canonical K retains its
strict dispatch. Optional-storage refusal keeps the original independent mixed
fallback. The mixed recurrence counter counts actual evaluations in either
domain, resets for every enqueue and excludes strict K work.

A second provider-selection bug persisted after the recurrence fix: the
trajectory-wide `any_lower_precision()` predicate disabled the optional strict
resident-value lease even after the solver entered its FP64 epoch. A cap-one
caffeine diagnostic profile attributed about 87 s of device time to strict
canonical J/K kernels but only 2.4 s to mixed shared J. These are profiler
observations, not complete endpoint benchmark times. Restore resident admission
at the completed handoff, preserving the original cold/native-cold provenance,
resource preconditions, budget and fallback. Publish no strict value lease in
the mixed epoch, and explicitly prevent mixed J from borrowing a strict lease.
The normal precision ledger includes FP64 value construction and all refinement
actions. Begin/restart retires the old lease before any new approximate work.
Keep AUTO's host-controlled path and its separate mixed-counter scratch;
restoring strict resident values does not admit device chunks or graph replay.

## Rejected alternatives

- Remove the generated-J guard: generated mixed Fock also narrows the density
  product and therefore implements a broader, unqualified precision contract.
- Use raw `float`: double literals would promote recurrence operations and
  undermine the intended arithmetic. Use the existing evaluation scalar.
- Share a mixed J integral with strict K: would silently change exact-exchange
  precision.
- Retain the old dense recurrence count: would misrepresent semantic work after
  source symmetry reduction.
- Enable strict chunks solely from a handoff flag: the mixed counter borrows
  chunk scratch, and control/history/budget accounting needs separate ownership
  qualification. Restoring phase-local source admission does not broaden chunk
  eligibility.

## Invariants

- Density storage/transforms, density products, accumulation and output
  projection remain FP64. Hermite reassociation and non-full-range mixed ERIs
  remain excluded by the shared scientific sources.
- Cartesian projection changes the placement of approximate summation relative
  to the old public-AO evaluator; qualify against independent ERIs, not bitwise
  agreement with the old approximation.
- Whole-shell mixed recurrence additionally shares cached FP64 Gaussian geometry
  and retains FP64 primitive sums. This is a separately oracle-qualified
  approximation, not bitwise parity with the component fallback.
- Zero density, exact UKS spin cancellation and fully screened domains evaluate
  zero mixed recurrences. Signed nonsymmetric densities retain both orientations.
- FP64 refinement, residual/energy gates and derivatives are unchanged.
- Complete endpoint timing, actual Fock counts and independent reference gates
  remain prerequisites to promotion; a corrected provider alone is insufficient.

## Evidence

The native mixed-census gate covers optional canonical storage and its zero-optional-budget
fallback, s/f fixtures, signed nonsymmetric RHF/UKS densities, spherical and
Cartesian s/p/d/f sources, strict K parity and independent CPU J error below
`3e-6`. Instrumented J agrees within `1e-13` FP64 atomic-scatter roundoff.
The host admission gate independently enumerates the complete eightfold orbit,
including coincident indices, offsets, reversed density entries and spin
cancellation. Source-matched endpoint captures are retained under
`benchmarks/results/dft-mixed-warmup-20261012/`; the earlier negative captures
are preserved unchanged.

The mixed phase-source probe checks that a strict lease is absent before the
first approximate observation and is admitted only at refinement. It compares
resident and exact-Direct fallbacks with an independent PySCF returned-density
energy for PBE0 and B3LYP. Warm/external seeds cannot acquire cold provenance;
an insufficient budget must retain convergence without optional values. Chunk
width requests do not bypass the AUTO host-control restriction.

## Revisit when

The generated streaming owner can express recurrence-only mixed J without
narrowing density products, or whole-shell reuse can preserve the same scalar
contract and improve complete endpoints. Evaluate strict-phase source/chunk
changes separately with lifetime, scratch, budget and provenance gates.
