# Decision: admit cross-shell AO tiles for the CCSD MO provider

Status: implemented
Date: 2026-10-03

## Problem

The CCSD provider already selected source tile widths and output batches under a
checked memory budget. However, `NativeBlockProvider` silently clipped every
candidate to the widest shell. The 28-AO STO-3G water cluster consequently used
10,000 width-three source reads even with 8 GiB available. Each read repeated the
cyclic MO transform and accumulation into complete output blocks. The CPU path
performed 10,853,326,848 transform FMAs, and the CUDA path launched 10,000 source
batches/70,000 block transforms. GPU source/transform preparation dominated the
approximately 65 s warm CCSD(T) endpoint after independent replay was optimized.

## Decision

Add an explicit basis-wide AO tile domain to `NativeBlockProvider`. CCSD
considers widths from one through the public AO dimension and continues to use
the existing generated block capacity and ordered source-reuse/tile selectors.
Candidates that do not fit the complete retained-output schedule are rejected
before numerical source work. Smaller tiles and multiple output batches remain
bounded fallbacks. Requests still use each AO value once per source scan.

The source APIs already accept arbitrary public-AO ranges: shell boundaries do
not restrict their recurrence scratch. The existing CUDA source writes directly
to the existing transform arena on the consumer stream. No new integral tensor
cache or host oracle is introduced. The planner can select a whole-basis tile
when admitted, and charges its full numeric storage; it does not require one.

Other MO consumers keep the shell-bounded default because some of their composed
resource contracts explicitly reserve that tile size. Expanding their candidates
without updating their planners would silently undercharge scratch. This change
is explicit at every CCSD candidate/execution provider construction.

## Numerical and resource invariants

- Preserve AO/component ordering, all source values, FP64 storage/accumulation,
  requested MO columns, coefficient layouts and cyclic contraction order.
- Larger source domains alter reduction and partial-block accumulation grouping.
  Validate complete energy/force endpoints against independent PySCF and energy
  finite differences; do not require arbitrary old rounding as a numerical gate.
- Use identical generated capacity formulas for every candidate and execution.
  Reject infeasible tiles before source reads; retain a width-one candidate.
- Do not infer a minimum feasible budget from the peak of the fastest admitted
  schedule. Tightening by one byte can legitimately select a smaller tile.
- Preserve independent expanded CC/Lambda residual checks and all tolerances.
- Public analytic forces remain limited to 12 AOs in this change.

## Evidence

The native provider contract test adds a five-AO s/p basis with unequal MO slot
shapes and compares widths 1,2,3,4,5 against a full-coordinate independent AO-to-MO
contraction. Width four exercises a partial tail crossing shell boundaries.
Every width executes at its exact admitted capacity, rejects one byte less
before any source read, and agrees with the exact source-read/value census.
The legacy shell-bounded construction remains width three on this fixture.

The public force capacity test now verifies both exact selected-peak success and
a tighter budget that increases source reads while preserving forces and staying
inside the smaller bound. The native force allocation-intercept tests separately
retain exact-cap success and one-byte-short refusal for the unavoidable force
stage. This distinguishes schedule flexibility from weakening admission.

## Rejected alternatives and consequences

Increasing the default memory budget would not remove the shell-width cap.
Caching an additional complete AO tensor would duplicate storage when the
existing source/transform interface already supports wider tiles. Unconditionally
widening the generic provider's default would invalidate other composed resource
contracts. Removing output/source work checks would hide incorrect schedules.

CCSD can use more scratch within the requested budget to reduce transform work
and launches. The existing checked planner remains the authority for feasibility.
Retain full cold, warm, repeated and changed-geometry endpoints and semantic work
counts when changing this selection policy. Revisit other consumers only with
their own complete admission and numerical gates, particularly the full force
Hamiltonian provider.
