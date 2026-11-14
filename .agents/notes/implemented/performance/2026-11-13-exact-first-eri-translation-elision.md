# Decision: omit exactly translation-zero frozen-density ERI derivatives

Status: implemented
Date: 2026-11-13

## Problem

The native CPU frozen-density nuclear Fock source evaluated primitive ERI
gradients for every canonical AO quartet in every direction. For a Cartesian
Hessian column, quartets with no relatively moving physical centers contribute
an identically zero directional integral. Recomputing their gradients amplifies
expensive primitive work with the number of nuclear columns.

## Decision

In `HostNuclearEriSchedule::SymmetryUnique`, omit primitive ERI differentiation
when the three displacement components agree exactly across all four physical
quartet centers. Supply zero to the same compiler weight/scatter programs.
Leave the ordered comparison path exhaustive. No recurrence, normalization,
spin/exchange factor, screening threshold, or approximate arithmetic changes.

Retain all ordered map incidences and the same bounded owner-validation cadence.
The direction-independent resource plans remain conservative upper bounds;
executed primitive-work counters decrease. The budget and failure contracts do
not become density-dependent, and no larger work cap is introduced.

## Rejected alternatives

- Do not use a tolerance for equal displacement vectors. Arbitrarily small
  distinct motions must retain their exact linear derivative.
- Do not prune the curvature quartet just because its pure integral direction
  vanishes. Explicit moving-density weights can require the first integral
  gradient, and value/first fields remain part of that source's contract.
- Do not call the 32-site frozen-density probe a 32-atom complete electronic
  endpoint. It has no SCF, response solve, XC/grid, residual ECP, or D3 closure.

## Evidence

The first targeted run passes 45 executable tests, including independent
three-step normalized libcint primal differences through f, both spins and AO
representations, ordered/symmetry-unique source parity, failure/recovery, and
larger-site frozen-curvature controls. New sparse-direction controls use finite
symmetric indefinite RKS/UKS densities, translations, and tiny distinct motions.

For 32 one-primitive s sites and one moving physical atom, the conservative
canonical plan is 139656 primitive gradients and actual execution visits 16399;
the exhaustive ordered path visits 1048576. This includes pre-existing eightfold
reuse, not a speedup attributable entirely to the new omission. Actual complete
frozen-source times were 0.438/0.530 seconds versus 4.587/4.649 seconds for the
two spin layouts on that run. These are source-only timings, not endpoint claims.
Raw matrices, directions, densities, timings, work counts, generated/compiled
probe, and actual exits are retained in
`build/qualification/host-exact-eri-motion-20261113`.

The complete source suite subsequently passes 111 tests, including an
independent mixed-energy oracle for common nuclear translation with nonzero
explicit D/W motion: its curvature contribution must not disappear. Six native
API/Hessian/CLI contract groups and 106 public/source-binding/saddle regressions
also pass against the rebuilt core. The first failed public recovery attempt
remains failed in the evidence directory; its corrected test has a separate
successful exit and XML report.

## Consequences

The first public regression attempt retains a failed budget-recovery test:
that test assumed that six measured directional work counts exactly equal
the conservative whole-matrix admission bound. The correction recovers using
the already admitted explicit budget, checks actual work stays below it, and
still requires all six columns, small/per-column-cap refusals, and H@v parity.
No production budget or numerical gate changes to make that test pass.

Sparse nuclear columns avoid unnecessary primitive differentiation, especially
for quartets localized on unchanged atoms. AO quartet enumeration and compiler
map work remain quartic; curvature, primal SCF and coupled response retain their
existing costs. Practical requested-level 25--40-atom electronic qualification
and representative Ni/ECP TS acceptance remain separate outstanding work.

## Revisit when

A source-driven shell/radial or multi-column design has independent numerical,
resource/lifetime, and complete-electronic endpoint evidence. Preserve the
exhaustive bounded path and exact no-screening semantics during that work.
