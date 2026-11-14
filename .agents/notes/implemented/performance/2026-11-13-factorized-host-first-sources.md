# Decision: share the canonical factorized first leaf in native integral sources

Status: implemented
Date: 2026-11-13

## Problem

The shared-XC change made the actual ICH3I complete map-work plan fit its unchanged
cap, but the changed complete native attempt timed out after 1,800 seconds
without publishing a raw Hessian. Its stationary energy/gradient/root checkpoint
qualified independently. Later preparation lacked phase-resolved progress, so
that attempt alone cannot identify which later stage consumed the remaining time.

A bounded investigation compared two already-linked first-derivative leaves on
eight s--f Cartesian quartet classes. The older derivative entry evaluates the
canonical recurrence three times, once for each independent center. The existing
factorized first specialization shares its primal factors and returns value plus
all twelve center derivatives. Isolated leaf ratios were 3.15--14.17, with maximum
gradient difference `7.55e-15`. These are a hypothesis and compatibility control,
not independent scientific qualification or complete-endpoint speed evidence.

## Decision

`HostKsNuclearIntegrals` now reuses
`integrals::full_range_primitive_first_cpu`, already generated and linked for
first-only mixed-source contractions. Each actual primitive evaluation uses the
same public Cartesian component packing and a seventeen-field, unit-scale record.
Public AO polynomial coefficients and radial contractions remain in the external
scale exactly once. Both streamed and admitted coordinate-panel schedules use
the same first specialization; no new native recurrence or derivative algebra
is introduced. All twelve mathematical-center partials remain separate until the
existing generated physical-owner projection. The primitive value is not
substituted for any derivative.

The source no longer includes or generates the older derivative declaration.
CMake and the independently compiled test fixture migrate directly to the current
linked first specialization. The coupled target's existing stationary-force-AOT
prerequisite remains: the ordinary native energy/gradient owner still uses its
incumbent implementation. That owner, public basis, S/T/V, SCF/CPKS, frozen
curvature representation, XC, residual ECP and D3 are not changed here.

The native CPU Hessian bridge now owns optional journal scopes across complete
first-source preparation, frozen-integral preparation and each nuclear column.
It records the actually selected schedule and admitted work before execution.
Selection is explicit rather than inferred from equal plan counts. A value named
`admitted_*` is not executed work; a completed direction is not an accepted full
matrix. Output/receipt transactionality and the raw symmetry gate remain intact.
Best-effort journal I/O does not affect scientific results or exceptions.

## Invariants

- Preserve normalized public AO topology, Cartesian component order, signed
  RKS/UKS density weights and every generated J/K incidence and coefficient.
- Preserve exact common-translation omission, physical-center projection and
  complete-request work/resource caps. Logical primitive visit counts do not
  become smaller merely because a leaf shares its internal primal work.
- Preserve owner snapshots, epoch/alias/reentry checks, finite rejection and
  transactional preparation/output publication. No reference, finite difference
  or runtime compiler enters production.
- Do not change the stationary force provider or public ECP saddle admission
  from this local first-source optimization.
- Qualify changed FP64 reduction order with independent normalized-AO/primal
  source controls, then actual complete endpoints. A leaf benchmark is not the
  complete method's runtime or scientific acceptance.

## Evidence

The new source/core/validation cohort is
`build/qualification/host-factorized-first-source-20261113`. Production identity
after the implementation build is
`0e0a1372969364c19c7c9901b62c8dc4d85ed1272f58d394abc8cef57d0d361b`.
Its tests include `test_host_nuclear_integrals.py`'s independent s--f
Cartesian/spherical RKS/UKS streamed/resident sources, signed densities, moving
D/W, alias/lifetime/transactional/cap failures, and actual contracted-basis raw
curvature controls. Scientific tolerances are unchanged. Component qualification
and any complete molecular outcome are retained separately in that cohort.

Final component qualification passes all 231 integral tests, 66 public/source/
CMake tests, ten native/API/CLI groups and six documentation tests, without
skips. Sphinx warnings-as-errors, compiler/SCF/native ownership, Ruff, C++
formatting and whitespace checks pass. The actual native and independent XC/ERI
work plans remain unchanged; reduced internal leaf work is not a reduction in
the logical primitive-visit census.

The public CPU regression additionally executes resource refusal, same-owner
recovery, admitted-versus-observed work bounds and completed-direction evidence.
It verifies that an unwritable journal preserves the actual detached raw matrix;
it does not pin native function names or internal implementation call counts.

Sealed parent shared-XC cohort:
`254e90d11b7df30f0c6b68d6df929ce76492d5890be1b9483421bf5ef7f1983f`.
The linked-leaf investigation is separately retained in
`build/qualification/host-first-leaf-hypothesis-20261113` with archive SHA256
`5763bdfbc9afcb6231f695e072a394807428c3aaf92ce535428f92f27832b445`.

This follows `2026-11-13-shared-host-xc-contractions.md` in this directory. That
note's unchanged-first-source boundary describes its earlier XC-only change;
this separately qualified decision changes the first-source backend.

## Rejected alternatives

- Increasing the full endpoint time/work caps or repeating the unchanged
  expensive timeout does not remove repeated first-leaf work.
- Replacing the ordinary force owner too would expand the scientific scope and
  obscure whether the first-source change improves the actual later schedule.
- Returning to a reference engine or adding a handwritten first-derivative
  formula would violate native ownership and scientific independence.
- Using changed reduction order without independent source gates, or declaring
  the full Hessian/TS qualified from leaf parity and map admission, is invalid.

## Revisit when

Source-matched complete-endpoint phase evidence identifies further repeated
primitive, contraction or response work. Preserve explicit bounded schedules and
complete endpoint/independent scientific gates for every subsequent improvement.

## Terminal complete-attempt observation

The source-bound attempt ended on November 13, 2026 at the unchanged 1,800-second
limit (wall time 1,800.03 seconds, maximum RSS 427,336 KiB). Its journal records
the stationary energy/gradient checkpoint at 546.189423352 seconds and completed
first-source preparation in 556.726212999 seconds. It then entered frozen-integral
preparation with the actually selected `projected_matrix` schedule. Polynomial
density projection completed in 0.269355 milliseconds, but that enclosing
preparation did not return before timeout. No raw Hessian or nuclear column
completed. Unpublished completed-API counters do not imply zero unfinished work.

The independent saved-state check passes: energy error `3.410605131648481e-13`,
gradient error `3.928496281535229e-12`, and maximum native gradient
`1.1870708816852198e-08`. It makes no reference SCF or displaced-state calls.
No raw-Hessian or unscaled-mode acceptance follows from these checks.

The immutable cohort archive SHA256 is
`af1dbbd2290661022679b3b3683b08fb6aaeb83ba737e0b69549f9ce97ec650a`.
All 313 final component tests/groups pass without skips, and the retained archive
also preserves the initial build failure and narrower first public selection.
Both public ECP saddle guards remain closed; the requested Ni TS, mixed-metal
stationary controls, practical complete-scale endpoint and full issue set remain
unqualified. The next investigation must address actual frozen-integral work,
not repeat this unchanged full attempt or increase its limits.

The separate follow-up decision is
`2026-11-13-factorized-host-matrix-sources.md` in this directory. It changes the
frozen-matrix leaf only and requires a new production qualification cohort.
