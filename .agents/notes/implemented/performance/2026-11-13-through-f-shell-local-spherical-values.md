# Decision: extend shell-local spherical value preparation through f

Status: implemented
Date: 2026-11-13

## Problem

The canonical Ni def2-SVP records include f polarization. Native CPU value-only
preparation previously admitted shell-local spherical projection only when all
shells were through d. A single f shell therefore selected a molecule-wide
Cartesian rank-four ERI allocation followed by global spherical projection,
including for all the s/p/d quartets in the same molecule. Bounded derivative
sources do not remove this complete-SCF-preparation allocation cliff.

## Decision

Expose the existing f-shell Cartesian value recurrence as a shared quartet
producer with separate Cartesian and spherical consumers. Do not duplicate its
Hermite/Coulomb mathematics, primitive normalization or accumulation order.
Share the existing local projection consumer between the generated s/p/d and
retained f producers. Extend its invocation-local Cartesian block from 6**4 to
10**4 doubles (80,000 bytes); f component storage remains bounded by one shell
quartet, not the molecule's AO count. Restore only the symmetry-orbit members
belonging to the current ordered shell block before projection.

The local schedule admits value-only, real-spherical bases entirely through f.
Cartesian values, derivatives, range-separated values and g+ bases retain their
existing schedules. An f quartet in a g+ basis still uses the same retained
Cartesian producer; the higher-angular fallback remains disjoint.

## Invariants

- Preserve contraction normalization, real-spherical component ordering, existing
  primitive/component sums and the recurrence owner.
- Scatter canonical ERI symmetry only for producers that own that symmetry;
  arbitrary user tensors still use the ordered generic transformation.
- Do not infer a complete memory/work bound: the final public ERIs remain AO**4.
- Keep conservative host setup resource allowances until allocation lifetimes
  are independently accounted for; this change does not tighten the planner.
- Do not widen the public ECP saddle gate or change requested chemistry,
  precision, numerical acceptance gates or optimizer controls.

## Evidence

The native projection test retains spherical/Cartesian parity, derivatives,
asymmetric ordered-adapter behavior and independent scalar-source checks. Its
four-distinct-f case limits each allocation during actual preparation to 8 MiB:
the public 28**4-double tensor fits, while the old 40**4-double Cartesian tensor
does not. This is a per-allocation contract, not a peak-memory measurement.

The linked native probe exports every production ERI for four independent
Libcint comparisons: distinct all-f centers or contracted mixed s/p/d/f shells,
each with forward and reversed shell ordering. Acceptance is absolute 3e-12
plus relative 2e-12, together with exact public permutation symmetry.
Terminal results and producer identity are recorded separately in
`build/qualification/host-f-shell-local-values-20261113`.

The final source-bound run passes 113 Python cases without skips and all 11
selected native/API/CLI groups. The exact same native projection executable
passes against the new core but exits 1 with `std::bad_alloc` against the frozen
predecessor; loader resolution and predecessor hash are retained. This negative
control is one-off evidence, not a permanent dependency on an old binary.

A separate atomic Ni preparation uses the pinned canonical def2-SVP records
(11 shells, 31 spherical versus 36 Cartesian AOs). Every overlap, core-Hamiltonian
and ERI element agrees with independent Libcint: maximum absolute errors are
8.89e-16, 5.12e-13 and 2.85e-14, respectively. No SCF, old Ni optimization or
stationary calculation is restarted. Its child-process RSS includes process
startup and possible inherited fork memory; it is not isolated preparation
memory or a measured reduction. Sphinx warnings-as-errors and final formatting,
lint and source-ownership checks are retained with the cohort.

## Rejected alternatives

Keeping molecule-wide Cartesian storage for f discards a safe local consumer
already available to the same recurrence. Replacing the retained recurrence
merely to change scheduling would introduce unnecessary scientific risk.
Recomputing all source quartets on every Fock action could bound memory but
amplify work, so it is not a substitute for complete endpoint qualification.

## Consequences

The fixed projection block is larger even for s/p/d-only systems. It remains
independent of molecule size and is reused across both producers. The final
dense AO**4 tensor, response work and complete 25--40-atom qualification remain
separate unresolved scaling obligations. No general wall-time speedup, total
peak-memory reduction, stationary fixture or Ni transition-state acceptance is
inferred from the local allocation test.

## Revisit when

Complete preparation allocation/lifetime measurements justify a smaller resource
reserve, or a geometry-backed streamed Fock plan can demonstrate bounded source
work as well as memory at the complete requested endpoint.
