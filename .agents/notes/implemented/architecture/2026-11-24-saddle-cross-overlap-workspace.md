# Decision: Charge cross-geometry overlap to the saddle root budget

Status: implemented
Date: 2026-11-24

The root-tracking matrix allowance already covers the caller's rectangular AO metric,
but not the CPU provider that constructs it. Query a separate modeled payload from
the provider's validated basis layouts: both sets of Cartesian AO views, sparse
public-AO maps and centres, plus the larger of the single-shell build and one-axis
high-angular Hermite fallback workspaces. This sizes only provider allocations, not
the input systems, caller metric or allocator overhead.

In the molecular corrector the two layouts are immutable across frames and mass
registration changes only positions. After the first accepted state, query the layout
before preparing the trial SCF, then recheck on the actual registered system when
executing the overlap. Keep the existing kernel and ordering, including the angular
fallback; the budget does not authorize an alternative numerical path. The first
frame needs no cross-geometry overlap and its root allowance does not change.

This makes the provider workspace visible to later lease composition. It does not
account for the lifetime of the registered system copy or enable a unified saddle
host limit. Qualification includes mixed Cartesian/spherical high-angular overlap,
exact capacity and one-byte-short refusal, and preservation of the accepted root
and work counters when a trial is refused before SCF.
