# Decision: explicitly budgeted coordinate first-source reuse

Status: implemented
Date: 2026-11-13

## Problem

The complete native Hessian asks the nuclear-response owner for one source per
Cartesian column. Every source previously repeated the same normalized first
integral derivatives and the same ordered compiler incidence maps. Lowering the
derivative order of special curvature quartets alone did not establish a material
complete-endpoint acceleration. Do not repeat that claim from source timings.

## Choice

This extends the streamed owner recorded in
`../architecture/2026-11-12-native-frozen-integral-nuclear-source.md`; it does
not replace that owner's original bounded default or its historical evidence.

Keep the original direction schedule as the default and resource fallback. An
explicit coordinate schedule computes all first S/F columns in one primitive
pass, reusing gradients across their physical owners. The compiler's sole
integral-tangent JVP is linear; evaluate its unit coefficients once per ordered
incidence and scatter their products with the physical partials. No scientific
spin/exchange equation is duplicated in native code. Both schedules use the
same incumbent first-derivative primitive, not a new recurrence or screening.

The full matrix may admit this additional owner only from leftover integral
storage and inside the response reservation, including the selected GMRES
workspace. The original streamed source remains allocated and accounted for;
its numeric cap is never enlarged. Admission performs allocation only. The
complete selected work envelope is checked before preparation, so an allocation
fallback cannot execute under the smaller resident work allowance.

Unit coefficients introduce an unscaled intermediate not required by the
direction schedule. Restrict panel admission to absolute density entries at
most 1e50, keeping even quadratic density products below 1e100. Larger finite
indefinite densities retain the original streamed contract. This precondition
is checked before panel construction, not by catching a numerical failure.

Only optional-construction `bad_alloc` is a resource fallback. Owner errors,
nonfinite arithmetic and failed preparation propagate. Bind the owner before
preparation so failed actual work remains visible. Snapshot density, packed
basis and ionic charges and retain full owner/lifetime validation. Replays
snapshot external directions and publish both outputs transactionally. Failed
preparation never makes a partial bank usable.

## Boundaries and evidence

This deliberately relaxes the first source's no-coordinate/AO-storage policy,
not the no-AO-fourth-power-derivative-tensor rule. Complete curvature still
streams columns; neither response iterations nor its J/K/XC work disappear.
Source-only 3/8/32-site work tests are not 25--40-atom complete-electronic
qualification. Public ECP saddle gates remain closed until their independent
molecular-state, raw-Hessian and mode criteria are actually met.

Current validation belongs to
`build/qualification/host-coordinate-first-panel-20261113`; the predecessor
scalar-lowering and terminal original IHH seals must not be restamped or
modified. The terminal IHH timeout is not a stationary seed or TS acceptance.

## Subsequent curvature reuse

`2026-11-13-source-local-frozen-integral-matrix.md` extends this first-source
panel with compiler-owned D'/W' contractions for an optional frozen nuclear
matrix. The original first-only and fully streamed bounded fallbacks remain;
this note's source-matched historical evidence is not restamped.
