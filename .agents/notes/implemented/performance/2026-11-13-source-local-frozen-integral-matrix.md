# Decision: source-local frozen integral matrix with explicit response contraction

Status: implemented; independent component/endpoint qualification remains scoped
Date: 2026-11-13

## Problem

The coordinate first-source bank removes repeated first gradients, but complete
Hessians still replay frozen-D/W integral curvature and ordered weight maps for
every global right coordinate. Most quartets depend on at most four physical
atoms. Repeating that work 3N times prevents treating small primitive lowering
wins as complete-electronic scale qualification.

## Decision

Retain the original mixed and streamed-column schedules. Add a Matrix schedule
that consumes each ordered weight incidence once, evaluates the same incumbent
primitive over independent local right motions and scatters both physical legs
directly into a coordinate-only frozen Hessian. The unique schedule uses each
non-origin physical owner at most once, with three axes: at most nine primitive
HVPs per radial quartet, independent of global N. Four identical physical owners
need the scalar value only. Ordered retains nine mathematical-slot HVPs as an
exhaustive comparison. Never merge distinct coincident owners. Raw blocks are
not symmetrized or rigid-projected.

S/T use three relative mathematical coordinates on each leg; V uses six. The
incumbent mixed primitives therefore run nine and 36 times per radial expansion.
Scalar value and left gradient publish once, not once per local right motion.
The source owns compensated [L, gradient, raw H] staging, never an AO4 derivative
tensor. Complete primitive and map work is planned before allocation/execution.

## Electronic response is not frozen away

For the same stationary quadratic integral objective, commuting its density VJP
and nuclear JVP gives the moving-density contribution <D', F'_integral>. The
existing overlap-constraint objective supplies -<W', S'>. The new compiler map
consumes the prepared first-integral coordinate panels and the actual coupled
nuclear D'/full W', without primitive work or a second native J/K formula.
Both response inputs and all map workspace are reserved/snapshotted; their
contents and the original frame are authenticated before publication.

The full matrix may allocate this additional owner and its replay staging only
from leftover integral capacity and the remaining aggregate numerical reserve.
Both original streamed children remain allocated/accounted with unchanged caps.
Admission allocates only. The complete selected work envelope, including every
D'/W' map application, is rechecked before numerical preparation. Allocation
failure may retain first-panel+streamed-curvature or fully streamed execution;
each selected fallback must fit its own original work allowance. Preparation,
owner and arithmetic failures propagate and retain work rather than retrying.

## Cross-artifact registry ownership

Adding the density-response maps exposed a real dynamic-linkage boundary in
source fixtures linked against the retained policy core. GNU-unique
`generativeqc_host_integral_maps::programs()::entries` symbols from different
inventory layouts interposed despite local DSO loading, causing missing UKS
weight-map failures. Keep the failed original run as evidence, not a numerical
pass. Qualify the private generated namespace with the complete emitted header
identity, then expose only the expected native namespace alias. Separate
inventories cannot share registry metadata; mathematical hashes/equations do not
change merely to solve linkage. A compiled two-inventory DSO coexistence test
exercises the actual boundary rather than pinning generated source spelling.

An alias-test buffer was also found too short for its declared output span and
corrected. The first full run was explicitly interrupted before reaching that
test; it already contained the registry failures above. No gate is relaxed and
no unrelated shared compiler-cache server is stopped.

## Scope and evidence

This extends `2026-11-13-host-coordinate-first-source-panel.md` and the original
`../architecture/2026-11-12-native-frozen-integral-curvature.md` decision. Current
source-matched measurements belong under
`build/qualification/host-local-curvature-matrix-20261113`. Predecessor seals
remain immutable. Source-only 3/8/32-site tests do not qualify complete electronic
25--40-atom work, memory or wall time. Neither this schedule nor a passed small
Cs raw matrix opens public ECP saddle gates or supplies a representative Ni TS.

The final source suite passes 183 cases, including direct independent three-step
four-corner energies with moving D/W, frozen matrices, mixed columns, Cartesian
and spherical through-f sources, density-response alias/finiteness/cap checks,
physical-owner degeneracies and compiled cross-inventory registry coexistence.
The new native full-matrix controls retain three independently admitted schedules:
first-bank plus matrix, first-bank plus streamed curvature and fully streamed.
Low-storage fallback cannot borrow the smaller matrix schedule's work allowance.

The real Cs matrix receipts reduce curvature ERI primitives from 4,398,324 to
1,400,544 while first preparation remains 333,745. Curvature S/T/V primitives
change from 207,360 to 186,624. This does not reduce the coupled solves: RKS/UKS
still require 164/152 induced Fock actions and 5,496,624/5,094,432 response XC
point visits. Do not turn the 68.16% integral-curvature work reduction into an
equivalent complete-endpoint speedup claim or practical large-system acceptance.
The source-driven matrix is one bounded composition improvement, not a substitute
for qualifying the unchanged electronic response and grid work at scale.

Both new real public Cs RKS/UKS endpoints pass raw H, two independent HVP
directions, matching central/displaced states and all three cold-gradient
matrices. Finest raw-H errors remain 9.0146214e-9 and 5.2988553e-9. Full raw-H
endpoint times are 111.23 and 209.79 seconds in concurrent qualification runs,
not controlled speedup experiments. Inclusive native-plus-oracle driver VmHWM
is 578756 KiB, not isolated endpoint RSS.

The additional response/stationary suite is split into finite groups: 14
response and five stationary-control cases pass. Its separate six-case
independent reconverged-energy group reaches the unchanged 900-second deadline
with actual exit 124 after five numerical artifacts; this is not a full six- or
25-case pass. Preserve that timeout and incomplete last-case work. These groups
exercise the original streamed stationary composition; the new optional matrix
path is independently exercised by the native three-schedule controls and both
real public Cs endpoints. Nothing here restamps the predecessor's timed-out
25-case run or completed Ni/terminal IHH evidence.
