# Decision: native semilocal nuclear columns from shared spatial jets

Status: implemented
Date: 2026-11-12

## Problem

The mixed-pair XC source is a useful independent composition control, but
replaying its entire AO/grid/feature source for every left coordinate does not
provide a work-bounded molecular column. The moving-grid column owner already
retains the necessary order-three spatial jets and full weight gradient/HVP.
It needs an XC consumer without another derivative authority or a molecular
coordinate-squared AO tensor.

This implements the next XC composition described in
[the geometry-column decision](2026-11-12-native-moving-grid-curvature-columns.md).
It does not complete the stationary electronic Hessian or native TS endpoint.

## Decision

`HostGgaNuclearColumn` composes the moving-grid column reader, canonical TensorIR
feature/pullback maps, and the exact linked `gga_energy_mixed` MethodIR/spin
closure. One baseline and `4*spins+1` fixed field/weight left seeds recover the
weighted field adjoints and their right directions. Therefore the complete
scalar schedule charges `(4*spins+2)*points`, not `3N*points`. Each spin/tile
evaluates its primal/right features once and uses a feature VJP, its JVP, and a
VJP of the moving field coefficients. Three fixed spatial unit seeds per AO
scatter the resulting gradient/HVP to the point owner and AO center with
opposite incidence. The geometry source is not replayed per left coordinate.

The left partial holds ground D fixed. The right action accepts either an
explicit empty span for frozen D or an arbitrary finite symmetric spin/AO
`D_right`. Linearizing the canonical feature and pullback maps with respect to
both D and AOs includes that supplied response exactly once. Cotangents stay
read-only; requesting their tangents here would double count the separately
applied moving field adjoints. In particular, `energy_right` includes the
density term and is not generally `partial_R dot right`. Supplied D_right is
not evidence of SCF stationarity or a certified solution of CPKS.

Ground RKS uses total D; UKS uses occupation-one spin Ds. The production linked
full ingredient-Hessian domain, vacuum/spin endpoint branches, signed atomic
measures and compensated FP64 accumulation are preserved. No endpoint repair,
translation projection, Hessian symmetrization, displaced production geometry,
reference engine or runtime compilation is introduced.

## Ownership and resource contract

The immutable owner authenticates ground density and geometry. Both external
right directions are snapshotted and authenticated again before publication;
output may alias either external input, but not ground D or geometry. Failure
publishes nothing and releases the exclusive invocation guard for recovery.

The numeric plan includes composed readers, staged panels, map workspace,
AO-center metadata, the spin/AO D_right snapshot and compensated O(N) outputs.
It is not an RSS or time bound. Scalar-point, AO-point/motion, generated-map,
grid-pair and tape visits are explicit semantic ceilings, not scalar FLOPs.
The child geometry tape bound is per full tile: the parent conservatively
charges that bound for each tile rather than dividing the child's actual tape
capacity. An upper bound can be conservative without authorizing a smaller
internal allocation.

The scalar-pair source explicitly queries only its original three maps, so
adding column/response maps does not inflate its unrelated resource plan.

## Rejected alternatives

- Repeating the mixed source per molecular coordinate hides invariant AO/grid
  work behind a bounded-memory claim.
- Native hand-derived XC or feature Hessians duplicate the compiler authority.
- Treating arbitrary D_right as a stationary response silently omits W and
  other electronic/classical sources.
- Reusing the left field tangent as the supplied right density tangent changes
  the frozen left nuclear partial and double counts density relaxation.

## Independent validation and finite-difference diagnosis

The 26-case column suite checks complete-coordinate libcint/Libxc displaced
primal energies, grids 1--4, spherical/Cartesian AOs through f, and RKS/UKS.
Four additional primal controls explicitly displace D only on the right leg;
frozen plus density-only actions equal the combined action. It also checks
caps, immutable owners, both right-input mutations/aliases, transactional
recovery, raw frozen symmetry/linearity/rigid zeros and Python-free execution.
Supplied random symmetric D_right controls are not real SCF responses.

An initial Cartesian-f UKS gradient comparison failed at one component because
a four-corner gradient averages frames displaced along the right leg. That
adds O(h^2) right-curvature bias to the left derivative. A six-step independent
primal study isolated this stencil bias: at h=1e-3 the maximum four-corner
gradient error was 1.15293e-6, versus 2.02068e-7 for an independent left-only
two-point gradient; at h=3e-4 those errors were 1.03911e-7 and 1.80847e-8.
At h=1e-4 the mixed subtraction error grew to 4.57437e-6, so blindly reducing
the mixed step is not a sound remedy.

Tests retain h=1e-2/3e-3/1e-3, left unit scale .2, independent left-only primal
gradients and four-corner mixed derivatives. The unchanged combined derivative
gate is atol3e-7/rtol2e-6 with twelvefold convergence/3e-8 floor; the primal
gate is atol3e-11/rtol2e-12. Both corner families and the six-step diagnostic
are retained under `build/qualification/host-xc-column-20261112/`. No native
mathematics or numerical acceptance threshold was changed for this diagnosis.

## Remaining endpoint work

These component controls do not qualify a stationary molecular Hessian,
complete electronic 25--40-atom resource/time gates, a current-source installed
native CPU TS CLI, actual Ni/I/Cs electronic endpoints, a requested Ni saddle
or final unscaled harmonic frequencies. Actual S/F/D/W stationary assembly and
all other Hamiltonian, nuclear and D3 sources remain required.
