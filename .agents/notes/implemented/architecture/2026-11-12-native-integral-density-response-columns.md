# Decision: explicit D/W response in native integral columns

Status: implemented
Date: 2026-11-12

## Problem

The source-driven frozen integral column does not include the stationary
density and energy-weighted-density right response. Replaying first F/S
sources for each left molecular coordinate would lose the column schedule's
source reuse. The stationary compiler already supports differentiable D/W
weights, so no new native derivative equation is necessary.

This extends [the frozen-column design](2026-11-12-native-frozen-integral-columns.md).
It is not a complete stationary Hessian or an SCF/TS endpoint admission.

## Decision

`HostKsNuclearIntegralCurvature::apply_column` takes explicit empty frozen spans
or finite symmetric supplied spin/AO D_right/W_right. The left partial of
`Tr(D h)+E_JK(D)-Tr(W S)` holds D/W fixed. Canonical TensorIR weight JVPs
generate both primal and right weights for each ordered pair/quartet. The same
primitive gradient/JVP supplies the geometric action and the weight-response
gradient contraction, without extra primitive evaluations or molecular-left
source replay. All spin/occupation/exchange/Pulay signs remain compiler-owned.
The scalar mixed API retains frozen semantics and its original map envelope.

The existing first-gradient one-electron and overlap weights intentionally mark
D/W non-differentiable for historical artifact identity. Linearizing those
frozen programs directly correctly refuses. Request
`StationaryGradientPlan.integral_block(...differentiate_densities=True)` before
the weight JVP; the canonical ordered quartet already preserves differentiable
density incidences. Differentiate those incidence inputs, not bar_energy.

The column's right scalar includes the supplied weight tangent, so it need not
equal its fixed-D/W gradient dotted with the nuclear direction. Arbitrary
D_right/W_right is not evidence of CPKS convergence or stationary symmetry. The
caller must authenticate the actual accepted stationary response before making
such a claim. ECP/XC, nuclear/D3 and full endpoint assembly remain separate.

## Ownership and resource consequences

The exclusive invocation snapshots and authenticates all three right inputs
through final publication. Output may alias external right inputs, not ground
D/W or system/basis state. Failure publishes no fields and permits recovery.
The numeric plan now includes two spin/AO response snapshots and the nuclear
right snapshot as well as compensated physical-coordinate outputs/map arena.
Consequently it has an explicit O(spins*nao^2) term, not only O(natoms) output
growth. This is not an RSS or endpoint time bound.

Primitive ceilings remain unchanged, but column map work now explicitly
charges the canonical weight JVPs. Comparing column map terms with the old
scalar-pair terms for exact equality would hide this new work; tests instead
compare actual scheduled column terms with its own admitted envelope.

## Validation contract

Independent libcint displaced-primal controls use Cartesian/spherical f AOs
and RKS/UKS with D/W displaced only on the right leg. Left-only primal gradients
avoid the right-frame averaging bias identified in
[the XC-column note](2026-11-12-native-xc-nuclear-columns.md). Retain the existing
integral derivative gates (atol3e-8/rtol2e-6, eightfold convergence/5e-9 floor)
and h=3e-3/1e-3/3e-4. Explicit split D-only/W-only, frozen-plus-response,
no-extra-primitive-work, malformed/null/nonfinite/asymmetric inputs, aliases,
actual final input mutation and recovery are separate executable controls.
Actual static-target linkage must execute supplied responses without Python.
These controls do not replace real stationary S/F/D/W molecular qualification.
