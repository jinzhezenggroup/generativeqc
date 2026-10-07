# Decision: bind CPU HF force weights to the final physical Fock

Status: implemented
Date: 2026-10-07

## Problem

Ordinary CPU RHF/UHF finalization diagonalized a physical Fock evaluated at the
converged density, projected a new determinant density from that frame, rebuilt
the physical Fock at the projected density, and returned forces using a mixed
state.  The density, final energy, and two-electron derivative used the projected
D and rebuilt F[D], while the Pulay energy-weighted density still used the
orbital energies from the preceding F[D_old].

Issue #1790 exposed this as a standard-control UHF H2O+ torque residual that
shrinks under tighter SCF controls.  That behavior is consistent with a lagged
final-state witness rather than a missing derivative term.

## Decision

CPU HF force finalization constructs the energy-weighted density from the same
returned determinant and physical Fock used by the remaining force inputs:

```
W = D F[D] D / spin_weight
```

The restricted spin weight is 2 and each unrestricted spin weight is 1.  This is
the same physical-state ownership rule already used by the shared strict
final-state selector.

## Rejected alternatives

- Do not tighten the public SCF tolerances.  That reduces the lag numerically but
  changes the requested solve and only masks the provenance mismatch.
- Do not relax or project the torque acceptance gate.
- Do not retain orbital-energy W after rebuilding F at a different density.
- Do not add another post-SCF SCF loop solely for this correction; the existing
  final physical Fock already supplies the consistent stationary weight.

## Invariants

- Density, Fock, weighted density, and two-electron derivative supplied to one HF
  force evaluation refer to the same returned determinant.
- The existing SCF controls, iteration policy, and two post-SCF physical Fock
  builds remain unchanged.
- Reference and scalar CPU eigensolver routes share the same finalization
  semantics.
- No force or torque projection is applied after assembly.

## Evidence

The regression uses the #1790 standard H2O+ input: def2-SVP, real spherical
24 AOs, charge +1, multiplicity 2, exact FP64 integrals, 150 maximum iterations,
energy tolerance 1e-12, density tolerance 1e-10, and screening tolerance 1e-14.
It applies the unchanged max-absolute torque gate of 1e-9 Eh to both ordinary CPU
reference and scalar eigensolver routes.

Historical #1790 evidence records approximately 2.664e-9 Eh torque at the
standard controls and approximately 3e-11 Eh only after much tighter controls,
which motivated checking final-state provenance instead of changing the gate.

## Consequences

The final determinant is unchanged and no additional Fock build is introduced.
Only the Pulay weighted-density construction changes.  RHF receives the same
correction because it had the identical generation mismatch.

## Revisit when

A future finalization path replaces the projected-density/rebuilt-Fock sequence
with a validated final-state owner that directly provides a same-generation W,
or when a non-idempotent mean-field state requires a different response contract.

## References

- #1790
- #2043
- `src/scf/solver/final_state.hpp`
- `src/scf/solver/mean_field_driver.cpp`
