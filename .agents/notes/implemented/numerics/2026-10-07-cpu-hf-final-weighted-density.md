# Decision: close CPU UHF force state before publication

Status: implemented
Date: 2026-10-07

## Problem

Ordinary CPU UHF finalization diagonalized a physical Fock evaluated at the
converged density, projected a new determinant density from that frame, rebuilt
the physical Fock at the projected density, and immediately published a
stationary analytic force.  Two distinct finite-convergence effects remained:

1. the Pulay energy-weighted density used orbital energies from the preceding
   F[D_old] while the other force inputs used the projected D and rebuilt F[D];
2. even after replacing that lagged weight by D F[D] D, the projected
   determinant was not necessarily a sufficiently stationary fixed point of its
   rebuilt physical Fock.

Issue #1790 exposes the second effect as a standard-control UHF H2O+ torque
residual that shrinks under tighter SCF controls.

## Decision

CPU UHF force finalization reuses the shared bounded final-state selector after
the ordinary converged solve.  The initial physical Fock is diagonalized and
projected exactly as before; the selector then evaluates the physical Fock at
that projected determinant and validates its commutator/fixed-point state under
the caller's existing density and energy tolerances.  If necessary it performs a
bounded physical correction and advances determinant generations.

Only after that state is accepted is the Pulay weight formed from the same
determinant and physical Fock:

```
W_sigma = D_sigma F_sigma[D_alpha, D_beta] D_sigma
```

RHF keeps its existing finalization schedule but uses the same-generation
`D F[D] D / 2` weight rather than lagged orbital energies.

## Rejected alternatives

- Do not tighten the public SCF tolerances.  That can reduce the residual but
  changes the requested solve and masks finalization semantics.
- Do not relax, average, or project the torque gate.
- Do not stop at the weighted-density provenance fix.  On CI that change alone
  reduced the reference-route #1790 torque only to
  2.5885168305900348e-9 Eh, still above the unchanged 1e-9 Eh gate.
- Do not add a second unbounded SCF driver.  The shared final-state owner already
  provides bounded correction, fixed-point checks, generation tracking, and
  same-generation weighted densities.

## Invariants

- Density, Fock, weighted density, and two-electron derivative supplied to one
  accepted UHF force evaluation refer to the same returned determinant.
- Public max-iteration, energy, density, and screening controls are not rewritten.
- The correction is bounded and force-specific; energy-only UHF preserves the
  existing finalization schedule.
- Reference and scalar CPU eigensolver routes share the same final-state policy.
- Post-SCF physical Fock work is reported in the existing Fock-build diagnostics.
- No force or torque projection is applied after assembly.

## Evidence

The regression uses the #1790 standard H2O+ input: def2-SVP, real spherical
24 AOs, charge +1, multiplicity 2, exact FP64 integrals, 150 maximum iterations,
energy tolerance 1e-12, density tolerance 1e-10, and screening tolerance 1e-14.
It applies the unchanged max-absolute torque gate of 1e-9 Eh to both ordinary CPU
reference and scalar eigensolver routes.

Historical #1790 evidence records approximately 2.664e-9 Eh torque at the
standard controls and approximately 3e-11 Eh only after much tighter controls.
The first same-generation-W CI experiment still measured
2.5885168305900348e-9 Eh on the reference route, distinguishing the dominant
finite-stationarity closure from the smaller lagged-W contribution.

## Consequences

UHF force publication may perform additional bounded physical Fock/eigen work
when the converged SCF iterate is not yet a valid final fixed point.  Energy-only
UHF does not pay that correction cost.  The returned force state gains explicit
physical-state validation rather than depending on accidental convergence
tightness.

## Revisit when

The ordinary CPU SCF solver itself returns an explicitly validated final-state
owner with the same generation and fixed-point guarantees, allowing this
post-convergence selection to become pure reuse.

## References

- #1790
- #2043
- `src/scf/solver/final_state.hpp`
- `src/scf/solver/mean_field_driver.cpp`
