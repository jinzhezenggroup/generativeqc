# Decision: bind the native saddle owner to one fixed molecular RKS model

Status: implemented
Date: 2026-11-11

## Problem and decision

The native callback controls have independent light-molecule evidence, but a
scratch callback harness is not a public molecular endpoint. Expose that same
C++ owner through `generativeqc.saddle`, with a separately admitted molecular
adapter. Do not duplicate optimizer mathematics, replace the Python-free CLI
launcher, or silently use an external optimizer to fill the remaining CLI gap.

The generic `correct_saddle` binds real synchronous callbacks. The molecular
`optimize_saddle` requires the existing public Hessian/forces capability and an
additional narrow CPU, direct, all-electron, unpolarized full-range hybrid RKS
domain. A qualified D3(BJ) source stays in the same total MethodIR. The adapter
is not admission for UKS, ECP, Ni or the requested mixed-element TS task.

This supersedes only the absence-of-Python-molecular-adapter part of the
[native callback note](../numerics/2026-11-10-native-saddle-callback-and-rigid-subspace.md).
Its distinction between mathematical tests, independent molecular qualification
and the requested installed Ni/ECP CLI endpoint remains in force.

## Invariants and alternatives

- One prepared SCF supplies energy/gradient and the next raw Hessian at a given
  geometry. Re-running a public cold Hessian after separately computing forces
  would obscure work accounting and permit a different electronic root.
- Preserve a detached occupied subspace only when the native optimizer reports
  that geometry accepted. Compare each cold-SCF occupied subspace through native
  cross-geometry AO overlaps to that reference, not the last rejected trial.
  Minimum principal overlap is an explicit fail-closed local policy, invariant
  to rotations within the occupied space. It is not a stability certificate or
  an unrestricted root-tracking implementation.
- Freeze the calculator physical signature and total/electronic MethodIR
  identities, including across observer callbacks. No preliminary method change
  or implicit compatibility route is permitted.
- Release response, AO and prepared-batch owners before geometry-only D3
  publication. A correction failure cannot publish an electronic-only total.
- No Python exception, including an interrupt or observer exception, can unwind
  through a ctypes callback. Preserve diagnostic strings and native failure
  state; initial failure publishes no derivative arrays or numerical metrics.
- Keep final raw Hessians and all unscaled vibrational modes. Harmonic
  publication failure preserves the native result without molecular success.
  Partial failed work counts remain explicitly incomplete, never inferred.

## Validation boundary

Executable ABI tests cover native stepping, fresh final derivatives, capacity,
callback failures, cancellation at the final observation and detached outputs.
Adapter tests cover admission before SCF, model mutation, last-accepted root
reference across a rejected trial, and electronic cleanup before additive
publication. Synthetic lifecycle providers are not scientific qualification.
The existing independently checked ammonia callbacks supply prerequisites, not
an automatic receipt for this new public adapter or an installed TS CLI.

The live serial Ni full-Hessian experiment retains its original frozen numerical
sources and native binary. This new module does not edit those owners. Its
complete public molecular endpoint still needs source-matched real runs and
independent acceptance before claiming that endpoint qualified.
