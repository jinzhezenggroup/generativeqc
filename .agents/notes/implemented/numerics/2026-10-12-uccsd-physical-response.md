# Decision: compose unrestricted raw Hamiltonians before AD and reuse UHF

Status: implemented
Date: 2026-10-12

## Problem

Canonical fixed-input CC bars alone do not include Fock normal ordering, the HF
reference, metric transport, independently owned spin rotations or nuclear
integral derivatives. Treating alpha/beta mixed ERIs as pair-exchange symmetric
or importing the restricted response sign silently breaks complete forces.

## Decision

Compose the existing UCCSD DAG with the compiler-owned raw spin Hamiltonian,
then differentiate once with generic TensorIR AD. Retain only three raw-MO
Coulomb owners, with `ba` as a pair-transposed view of `ab`. Use the existing
checked shared UHF response contract; it rotates `C exp(-K)`, so the generated
direction is `-K`, the orbital RHS is positive antisymmetric rotation bar, and
the final Fov constraint seed is `+z`. Metric transport gives `-sym(G_U)/2`.

Bind final weights to shell derivatives through the existing ordered shared
four-coefficient transform. CPU ERIs use packaged s/p/d kernels. CPU one-electron
derivatives deliberately retain the named native compatibility recurrence;
CUDA uses generated derivative consumers without CPU orbital/derivative
fallback. Keep this distinction in capabilities, options, and provenance.

## Rejected alternatives

- Replaying the whole CC VJP for every canonical input repeats a large DAG.
- Handwritten unrestricted RDM/normal-ordering algebra would create a second
  scientific implementation and obscure excitation-dual metrics.
- Assuming an SPD orbital Hessian excludes legitimate stationary broken-spin
  branches. Checked residual/stationarity gates are the authority instead.
- A dense AO rank-four weight or coordinate derivative tensor violates the
  shell-consumer boundary. Shell streaming trades storage for explicitly
  admitted transformation work; it is not automatically work-efficient.

## Invariants

Freshly qualify standard CCSD Lambda, raw bindings, energies, generated/shared
Fov actions, true Z residual, and all rotation blocks before publishing. Bind
borrowed lifetimes/configurations, add nuclear repulsion once, keep mixed-spin
owners distinct, and publish only detached immutable results. Primitive/source
tasks and generated scalar work have separate finite admission. Do not claim
an allocator cap for the legacy CPU one-electron recurrence.

## Evidence

The first independent raw compiler gate passed seven cases. The NH2 physical
probe gave Z residual `1.34e-12` and full orbital stationarity `3.12e-11`.
The installed PySCF analytic gradient differed by `9.62e-7` even after tightening
its CPHF tolerance. Independent reconverged UHF/UCCSD total-energy derivatives
at steps `1e-3` and `2e-4` agreed with the physical gradient to `5.18e-10` and
`4.94e-11`, respectively. Therefore keep distinct analytic (`2e-6`) and strict
finite-difference (`2e-7`) acceptance gates rather than changing production
weights to fit an approximate independent analytic oracle. Reproduction is
in the committed molecular response and gradient tests; ignored diagnostic
logs are not a release or external backup.

## Revisit when

Replace the explicit CPU one-electron compatibility consumer when a shared
generated CPU weighted derivative seam has independent qualification and
allocator-enforced staging. Improve source-driven MO projections when larger
complete-endpoint work counts demonstrate shell-transform amplification; do
not optimize isolated kernels without retaining all scientific gates.
