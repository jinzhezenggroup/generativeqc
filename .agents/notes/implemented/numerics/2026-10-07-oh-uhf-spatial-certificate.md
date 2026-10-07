# Decision: keep raw OH UHF comparisons separate from spatial diagnostics

Status: implemented
Date: 2026-10-07

## Problem

Issue #1791 retains four CPU OH/STO-3G UHF raw density/projector failures from
e3f66381/c9d9f8ef. The public selective capsule has hashes and scalar outcomes,
not recoverable original density matrices. A later independent oracle can select
a different orientation in the degenerate transverse pi manifold. Equal energies
alone cannot establish that two determinants describe symmetry-related states.

## Decision

The issue-specific validation tool has independent `raw` and `symmetry` results.
Raw spin densities and metric projectors keep the strict absolute 1e-7 gate.
The diagnostic considers only an SO(2) spatial rotation about the nuclear z axis
for O(0,0,0), H(0,0,1.834) or H(0,0,1.85234), with the exact six spherical AOs
O 1s, 2s, 2px, 2py, 2pz, H 1s. The same rotation acts on both spin blocks.

Fitting the pi-plane angle is only a candidate-generation step. Qualification
requires geometry, S/H/full chemist ERI invariance, independently reconstructed
Fock covariance, spin counts (5,4), Hermiticity, metric idempotency, physical
commutators for both original states and the transformed state, reconstructed
and endpoint/reference energy agreement, and transformed full-D/projector
agreement. Every measured error and its gate is exposed.

The runner captures densities retained by the actual primary energy+forces
endpoint before batch closure, then runs moved geometry through the same batch.
It neither calls reference export nor starts another native solve to substitute
for that density. Its independent PySCF solve uses the actual bundled primitives,
explicit spherical AO order, and frozen oracle controls. New executions retain
their actual source/library/tool identities and raw outcomes, even when the
historical raw error is not reproduced exactly.

## Rejected alternatives

- Relabeling historical FAIL as PASS: the matrices cannot be reconstructed from
  their hashes, and a separate symmetry diagnosis does not pass a raw comparison.
- General AO/occupied-subspace Procrustes alignment or separate spin rotations:
  these do not establish a symmetry of the physical Hamiltonian.
- Rotating production densities or changing occupations, SCF guesses or tolerances:
  that changes the endpoint being diagnosed.
- Applying a symmetry fallback to a nondegenerate fixture: strict raw comparisons
  remain necessary outside this explicitly bound OH domain.

## Evidence and boundaries

`tests/python/test_oh_uhf_symmetry.py` includes algebraic positive and adversarial
checks, including a stationary integer-projector state with different energy,
broken S/H/ERI symmetry, incompatible spin rotations, wrong geometry/AO order,
nonfinite inputs, and a nondegenerate raw-comparison rejection. Synthetic checks
do not establish molecular CPU acceptance. The companion runner and recorded
molecular fixture provide separate source-matched evidence. The capture at
`7c07fa309cf7f9123cde697e460169e481bfca01` used distinct scalar/OpenBLAS
libraries with matching native source identity and an independently bound
PySCF 2.14.0 oracle. Both providers' original/moved rows retain raw FAIL and
separate symmetry/force PASS: raw D errors are 0.8194010709038728 and
0.467758639658178, while transformed D errors are at most 7.86223e-11.
Physical residuals are at most 1.62936e-11 and energy discrepancies at most
5.68435e-14 Hartree. The dedicated qz suite passed 14 tests, including
non-equivalent molecular determinant rejection. See the fixture card at
`tests/data/oh_uhf_symmetry_1791.md` for work counts and reproduction.

The diagnostic is a validation tool, not an adopted canonical determinant policy.
Whether this domain should require a unique determinant or accept independently
certified equivalence classes remains unresolved in #1791. This slice uses
`Refs #1791`; it does not close that policy question, #1002, or any GPU gate.

## Revisit when

A reviewer agrees on the determinant policy, AO representations or supported
geometries change, or a genuine non-equivalent state is found that passes all
current Hamiltonian, physical-state and transformed-density checks.
