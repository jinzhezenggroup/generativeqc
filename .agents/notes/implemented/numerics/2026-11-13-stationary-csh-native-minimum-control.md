# Decision: retain a stationary CsH minimum as an ECP molecular negative control

Status: implemented (qualification control only, not public saddle admission)
Date: 2026-11-13

## Problem

Nonstationary ECP Hessian agreement does not qualify mode counts or prove that
the shared molecular saddle consumer handles a stationary electronic state.
The original IHH search and independent NiI discovery have terminal incomplete
or failed receipts; repeating or relabelling those calculations is not evidence.
A same-level minimum is a useful negative control, not the requested Ni TS.

## Decision

Retain a distinct neutral-singlet CsH minimum candidate and qualify its actual
shared native molecular energy/gradient/orbital/Hessian owner against independent
cold reference states. Preserve canonical real-spherical def2-SVP, Cs46 ECP,
Gaussian/VWN-RPA B3LYP-D3(BJ), two-body/ATM-off and native-default v4/level3 grid.

The independent test-only fixture discovery brackets the radial gradient on
4.2--5.6 Bohr and solves it with bounded scalar root finding. It is not native
optimization, a target-Ni preoptimization stage or production fulfillment of
the fixed-method saddle request. Sixteen actual cold states are retained.
The candidate distance is 4.90221668973618 Bohr. The independent gradient maximum
is about 1.90e-14 Eh/Bohr, orbital gradient8.01e-11 and electron count10.

## Evidence

Evidence is retained in
`build/qualification/host-csh-stationary-discovery-20261113`. Its producer is
source `04226e07d04647fccf604519189b3e0a98815cf4b7ed6f37723030bec65390a6`,
core `91861368aaf934af6848ec135526613d0fab9d78ec6c0a1a3877f0e428c2f001`.
The frozen parent and source inventory are independently verified, not restamped.

- Actual native orbitals match the saved independent C/D/S: density error
  2.51e-10, orthonormality error1.91e-14 and minimum occupied overlap
  0.9999999999999994. Native coefficients are never reconstructed from a reference.
- Total energy and gradient errors are1.71e-13 Eh and3.11e-12 Eh/Bohr. Native
  gradient maximum3.13e-12 establishes stationarity for this control, not the
  five criteria of an executed optimizer.
- Thirty-six cold displaced states form every raw Hessian column at0.003,
  0.001 and0.0003 Bohr. Native maximum raw-matrix errors are5.99e-8,6.64e-9
  and6.43e-10, respectively, below the unchanged1e-5 gate. Independent state,
  root, partition, asymmetry and translation gates all pass without cross-seeding.
- Native raw asymmetry2.78e-17 and translation residual5.56e-17 are checked
  before mode analysis; response residual1.13e-12. No production matrix repair
  or finite-difference production Hessian is used.
- With explicit Cs-133/H-1 masses132.90545196/1.00782503223 dalton, both native
  and independent analyses give five rigid modes and one positive Cs-H stretch,
  unscaled855.03177588 cm^-1, with zero imaginary modes. All three reference
  stretch overlaps exceed0.99. No mode is deleted to manufacture an index.

The accepted native preparation-through-mode endpoint takes75.893746s in this
particular run:14 SCF iterations,16 Fock builds, six response directions,
58 response iterations and128 induced Fock actions. Detailed semantic work
and separate process memory/owned-reservation records remain in raw receipts.
This is not a25--40-atom measurement or a controlled speedup claim.

## Validation-instrumentation failure

The first one-off frame wrapper value-initialized the C saddle-controls struct
but did not assign its raw-symmetry tolerance. The resulting exact-zero gate
rejected harmless nonzero asymmetry after computing H. Its exit1, actual native
state, failure receipt, source and binary remain retained; no matrix was published.

The second wrapper explicitly copies the existing native saddle default1e-10
from `geometry::saddle::Options`; its core, method, geometry, SCF, response and
oracle gates are unchanged. This fixes wrapper initialization, not a production
numerical gate or scientific failure. Future standalone consumers must initialize
every C control they use; zero initialization does not apply C++ optimizer defaults.

## Boundaries

This qualifies only a stationary CsH RKS minimum negative control through the
actual shared internal molecular owner. No native optimizer is run and no
first-order-saddle or five-gate search is accepted. Public ECP saddle guards
remain closed. I/Cs positive saddle consumers, a stationary mixed Ni/ECP mode
fixture, complete25--40-atom endpoint qualification and the representative
requested-level Ni TS remain separate unmet obligations.
