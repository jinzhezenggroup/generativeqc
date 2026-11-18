# Decision: test raw derivatives under actual discrete-grid symmetries

Status: implemented
Date: 2026-11-13

## Problem

Raw Cartesian Hessian rotation checks have two distinct pitfalls. Away from a
stationary geometry, even a continuum rotationally invariant energy satisfies
`H (omega cross R) = omega cross gradient`; rotations are not zero modes of the
raw Hessian. A finite laboratory-oriented angular grid also need not preserve
an arbitrary rigid rotation of the molecule. Neither issue permits modifying
the raw Hessian or weakening its independent matrix gates.

## Decision

Use an explicitly grid-preserving proper rotation, a translation and an atom
permutation as a metamorphic check of actual public endpoints. For Cartesian
transform `T = P tensor Q`, require `H_new = T H_old T^T` and transform HVP
directions and values with the same `T`. A quarter-turn about z preserves the
compact product grid used by the executable RKS/UKS regression. That regression
also changes mixed atomic identities' order and covers bare and D3(BJ) methods.

The separate installed Cs/H qualification uses a proper cyclic permutation of
Cartesian axes, preserving the native default level-3 angular grid, and reverses
the atom order. It keeps canonical spherical def2-SVP, Cs's 46-electron scalar
def2-ECP, Gaussian/VWN-RPA B3LYP and uncut two-body BJ/ATM off. Exact discrete
covariance uses the existing `2e-8` raw-symmetry gate; comparisons against all
three independently qualified gradient stencils keep their `1e-5` gates.

## State matching without an additional SCF

Fresh independent AO evaluations at corresponding spatial probes determine
`phi_new(Q x + t) = phi_old(x) A`. The fit must have full rank and pass separate
held-out AO and overlap checks at `1e-10`. Check each spin density through
`D_old = A D_new A^T`, with the original `2e-7` density gate. This is independent
basis evaluation, not an electronic solver or a production reference fallback.

The archived independent state contains total energy, occupations and density,
but not orbital coefficients. Its occupied space can be recovered without SCF
from the eigenvectors of `S^(1/2) D S^(1/2)`. First require positive-definite
independently evaluated `S` and the correct integer natural occupations. Then
compare every occupied spin subspace to the mapped new native orbitals using
the original `0.99` minimum singular-value gate. Do not fabricate missing
orbital arrays, seed an SCF or infer state identity from energy alone.

An initial validator incorrectly assumed a newer archive's split electronic/D3
energy and orbital fields. Its actual KeyError, failed receipt and source remain
unchanged. The distinct v2 validator explicitly reads the original total-energy
and density schema and records its own acceptance; it never rewrites a producer.

## Evidence and boundaries

The four new compact native regression cases pass. New isolated installed Cs/H
RKS execution takes 227.240 seconds and has raw covariance error `1.621e-14`,
maximum independent matrix error `7.067e-7`, spin-density error `1.632e-10`
and occupied-root minimum `0.9999999999999981`. Separate UKS execution takes
364.650 seconds, with covariance error `6.766e-14`, maximum independent matrix
error `5.296e-7`, spin-density error `1.445e-10` and occupied-root minimum
`0.9999999999999978`. Both producers actually exit zero and remain unaccepted;
their distinct read-only v2 validation receipts each accept independently.
These are new endpoints at the transformed geometry against retained
independent references, not new independent displaced SCFs. The subsequent
native/public/policy/auditor regression has 97 passing tests.
The native installed CLI suite additionally has 11 passing tests and one
explicit positive-endpoint opt-in skip. Its new admission-only scalar-ECP
control requires rejection without a molecular-state result for both RKS and
corrected UKS, guarding the still-unqualified saddle boundary before SCF.

Evidence lives under `build/qualification/host-ecp-covariance-20261113/`.
The measured native source identity is
`9f648efbe1d578fe5c7a5ecae11ce6ede0175bf1872cdb70b59e01382ac47656`;
the installed core SHA-256 is
`f3f2927c1c897af1a3304097b6caf5507cbe01302009d6b75500be0f58d89b55`.

This does not qualify arbitrary-rotation covariance, an ECP molecular saddle,
a stationary mixed Ni/ECP fixture, the requested Ni TS or practical complete
25--40-atom work. Existing molecular ECP guards stay closed. In particular,
the native CLI saddle's current all-electron-only frame constructor must bind
the ECP core/potential metadata before any separately qualified ECP promotion;
removing its constructor guard alone would silently change the Hamiltonian.

## Rejected alternatives

- Forcing `H (omega cross R)` to zero away from stationarity.
- Calling finite-grid anisotropy an analytic-derivative error without comparing
  the same discretized operator to independent references.
- Projecting, symmetrizing or repairing raw outputs before covariance checks.
- Treating one accepted spin layout or public ECP curvature as ECP saddle,
  stationary Ni/ECP or large-endpoint qualification.
