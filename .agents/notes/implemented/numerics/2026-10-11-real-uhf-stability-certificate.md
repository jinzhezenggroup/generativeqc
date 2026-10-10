# Decision: bound full real UHF internal diagnostics to shared response science

Status: implemented
Date: 2026-10-11

## Problem

A canonical converged UHF state can be an internal real orbital saddle. The
shared response Jacobian had no consumable stability contract or immutable
negative direction. Canonical-looking Focks alone do not bind a reference to
the current Hamiltonian, and diagonalizing an unchecked matrix cannot establish
stability or lifecycle qualification.

## Decision

Use a strictly admitted tiny full-OV dense diagnostic over the shared UHF action.
For C exp(-t K), the physical Hessian is twice the density-symmetric-OV Jacobian;
the independent rotated energy tests protect this sign and normalization.
Reuse the shared Fock seam to verify the reference/provider Hamiltonian before
diagnosis, and freshly reapply every eigenvector after diagonalization. Bind
source/reference/operator/layout identities and thresholds to immutable evidence.
Report an unresolved near-singular state instead of treating near-zero curvature
as positive. Empty domains have their own status and still validate lifetime.

## Rejected alternatives

- A bare to_dense/eigvalsh label has no resource, reference or failure contract.
- Limited-subspace Lanczos cannot certify full-domain stability; a large-system
  spectral redesign is unnecessary for this independent prerequisite slice.
- A second UHF Hessian/J/K implementation would split production scientific
  ownership. Explicit energy algebra exists only in tests as an independent oracle.
- Automatically rotating/reconverging or publishing energies/forces would claim
  a complete root workflow before recovery and intended-root gates exist.

## Invariants and evidence

The current contract and runnable gates are in
`docs/developer/uhf_stability.md`. `tests/python/test_uhf_stability.py` checks
independent full energy curvature, mixed-spin coupling, sign/factor controls,
pinned PySCF H2 saddle/broken root and Li, admission, self-adjointness, fresh
residuals and provider failure. Skipped molecular tests are not qualification.
Runtime completion and a green generic test suite are not scientific acceptance.

At the exactly symmetric stretched H2 saddle, PySCF 2.14.0's default internal
Davidson seed returned stable even though the full independent energy spectrum
contains negative spin-antisymmetric curvature. The full PySCF `2*hop` spectrum
and `with_symmetry=False` status are therefore the aligned additional oracle;
never replace full-domain evidence with the default limited seed's boolean.

## Consequences and revisit conditions

The admitted domain is small and host controlled; its complete work is 2*n
response applications and one reference Fock evaluation. Accounted numeric
storage does not promise a hard process RSS cap. No GPU or force speedup is
claimed. Revisit when a separately qualified large-system spectral method or
UHF/UKS recovery/root-publication consumer can preserve the same evidence and
failure semantics. Parent #1826 remains broader than this slice.
