# Decision: Serialize physical orbital export on the Hessian lease

Status: implemented
Date: 2026-11-24

Orbitals used in saddle root tracking are certified against the prepared physical Fock
state, not copied from the last SCF proposal. The detached export receipt already
counts verified D/F/C, staging C and the caller's C, but deliberately excludes the
physical owner's diagonalization and verification scratch. Treating it as a full host
inventory would permit a leased export to exceed its declared workspace.

The CPU export now checks a separate native workspace bound against the leased batch's
declared derivative slot and takes the pool workspace lock before authenticating the
state. The model takes the largest phase across overlap orthogonalization and spin
diagonalization, determinant verification, verified-frame construction, and output
staging. It counts live matrix and spectrum payloads, including the Jacobi permutation,
but not allocation metadata. Existing retained SCF sources and caller-owned coefficients
are not charged to that slot. The old root-panel cap still applies independently, so
an unleased molecular calculation does not see a change in `--root-budget-bytes`.

No scientific provider, validation gate, or eigensolver changes. Exact and one-byte-
short leased exports are tested on RKS and UKS; refusal must not write the output or
receipt. This does not admit overlap integrals, optimizer panels, warm seed storage or
cold damping, so it does not provide an aggregate saddle host limit.
