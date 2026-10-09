# Decision: accept certified spatial equivalence in the frozen OH domain

Status: implemented
Date: 2026-10-08

## Decision and scope

Resolve #1791's remaining policy choice in favor of independently certified
spatial equivalence, restricted to the original/moved neutral OH/STO-3G six-AO
CPU fixture already qualified by #2087. No unique canonical determinant is
required in this domain. Raw density/projector errors keep their independent
1e-7 gates and original FAIL decisions. Production densities, SCF controls,
occupations and force/energy acceptance are untouched.

This supersedes only the unresolved-policy paragraph in
[the original diagnostic decision](2026-10-07-oh-uhf-spatial-certificate.md).
That note and the capture's unresolved-policy text remain historical evidence.
The authoritative current contract is
[the maintainer policy](../../../../docs/maintainer/oh_uhf_comparison.md).

## Why

The independent Hamiltonian has an axial spatial symmetry and the retained
scalar/OpenBLAS original/moved states satisfy the same full-tensor spatial
certificate. Selecting an arbitrary orientation of the degenerate pi determinant
is not a physical invariant. Requiring canonicalization would change production
state policy to satisfy a reference gauge, despite the already verified common
spatial symmetry. It is unnecessary for this bounded comparison domain.

Equal energy, equal occupation counts, or arbitrary occupied-space rotations
are insufficient alternatives: they could accept a genuinely different root.
The exact spatial transformation and all Hamiltonian, Fock, physical residual,
energy and transformed-density gates remain mandatory. This is not a stability
or intended-root selection solution to #1002 or #1826.

## Evidence and compatibility

The independently captured tensors, hash-bound certificate and runner remain
unchanged. A separate policy evaluator recomputes certification from actual
inputs rather than trusting recorded PASS strings. Retained #2087 evidence
covers both configured CPU providers and both geometries. Existing molecular
promotion, Hamiltonian mutation, incompatible-spin-rotation and nondegenerate
negative tests remain enforced, supplemented by current-policy regressions.
No new native execution, historical-matrix recovery, GPU or speedup is claimed.

## Revisit when

Expanding the domain requires independently bound molecular evidence and a
correct spatial representation, plus non-equivalent-state rejection. Revisit
if a caller requires a canonical orientation as an explicit observable/state
contract or a non-equivalent state passes every existing certificate gate.
