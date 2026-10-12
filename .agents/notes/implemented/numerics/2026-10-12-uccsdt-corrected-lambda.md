# Decision: distinct corrected UCCSD(T) amplitude stationarity

Status: implemented
Date: 2026-10-12

## Problem

Unrestricted triples VJPs must enter the amplitude adjoint RHS and the fresh
actual stationarity gate. Passing corrected multipliers as standard Lambda
would disable or mislabel the very scientific qualification needed by a force
consumer. Sharing native actions should not create a second CC algebra or a
dense production Jacobian.

## Decision

Borrow matching standard adjoint and bounded triples owners. Share their native
J* action, physical dual lift, parameter VJP and checked GMRES, but introduce
separate corrected owner/result types. Solve the combined CCSD+(T) RHS and
qualify the actual U* pullback of both amplitude gradients plus J*lambda.
Raw tile bars are unconstrained Frobenius bars: same-spin amplitude symmetry
is enforced by U*, not an extra quarter or an unphysical full-array gauge test.

Every qualification replays all triples channels, the physical parent and
corrected stationarity. State/template/backend identities are exact; numerical
correction values have a bounded drift gate rather than a byte-identical
replay assumption. Validate all energies, each channel and every source bar,
not only the packed amplitude gradient or total energy. Retain a qualified
correction with the checked dual for a later physical consumer.

Aggregate numeric/work admission includes both borrowed consumers, retained
old/fresh corrections, coordinate/Krylov/publication buffers and each extra
parameter action. Borrowed owner closure/configuration changes fail closed.
No owner closes borrowed resources. Standard Lambda qualification remains
standard, including when a corrected record is forged into a standard record.

## Important Remaining Fock Boundary

A canonical eps-to-Fock-diagonal pullback is necessary but **not sufficient**
for complete orbital/nuclear response. Differentiating a covariant triples
resolvent requires `d D^-1 = -D^-1 (dD) D^-1`; dD acts on each occupied/virtual
slot and includes same-spin off-diagonal oo/vv Fock elements. The restricted
compiler already implements the complete same-space resolvent moment contract
in `cc/triples_fock_response.py` and `cc/occupied_triples_fock.py`.

The unrestricted analogue can reuse the existing nine-term moments: at the
canonical reference X=C/D and Y=(C+V)/D, with the existing channel multiplicity.
Contract paired bounded tiles over five common coordinates, leaving the two
coordinates of one chosen same-spin axis open. Sum all three occupied and
three virtual slot moments with opposite signs. Their diagonal must reproduce
the already qualified eps bars exactly within numerical tolerance. Replace the
diagonal-only denominator contribution with the full moments, rather than
adding both and double-counting its diagonal. Avoid full
molecular T3 storage; admit complete paired traversal and recomputation work,
not merely resident bytes. Do not divide same-occupancy orbital gaps or assume
those gaps are nonzero. Degenerate internal spaces still require full response.

Until this chain is qualified, the standard physical owner must continue
rejecting corrected owners/results. Disabling all-rotation stationarity or
using an ov-only gate would conceal the missing oo/vv response. No complete
UCCSD(T) force contract is promoted by this amplitude-response implementation.

## Rejected Alternatives

- Corrected records inheriting from standard Lambda: accidental type acceptance
  and false method/equation provenance, including in a zero-triples limit.
- Add triples only to the RHS: Krylov convergence would not prove the actual
  corrected Lagrangian stationary.
- Trust detached correction diagnostics or exact replay hashes: hides forged
  source bars, or rejects valid finite native replay roundoff respectively.
- Diagonal denominator bars as a complete physical Fock response: omits
  same-space off-diagonal resolvent moments and internal-gauge stationarity.

## Evidence

`tests/python/test_uccsd_t_lambda.py` compares corrected multipliers on broken
H4, NH2 and asymmetric H6 with a test-only dense Jacobian/RHS from independent
PySCF spin-orbital CCSD residual and triangular GCCSD(T) energy central
differences. Fixed-canonical Fov cotangents have independent Lagrangian FD
closure. The collapsed-spin corrected dual agrees with the existing RCCSD(T)
Lambda owner at 2e-8. Tests also reject corrected records through standard qualification,
forged standard multipliers through actual standard stationarity, altered
identities/energies/channel sums/backends/bars, resource overruns and expired
owners. Empty excitation/triples domains retain a distinct zero corrected
contract. The real-CUDA tier requires finite Slurm and forbids CPU equations.

## Revisit When

The unrestricted full Fock resolvent is bounded and independently qualified,
then compose corrected Hamiltonian/UHF/nuclear response with explicit method
types, all-rotation stationarity and reconverged force closure.

## References

- #1825 and dependency PR #2340.
- `docs/developer/uccsd_triples.md`.
- `2026-10-12-uccsd-triples-cotangents.md`.
