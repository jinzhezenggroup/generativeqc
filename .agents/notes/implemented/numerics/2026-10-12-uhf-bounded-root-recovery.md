# Decision: use certificate-bound nonlinear UHF restarts, not a second SCF solver

Status: implemented (internal CPU slice)
Date: 2026-10-12

## Problem
A complete real internal UHF Hessian can detect a spin-broken saddle and return
a certified negative direction (#2240), but such a direction alone is not a
converged electronic state. Publishing the lowered trial energy or continuing
a downstream method from unvalidated rotated orbitals would be unphysical.

## Decision
Introduce a bounded CPU recovery *controller* that reuses the exact existing
UHF response J/K Fock source, rotates both spin orbital frames according to the
certified Hessian convention, computes true nonlinear trial energies and sends
downhill AO density guesses to an explicit pre-existing SCF callback.

For each returned canonical state, validate source/geometry/spin identity,
reported energy versus an independent provider evaluation, strict energy
descent and a fresh full-real-OV stability certificate. Distinguish identical
AO roots and global spin flip at equal spin counts. Block any ambiguous
classification, stale provider or inconsistent energy.

This is an internal CPU proof of the response-to-SCF handoff and does **not**
activate public root/force publication, unrestricted Kohn-Sham XC stability,
complex or spin-flip excitations, or geometry-to-geometry root tracking.

## Rejected alternatives
- Do not implement a Python fixed-point/DIIS production UHF solver beside
  the existing native source; that would duplicate numerical ownership.
- Do not treat the quadratic Hessian prediction as accepted nonlinear energy.
- Do not use PySCF as a production recovery solver or fall back to an RHF root.
- Do not identify all symmetry-related roots by merely swapping alpha/beta;
  global spin reversal is the only explicitly admitted equivalence here.

## Invariants
A published recovery result requires the original immutable certificate and
the recovered immutable certificate to remain source-current. The callback
cannot change charge, alpha/beta populations, geometry, overlap, hcore,
physical Hamiltonian or AO representation. All SCF attempts have finite
predefined rotation angles, and a failed/nonconverged trial cannot supply
a partial success.

## Evidence
The independent two-site UHF Hubbard dimer with U=1.2, t=0.2 has a symmetric
internal saddle at E=0.2 Eh and a broken-symmetry solution at
E=-2*t^2/U=-0.0666666666666667 Eh. A separate numerical fixed-point solver
using exact onsite interaction converges signed rotated seeds to that root
with <S^2> about 0.888889 and unaltered M_S=0, and root-equivalence checks
collapse the two spin-flipped determinants. Additional CPU negative tests
cover insufficient budgets, stale provider, false energy, invalid callback,
unchanged saddle and previously stable references.

The focused regression is tests/python/test_uhf_recovery.py. A model gate
does **not** replace independent molecular H2/PySCF or native SCF CI evidence.

## Consequences and revisit conditions
The CPU controller is intentionally small, dense and limited to real internal
rotations. It must remain internal until native source-to-reference export and
public intended-root/force gate policies are independently qualified.
Integrate UKS using its actual XC response owner, not by relabeling UHF.

## References
- https://github.com/jinzhezenggroup/generativeqc/issues/1826
- https://github.com/jinzhezenggroup/generativeqc/pull/2240
- docs/developer/uhf_stability.md

Agent: ChatGPT
Model: GPT-6
