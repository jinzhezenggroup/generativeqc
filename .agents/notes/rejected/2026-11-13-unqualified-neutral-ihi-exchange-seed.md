# Set aside: the unqualified cold neutral I-H-I exchange seed

Status: rejected as a qualified initializer; not a statement of nonexistence
Date: 2026-11-13

## Purpose and boundary

A distinct neutral-doublet I-H-I candidate was explored for a real ECP exchange
saddle control. It is not the original I-H-H geometry or its timed-out native
search. Weaver et al., J. Phys. Chem.1988,92(20),5558--5560,
DOI10.1021/j100331a004 motivates the I+HI transition-state region through
photodetachment of IHI-. That paper does not supply a stationary geometry for
the present scalar-ECP B3LYP/def2-SVP/grid definition.

The attempted model is canonical real-spherical def2-SVP, two I28-core ECPs,
Gaussian/VWN-RPA B3LYP-D3(BJ), uncut two-body/ATM off, native-default v4/level3
grid, charge0/multiplicity2. Initial I-H distances were3.5Bohr on both sides;
a proposed4.5Bohr bracket endpoint was never executed. Every solver starts
from its own cold default guess, not native/saved/neighbor orbitals.

## Terminal evidence

`build/qualification/host-ihi-stationary-discovery-20261113` retains all sources,
actual failed states, logs, statuses and the source-bound provenance.

- The original independent DIIS driver exhausts240 iterations, converged=false,
  actual exit1 in53.17s. No nuclear gradient, bracket, root or stationary
  geometry is accepted. Its actual D/C/occupations/energies/S are retained.
- A separate read-only audit recomputes the saved state's physical Fock/energy
  without running SCF. Orbital-gradient norm7.7820e-8 exceeds both the declared
  1e-10 SCF target and independent2e-9 gate. Energy matches the saved value and
  actual spin electron counts are26/25. The convergence flag stays false.
- A distinct fresh cold PySCF Newton probe, without any saved-state seeding,
  uses the same240-macroiteration allowance, energy1e-13 and orbital1e-10
  targets. It exits1, converged=false after144.67s with physical orbital
  gradient4.7650e-7. It computes no nuclear gradient. The Newton wrapper's
  `cycles` field is0; this is not an actual zero-work count. Macro/micro/Fock
  iteration counts were not instrumented and must not be inferred from it.

Both independent energies are close to-596.15140035236Eh, but energy proximity
does not establish a qualified state. No native electronic evaluation, Hessian,
frequencies, mode index or native optimizer executes for this candidate.

## Decision

Do not promote this seed or rerun either failed attempt to manufacture missing
evidence. Both failed flags/receipts remain unchanged. Neither failure proves
that an I-H-I saddle does not exist at the requested level. A future distinct
investigation needs a justified electronic-state/convergence strategy with the
same physical Hamiltonian and explicit unchanged acceptance gates, not a looser
residual, an unreported warm guess or a successful-looking energy plateau.

The requested Ni TS, real positive I/Cs saddle controls, stationary mixed Ni/ECP
modes and25--40-atom complete-endpoint qualification remain separate unmet gates.
No public ECP saddle admission or production numerical implementation changes.
