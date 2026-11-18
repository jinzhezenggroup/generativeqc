# Set aside: the unqualified cold neutral NiI fixture seed

Status: rejected as current qualification evidence, not a nonexistence claim
Date: 2026-11-13

## Purpose and boundary

A stationary mixed Ni/ECP mode fixture is still missing. A separate neutral
doublet Ni/I dimer discovery trial used canonical real-spherical def2-SVP,
all-electron Ni, I28-core def2-ECP, Gaussian/VWN-RPA B3LYP with uncut two-body
D3(BJ), ATM off, and the actual native default v4/level3 quadrature. Distances
4.1/4.4/4.7 Bohr were explicit discovery hypotheses, not authoritative equilibrium
geometries, a known transition state, the requested Ni reaction, or replacements
for any supplied chemistry. No old completed NiH3 calculation was restarted.

## Terminal outcome

Evidence belongs in
`build/qualification/host-ni-i-stationary-discovery-20261113`.
V1 exits1 before an electronic solve because PYTHONPATH lacks the repository root
needed by the existing basis-reference importer. Its original script, receipt,
error and status remain. V2 adds the root only in its environment and writes to a
separate directory. It does not change physical inputs or numerical gates.

The first4.1-Bohr cold independent PySCF SCF exhausts600 iterations with
`converged=False`, actual exit1 after111.02 seconds in the diagnostic driver
(111.54 seconds for the timed wrapper). Its strict policy is energy1e-13,
orbital-gradient1e-10 with ordinary final diagonalization. The actual failed
density, coefficients, occupations, orbital energies, overlap, energy, geometry
and solver status are retained. No gradient frame or other distance completes;
no root-continuity bracket, stationary point, native state matching, raw H,
vibrational classification or native electronic evaluation is accepted.

## Decision

Do not promote this candidate or relabel its nonconverged state as a reference.
No tolerance or convergence flag was weakened and no production external-engine
fallback was introduced. The trial does not establish why SCF fails, ground-state
character, saddle nonexistence, or impossibility of another independently
specified stationary fixture. Do not restart this completed calculation simply
to fill the evidence gap. A future fixture needs a genuinely independently
qualified stationary state and all original native/oracle/mode gates; this failed
seed supplies none of them. The requested Ni TS and full issue goal remain open.
