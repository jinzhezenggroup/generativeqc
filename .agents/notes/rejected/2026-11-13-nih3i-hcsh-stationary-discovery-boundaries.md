# Set aside: unqualified NiH3I and H-Cs-H exchange fixtures

Status: rejected as positive saddle controls, not a nonexistence result
Date: 2026-11-13

## Scope

These are separate test-only stationary-fixture hypotheses, not restarts of the
old neutral NiI, original I-H-H search, or target Ni reaction. Neither changes
the production Hamiltonian, initializes a production optimizer from an external
optimizer, nor qualifies the requested Ni TS or a 25--40-atom endpoint.

Both use Gaussian/VWN-RPA B3LYP-D3(BJ), uncut two-body dispersion with ATM off,
canonical real-spherical def2-SVP, I28/Cs46 scalar ECPs where present, and the
actual native-default v4/level3 grid. All independent SCFs use their own cold
guesses. Energy, orbital-gradient and subsequent independent acceptance gates
remain unchanged. Failed solver flags are never promoted.

## NiH3I dianion singlet

`build/qualification/host-nih3i-stationary-discovery-20261113` records an attempted
C2v four-parameter stationary fixture with Ni-I/trans-H/side-H-x/side-H-z seed
`[4.6, 2.9, 2.9, 0.05]` Bohr, charge -2 and multiplicity 1. A bounded independent
least-squares discovery allows eight residual evaluations, 48 total frames and
600 SCF iterations per frame; this is not the native optimizer's iteration cap.

The discovery exits 1 after 288.72 seconds. Three independently converged states
are retained (initial, Ni-I +0.001 Bohr, trans-H +0.001 Bohr). The next attempted
side-H-x +0.001 Bohr state reports nonconvergence after 222 iterations: it did
not exhaust the 600-iteration allowance. Its actual failed density, orbitals,
occupations, eigenvalues and overlap remain available. Its energy also differs
substantially from the neighboring branch, so it must not be reinterpreted as
harmless final-diagonalization noise.

The initial state has 58 electrons, orbital-gradient norm 2.65e-10 and a positive
0.1558 Eh occupied/virtual gap, but its highest occupied energy is +0.20356 Eh.
No bound ground-state or stability claim follows. Nuclear gradient maximum is
0.03888 Eh/Bohr; no stationary geometry, Hessian, frequencies or native state
matching is accepted. Do not repeat this failed discovery to fill missing gates.

## Neutral-doublet H-Cs-H

`build/qualification/host-hcsh-stationary-discovery-20261113` records a distinct
symmetric H-Cs-H hypothesis, not the already qualified RKS CsH dimer. Discovery
exits 0 after 101.01 seconds and 21 cold independent states. At H distances
5.059286709888577 Bohr about central Cs, all Cartesian gradient components are
below 6e-16 Eh/Bohr, orbital-gradient norm is 3.07e-11, and alpha/beta populations
are 6/5 in the independent overlap metric.

Central-Cs transverse curvatures are positive, +0.002524415 Eh/Bohr^2. Its axial
curvatures are also positive, +0.035484215/+0.035484120/+0.035484111 Eh/Bohr^2 at
0.003/0.001/0.0003 Bohr. Occupied-root overlaps exceed 0.999997. Thus this control
does not provide the sought negative exchange direction. These few positive
directional curvatures are not a full nine-by-nine Hessian or a complete mode
count: even a minimum classification remains pending full independent and native
qualification. No native electronic state, Hessian or optimizer ran here.

## Executed-source identity

Both discoveries execute before fixing the independent helper's restricted-vs-
unrestricted dispatch for multiplicities other than doublets. Their actual old
helper bytes are retained under `executed-reference-sources/tests/python/` and
match each discovery receipt's recorded digest. Retention must resolve those
digests to those snapshots rather than falsely rebinding old results to the
new helper. Their actual multiplicities (one and two) were already dispatched
correctly, but that fact does not authorize rewriting executed provenance.

Public ECP saddle guards remain closed. Real positive molecular saddle control,
stationary mixed Ni/ECP modes, representative requested-level Ni TS with all five
criteria and at most 100 attempts, and practical complete-scale endpoint work
remain distinct obligations. Additional negative fixtures cannot replace them.
