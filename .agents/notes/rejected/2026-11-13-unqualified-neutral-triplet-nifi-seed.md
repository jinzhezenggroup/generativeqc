# Set aside: unqualified cold neutral-triplet NiFI seed

Status: rejected as a qualified initializer, not a statement of nonexistence
Date: 2026-11-13

## Hypothesis and model

A distinct linear F-Ni-I neutral-triplet fixture was proposed at Ni-F 3.4 and
Ni-I 4.4 Bohr. It is neither the old neutral NiI nor NiH3I dianion attempt, nor
the requested Ni reaction TS. The test-only reference uses canonical spherical
def2-SVP, I28 scalar ECP, Gaussian/VWN-RPA B3LYP-D3(BJ), ATM off and the actual
native-default v4/level3 grid. The fluorine record is a new cohort-local version-1
export from the clean pinned BSE checkout; existing canonical fixtures are not
modified. Upstream data, export, original manifest and license hashes are kept.

Before this attempt, the independent oracle was corrected to select UKS for
every non-singlet, rather than only doublets. PySCF's RKS factory returns ROKS
for higher open-shell multiplicities: a two-component density alone therefore
does not prove independent unrestricted orbitals. The permanent regression also
checks the orbital layout and overlap-metric alpha/beta populations. This is a
test-oracle correction, not a production functional or electronic-state change.

## Actual evidence

`build/qualification/host-nifi-stationary-discovery-20261113` preserves:

- The cold independent DIIS discovery exits 1 after 78.95 seconds and 240 SCF
  iterations at its first frame, with converged=false. No nuclear gradient,
  radial fit, stationary geometry, Hessian or native electronic work is accepted.
- A read-only physical audit, without an SCF kernel call, reproduces the saved
  energy -1905.7527771308698 Eh and alpha/beta electron populations 32/30. The
  physical orbital-gradient norm is 1.12462e-7, failing both the 1e-10 declared
  SCF target and 2e-9 independent gate. The original failed flag stays false.
- Independent occupied/virtual Fock-block gaps are finite, 0.21741 and
  0.06825 Eh. This motivates one distinct cold Newton diagnostic without saved
  density/orbital seeding; it does not establish a qualified or stable state.
- The cold Newton probe also exits 1, converged=false, after 258.54 seconds.
  Its physical orbital-gradient norm is 1.02696e-7 and its energy is
  -1905.6724180297795 Eh, materially different from the DIIS branch. No nuclear
  gradient is computed. Its actual state and macro callback trace are retained.
  Callback invocations may repeat an `imacro` value and the wrapper's `cycles`
  field is not a work count; Fock/microiteration counts were not instrumented.

## Decision

Do not rerun or promote either failed state, loosen orbital gates, silently use
ROKS instead of UKS, import native orbitals as an independent initial guess, or
call these energies a qualified ground state. No stationary mixed-Ni/ECP mode
count or character follows. The frozen production source identity remains
`04226e07d04647fccf604519189b3e0a98815cf4b7ed6f37723030bec65390a6`.

Both public ECP saddle guards remain closed. This diagnostic does not discharge
the positive ECP saddle, stationary mixed Ni/ECP fixture, practical complete-
scale endpoint, or representative requested-level Ni TS obligations.
