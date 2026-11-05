# Decision: qualify canonical def2 scalar-ECP forces without promoting the domain

Status: implemented
Date: 2026-11-05

## Problem and decision

Issue #2846 requests physical Gaussian/VWN-RPA B3LYP, spherical def2-SVP,
I/ECP28 and Cs/ECP46, including mixed Ni/ECP molecules. Existing Stuttgart or
synthetic ECP gates cannot establish this potential/basis/functional combination.
The current native owner already returns some of these small force endpoints;
adding another functional-specific force equation or reference fallback would
duplicate scientific ownership instead of qualifying it.

Extend the existing pinned offline BSE exporter with a separate optional ECP
fixture. Preserve the original Ni/H export byte-for-byte. Bind every extra
primary source file, the license, contractions, radial/projector convention and
complete canonical export by SHA-256. Runtime consumers load retained data;
BSE remains an offline exporter and PySCF/simple-dftd3 are independent test
oracles only. I has 25 valence electrons, not seven; Cs has nine. Dispersion
continues to receive elemental I=53/Cs=55, not effective ionic charges.

Freeze the small-fixture force gate at `1e-5 Eh/Bohr`, energy at `2e-8 Eh`, AO
density at `2e-7` and overlap at `2e-11` before qualification. Compare converged
states, spin electron counts, all forces with moving-grid response, and
multistep reconverged energies. These tests do not change capability metadata,
production numerical tolerances, native derivative equations or resource caps.

## Evidence and retained exclusions

The declared two-atom fixtures have heavy center `(0.03,-0.02,0.01)` Bohr and H
at `(0.17,0.11,3.1)` for I or `(0.17,0.11,3.4)` for Cs. They use charge/multiplicity
`(0,1)` and `(-1,2)`, direct CPU FP64, an explicit `16 x 6 x 12` grid per atom,
and no production finite differences or oracle work. Native SCF controls are
energy `1e-13`, density `1e-11`, maximum 240 iterations. Their actual spherical
spaces differ from Cartesian spaces; no representation substitution is allowed.

A matched-state diagnostic returned maximum independent force errors
`9.8411e-13`, `2.2338e-13`, `7.3859e-11` and `7.3044e-11 Eh/Bohr`, respectively.
The largest density error was `4.7865e-10`; largest energy error was
`2.4941e-12 Eh`. These diagnostic results are not a complete issue receipt.

CsH(-) at looser energy/density controls `1e-12`/`1e-10` converged its energy
but its direct energy+force call returned numerical failure. The retained split
probe measured residual `3.29443e-12`; the tighter unchanged-chemistry solve
returned residual `5.79541e-13` and complete forces. Tightening SCF controls is
not permission to weaken a stationary force check. The exact internal rejection
cause is not established by the generic native error. Keep this failed call in
the evidence; do not claim the looser/default controls qualified.

The first combined force/energy-difference run passed the IH singlet, CsH
singlet and Cs covariance/replay tests, but failed a cold displaced IH(-) SCF.
A separate native-only warm-seed diagnostic also retained a failed displaced
IH(-) entry, despite a small reported residual; that residual is not permission
to consume a result whose convergence status is false. Its force test therefore
declares independent PySCF reconverged energies in advance, not an exception-time
fallback, and does not claim native IH(-) geometry continuation qualified. Native
displaced-energy coverage remains for the other small force cases. No force gate
or SCF tolerance is relaxed, and neither failed energy is used in differences.

Standalone native/public composition tests for CsH and CsH(-) compare the
two-body BJ correction to simple-dftd3, retain elemental Cs=55, and reject omission,
duplication and effective-charge substitution as numerical negative controls.
These compose first gradients exactly once; they do not qualify total-method
curvature or extend the failed mixed Ni/I/H domain.

The mixed fixture Ni `(-2.7,0.2,-0.1)`, I `(0.03,-0.02,0.01)`, H
`(0.17,0.11,3.1)` failed the existing two-grid ECP quadrature convergence gate
before SCF for both `(charge=0,multiplicity=1)` and `(charge=1,multiplicity=2)`.
Neither a successful pure-ECP dimer nor input-space AO parity qualifies this
mixed Hamiltonian. Do not lower that gate, discard its failed entries, move the
atoms merely to relabel this fixture as passing, or generalize to universal ECP
support. A future bounded refinement needs its own independent matrix/derivative
evidence and resource/work accounting.

## Default-grid oracle charge pitfall

A separate default-grid CsH singlet probe matches independent energy and density
but initially showed a `2.35846e-5 Eh/Bohr` full-grid gradient difference. Native
reconverged energy differences at `1e-3` and `3e-4` Bohr agree with the native ECP
center force to `3.24159e-8` and `2.93722e-9 Eh/Bohr`; the discrepancy does not
justify changing native derivatives or the `1e-5` acceptance gate.

In PySCF 2.14, the primal Treutler adjustment indexes elemental symbols, while
`grad.rks.grids_response_becke` indexes `mol.atom_charges()`, which contains ECP
ionic charges. For Cs this confuses elemental radius index 55 with ionic index 9.
The independent oracle therefore uses a separate grid-only molecule view whose
charge query supplies elemental indices. Its actual SCF/gradient Hamiltonian
retains the original ECP molecule, ionic charges `[9,1]`, electron count and
integrals. Do not patch the physical molecule, global PySCF classes or production.

The corrected independent grid reconstructs identical points and partition weights
(maximum weight difference `8.641e-12`, including large far-field weights). The
matched force difference becomes `7.36374e-11 Eh/Bohr`. Retain the original failed
reference receipt and the independent partition and energy-difference checks;
do not silently relabel it as accepted. This only qualifies the small default-grid
CsH singlet, not I, UKS, mixed Ni/ECP or a production-scale default-grid endpoint.

## Boundaries and revisit conditions

The retained inputs and small tests are a draft slice, not completion of #2846.
The remaining default-grid domains, representative 25–40-atom Ni/ECP endpoint,
installed artifact identity and mixed-center force qualification remain separate gates.
A discrete azimuthal-grid-preserving rotation is not proof of arbitrary-grid
rotation invariance. Scalar channels do not admit spin-orbit potentials or
unqualified higher angular momentum. ECP second order, total-method curvature,
transition-state optimization and frequencies remain separate consumers.

Revisit public promotion only after the requested physical mixed domains,
default-grid complete endpoints, bounded resources and independent whole-molecule
gates pass together. Preserve the current fail-closed behavior in the meantime.

## References

- Issue #2846 and second-order dependency #2847.
- `tools/generate_global_hybrid_hessian_basis.py --include-def2-ecp --check`.
- `tests/data/global_hybrid_ecp/manifest.json`.
- `tests/python/test_global_hybrid_ecp_basis.py`.
- `tests/python/test_global_hybrid_ecp_forces.py`.
- `tests/python/test_global_hybrid_ecp_d3.py`.
