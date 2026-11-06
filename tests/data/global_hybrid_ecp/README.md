# Pinned Ni/H/I/Cs def2 inputs

`def2-svp-ni-i-cs-h.json` is a canonical Basis Set Exchange def2-SVP version-1
export at the immutable source commit recorded in `manifest.json`. The shared
BSD-3-Clause license, source-file digests and complete export digest are bound
by that manifest. Runtime consumers load these retained records, not BSE/PySCF.

- Real-spherical orbital shells retain all contractions, including Ni d/f space.
- H and Ni are all-electron; I removes 28 electrons and Cs removes 46.
- Elemental nuclear identities remain H=1, Ni=28, I=53 and Cs=55. Effective ionic
  charges/electron counts account for core removal separately; dispersion must
  continue using the elemental identities.
- Potentials are scalar semilocal, without spin-orbit terms. The highest supplied
  angular channel is local; the others are nonlocal projector differences.
- Residual radial terms are `c*r^(r_exponent-2)*exp(-exponent*r^2)` in atomic units.
  The Coulomb tail `-Z_effective/r` is separate nuclear attraction.

Regenerate offline with an independent BSE installation whose seven primary
data-file digests match the recorded source pin:

```bash
python tools/generate_global_hybrid_hessian_basis.py --include-def2-ecp
python tools/generate_global_hybrid_hessian_basis.py --include-def2-ecp --check
```

The default exporter mode and existing Ni/H fixture remain byte-identical.
The optional `--bse-package` flag selects an independent source package explicitly.

`tests/python/test_global_hybrid_ecp_basis.py` checks provenance, core/nuclear
identities, true spherical AO jets and independent valence counts on I/H, Cs/H
and mixed Ni/I/H inputs at singlet/doublet electron counts. These are input-space
checks, **not** converged B3LYP state/force checks or installed endpoint receipts.

`tests/python/test_global_hybrid_ecp_forces.py` separately exercises the source-tree
public CPU/direct/FP64 B3LYP force endpoint on declared I/H and Cs/H singlet/doublet
fixtures. It compares independent converged states and moving-grid analytic
forces, all-center forces, ECP-only motion and multistep reconverged energy
differences at a frozen `1e-5 Eh/Bohr` gate. The grid is explicitly `16 x 6 x 12`
per atom, **not** the native default grid. The IH(-) energy-difference oracle is
explicitly independent because retained native displaced-SCF failures leave its
native geometry-continuation gate open. Failed native energies are never used.

The Cs covariance check uses a discrete azimuthal-grid-preserving rotation plus
translation; it is not arbitrary-grid covariance. Separate Cs first-gradient
composition tests in `tests/python/test_global_hybrid_ecp_d3.py` use an independent
two-body BJ oracle and keep elemental Cs identity despite ECP core removal.

A separate Cs/H singlet test uses the unmodified native default grid. The
independent Treutler grid response needs a grid-only elemental-charge view for
PySCF 2.14; its actual ECP Hamiltonian keeps ionic charges. The test verifies
partition parity, matched state/force and multistep native displaced energies.
This is not default-grid qualification of I, UKS, mixed molecules or larger inputs.

The actual mixed Ni/I/H fixture currently fails the unchanged native ECP
quadrature convergence guard before SCF. Neither basis-input parity nor successful
dimers qualify that mixed domain. Issue #2846 still requires its remaining native
default-grid, mixed-molecule and installed-artifact qualification; its production-scale
25–40-atom gate is not completed by these fixtures. See
`.agents/notes/implemented/numerics/2026-11-05-def2-ecp-global-hybrid-force-gates.md`
for declared coordinates, controls, retained failures and promotion boundaries.
