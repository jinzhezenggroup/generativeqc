# Native Modified Hückel reference

`huckel_reference.json` contains independent CPU PySCF
`RHF.init_guess_by_mod_huckel()` densities and molecular overlaps. The provider
version is recorded in the JSON. Coordinates are in Bohr, shell primitives
are raw (not GenerativeQC-normalized) contraction coefficients, and each
contraction is represented by its own shell. Generation uses the neutral
spherical-average atomic RHF provider and updated GWH rule, not a converged
molecular SCF density.

The single-atom cases request closed-shell electron counts filling the entire
occupied atomic subspace. Some charges are deliberately nonphysical: these
are construction/admission tests, not ground-state or atomic-energy claims.
Filling the subspace avoids treating a choice of eigenvectors at an exactly
degenerate partially occupied atomic p level as an observable density error.
Heterogeneous water/HF/NaCl cases test the molecular GWH coupling and occupation.
Water includes Cartesian/spherical def2-SVP polarization and a permutation of
shells that interleaves atom groups; its overlap and density are permuted together.

To independently regenerate the ordinary cases, reconstruct each atom's basis
from its raw shells, give different basis problems distinct PySCF atom labels,
and construct `gto.M(..., unit="Bohr", cart=not spherical, charge=charge)`.
Use `scf.RHF(mol).init_guess_by_mod_huckel()` and
`mol.intor_symmetric("int1e_ovlp")`. PySCF canonicalizes shells within each atom;
apply the same AO permutation to both matrices when reproducing an interleaved
target order. Do not import a final molecular SCF solution into this fixture.

`tests/python/test_huckel_initial_guess.py` compiles the native CPU-only probe,
compares densities with a `2e-7` entrywise gate, independently checks metric
trace/occupations at `1e-9`, and verifies native/planner capacity equality and
unique atomic solve counts. Separate analytic Gaussian hydrogen cases exercise
same-element/different-basis cache keys without another electronic-structure
package. Complete target energy/force performance qualification is separate.
