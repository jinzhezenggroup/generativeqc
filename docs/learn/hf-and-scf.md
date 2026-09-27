# Hartree-Fock and SCF

Hartree-Fock (HF) approximates the many-electron wavefunction with a single Slater determinant,

$$
\Psi_\mathrm{HF}
=
\frac{1}{\sqrt{N!}}
\det[\psi_i(x_j)].
$$

For closed-shell RHF, the effective one-electron Fock operator can be written

$$
\hat f
=
\hat h
+
\sum_j^\mathrm{occ}
\left(2\hat J_j-\hat K_j\right),
$$

where $\hat h$ contains one-electron kinetic and electron-nuclear terms, while $\hat J$ and $\hat K$ are Coulomb and exchange operators.

In a finite nonorthogonal AO basis, the orbitals satisfy the Roothaan-Hall equation

$$
\mathbf F \mathbf C
=
\mathbf S \mathbf C \boldsymbol{\varepsilon}.
$$

For canonical closed-shell orbitals, the RHF energy is

$$
E_\mathrm{RHF}
=
2\sum_i^\mathrm{occ} h_{ii}
+
\sum_{ij}^\mathrm{occ}
\left(2J_{ij}-K_{ij}\right)
+
E_\mathrm{nn}.
$$

The Fock operator depends on the occupied orbitals or density being solved for, so HF is normally solved iteratively. The **self-consistent field (SCF)** loop starts from a density/orbital guess, builds a Fock operator, solves for updated orbitals, forms a new density, and repeats until the convergence contract is satisfied.

Schematically,

$$
\mathbf P^{(0)}
\rightarrow
\mathbf F[\mathbf P^{(0)}]
\rightarrow
\mathbf C
\rightarrow
\mathbf P^{(1)}
\rightarrow
\cdots
$$

until both the chosen update criteria and physical residual requirements are satisfied.

“SCF converged” means the iterative equations converged. It does not prove that the chosen method and basis are chemically accurate.

RHF is a restricted form; UHF permits different alpha and beta spin orbitals and therefore separate spin densities and Fock operators. Kohn-Sham DFT uses a related self-consistent structure.

Next: [DFT and functionals](dft-and-xc.md).
