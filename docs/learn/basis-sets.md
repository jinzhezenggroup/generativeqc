# Basis sets

Gaussian-basis methods represent each molecular orbital as a linear combination of atom-centered basis functions,

$$
\phi_i(\mathbf r)
=
\sum_\mu C_{\mu i}\chi_\mu(\mathbf r).
$$

A contracted Cartesian Gaussian basis function centered on nucleus $A$ has the schematic form

$$
\chi_\mu(\mathbf r)
=
\sum_p d_{\mu p} N_p
(x-A_x)^{l_x}(y-A_y)^{l_y}(z-A_z)^{l_z}
e^{-\alpha_p |\mathbf r-\mathbf A|^2},
$$

where $\alpha_p$ is a primitive exponent, $d_{\mu p}$ is a contraction coefficient, $N_p$ is a normalization factor, and $l_x+l_y+l_z$ determines the angular momentum.

The basis set therefore limits the finite space in which the electronic problem is solved. In a nonorthogonal atomic-orbital basis the overlap matrix is

$$
S_{\mu\nu} = \langle \chi_\mu | \chi_\nu \rangle,
$$

which appears explicitly in the generalized eigenvalue problems used by HF and Kohn-Sham DFT.

A larger or more flexible basis usually costs more computation and can reduce basis-set error. Energies from different basis sets should not be compared as though only geometry changed.

Angular-momentum support also matters: a backend must support every shell required by the chosen basis.

Effective core potentials (ECPs) replace selected core-electron physics with an effective operator, so an ECP calculation is not automatically equivalent to an all-electron calculation.

See [external basis data](../user/external_basis.md), [higher angular momentum](../user/high_angular_momentum.md), and [ECPs](../user/ecp.md).

Next: [Hartree-Fock and SCF](hf-and-scf.md).
