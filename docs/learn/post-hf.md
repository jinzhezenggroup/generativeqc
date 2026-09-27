# Post-HF methods

Hartree-Fock misses electron correlation beyond its mean-field treatment. Post-HF methods add correlation using an orbital reference.

For canonical spin orbitals, MP2 adds the second-order correlation energy

$$
E_\mathrm{MP2}
=
\frac{1}{4}
\sum_{ij}^{\mathrm{occ}}
\sum_{ab}^{\mathrm{vir}}
\frac{
|\langle ij \Vert ab\rangle|^2
}{
\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b
},
$$

where $\langle ij\Vert ab\rangle$ are antisymmetrized two-electron integrals, $i,j$ label occupied orbitals, and $a,b$ label virtual orbitals.

Coupled-cluster theory uses an exponential wavefunction,

$$
|\Psi_\mathrm{CC}\rangle
=
e^{\hat T}|\Phi_0\rangle,
$$

with excitation operator

$$
\hat T = \hat T_1+\hat T_2+\hat T_3+\cdots.
$$

CCSD truncates the iterative cluster operator to singles and doubles, $\hat T=\hat T_1+\hat T_2$. Defining the similarity-transformed Hamiltonian

$$
\bar H = e^{-\hat T}\hat H e^{\hat T},
$$

the amplitudes satisfy projected equations

$$
\langle \Phi_\mu | \bar H | \Phi_0\rangle = 0,
$$

while the energy is

$$
E_\mathrm{CC}
=
\langle \Phi_0 | \bar H | \Phi_0\rangle.
$$

CCSD(T) adds a perturbative triples correction to the converged CCSD reference,

$$
E_\mathrm{CCSD(T)}
=
E_\mathrm{CCSD}
+
\delta E_{(T)}.
$$

The compact equation above describes the method structure, not the full implementation formula for the triples correction.

MP2, CCSD, and CCSD(T) are generally much more expensive than HF because their tensor contractions involve occupied and virtual orbital spaces at higher polynomial cost.

Higher cost does not mean universal correctness: multireference character, basis limitations, relativistic effects, and other modeling choices can dominate.

For current availability, use the generated [public method table](../public_methods.md). Implementation details belong in the [Developer Guide](../developer/index.md).
