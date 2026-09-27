# Quantum chemistry in ten minutes

A molecular electronic-structure calculation starts from nuclei, electron count, a representation of the electronic problem, and a chosen approximation.

For ordinary molecular calculations the nuclei are treated as fixed while the electronic problem is solved. In atomic units, the clamped-nuclei electronic Hamiltonian is

$$
\hat H_\mathrm{e}
=
-\frac{1}{2}\sum_i \nabla_i^2
-\sum_{iA}\frac{Z_A}{r_{iA}}
+\sum_{i<j}\frac{1}{r_{ij}}.
$$

The electronic problem is

$$
\hat H_\mathrm{e}(\mathbf R)\Psi(\mathbf r;\mathbf R)
=
E_\mathrm{e}(\mathbf R)\Psi(\mathbf r;\mathbf R),
$$

and the Born-Oppenheimer energy used for molecular structure adds the nuclear repulsion,

$$
E(\mathbf R)
=
E_\mathrm{e}(\mathbf R)
+
\sum_{A<B}\frac{Z_A Z_B}{R_{AB}}.
$$

Approximate methods differ mainly in how they represent or approximate the many-electron state and its energy. For a normalized variational wavefunction, the corresponding expectation value is

$$
E[\Psi] = \langle \Psi | \hat H | \Psi \rangle.
$$

A practical calculation therefore requires explicit choices:

- **System:** elements, coordinates, total charge, and spin state.
- **Representation:** for Gaussian-orbital methods, a basis set.
- **Method:** for example HF, a DFT functional, MP2, or CCSD(T).
- **Property:** energy, forces/gradient, Hessian, or another response.
- **Numerics/backend:** tolerances and execution strategy should change numerical cost/error, not silently change the scientific method.

The most important beginner rule is that an energy is meaningful mainly through **consistent comparisons**. For two states evaluated with the same scientific model,

$$
\Delta E = E_B - E_A.
$$

Changing method, basis, charge/spin, Hamiltonian, or numerical accuracy can change the scientific problem rather than only the numerical value.

Next: [molecules, charge, and spin](molecules-charge-spin.md).
