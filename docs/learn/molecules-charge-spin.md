# Molecules, charge, and spin

Geometry alone does not define an electronic problem.

For nuclei with charges $Z_A$ and total molecular charge $q$, the electron count is

$$
N_\mathrm{e} = \sum_A Z_A - q.
$$

Neutral H2, H2+, and H2- therefore contain the same nuclei but are different electronic systems.

The spin multiplicity is

$$
M = 2S + 1,
$$

where $S$ is the total spin quantum number. Singlet, doublet, and triplet correspond to multiplicities 1, 2, and 3. In a collinear spin-orbital description,

$$
N_\alpha + N_\beta = N_\mathrm{e},
\qquad
M_S = \frac{N_\alpha - N_\beta}{2}.
$$

$M_S$ is the spin projection of the determinant; it should not be confused with the total-spin quantum number $S$ in cases where the determinant is not an eigenfunction of $\hat S^2$.

Restricted methods constrain spin channels in method-specific ways; unrestricted methods allow them to differ. RHF/UHF and RKS/UKS encode this distinction.

Charge and multiplicity are part of the scientific model, not convergence knobs.

Next: [basis sets](basis-sets.md).
