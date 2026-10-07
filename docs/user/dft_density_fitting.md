# Density-fitted DFT energy and force interface

`Calculator(method="pbe-rks", device="cuda", density_fitting="auto",
auxiliary_basis="def2-svp")` selects the shared native DF Coulomb provider for
KS calculations. `auto` follows the calculation backend; explicit
`cpu`/`cuda` DF selections must match `device`. Omitting the auxiliary basis
uses the orbital basis as the auxiliary basis, as in the existing HF interface;
choose an appropriate fitting basis for scientific production calculations.

CPU and CUDA support local/semilocal RKS/UKS density-fitted energies and
analytic forces. Full-range global hybrids use matching DF-JK on both backends;
the CUDA path reuses the prepared fitted J/K provider and its occupied-RI-K
trajectory optimization. Analytic forces differentiate the same auxiliary
basis and Coulomb metric used by the energy, including auxiliary-center and
metric response. Range-separated/nonlocal DF compositions, ECP DF and
automatic mixed precision remain rejected. This interface does not change the
selected functional or grid.

```python
from generativeqc import Calculator

calc = Calculator(method="pbe-rks", basis="sto-3g", device="cuda",
                  density_fitting="auto", auxiliary_basis="def2-svp")
result = calc.singlepoint([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
                          properties=("energy", "forces"))
```

The prepared owner copies auxiliary shells before the public descriptor is
released. Batch geometry changes rebind both orbital and auxiliary centers.
CUDA iterations enqueue DF J and XC on the DF owner's stream, retaining density
and Fock matrices on the device; no CPU integral/reference retry is selected.
The ordinary host-controlled KS convergence loop is used for DF. The opt-in
multi-iteration direct-J solver region remains restricted to direct J.

`density_fitting_relative_threshold` controls metric rank selection.
`density_fitting_memory_budget_bytes` bounds the native CUDA DF provider's
explicit source/value storage through its existing bounded tile planner; it
is not a cap on the full KS calculation. Infeasible budgets fail explicitly.
Prepared CUDA batches expose the provider's metric diagnostics. Whole-KS
`estimate_resources`/resource-plan admission is rejected for DF until its
combined inventory is qualified; conventional inventories must not describe DF.

Qualified force calculations use token-checked derivative snapshots and the
prepared DF response provider. The CPU diagnostic requires `execution="native"`
for a fitted state; the Direct-only reference derivative path is rejected rather
than differentiating a different Hamiltonian.

The CPU bridge contracts retained host H'/S' derivatives for the one-electron
and Pulay sources. CUDA first borrows the token-checked final stationary D/W
already retained by the KS owner and runs the bounded paired one-electron
consumer without uploading those AO matrices again. If that optional device
consumer cannot be admitted under the caller's budget, the exact host
contraction remains the bounded fallback.

For restricted CUDA DF response, the Coulomb J' component also borrows the exact
final resident density under the same live KS token. The detached host density
remains the finite/symmetric scientific witness, but the response bridge reads
the device matrix directly and therefore reports zero density H2D bytes for
that J-only call. Exchange K' deliberately keeps its existing density/projection
path in this change, and unrestricted/multi-term response retains the ordinary
upload path. This is therefore not a zero-upload resident whole-force path.

Resource metadata distinguishes the resident one-electron path from its host
fallback; `density_fitted_response_resources_included=0` still explicitly
excludes unmeasured DF-provider scratch and transfers. These partial diagnostics
cannot establish a whole-force memory or transport bound. Full DF resource-plan
admission remains unqualified.

`tests/python/test_dft_df_public.py` compares independently converged PySCF
energies and analytic gradients with copied orbital/auxiliary primitives and
identical moving quadrature (absolute energy gate `1e-8 Eh`, force gate
`3e-7 Eh/bohr`, physical residual below `1e-9`). GPU acceptance
requires `GENERATIVEQC_DFT_CUDA_TEST=1` and a scheduler-allocated CUDA device.

See the [ownership note](../../.agents/notes/implemented/architecture/2026-09-22-public-df-ks-provider.md)
for the retained boundaries.
