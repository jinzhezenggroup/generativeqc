# Density-fitted DFT interface

`Calculator(method="pbe-rks", device="cuda", density_fitting="auto",
auxiliary_basis="def2-svp")` selects the shared native DF Coulomb provider for
KS calculations. `auto` follows the calculation backend; explicit
`cpu`/`cuda` DF selections must match `device`. Omitting the auxiliary basis
uses the orbital basis as the auxiliary basis, as in the existing HF interface;
choose an appropriate fitting basis for scientific production calculations.

CPU supports the existing local/semilocal RKS/UKS methods and full-range global
hybrids (DF-JK) for energy. CUDA supports strict-FP64 LDA, PBE and r2SCAN
RKS/UKS with DF-J for energy and analytic forces. CUDA DF hybrids,
range-separated/nonlocal DF compositions, ECP DF and automatic mixed precision
remain rejected for public forces. This interface does not change the selected functional or grid.

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

CUDA semilocal DF forces reuse the prepared DF provider's first-derivative
contract: orbital three-center response, auxiliary-center response and the DF
metric response are evaluated under the same auxiliary basis, rank threshold,
geometry and provider identity as the converged KS state. The stationary force
consumer receives that result as its Coulomb derivative source; the semilocal
exchange slot remains exactly zero. No Direct AO-quartet derivative is relabeled
as a DF result.

The force-capable CUDA owner reserves bounded DF response capacity at
preparation so an energy replay can later request forces without changing the
prepared scientific model. Whole-KS `estimate_resources` remains unsupported
for DF until the combined inventory is qualified. CPU DF forces, CUDA DF hybrid
forces, RSH/nonlocal DF forces and ECP DF forces remain fail-closed.

`tests/python/test_dft_df_public.py` compares independently converged PySCF
energies with copied orbital/auxiliary primitives and identical quadrature. The
CUDA force gate additionally uses PySCF density-fitted analytic gradients with
moving-grid response, so auxiliary/metric response is checked independently of
the native stationary implementation. GPU acceptance requires
`GENERATIVEQC_DFT_CUDA_TEST=1` and a scheduler-allocated CUDA device.

See the [ownership note](../../.agents/notes/implemented/architecture/2026-09-22-public-df-ks-provider.md)
for the retained boundaries.
