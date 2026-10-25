# Hessians and Hessian-vector products

`Calculator` provides analytic Cartesian Hessians and Hessian-vector products
(HVPs) for selected DFT methods. These results do not use finite differences
of the energy.

```python
from generativeqc import Calculator, GridSpec, KsOptions

atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]

calc = Calculator(
    method="pbe-rks",
    basis="sto-3g",
    device="cpu",
    precision="fp64",
    ks_options=KsOptions(grid=GridSpec()),
)

print(calc.second_order_capabilities)  # frozenset({'hvp', 'hessian'}) when qualified

hvp = calc.hessian_vector_product(atoms, [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
hessian = calc.hessian(atoms)

print(hvp.value)      # shape (natoms, 3), Hartree/Bohr^2
print(hessian.matrix) # shape (3*natoms, 3*natoms), Hartree/Bohr^2
```

## Check support

Check `calc.second_order_capabilities` for the chosen device, basis and method.
Unsupported requests raise an error; they do not switch methods or fall
back to finite differences.

`Calculator.capabilities.supported_second_order` exposes the same frozenset.
The general rule for how a capability source relates to a method identity is in
[Capability sources and interpretation](../reference/capabilities.md); the
first-order equivalent is described under
[DFT forces](methods.md#dft-forces).

## Supported methods

Two sets of DFT calculations currently support second derivatives:

**CPU LDA/PBE RKS.** Direct all-electron Cartesian closed-shell strict-FP64
LDA and PBE, with no exact exchange, no density fitting, no dispersion
correction, no nonlocal correlation and no ECP. Both `hvp` and `hessian` are
advertised for this context.

**Canonical WB97M-V RKS/UKS.** Both spins, Cartesian or spherical
representation, CPU or CUDA, with full-rank density fitting and scalar semilocal
ECPs admitted. On CUDA the numerical work — grid/DF/ECP automatic
differentiation and coupled response — stays in linked native code, and an
AOT Hessian package must be built; Python, CuPy and runtime compilation are not
numerical dependencies.

Other functionals, truncated DF metrics, mixed precision and independent
XC/VV10 grids are not supported for these derivatives. The generic
fixed-density route also excludes meta-GGA density response.

## Resource arguments

`hessian` and `hessian_vector_product` bound separate resources rather than
exposing one global cap:

| Argument | Bounds |
| --- | --- |
| `integral_budget_bytes` | generated second-order integral work |
| `grid_budget_bytes` | WB97M-V only: numeric AO/grid/nonlocal-pair workspace |
| `max_nonlocal_pairs` | WB97M-V only: ordered VV10 pairs per response action |
| `output_budget_bytes` | (`hessian` only) the dense matrix plus immutable publication |

The LDA/PBE endpoints do not consume `grid_budget_bytes` or
`max_nonlocal_pairs`; do not use these arguments as a grid-memory cap for those
methods. An applicable budget that cannot hold its declared work raises before
publication. Response and provider diagnostics are reported separately and are not a combined
process-RSS or GPU-memory measurement; see
[Resource planning](../maintainer/resource_planning.md) for how those capacity
records relate to actual allocation.

## Result shape and symmetry

- `hessian_vector_product(...).value` has shape `(natoms, 3)`.
- `hessian(...).matrix` has shape `(3*natoms, 3*natoms)` with atom-major,
  xyz-minor ordering.
- Both are in Hartree/Bohr².
- The matrix is **raw**. GenerativeQC never post-hoc symmetrizes it, so a
  genuine asymmetry in the assembled derivative remains visible instead of
  being hidden.

Exact symmetry is a qualification property, not a publication step. Independent
checks compare components against finite differences of reconverged energies and
against a higher-precision direct-product oracle; a passing symmetry check never
substitutes for those comparisons.

## Solvers and strategies

LDA/PBE use the shared bounded response path with a `block_size` and
`strategy` controlled Krylov solve. WB97M-V assembles scalar columns:
`block_size=None` or `1` is required, and CUDA accepts only
`strategy="scalar"`. Requesting recycling or block solves on CUDA raises rather
than being silently downgraded.

`solver_options` accepts the shared `GMRESOptions`. Tight SCF and response
tolerances and a converged quadrature remain necessary regardless of solver
settings; a converged SCF flag alone does not authorize a Hessian.

## Native CLI

The C++ executable also exposes `hessian` and `hvp` for canonical
`wb97m-v-rks`/`wb97m-v-uks` on CUDA, with a build-time AOT package. The CLI
contract, inventory selection and worked commands are in the
[native CLI guide](native_cli.md#compiler-ad-wb97m-v-hessian-and-hvp).

## Where the details live

- [DFT Hessian and HVP execution contract](../developer/dft_hessian.md) —
  MethodIR source planning, AOT packaging and the qualified execution boundary.
- [Analytic HF Hessians and HVPs](../developer/hessian.md) — the conventional
  RHF term map, response RHS and bounded tools HVP.
- [Stationary problems](../developer/stationary_problem.md) — first-order
  stationary assembly from which the second-order chain is built.
