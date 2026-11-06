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

Three sets of DFT calculations currently support second derivatives:

**CPU LDA/PBE RKS.** Direct all-electron Cartesian closed-shell strict-FP64
LDA and PBE, with no exact exchange, no density fitting, no dispersion
correction, no nonlocal correlation and no ECP. Both `hvp` and `hessian` are
advertised for this context.

**CPU full-range PBE0/B3LYP RKS compositions.** Direct all-electron strict-FP64
restricted calculations admit Cartesian or real-spherical AOs through f.
The resolved semilocal composition must equal PBE0 or Gaussian-definition
B3LYP (VWN-RPA); its one full-range exchange coefficient may be in `(0, 1]`.
A changed MethodIR identifier does not change admission or the physical method.
Unrestricted, density-fitted, ECP, dispersion-corrected and CUDA variants do
not inherit this support. Imported canonical basis records retain their actual
spherical AO dimension; no Cartesian replacement is used.

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
| `grid_budget_bytes` | full-range hybrids and WB97M-V: numeric AO/grid/AD workspace |
| `max_nonlocal_pairs` | WB97M-V only: ordered VV10 pairs per response action |
| `output_budget_bytes` | (`hessian` only) the dense matrix plus immutable publication |

The LDA/PBE endpoints do not consume `grid_budget_bytes` or
`max_nonlocal_pairs`; do not use these arguments as a grid-memory cap for those
methods. An applicable budget that cannot hold its declared work raises before
publication. Response and provider diagnostics are reported separately and are not a combined
process-RSS or GPU-memory measurement; see
[Resource planning](../maintainer/resource_planning.md) for how those capacity
records relate to actual allocation.

Full-range hybrids do not have a nonlocal pair contribution, so their
`max_nonlocal_pairs` setting does not limit semilocal grid work. The native
default quadrature is preserved. A grid that does not fit the default 128-MiB
AO/grid lease raises `MemoryError` naming `grid_budget_bytes`; increase that
explicit allowance when appropriate rather than reducing the grid silently.
The lease is not a whole-process memory limit and excludes compiler subprocesses
and borrowed SCF storage. Installed endpoint qualification measures process-tree
RSS separately from provider capacity diagnostics.

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
`strategy` controlled Krylov solve. Full-range hybrids and WB97M-V assemble scalar columns:
`block_size=None` or `1` is required, and CUDA accepts only
`strategy="scalar"`. Requesting recycling or block solves on CUDA raises rather
than being silently downgraded.

CPU composite calls accept the default scalar strategy (`strategy=None` or
`"recycled"`); this does not promise multi-right-hand-side response recycling.
Diagnostics distinguish the actual public entry point from a low-level owner
and include SCF energy, iterations, Fock builds, executed backend and complete
SCF/response/assembly wall time. No displaced SCF or reference engine runs in
the production Hessian.

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
