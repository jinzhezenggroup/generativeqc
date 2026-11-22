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
back to finite differences. On CUDA, the loaded library must expose the linked
Hessian ABI before an endpoint is advertised. A custom partial AOT inventory
can still reject a molecule whose required shell or radial leaves are absent.

`Calculator.capabilities.supported_second_order` exposes the same frozenset.
The general rule for how a capability source relates to a method identity is in
[Capability sources and interpretation](../reference/capabilities.md); the
first-order equivalent is described under
[DFT forces](methods.md#dft-forces).

## Supported methods

The following DFT calculations support second derivatives:

**CPU LDA/PBE RKS.** Direct all-electron Cartesian closed-shell strict-FP64
LDA and PBE, with no exact exchange, no density fitting, no dispersion
correction, no nonlocal correlation and no ECP. Both `hvp` and `hessian` are
advertised for this context.

**CPU full-range PBE0 RKS and B3LYP RKS/UKS compositions.** Direct all-electron
strict-FP64 calculations admit Cartesian or real-spherical AOs through f.
B3LYP UKS preserves the supplied charge/multiplicity and both physical spin
densities, including coupled same-spin exchange and cross-spin XC response.
The resolved semilocal composition must equal PBE0 or Gaussian-definition
B3LYP (VWN-RPA); its one full-range exchange coefficient may be in `(0, 1]`.
A changed MethodIR identifier does not change admission or the physical method.
An optional **uncut two-body D3(BJ), ATM off** correction is added once to the
qualified electronic result, including coordination-number/C6 response. It uses
the supplied canonical `D3Spec`, not a second functional-specific Hessian solver.
The exact canonical B3LYP graph instead uses the linked native CPU aggregate
owner for RKS/UKS, including scalar ECPs within its operator/resource bounds.
Its response, moving-grid assembly and optional D3 composition do not require
Python numerical kernels or runtime compilation; missing linked electronic or
total-method closure fails before SCF without a fallback. ECPs on PBE0 or
coefficient-variable graphs remain excluded. This CPU admission does not extend to PBE0 UKS or density-fitted variants.
On CPU, D4, zero damping, ATM and finite-cutoff corrections remain unsupported for
second order, even when their energy and first gradient are available.
Imported canonical basis records retain their actual
spherical AO dimension; no Cartesian replacement is used.

**CUDA LDA/PBE/PBE0/B3LYP RKS/UKS.** Canonical strict-FP64 calculations use
the linked native stationary endpoint with direct integrals or CUDA density
fitting, both all-electron and with scalar semilocal ECPs. A fitted metric
must be separated from the relative eigenvalue cutoff. Cartesian
and real-spherical AOs require an integral inventory that covers their shell
classes. These bare models have no VV10 work and do not admit noncanonical
exchange fractions. B3LYP uses the Gaussian-definition VWN-RPA composition.
The CUDA package and fallback constraints are the same as for WB97M-V.
With `ResourceBudget`, canonical Direct PBE0 and B3LYP require only the
energy SCF inventory, including when the basis has f shells and the standalone
force owner is not available.

For all-electron Cartesian LDA/PBE RKS with Direct integrals, the native
FP64 response keeps the CPKS iterations and molecular accumulation on the
device. The default dense solve reuses one response context for the raw
columns; it does not average the matrix with its transpose. Both endpoints
can start from an energy-only SCF plan and do not run a standalone force
calculation. An unavailable linked device provider rejects the request rather
than switching to CPU.

The CUDA scalar-ECP source first compares the fixed 160-radial/32-polar
and 224-radial/44-polar quadratures. If a local or nonlocal channel misses
the tolerance with finite results, the source can check that channel once
against 320 radial and 60 polar points, provided the remaining workspace and
work budgets permit it. A successful check still returns the 224/44 result;
an unresolved mismatch rejects the derivative rather than publishing an
unchecked result or switching to a CPU calculation.

**CUDA PBE with explicit range-separated HF exchange.** Via Python, pass a
`KsOptions(composition=...)` `MethodIR` containing unscaled PBE and distinct
short- and long-range HF fractions in `(0, 1]` at the same positive `omega`
(inverse Bohr). This supports RKS and UKS, all-electron or with scalar
semilocal ECPs, in strict FP64 with the fused XC schedule and three Becke
partition iterations. The AOT inventory must cover the orbital shell classes
and the requested radial operator. With an orbital ECP, CUDA DF requires an
explicit ECP-free auxiliary basis on the same centers.
With a CUDA DF primary, `GENERATIVEQC_DF_RANGE_EXCHANGE=direct` fits only the
full-range primary; `df` also fits LR exchange with its own erf metric.
When unset, range-separated PBE retains exact LR. Explicit `auto` follows
SCF's fit admission and may select exact LR before the first energy. The
Hessian differentiates the accepted model, not a new fit.
Both metrics must be separated from the relative eigenvalue cutoff. Off-baseline
`omega` requires parameterized factorized AOT leaves; fixed expanded leaves
cannot substitute. Nonlocal correlation and dispersion are not supported for
this composition.
`ResourceBudget` accounts for both radial sources and the analytic response;
the SCF prerequisite is an energy-only plan.

For CLI `hessian` or `hvp`, write the composition's `MethodIR.to_payload()`
as JSON and pass `--method-ir FILE.json` instead of `--method`:

```bash
generativeqc hessian geometry.xyz --backend cuda --basis def2-svp \
  --method-ir range-pbe.json --json
```

The CLI checks the semilocal graph against its linked PBE catalog and takes the
RKS/UKS selection from the file; set `--charge` and `--multiplicity` for the
molecule. Use a payload from the matching compiler build. This input accepts
only the unscaled PBE + SR/LR composition above, without a method-bound basis;
the positive fractions and `omega` use the canonical rational strings from
`to_payload()` with numerator and denominator at most `2**53`. The ordinary
CLI `--method pbe-rks` and `--method pbe-uks` selectors remain pure PBE.

**CUDA PBE/PBE0 with named D3.** PBE-D3(BJ), PBE-D3(BJ)-ATM, PBE-D3(0),
PBE0-D3(BJ) and PBE0-D3(0) add their analytic geometry-only curvature on
device to the qualified electronic model. Use the catalog MethodIR in Python
or the corresponding named `-rks` / `-uks` selector in the CLI, for example
`--method 'pbe0-d3(bj)-uks'`. Direct and CUDA DF use the same
correction, including the coordination dependence of C6. The correction must
belong to the native batch; arbitrary Python-side D3 specifications do not
inherit GPU Hessian support. A geometry exactly on a hard cutoff is rejected
without publishing a derivative.
Python's `ResourceBudget` includes the native D3 owner with the electronic
SCF and Hessian/HVP response. A standalone D3 budget cannot substitute for it.

**CUDA PBE/PBE0/B3LYP/r²SCAN with named D4.** The catalog D4(BJ-EEQ-ATM)
compositions support RKS and UKS with direct integrals or CUDA DF.
They require strict FP64, the fused XC schedule, no application precision
budget, and a matching Hessian AOT inventory. r²SCAN-D4 also requires the same
variational SCF domain as bare r²SCAN. Use the catalog MethodIR in Python or
the corresponding CLI selector, for example `--method 'pbe0-d4(bj-eeq-atm)-uks'`.
The `pbe-d4-rks` selector is also supported. The native KS owner adds the
analytic D4(BJ) pair, Axilrod–Teller–Muto (ATM), and EEQ charge response to the
electronic result on device. A separately evaluated Python correction, a
modified fit, or an unqualified electronic model cannot supply this endpoint.
Python's `ResourceBudget` covers the electronic SCF, native D4, and bounded
response on one device ledger. This works with a catalog MethodIR or the
exact `pbe-d4-rks` selector, which uses the catalog owner for budgeted
execution. The SCF prerequisite is an energy-only plan, not a force plan.

**CUDA r²SCAN and r²SCAN-3c RKS/UKS.** The linked NVIDIA native stationary
endpoint admits strict FP64 with the fused XC schedule and no application
precision budget, on grids with three Becke partition iterations. It supports
Direct or CUDA DF with a matching Hessian AOT inventory. Bare r²SCAN
admits scalar semilocal ECPs and uses the prepared representation, with
orbital angular momenta through f for either provider. Named r²SCAN-3c
uses its defining spherical basis and differentiates the D4 and gCP
corrections on device as part of the same result. The incumbent r²SCAN point
policy (including AUTO and host-unfused execution) does not provide these
second derivatives.

**Experimental CUDA M06-2X and MN15 RKS/UKS.** Builds with native CUDA
forces and a matching Hessian AOT inventory can opt in independently using
`GENERATIVEQC_ENABLE_M062X_STATIONARY_POINT=ON` or
`GENERATIVEQC_ENABLE_MN15_STATIONARY_POINT=ON`. Select `m06-2x-rks`,
`m06-2x-uks`, `mn15-rks`, or `mn15-uks` in Python or the native CLI. These
endpoints accept all-electron or scalar semilocal ECP orbital bases and require
strict FP64, fused XC, and three Becke partition iterations. Use Direct
(`density_fitting="none"`) or explicit CUDA DF with a supplied ECP-free
auxiliary basis. The angular range is through f for orbitals
and g for auxiliary shells, with spherical representation for d and higher
and subject to the linked derivative inventory and work budgets. Each canonical
model uses its own energy-derived point domain consistently for SCF, Hessian,
HVP, and any standalone force calculation within its force domain. The
standalone generated Direct force endpoint still requires all-electron orbital
shells through d. The ECP Hessian/HVP do not enable standalone ECP forces.
AUTO precision, automatic fitting-basis selection, and empirical corrections
are outside this scope. Both build options default
to off; the incumbent split-hybrid SCF domain cannot supply these second derivatives.
Python Hessian/HVP calls also accept `ResourceBudget` for these linked Direct
and explicit-DF contexts. An energy-only prepared plan cannot replay forces.

**CUDA r²SCAN+rVV10 RKS/UKS.** Select `r2scan+rvv10-rks` or
`r2scan+rvv10-uks` for strict-FP64, device-fused calculations with all-electron
or scalar semilocal ECP orbital bases.
The Hessian and HVP differentiate the self-consistent r²SCAN+rVV10 energy,
using Direct or CUDA DF with the linked variational r²SCAN point domain. Orbital
shells through f and fitted auxiliary shells through g are supported, subject
to the linked inventory and work budgets.
Both a shared grid and `KsOptions(nonlocal_grid=...)` are supported; both
grids require three Becke iterations. The rVV10 term is not a post-SCF
correction. `ResourceBudget` covers the ECP core setup and response alongside
the nonlocal owner. This composition does not expose a standalone native force
endpoint.

**Canonical WB97M-V RKS/UKS.** Both spins, Cartesian or spherical
representation, CPU or CUDA, with density fitting and scalar semilocal
ECPs admitted. CPU density fitting still requires full rank and exact
long-range exchange. On CUDA the numerical work — grid/DF/ECP automatic
differentiation and coupled response — stays in linked native code, and an
AOT Hessian package must be built; Python, CuPy and runtime compilation are not
numerical dependencies. The default AOT inventory includes the erf-Coulomb
metric and three-center derivative leaves for fitted long-range exchange at
`ω = 0.3 bohr⁻¹`. Each CUDA fitting metric can be full rank or rank-truncated;
the response differentiates its actual retained subspace, including its motion
with geometry. A root too close to the cutoff makes the derivative request
fail. The primary and erf metrics pass this gate independently. An unsuitable
fitted source or a custom inventory missing a required leaf does not switch
the accepted calculation to exact long-range exchange.

CUDA WB97M-V accepts `KsOptions(nonlocal_grid=...)` for both `hvp` and
`hessian`. It differentiates the independent, device-resident VV10 quadrature
used by the accepted SCF state, not the primary XC grid. The CPU Hessian path
does not support independent XC/VV10 grids. For example, `KsOptions(grid_accuracy="tight")`
uses Treutler level 4 for XC and retains level 3 for VV10 unless you override it.
If the retained mesh exceeds `max_nonlocal_pairs`, the request reports the active
point count and the cap without changing either grid or publishing a derivative.
See [native CLI options](native_cli.md#compiler-ad-dft-hessian-and-hvp)
for the equivalent command-line controls.

**AUTO SCF on CUDA.** Direct, all-electron LDA, PBE, PBE0, B3LYP and
WB97M-V RKS/UKS can request `precision="auto"` with device-fused XC. The
same option is available for all-electron PBE RKS with CUDA density fitting.
On the stock Treutler level-3 grid, fitted PBE can use mixed XC matrix products
for SCF proposals; its fitted integrals, metric and Coulomb build stay FP64.
Outside the mixed proposal domain it uses FP64 SCF on the same model and grid.
`scf_mixed_fock_builds` distinguishes these paths. Every CUDA Hessian/HVP
requires a fresh FP64 physical Fock from the converged generation and runs
entirely in FP64. The public AUTO frontend does not admit other fitted models,
application precision budgets, r²SCAN variants or post-SCF corrections.

**CUDA dense response cache.** `Calculator.hessian(..., response_cache_precision="fp32-proposal")`
stores the bounded repeated-column CPKS projection in FP32 instead of FP64.
Its solution is a proposal only: physical FP64 GMRES corrects it and the
physical stationarity residual gates every column at the requested tolerance.
The two phases share the iteration budget. If there is no space for a cache,
the calculation runs the uncached FP64 solver. The default stays `"fp64"`;
one-shot HVPs do not build a repeated-column cache. Use
`cpks_cache_precision`, `cpks_operator_bytes`, `cpks_proposal_seeds` and
`cpks_physical_checks` to distinguish the executed paths. This option halves
the projected matrix storage, not the entire Hessian workspace, and does not
promise a faster solve.

Other functionals and lower-precision physical Hessian arithmetic are not supported.
The generic fixed-density route also excludes meta-GGA
density response.

## Native SDK

Prepare and successfully execute an energy-only CUDA KS batch first
(`batch.execute({}, false)` in C++). C++ callers can then use
`Batch::ks_hessian` or `Batch::ks_hvp` on the original
input index. They take caller-owned output spans and return a detached work
receipt. In C, include `generativeqc/ks_hessian.h`, initialize the options with
`generativeqc_ks_hessian_default_options_v2`, and call
{ref}`generativeqc_ks_hessian_cuda_v2 <native-c-generativeqc-ks-hessian-cuda-v2>`.
C callers must also set the receipt's `struct_size` and `abi_version`.
The linked-version query checks the library without opening a device; method,
state and integral support are checked at execution.

For `N` atoms, the dense output is `(3N) × (3N)` in row-major order; an HVP
accepts and returns `3N` atom-major Cartesian values. Input and output buffers
must not overlap. The SDK uses the batch's accepted physical SCF state and, if
present, its shared device ledger. It does not rerun SCF, expose response
panels, or switch providers. On failure it leaves caller outputs untouched.

## Resource arguments

`hessian` and `hessian_vector_product` bound separate resources rather than
exposing one global cap:

| Argument | Bounds |
| --- | --- |
| `integral_budget_bytes` | generated second-order integral work |
| `grid_budget_bytes` | CUDA or CPU composite: numeric AO/grid/AD workspace |
| `max_nonlocal_pairs` | WB97M-V and r²SCAN+rVV10: ordered nonlocal pairs per response action |
| `output_budget_bytes` | (`hessian` only) the dense matrix plus immutable publication |
| `max_eri_primitives` | linked native CPU only: optional positive-int64 whole-request nuclear primitive-work cap |
| `max_map_terms` | linked native CPU only: optional positive-int64 whole-request contraction/assembly work cap (default `2**40`) |
| `numeric_budget_bytes` | linked native CPU only: aggregate derivative-owned numeric reservation (default 512 MiB) |
| `response_budget_bytes` | linked native CPU only: response subreservation, including optional coordinate source panels (default 128 MiB) |

The linked native CPU owner's reservations, default work allowance and separate
induced-Fock work are described in
[Linked native CPU B3LYP endpoint](../developer/dft_hessian.md#linked-native-cpu-b3lyp-endpoint).
The optional nuclear work override is not a time/RSS or induced-Fock integral
bound. Other owners reject it rather than ignoring it.
The explicit byte limits are positive signed-int64 values; the response
subreservation must fit in the aggregate reservation. They exclude primal
SCF storage, BLAS scratch and process RSS. GMRES has a separate workspace
limit. Raising a byte limit can admit reusable coordinate sources rather
than the bounded column fallback; it does not raise either work cap.

For other corrected CPU composite endpoints, `output_budget_bytes` holds three dense matrices:
the detached electronic result, the private total, and immutable publication.
The constructor's `dispersion_memory_budget_bytes` additionally bounds the
simultaneously retained electronic arrays, total publication and D3 workspace.
Correction columns are streamed; no separate dense D3 matrix is retained.
An insufficient correction budget raises without returning an electronic-only
result under the corrected method's identity.

The native CPU aggregate publishes immutable raw H/HVP results and scalar
electronic/total/D3 energies, physical response residuals, source consistency,
SCF iterations/Fock builds and actual derivative work. It does not export the
private source component panels or a central density/grid fingerprint. Its
`endpoint_seconds` includes the fresh native SCF and complete derivative call.

Other corrected composite results retain the electronic response diagnostics and add one
`d3_bj_two_body` source. HVP `components` includes that positive gradient
derivative, not a force derivative. `diagnostics["dispersion"]` reports the
parameter/source identities, actual correction actions/scalar work, workspace
and correction preparation/assembly time. No additional electronic response
solve or displaced SCF is introduced. `endpoint_seconds` includes the native
SCF, electronic curvature and correction, while `scf_energy`, `dispersion_energy`
and `total_energy` distinguish the energy components.

Snapshot-based CPU composite diagnostics additionally identify the central SCF state:
`scf_density_sha256` hashes its little-endian float64, C-order spin-major density
bytes, with shape in `scf_density_shape`. `scf_occupied_orbitals` counts occupied
orbitals in each stored spin block (one doubly occupied block for RKS, separate
alpha/beta blocks for UKS). `scf_physical_residual` and `scf_basis_identity`
accompany that fingerprint. `scf_grid_identity` includes the native resource
owner and need not match across calls. To compare actual quadratures, use
`scf_grid_sha256`: SHA256 of the bytes `generativeqc.scf-grid-v1\0`, the ASCII
point count followed by a null byte, then the C-order little-endian points
(float64), weights (float64) and atom owners (int32), in that order.
`scf_grid_point_count` reports the count explicitly. These detached scalar/tuple
records retain no native owner. A fingerprint identifies exact contents; it
does not replace independent numerical state/root matching or qualify a new
method, geometry or artifact.

The CPU LDA/PBE endpoints do not consume `grid_budget_bytes` or
`max_nonlocal_pairs`; do not use these arguments as a grid-memory cap for those
methods. An applicable budget that cannot hold its declared work raises before
publication. Response and provider diagnostics are reported separately and are not a combined
process-RSS or GPU-memory measurement; see
[Resource planning](../maintainer/resource_planning.md) for how those capacity
records relate to actual allocation.

Full-range hybrids do not have a nonlocal pair contribution, so their
`max_nonlocal_pairs` setting does not limit semilocal grid work. The native
default quadrature is preserved. A grid that does not fit the default 128-MiB
AO/grid lease raises before publication. Python composition paths use
`MemoryError` naming `grid_budget_bytes`; a linked native owner can instead
raise `RuntimeError` carrying its geometry/grid or response workspace refusal.
Increase the applicable explicit allowance when appropriate rather than
reducing the grid silently.
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

CPU LDA/PBE use the shared bounded response path with a `block_size` and
`strategy` controlled Krylov solve. The native CUDA endpoint and CPU composite paths assemble scalar columns:
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

The C++ executable also exposes `hessian` and `hvp` on CUDA with a build-time
AOT package, for the same canonical methods. The CLI
contract, inventory selection and worked commands are in the
[native CLI guide](native_cli.md#compiler-ad-dft-hessian-and-hvp).

## Where the details live

- [DFT Hessian and HVP execution contract](../developer/dft_hessian.md) —
  MethodIR source planning, AOT packaging and the qualified execution boundary.
- [Analytic HF Hessians and HVPs](../developer/hessian.md) — the conventional
  RHF term map, response RHS and bounded tools HVP.
- [Stationary problems](../developer/stationary_problem.md) — first-order
  stationary assembly from which the second-order chain is built.
