# Compact XC contractions

`generativeqc_compiler.xc` now has one contraction owner between the audited scalar
functional and AO potential, response and explicit geometric consumers. This
covers the value/gradient contraction path for real LDA/GGA on a fixed
quadrature and screening branch. The native CPU candidate is explicit; method
capability registration, stationary molecular forces, GPU XC and Hessians
remain separate owners.
The value-only Becke partition routine currently supplies no physical weight
Jacobian, so geometry returns independent source partials for that later adapter.

## Scientific contract and ownership

`IngredientContract`, `DerivativeRequest` and `DiscreteEnergyContract` identify
`E = sum_g w_g e_xc(z_g)`, where the scalar returns energy per volume. Density
and spatial AO jets are real. A total D splits as Da=Db=D/2; polarized output
keeps two functional-spin channels. Unpolarized output has one total-density
channel and requires equal spin matrices/directions. Potentials and response
use the full symmetric matrix trace, with both off-diagonal entries present.
`PairSpace` svec coordinates put sqrt(2) on each off-diagonal element.

| Term | Scientific owner | Independent check |
| --- | --- | --- |
| Scalar e, feature gradient/Hessian | Existing audited expressions + Graph AD | Pinned Libxc/formula blocks |
| rho, gradient, sigma, tau | `dft.features` | Native AO and independent orbital/PySCF fixtures |
| Compact potential/response coefficients | `xc.coefficients`, existing Graph AD/CSE | Direct bilinear factor and directional tests |
| AO matrix assembly | `xc.potential.assemble_coefficients` | Pinned independent PySCF E/V matrices |
| Both AO-leg geometric adjoints | `xc.coefficients.jet_pullback_program` | Displaced centers/points/weights, separately and together |
| CPU point lowering/cache | Existing ScalarCEmitter, CppCompilerAdapter, compile_runtime | Native/interpreter gates and standalone ASan/UBSan ABI harness |
| Solver actions | Existing `FixedDensityXCDerivativeKernel` adapter | Full-spin actions, restricted reduction and svec transpose |

The previous hand-written `potential_coefficients` and response coefficient
formulas are retired. Compatibility functions delegate to these common owners;
there is no second production contraction formula. The generic tau-half path
remains for synthetic tests: tau = 1/2 sum_k grad(phi) D grad(phi). It does not
add a meta-GGA. Cross-spin sigma is grad(rho_a) dot grad(rho_b), without an
extra factor two. Nonlinear XC follows complete-density reduction; splitting
D into contributions and summing their separate nonlinear energies is invalid.

For LDA, coefficient generation requests only rho and value AO jets. GGA adds
first jets and sigma, with no tau reduction. Response requests only the active
feature Hessian and includes both its action and the changing sigma Jacobian.
Geometry adds one AO derivative order. Unsupported observables, complex or
asymmetric D, unvalidated physical boundary domains and stale identities fail
explicitly. A structurally empty fixed AO mask contributes constant zero;
nonempty zero density retains the audited vacuum derivative error.

The old potential baseline already used compact panels. The new generator
canonicalizes the same patterns rather than claiming an invented symbolic
compression speedup. LDA assembly uses one matrix product per spin; GGA forms
one combined spatial panel and uses two. Reported coefficient SSA before/after
CSE is distinct from an AO-pair expansion, which is never constructed.

## Native CUDA matrix schedule

The resident CUDA semilocal owner uses compiler-generated 16-by-16 shared-memory
contractions for admitted AO/point shapes. Density products preserve explicit
symmetrization of the input. Potential assembly derives compact weighted panels
from `dft.xc_bilinear` using Graph differentiation, then contracts both symmetric
AO legs. LDA/GGA reuse one panel per spin; meta-GGA uses four. The original
density-product work panels are dead after feature/response coefficients have
been formed, so this schedule adds no global buffer or provider allocation.

The compiler owns tiling, the shape admission and the scalar fallback. Native
code supplies the stream and borrowed pointers. Shapes smaller than one matrix
tile or outside the two-dimensional launch domain keep the scalar grid-stride
schedule. Partial blocks use zero padding and still participate in all barriers;
only the authoritative triangle writes and mirrors the final potential.
Quadrature, point-domain rules, precision, screening and convergence gates do
not change. See the [schedule decision](../../.agents/notes/implemented/performance/2026-09-23-cuda-xc-matrix-tiles.md).

Strict FP64 tiled density and potential inner products use explicit FP64
`fma`, even when the surrounding CUDA build disables implicit contraction.
Storage and accumulation remain FP64, with the same reduction index order and
one accumulator. Fusion changes rounding; cross potential products also add
their two legs sequentially. Cross products check the accumulator after each
shared-memory point tile. A finite result proves that no earlier FMA accumulator
in that tile became nonfinite. Otherwise the tile replays from its saved initial
accumulator, checking each step and retrying the original grouped two-product
expression when needed. A cancelling leg therefore retains the original finite
domain without a range check on every ordinary term. Independent E/V and
complete energy/force gates qualify the arithmetic on their recorded source. The range-sensitive diagonal fallback retains its
original two-product expression, and FP32/mixed products keep their separate
arithmetic contract.

## Explicit native execution

The CUDA KS owner automatically requests occupied-orbital density features for
eligible strict-FP64 meta-GGA proposals. Only the token-bound canonical frame
that actually formed the accepted density can supply the four orbital jets,
and a tile uses it only when its occupied rank is smaller than its active AO
count. Existing density scratch holds the projection; potential assembly still
uses the original AO jets. No extra allocation or screening change is required.

Every completion retires that frame lease. Held confirmations, rejected probes,
imported densities, mixed precision, nonlocal and response work, and missing or
stale frames retain matrix contraction. LDA/GGA's one-jet route is unchanged.
`GENERATIVEQC_CUDA_KS_OCCUPIED_XC=0` disables the request for comparison or
rollback; unset or `1` requests it without bypassing any admission check.
Other explicit values also retain matrix contraction.

The CUDA density-feature reduction uses one warp per spin/grid point for
32 or more AOs. Lanes traverse adjacent AO components and reduce the existing
generated rho/gradient/tau contributions in FP64; smaller AO spaces retain
scalar summation. The compiler chooses the launch, and native execution binds
the same borrowed buffers for physical density and signed response. Point and
AO tails are covered without extra shared/global storage. Changing reduction
order requires the independent complete potential and response gates.

```python
from pathlib import Path
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.resources import ResourceBudget
from generativeqc_compiler.xc import functional
from generativeqc_compiler.xc.native import NativeContractionProgram
from generativeqc_compiler.xc.prepared import PreparedXCContractions

program = NativeContractionProgram(
    functional("PBE"),
    "response",
    compiler=CppCompilerAdapter(Path("c++")),
    cache=Path(".artifacts/xc-cache"),
)
with PreparedXCContractions(
    program,
    basis,
    grid,
    tile_points=64,
    resource_budget=ResourceBudget(host_bytes=64 << 20),
) as prepared:
    delta_v = prepared.execute(density, delta_density=direction)["response"]
```

AO evaluation and CPU BLAS are native; generated scalar/coefficient/jet-pullback
functions use checked contiguous point ABIs. NumPy orchestrates bounded panel
reductions and products. There is no production PySCF/Libxc callback. The common
compile cache hashes actual source/compiler/options/binary; endpoint metadata
also binds host source owners, basis, quadrature, functional and mask identity.

`spatial=PreparedSpatialGrid(...)` borrows an existing CPU fixed-mask adapter.
Its certificate must cover the requested jet order. Execution holds the CPU map
lock during internal work, with no yield to caller code; reconfiguration cannot
mix new points and old maps. The prepared XC owner does not close borrowed
basis/spatial objects. Each call recomputes from the supplied D.

The existing response solver adapter accepts `prepared=prepared` for a matching
native response owner. `apply_spin()` keeps functional-spin channels;
`apply()` and `apply_transpose()` retain the restricted total-D convention.
This does not register a new complete KS/CPKS method capability.

## Resident CUDA spatial potentials

`PreparedXCContractions` can borrow a CUDA `PreparedSpatialGrid` and evaluate
fixed-density energy and potential without downloading AO jets or density
features. Point caching and AO assembly are independent of the source of the
functional. Ordinary grid artifacts include canonical LDA/PBE. The existing
`interior-v1` API binds an audited semilocal composition explicitly:

```python
from generativeqc_compiler.xc.spatial_cuda import compile_spatial_cuda

spec = functional("R2SCAN", spin="polarized")
artifact = compile_spatial_cuda(spec, cuda_compiler, cache)
with PreparedSpatialGrid(
    basis, grid, backend="cuda", artifact=artifact,
    ingredients=("rho", "gradient", "tau"), tile_points=128,
) as spatial:
    program = NativeContractionProgram(spec, compiler=cpu_compiler, cache=cache)
    with PreparedXCContractions(program, basis, grid, spatial=spatial) as prepared:
        result = prepared.execute(source, stamp=source.stamp)
```

`source` is a `DensitySource`. Both density-matrix and validated orbital-factor
routes are supported. The CPU program supplies the shared contract and explicit
host alternative; the selected device execution performs no CPU XC evaluation.

The generated extension admits linear combinations of the audited LDA/PW,
PBE and SCAN/r²SCAN components in `spec.PUBLIC_COMPONENTS`. Full expression
identity, coefficients and the interior domain bind the artifact. Restricted
and polarized layouts share the polarized evaluator, with exact equality checks
on restricted spin features.
An unavailable exact artifact selects `host_unfused` only when the existing
host ingredient contract is satisfied; explicitly requesting `device_fused`
then fails at preparation. Execution failures propagate.

For Libxc worker semantics, use `SpatialContractionProgram`. It adapts the existing
source-derived rho/sigma/tau point lowerers, including automatic imported
components and the registered split-hybrid semilocal pairs, to the same grid ABI:

```python
from generativeqc_compiler.xc.spatial_program import SpatialContractionProgram

# MN15 selects only its semilocal XC contribution. Exact exchange is external.
program = SpatialContractionProgram("MN15", compiler=cuda_compiler, cache=cache)
with PreparedSpatialGrid(
    basis, grid, backend="cuda", artifact=program.artifact,
    ingredients=program.consumer.ingredients, tile_points=128,
) as spatial:
    with PreparedXCContractions(program, basis, grid, spatial=spatial) as prepared:
        result = prepared.execute(source, stamp=source.stamp)
```

This device program needs no CPU scalar artifact. An imported component name,
such as `GGA_X_B88` or `MGGA_X_MVS`, or a supported `FunctionalSpec` composition
uses the existing ingredient/work-policy admission, with no spatial name list.
Unsupported Laplacian, range/nonlocal operators and altered split-hybrid pairs
remain outside that contract. Explicit exact-exchange metadata is rejected.
`MN15` and `M06-2X` names select their semilocal pairs only; method owners supply
exact exchange independently. This API does not assemble a full hybrid energy.

Libxc point workers preserve vacuum screening, per-component density/sigma/tau
floors, Fermi-hole constraints and original-density energy weighting. Their vxc
is evaluated at work inputs; it is not differentiation through clipping or a
new physical empty-spin particle-birth continuation. Restricted density-only
workers use total-density U1 semantics. GGA/meta-GGA workers share equal-spin
arithmetic after validating equal physical channels, retaining independent spin
partials. Source bytes, coefficients and worker contracts bind the artifact
identity. A missing/mismatched artifact or host schedule fails at preparation;
this device program has no host scientific fallback.

Point evaluation occurs once per grid point. A bounded FP64 panel stores energy,
two density derivatives, six Cartesian gradient coefficients and two kinetic
coefficients. All AO pairs reuse these eleven rows; meta-GGA matrix assembly
includes tau's one-half factor exactly once. The grid resource plan charges
`88 * tile_points` bytes for these coefficients plus `8 * (tile_points + 6)`
bytes for weights and integration scratch in every local plan, including a
one-AO tile. Features and AO jets remain intact for the task lease. Statistics
report `xc_point_evaluations`, the per-tile `xc_point_cache_bytes`, and
`xc_total_point_cache_bytes` including optional batch slots.

Spatial XC uses the native compact potential panels and tiled symmetric matrix
contraction. Tiny or out-of-domain shapes retain the bounded scalar fallback.
Integration uses a fixed-warp compensated FP64 reduction; batch accumulation
retains its low words until final publication. The optional queue defaults to
eight tiles with a 32 MiB limit. It preserves original point/AO maps and uses the
native deterministic ordered scatter. The resource plan charges all copied AO,
feature, coefficient, weight, matrix, descriptor and error storage. Insufficient
budget or optional allocation failure selects the single-tile path; execution
errors propagate. `xc_batch_tiles`, `xc_batch_device_bytes`, `xc_batch_flushes`
and complete `owned_device_bytes` expose the selected work and storage.

Use `device_options=SpatialXcOptions(batch_tiles=1)` on `PreparedXCContractions`
to disable batching. Import `SpatialXcOptions` from
`generativeqc_compiler.dft.spatial_xc_batch`. Its explicit
`matrix_precision="fp32"` option uses the existing float compact matrix body,
while point evaluation, storage, energy/electron integration and global matrix
scatter remain FP64. The default is `"fp64"`; precision is part of prepared
identity and statistics. FP32 potential acceptance is `2e-6` absolute/relative
against independent Libxc on the spatial qualification fixtures. FP64 uses
`5e-10`; these fixture gates are not universal error bounds or SCF admission.

`SpatialContractionProgram` enables the existing AO radial/axis reuse emitter
by default (`ao_radial_reuse=False` selects its scalar alternative). Compatible
single-component registrations share parameterized compiled packages by default.
The native package planner must prove both polarized and restricted worker ABIs;
a common family name is insufficient. Each package contains immutable parameter
rows selected per call, so distinct prepared owners cannot overwrite each
other's coefficients. Functional metadata stays distinct from package identity.
Unmatched graphs and compositions retain specialized artifacts. Set
`share_workers=False` to request a specialized artifact without catalog worker
discovery. The first shared request incurs structural discovery; subsequent
requests in the process reuse the plan, and compatible programs reuse the same
compiled artifact.

Both APIs evaluate fixed-grid energy/potential, with their respective interior
or Libxc worker domains. Neither adds molecular force, response, hybrid exchange
or nonlocal-correlation capabilities. Native KS admission and its PBE launch
schedule remain independently owned.

`test_spatial_xc_codegen.py` and `test_spatial_libxc_codegen.py` compare emitted
adapters to independent Libxc, including worker boundaries. Their corresponding
CUDA test files check complete prepared endpoints, interior density directional
derivatives, resource accounting, leases and rejection/recovery. Run real CUDA
tests with `GENERATIVEQC_GRID_CUDA_TEST=1` inside a finite Slurm GPU allocation.

## Geometry and memory scope

`GeometryPartials.centers`, `.points` and `.weights` are independent gradients
at fixed D, exponents/coefficients, parameters, regularization and mask membership.
For an AO depending on r-R_A, point motion has positive spatial-jet sign and
its own center motion has the negative sign. The weight partial is unweighted
`e_xc`. `directional(centers=..., points=..., weights=...)` contracts all three
sources exactly once. A physical partition/grid adapter must supply its actual
motion. These partials exclude electronic response, Pulay and other energies;
force is minus the complete gradient, not this partial contribution alone.

The shared `ResourceBudget` preflight covers numeric capacities. For tile B,
N AOs, P points and A atoms, the XC workspace allowance is
`8*(20*N*N + 128*B*N + 256*B)` bytes, plus basis/grid storage,
`8*(16*P + 12*A)` for quadrature/output copies, and molecular setup scratch.
Borrowed spatial requests are composed into the same plan and may conservatively
double-charge inputs. The bound includes immutable copies and full spin D/V;
no global AO table, AO^4 tensor or solver-history tape is retained.

Python object headers/allocator rounding, BLAS workspace/thread stacks, native
scalar stack/code pages and caller-retained old results are explicitly outside
that numeric scope. It is not a whole-process memory cap. Benchmarks separately
record traced allocations and process RSS high-water observations, with their
measurement scopes; none is silently substituted for the numeric plan.

## Validation and reproduction

`test_xc_contractions*.py` extends pinned independent E/V checks with both spin
layouts, f Cartesian/spherical AOs, response at three finite-difference steps,
spin exchange, svec transpose, each geometry source and simultaneous motion,
minimal ingredients, stale masks and negative domains. Existing synthetic
bilinear tests retain tau-half/cross-spin/off-diagonal factor checks.

Run `tools/check_xc_native_sanitizer.py --output .artifacts/xc-sanitizer` to
compile 16 standalone CPU ASan/UBSan ABI harnesses. Its explicit instrumented
flags and source hashes are separate from the production compile cache.

`tools/benchmark_xc_contractions.py` records complete construction/execution,
scalar compilation, source/SSA size and liveness, contraction counts, numeric
plans and scoped observed memory. Independent reference work is outside timed
production calls. It requires a clean measured checkout and matching native
library. `tools/publish_xc_contractions.py` validates and publishes only durable
summaries/samples under the common evidence policy. See the retained evidence
README for exact commands and historical measured revisions. No significant
speedup or automatic production promotion follows from these measurements.
