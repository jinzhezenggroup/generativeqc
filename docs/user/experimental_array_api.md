# Experimental Array API

GenerativeQC exposes an Array API **2025.12 CPU eager namespace** at
`generativeqc.experimental.array_api`. The ordinary user path is deliberately
shape/dtype based: users write array expressions and do not need to construct
TensorIR `IndexSpace`, `Index`, or `TensorSpec` objects.

The public surface remains experimental. Concrete `Array` objects implement
`__array_namespace__` and the core standard, including all thirteen standard
dtypes, IEEE non-finite values, data-dependent shapes, and the `linalg` and `fft`
extensions. Storage and eager execution are explicitly NumPy-backed CPU work,
using `array-api-compat`; no GPU/native-compiler conformance is implied.

`VibeArray` and `compile` remain a bounded scientific TensorIR frontend. Symbolic
arrays deliberately do not advertise `__array_namespace__`: eager capabilities
must not be mistaken for qualified compiler/native operations. See
[the developer boundary](../developer/array_api_frontend.md) and
[the reproducible conformance gate](../maintainer/array_api_conformance.md).

```python
from generativeqc.experimental import array_api as xp

x = xp.asarray([1, 2, 3], dtype=xp.int32)
assert x.__array_namespace__(api_version="2025.12") is xp
assert x.device == "cpu"
y = xp.where(x > 1, x, 0)
```

## Ordinary use

```python
import numpy as np

from generativeqc.experimental import array_api as xp

@xp.compile
def observable(C, occupation, O):
    density = (C * occupation) @ C.T
    return xp.sum(density * O)

C = np.array(
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    dtype=np.float64,
)
occupation = xp.asarray([2.0, 1.0], dtype=xp.float64)
O = np.eye(3, dtype=np.float64)

value = observable(C, occupation, O)
```

The namespace also works eagerly on its NumPy-backed CPU arrays. The same
function body can therefore be evaluated directly and then compiled without
rewriting its array expressions. The first compiled call captures a
specialization from the concrete argument shapes and dtypes, lowers the
expression to canonical TensorIR, and compiles/caches a native CPU executable. Use
`observable.lower(C, occupation, O)` when the TensorIR `Program` itself is
needed for inspection or optimization. Native CPU currently requires float64.
`backend="reference"` explicitly selects the independent interpreter; unsupported
native lowering never falls back to it implicitly.

## Compile energies and derivatives

Users do not need to construct derivative IR or call compiler-internal AD:

```python
@xp.compile(backend="cpu")
def energy(density, hcore):
    return xp.sum(density * hcore)

density = xp.eye(2)
hcore = xp.asarray([[-1.0, -0.2], [-0.2, -0.8]])
energy_and_gradient = xp.value_and_grad(energy, argnums=0)
value, density_gradient = energy_and_gradient(density, hcore)
```

`grad` returns only gradients. Integer `argnums` selects one positional parameter;
a tuple returns gradients in that order, including with keyword-bound arguments.
Negative positions are accepted. Both transforms require one scalar array
output; use `vjp` for vector/dictionary outputs. Differentiability is inferred
from `argnums`. `value_and_grad` compiles primal and reverse nodes as **one shared
native endpoint**, not interpreter calls or finite differences.

`xp.jvp(function, (x,), (direction,))` returns primal and directional derivative.
`value, pullback = xp.vjp(function, x)` returns a reusable compiled pullback;
`pullback(cotangent)` returns a tuple of input cotangents. Pullbacks snapshot
their primal inputs, so later mutation cannot change the derivative. Call
`pullback.close()` to release native owners. Compiled and gradient functions
also provide `close()` and `compilation_info()`.

For first-order fixed external masks on generic finite FP64 arrays, public JVP
accepts an explicit opt-in and a matching tangent tuple with `None` at each
non-differentiated primal:

```python
def selected_values(x, y, mask):
    return xp.where(mask, x * x, y * y)

value, direction = xp.jvp(
    selected_values, (x, y, mask), (dx, dy, None),
    allow_external_mask_where=True,
)
```

This path is qualified for reference/materializing native CPU. `None` supplies
no numerical tangent, including for the Boolean mask; omitted real directions
are fixed parameters. At least one explicit real direction is required. Every
supplied direction, including a disconnected one, must match its primal shape
and dtype and remain finite. Requested disconnected FP32 or scientific-domain
inputs reject before compilation. All live primal and forward derivative
domains remain checked, including unselected branches and empty outputs. Results
are detached, inherited native byte/work/node budgets apply, and temporary
executors close on success or failure. Default JVP admission, computed predicates,
Boolean AD, scientific annotations, FP32/mixed/packed selection, higher-order
selection and CUDA admission remain unchanged.

`xp.vjp` additionally accepts `argnums` (an integer or nonempty tuple). It returns
only those input cotangents, always as a tuple in the requested order. Negative
positions count from the supplied positional primals, rather than any omitted
function defaults. Omitting `argnums` retains differentiation of all supplied
primals.

For fixed external masks, select only real arguments and explicitly enable
`allow_external_mask_where=True`:

```python
def selected_values(x, y, mask):
    return xp.where(mask, x * x, y * y)

value, pullback = xp.vjp(
    selected_values, x, y, mask,
    argnums=(0, 1), allow_external_mask_where=True,
)
dx, dy = pullback(xp.ones_like(value))
pullback.close()
```

This first-order opt-in is limited to generic finite FP64 arrays on reference or
materializing native CPU. All primal arrays, including the Boolean mask and
non-selected real inputs, belong to the forward snapshot. Mutating them after
`vjp` does not change a pullback. Every output needs a matching finite cotangent,
including empty/disconnected outputs. All live primal/forward domains and
reference retained-byte and native byte/work/node gates remain enforced.
Computed predicates, Boolean AD, scientific
annotations, FP32/mixed/packed policies, higher-order selection and CUDA are
unsupported; default `where` rejection is unchanged.

For a condition computed from real comparisons, explicitly select
`allow_comparison_where=True` on `jvp`, `vjp`, `grad` or `value_and_grad`:

```python
def piecewise_values(x, threshold):
    return xp.where(x > threshold, x * x, 3 * x)

value, direction = xp.jvp(
    piecewise_values, (x, threshold), (dx, None),
    allow_comparison_where=True,
)
```

This is first-order differentiation of the real branches with the condition
fixed at that call's primal value. It supports all six finite real comparisons,
external selectors and their typed views on generic FP64 arrays, using reference
or materializing native CPU execution. Only real branch edges carry derivatives;
Boolean values and comparisons have no tangents/cotangents. Real computations
used only by the predicate are evaluated and domain-checked as primals, without
evaluating their derivatives. If a value is also used by a real branch, its
branch derivative remains subject to the usual domain checks. All unselected
real branches retain the existing primal and forward-derivative checks.

At a switching point, the selected branch defines the reported directional and
adjoint values; this is **not a claim of mathematical differentiability** at a
jump or kink. Finite differences qualify points whose selectors do not switch
under the chosen perturbation. VJP snapshots include predicate inputs, so later
mutation does not change its selector. `None` JVP directions, `argnums`, retained
byte/work/node budgets, detached outputs and executor lifetime follow the fixed
external-mask contract above. The default policy and
`allow_external_mask_where=True` retain their narrower admission. Boolean
outputs/AD, scientific annotations, FP32/mixed/packed generated selection,
higher-order selection and CUDA remain unsupported. Low-level reference
`tensor.autodiff.jvp/vjp/dot_test` accepts the same explicit option on generic
finite FP32/FP64 graphs; generated `linearize/transpose_program` accepts it on
FP64 graphs.

For quantum-chemistry equations, pass `specs={parameter: TensorSpec(...)}` to
`compile` to retain AO, occupied, virtual, auxiliary and spin domains. Equal
extents never authorize mixing scientific spaces. `einsum` accepts exact
`Fraction` coefficients. The native RHF density derivative is checked against
independent Fock contractions and finite differences in
`tests/python/test_array_api_native_autodiff.py`.

`backend="cuda"` uses the existing native TensorIR compiler/planner. `cuda_arch`
and the Slurm-visible `device` are explicit. This interface stages host inputs
to the GPU and returns detached CPU arrays; it does not create a GPU eager
namespace or accept implicit foreign GPU conversions. CUDA failures never
switch to CPU/reference. Real-device qualification requires scheduled execution;
planner/source tests alone do not qualify a GPU endpoint.

Native compilation requires a C++/CUDA toolchain and a verified compiler-cache
launcher. `cache`, `max_bytes`, `max_work`, `max_nodes` and `compile_timeout`
control artifacts, resources and finite compiler lifetime. Forward and
derivative programs pass the same backend admission. Native CPU supports float64
contractions, shape/index transforms, reductions and finite-domain `exp`, `log`,
`sqrt` and exact `pow`; CUDA admits float32 and float64 for these scalar functions.
The materializing CPU backend also accepts one-byte Boolean inputs, constants,
shape/static-index views and six comparisons of finite float64 operands.
It also executes forward `where` on genuine Boolean selectors and finite generic
float64 branches, with explicit broadcasting and detached outputs.
Programs containing Boolean data use a separate checked byte ABI; pure float64
programs retain their existing ABI. FP32 comparison operands, Boolean AD,
float32/mixed selection branches, nonfinite comparison semantics and native CUDA
Boolean data remain
unsupported. This subset does not establish complete Array API conformance.
Other operations fail closed; full eager conformance does
not imply full compiler coverage. Higher-order public transforms are unsupported.

Inputs are non-differentiable by default. When the lowered program will be used
with TensorIR AD, declare that contract explicitly:

```python
@xp.compile(differentiable=("x",))
def norm2(x):
    return xp.sum(x * x)

program = norm2.lower(x)
```

This avoids silently treating every runtime array as a differentiable scientific
parameter while still giving the inferred public path a supported JVP/VJP route.

The compiled frontend supports finite, uniform array creation through `zeros`, `ones`,
`full`, and `zeros_like`, `ones_like`, `full_like`. Inside `@xp.compile`,
these constructors lower **statically known** shapes to a single exact TensorIR
scalar constant and broadcast (no size-proportional literal payload):

```python
@xp.compile
def shifted(x):
    return x + xp.ones_like(x) * xp.full(x.shape, 0.25, dtype=x.dtype)
```

Inside capture, `bool`, `float32` and `float64` dtypes with `device=None` are
admitted: an explicit device request never triggers a hidden host transfer.
Boolean `full`/`full_like` require a Boolean fill
value; `zeros` and `ones` produce false and true when `dtype=xp.bool` is explicit.
`full(shape, integer)` without an explicit floating dtype is rejected rather than
silently claiming an unsupported compiled integer dtype. Scientific
AO/occupied/etc. arrays cannot be passed to `*_like` without their explicit
TensorIR metadata.

Outside capture, creation follows the full CPU eager standard, including integer
and complex dtypes and explicit `device="cpu"`; these compiled restrictions do
not apply. See [execution boundaries](#execution-boundaries).

In both eager and captured programs, `.T` requires a two-dimensional matrix.
Use `.mT` / `matrix_transpose` for the final two axes of a matrix stack, or
`permute_dims` with an explicit permutation for arbitrary-rank arrays.

The current compiled frontend supports ordinary shape broadcasting for generic arrays,
`@`, `.T`, `.mT`, `matrix_transpose`, reshape with one inferred `-1`
dimension, `tensordot` with static contraction counts or pairs of axis sequences,
and `vecdot` with a static signed axis. These two contractions accept generic
real float32/float64 arrays (with ordinary promotion), require equal contracted
extents, and reuse TensorIR einsum. `vecdot` broadcasts only the remaining axes;
`tensordot` retains left free axes followed by right free axes. Capture permits
at most 52 distinct contraction labels; scientific index domains require explicit
`einsum`. The frontend also supports static indexing with integers, slices (including negative
strides), `None`/newaxis, and ellipsis. Common elementwise conveniences
`square` and `reciprocal` lower to existing TensorIR multiply/divide nodes.
Captured `positive(x)` and unary `+x` preserve float32/float64 values, signed
zeros and scientific index metadata. Both reject Boolean arrays and integer
control tensors; unary plus does not turn them into numeric data.
`sum` and `mean` accept static axes, and generic arrays may use
`keepdims=True` to retain reduced singleton axes. Shape utilities also include
`broadcast_shapes`, `broadcast_arrays`, `expand_dims`, `squeeze`,
`moveaxis`, `flip`, `roll` and `repeat`. `expand_dims` accepts the 2025.12 multi-axis tuple
form, and `permute_dims` accepts negative axes; all reuse existing TensorIR
reshape, transpose, broadcast and gather semantics.

Captured `repeat` accepts a static nonnegative Python integer count and a signed
axis, or `axis=None` to flatten in row-major order. Scalar inputs use `axis=None`.
It composes reshape/broadcast without an elementwise static gather map; per-element
count arrays are not admitted. Existing backend byte/work/node budgets still apply.

Six elementwise comparisons
(`equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`, plus
their Python operators) accept finite real operands and return genuine Boolean
arrays. Generic comparison operands broadcast and promote within float32/float64;
scientifically annotated operands must have identical domains. Boolean arrays
support the shape/view operations listed above in captured programs,
but remain excluded from compiled floating arithmetic and reductions. Generic arrays accept
finite Python float literals as ordinary scalar values, so expressions such as
`x + 0.5` have eager/compiled parity. The compiler records the exact binary value
of that Python float. During capture, negative-zero float literals are rejected because exact
rational constants cannot preserve their sign. Explicitly scientific arrays
retain the stricter exact-scalar spelling rules.

## Generic native molecular method sources

Use `xp.MolecularModel` to connect any **existing, supported** native
`Calculator` method selector (such as `rhf`, `uhf`, or a qualified
`pbe-rks`/`pbe0-rks` route) without writing a separate frontend or
first-order TensorIR rule for each functional:

```python
pbe = xp.MolecularModel(("O", "H", "H"), method="pbe-rks",
                        basis="sto-3g", backend="cpu")
energy_grad = xp.value_and_grad(lambda coordinates: pbe(coordinates))
value, gradient = energy_grad(coordinates)
energy_grad.close()
```

`method` passes through the established native method/MethodIR resolution,
using the native method's own analytic energy and force operators.
`multiplicity`, `density_fitting` and `auxiliary_basis` are part of the
immutable source identity; unknown or unsupported scientific combinations fail
through the native capability/resource gate, without CPU or finite-difference
fallback. This is an *interface generalization*, **not** evidence that all DFT
functionals, ECPs, unrestricted or CC methods already provide qualified analytic
nuclear gradients. Energy-only calls are allowed for energy-capable methods;
`grad`/`value_and_grad` require the method's genuine analytic forces.
The initial public method-neutral source remains host-staged FP64 and first
order; higher-order implicit/response gradients are not inferred.

### User-composed XC methods inside Array API

A user may combine **already audited** XC components through the public
`generativeqc.extensions` builders, then pass their canonical `MethodIR`
directly to `xp.MolecularModel`. This uses the existing native SCF and,
**only when qualified**, its analytic force. Explicit `KsOptions` may select
the grid and tile schedule, but cannot replace the method's XC composition.

```python
from generativeqc import GridSpec, KsOptions
from generativeqc.extensions import method, xc
from generativeqc.experimental import array_api as xp

functional = xc.compose(
    "my-pbe50",
    {"GGA_X_PBE": "1/2", "GGA_C_PBE": 1},
    exact_exchange="1/2",
)
method_ir = method.compose("my-pbe50", xc=functional)
model = xp.MolecularModel(
    ("H", "H"), method=method_ir, basis="sto-3g", backend="cpu",
    ks_options=KsOptions(grid=GridSpec()),
)
coordinates = xp.asarray(
    [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=xp.float64,
)
compiled = xp.compile(lambda positions: model(positions), backend="cpu")
try:
    energy = compiled(coordinates)
finally:
    compiled.close()
```

The MethodIR identity, not the custom label, determines source deduplication;
grid/scheduling options are also included in the immutable source identity.
The source passes the original MethodIR to `Calculator`; it does not
reconstruct the science from the label. Unsupported methods, device routes
and missing analytic-force support fail through the native capability gate,
not through a silent fallback. For methods with qualified analytic forces,
`xp.value_and_grad` reuses the existing first-order native-source AD path.

This does **not** allow an arbitrary new formula in \(\rho,\sigma,\tau\)
to run self-consistently. That separate compiler/SCF work is tracked in
[XC expression-to-SCF #2330](https://github.com/jinzhezenggroup/generativeqc/issues/2330).

### Diagnose small-endpoint overhead

The opt-in environment switch
`GENERATIVEQC_ARRAY_API_PROFILE=1` must be set **before constructing the
compiled function**. `compiled.compilation_info()` then exposes
`molecular_profile_ns` and `molecular_profile_counts`. The stages are
`source_input_scan`, `source_geometry`, `source_calculator_lookup`,
`source_native_singlepoint`, `source_result_publication`,
`source_diagnostics`, `tensor_snapshot_binding`,
`tensor_build_cold`, `molecular_dispatch`, and `tensor_execute`.
`molecular_dispatch` includes its nested source stages, so do **not** sum all
stage totals as if they were disjoint. Timings use CPU wall clocks around
synchronous calls and are **not** CUDA-event/kernel measurements.

For alternating complete energy-and-gradient endpoints with a separately
subtracted cold profile:

```bash
python benchmarks/array_api_rhf_endpoint.py --backend cuda --molecule h2 \
  --profile-molecular --repeats 10 --output /tmp/rhf-profile.json
```

This command still requires the documented bounded Slurm GPU allocation.
`warm_phase_profile` reports the mean time per profiled warm call; profile
sampling changes the measured timing and disables the ordinary 5% parity verdict.
Run again without `--profile-molecular` for fair endpoint comparisons. Neither
mode by itself certifies GPU peak memory or exact-J/K operator-work parity.

## Compose native RHF energies with the unified molecular operator

`MolecularModel` is a GenerativeQC extension, not a standard Array API function.
Selecting `method="rhf"` binds the native all-electron closed-shell RHF solver
and its complete analytic force endpoint. It composes inside ordinary captured
Array API expressions:

```python
rhf = xp.MolecularModel(("O", "H", "H"), method="rhf", basis="sto-3g", backend="cpu")

@xp.compile(backend="cpu")
def energy(coordinates, strength):
    separation = coordinates[1] - coordinates[2]
    return rhf(coordinates) + strength * xp.sum(separation * separation)

coordinates = xp.asarray(
    [[0.0, 0.0, 0.0], [0.0, -1.43, 1.11], [0.0, 1.43, 1.11]],
    dtype=xp.float64,
)
energy_and_gradient = xp.value_and_grad(energy, argnums=0)
value, nuclear_gradient = energy_and_gradient(coordinates, xp.asarray(0.0))
forces = -nuclear_gradient
energy_and_gradient.close()
energy.close()
```

Coordinates have shape `(number_of_nuclei, 3)` in **Bohr**, energy is total
**Hartree**, and nuclear gradients/forces are **Hartree/Bohr**. These are full
geometric derivatives, including orbital stationarity, basis/Pulay and nuclear
repulsion contributions. They are **not** the density derivative/Fock test
described earlier. Nonlinear surrounding energy expressions and compiled
geometry transformations use the same first-order chain rule. `grad`, combined
`value_and_grad`, JVP and snapshot VJP are supported. Changed geometries always
run a new native endpoint; identical operator/geometry expressions within one
capture share one source invocation. VJP replay uses the saved forward source
and does not rerun SCF after the caller changes an input.

The molecular method is an **explicit native source cut**. Capture records its
settings and geometry subprogram without loading a native library or executing
SCF. The derivative tape contains its local first-order jet; after AD lowering,
runtime binds that jet exactly at the snapshot geometry, preserving the native
energy without duplicate displacement arithmetic. TensorIR compiles the
surrounding algebra and its chain rule; the native RHF driver still owns the
solver and its analytic molecular derivative. This does not differentiate the
SCF iteration history, implement arbitrary electronic methods, or add higher
derivatives/response solvers. A molecular `lower()` result requires its public
compiled owner's source bindings; it is not a standalone TensorIR executable
with only the user's original inputs.

Configuration requires a nonempty tuple of at most 256 nuclei, a valid
method-specific electron/spin occupation, an admitted Gaussian basis, float64
coordinates, finite positive tolerances and bounded iteration counts. Restricted
RHF specifically requires a positive even electron count and singlet multiplicity.
Scientifically annotated coordinates must use complete, spin-independent `atom`
and `cartesian` index spaces; matching AO/orbital extents do not authorize a
nuclear-coordinate binding. These domains survive native snapshots and AD.
Density fitting, unrestricted methods and ECPs are admitted *only* when the
selected native Calculator supports the exact requested basis, backend and
analytic derivative; the generic interface does not create missing scientific
capabilities. Nested molecular dependencies in geometry remain unsupported.
Capture admits at most 16 distinct molecular sources. Native
nonconvergence, basis/derivative/resource errors propagate without publication
or fallback. `resource_budget` is the existing `ResourceBudget` for native method
preflight; the compiler's `max_bytes`/`max_work`/`max_nodes` bound **tensor algebra**,
not the independent SCF/J/K endpoint's work or resident memory.

`backend="cuda"` must be selected on both `MolecularModel` and `compile`/AD, with the same
logical `device`. The interface stages host coordinates/results, not resident
GPU arrays. CUDA failures do not rerun on CPU. The native CPU ABI calls its
scientific backend `cpu_reference`; this label means the **native C++ CPU method**,
not `backend="reference"`'s Array API interpreter. Molecular source execution
does not support the latter. `compilation_info()["molecular_calls"]` reports
native backend, source/force call counts, SCF iteration totals and last geometry
identity. `close()` releases retained native method and compiled tensor owners.

Run CPU endpoint acceptance with an actual built/installed native library:

```bash
GENERATIVEQC_LIBRARY=build/libgenerativeqc.so PYTHONPATH=python:. \
  pytest -q tests/python/test_array_api_molecular.py
GENERATIVEQC_LIBRARY=build/libgenerativeqc.so PYTHONPATH=python:. \
  python benchmarks/array_api_rhf_endpoint.py --backend cpu --molecule water \
    --repeats 10 --output /tmp/array-api-rhf-cpu.json
```

The benchmark compares identical FP64 exact-J/K complete energy/nuclear-gradient
calls with fresh Hcore starts, alternates order, separates cold JIT, and reports
numerical errors and actual SCF iterations. It reports a measured 5% timing
target separately from qualification: peak memory and a complete native
operator-work inventory are not measured by this script. **No GPU parity is
qualified by a CPU result.** CUDA tests/benchmarks require an explicit finite-time
Slurm `srun` on an allowed node (`n1`, `n2`, `n4`, `n5`), a compatible partition/GPU
request and the unchanged scheduler-assigned `CUDA_VISIBLE_DEVICES`. Select
`GENERATIVEQC_ARRAY_API_CUDA_TEST=1` for the real-device molecular test, or
`--backend cuda` for the benchmark; source/planner checks are not device evidence.
Pass `--cuda-arch` matching the selected GPU when it differs from the default
`sm_120`; this selects the tensor compiler target without changing Slurm visibility.
SSH host aliases and Slurm node names are distinct: these machines report
`node1`, `node2`, `node4`, and `node5` to the scheduler. Both verified naming
forms are admitted; `n3`/`node3` remains excluded during maintenance.

On an allowed remote machine with a source-matched CUDA library and the verified
compiler-cache launcher/toolkit on `PATH`, run the real-device acceptance and
complete endpoint benchmark in the same allocation. For example, after connecting
to `n4` and changing to that machine's checkout:

```bash
srun --partition=main --nodelist=node4 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G --time=00:20:00 \
  bash -lc '
    set -e
    export PYTHONPATH=python:.
    export GENERATIVEQC_LIBRARY="$PWD/build/cuda-release-sm120/libgenerativeqc.so"
    export GENERATIVEQC_ARRAY_API_CUDA_TEST=1 GENERATIVEQC_TENSOR_ARCH=sm_120
    python -m pytest -q tests/python/test_array_api_native_cuda.py \
      tests/python/test_array_api_molecular.py
    python benchmarks/array_api_rhf_endpoint.py --backend cuda --cuda-arch sm_120 \
      --molecule water --repeats 10 --output /tmp/array-api-rhf-gpu.json
  '
```

The scheduled molecular gate checks H2/STO-3G, HeH+/def2-SVP and water/STO-3G
gradients against independent central energy differences, plus compiled nonlinear
geometry composition and a mutation-safe reusable VJP. A CUDA library build or
planner test alone does not satisfy this gate.

## Scientific metadata remains explicit

The convenience above uses **generic array dimensions**: equal aligned extents
are compatible exactly as normal array programming expects. That does not weaken
GenerativeQC's quantum-chemistry type system.

Compiler/internal code and advanced users can still use `xp.trace(..., specs)`
with explicit AO, occupied, virtual, auxiliary, batch, or spin spaces. In that
scientifically annotated mode, equal integer extents do not make two axes
compatible, and reshape/broadcast operations that would erase those meanings
still require explicit TensorIR metadata.

This separation is intentional:

```text
ordinary public arrays        scientific annotated arrays
shape/dtype semantics         AO/occ/vir/aux/spin semantics
        \                         /
         \                       /
                  TensorIR
```

## Execution boundaries

The standard eager namespace supports Boolean, int8/int16/int32/int64,
uint8/uint16/uint32/uint64, float32/float64 and complex64/complex128 data on CPU.
Constructors infer standard dtype defaults; explicit `device="cpu"` is supported.
Non-finite values, negative power bases where mathematically defined, reduction
`dtype`, empty reductions, array-valued gathers and data-dependent results follow
the standard. `linalg` and `fft` results retain the same discovery protocol.

`copy=False` refuses an allocation when conversion requires one; wrapping an
existing NumPy allocation does not copy its data. `astype` with unchanged dtype
and `copy=False` preserves the input. `T` requires a matrix, `mT` transposes the
last two axes, indexing always returns an array, and Python scalar conversion
requires rank zero. Complex-to-real casts require an explicit component selection.
NumPy-style host interoperability is available, but foreign device protocols are
rejected before implicit conversion, including through operators or nested host
containers. CPU `from_dlpack` is explicit; `import_dlpack` retains the existing
same-device consumer handoff extension.

Capture accepts CPU bool/float32/float64 inputs and the compiler's bounded
operation set; backend execution has narrower dtype/primitive admission.
Captured `asarray(..., copy=False)` rejects dtype changes; `copy=True` or
`copy=None` permits an explicit supported real cast. Same-dtype conversion does
not imply aliasing of compiled outputs, which remain detached from inputs.
Shape/axis values must be static. Captured `flip` retains
its 65,536-index budget per reversed axis. Captured `roll` accepts static integer
shifts and signed axes (including tuples and repeated axes); `axis=None` rolls
the flattened row-major array and restores its shape. Its total gather-map budget
is 65,536 positions across axes with nonzero effective shifts, checked before
constructing any maps. Empty axes and shifts by whole axis lengths need no maps.
Scientific domains retain explicit
metadata requirements. Compiled reductions do not introduce implicit dtype
conversions, compiled empty means remain unsupported, and reference execution
requires finite inputs/results with positive `log`/`pow` and nonnegative `sqrt`
domains. Exact rational constants cannot represent negative zero. Integer/complex
compilation, dynamic indexing/shapes and arbitrary Python control flow are not
implied by eager support. Unsupported capture functions, including linalg/fft,
raise instead of evaluating a NumPy fallback during tracing.

Captured `minimum(x1, x2)` and `maximum(x1, x2)` accept generic finite
float32/float64 arrays, with at least one array operand. Array operands broadcast
and use existing float32/float64 promotion; static real scalars follow the weak,
exact scalar contract of the array dtype. Scalar overflow is rejected even for
empty arrays. Scientific indices, representation/symmetry/spin annotations,
other dtypes, implicit device transfers and nonfinite live inputs remain
unsupported. The existing eager standard operations retain their IEEE behavior,
including NaN propagation. Capture is a bounded subset, not full conformance.

Both captured operations retain the first operand on equality, including its
array signed zero. The [minimum](https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.minimum.html)
and [maximum](https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.maximum.html)
specifications leave signed-zero order implementation-defined, so equal numerical
results need not have the same zero sign as an eager backend. Static negative-zero
literals remain unsupported. Direct `xp.compile(xp.minimum)` / `xp.compile(xp.maximum)`
accept two concrete array arguments; put static scalar operands inside a function.
First-order JVP/VJP require `allow_comparison_where=True` and use the selected
branch, with broadcast adjoint reduction. Interior derivatives are qualified
separately from the tie selection convention; no mathematical differentiability
at ties or Hessian qualification is claimed. The operations reuse comparisons and
`where`, with shared broadcast values and unchanged resource budgets. Their
capability entries retain native CPU/CUDA qualification unset.

Captured `max(x, axis=None, keepdims=False)` and `min` accept finite generic
float32/float64 static arrays. Axis accepts a signed integer, a unique tuple of
axes, or `None`; an empty tuple is a detached identity. `keepdims` must be a
Python bool. An actual zero-length reduced axis raises; a zero-length retained
axis gives an empty output. Scientific annotations and nonfinite capture remain
unsupported. The existing eager IEEE behavior is unchanged. The
[max](https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.max.html)
and [min](https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.min.html)
specifications permit errors for empty reductions and implementation-defined
signed-zero order.

Capture pairs adjacent values in a balanced tree, preserving the earliest
C-order value on ties, including its zero sign. Before constructing the graph,
it limits total gather positions to 65,536, pairwise rounds to 64, and estimated
new retained tensor storage to 256 MiB. Ordinary backend node, work and byte
budgets still apply and may reject smaller programs. Put static options inside
a function, for example `xp.compile(lambda x: xp.max(x, axis=(0, -1)),
backend="reference")`, following the existing array-only runtime argument
contract. First-order comparison-selector AD requires
`allow_comparison_where=True` wherever comparisons remain in the graph; empty
axis and singleton identities use the existing identity AD. Tie gradients are
only a deterministic branch convention. CPU FP64 execution/AD uses the existing
opt-in native path; capability qualification flags remain unset, and native
float32 and CUDA reductions are not qualified.

Captured `clip(x, min=None, max=None)` accepts generic finite float32/float64
arrays. Each bound can be absent, a finite static scalar under the existing exact
scalar contract, or a same-dtype generic array. All three shapes must broadcast;
mixed array dtypes, scientific annotations and nonfinite live operands are
rejected. Both absent bounds give a detached numerical identity at execution.
Static negative-zero literals remain unsupported; array signed zeros are retained
by selection. Clamping applies the lower bound and then the upper bound, keeping
the current value on equality. Reversed bounds therefore select the upper bound;
the [2025.12 standard](https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.clip.html)
leaves reversed bounds and mixed array dtypes unspecified. These capture limits
do not restrict the existing eager standard `clip`, including its NaN behavior.
For direct `xp.compile(xp.clip, backend="reference")`, supply concrete array
arguments for all three parameters, as required by the compiled runtime input
contract. Capture absent or static scalar bounds inside a named function instead.
Reference first-order JVP/VJP use the existing explicit
`allow_comparison_where=True` option. Interior derivatives follow the selected
input or bound, including broadcast adjoint reduction. At a tie, the selected
current branch is only a convention, not a differentiability claim. No new AD
engine or TensorIR primitive is introduced. The clip capability entry keeps
native CPU/CUDA qualification unset; successful capture alone does not qualify
either backend, performance, or full standard compliance.

Captured `where(condition, x1, x2)` supports a genuine Boolean condition and
finite generic float32/float64 branches. The branches use explicit casts for
float promotion; condition and values each broadcast explicitly, including
scalars and empty axes. Python real branch scalars follow the existing exact
scalar contract (negative-zero literals are rejected; array signed zeros are
preserved). Reference outputs are detached and never mutate inputs. The reference
precision transform preserves the Boolean condition and casts only the real
branches; generated precision variants remain unqualified reference candidates.
The reference interpreter counts all live operands, selection results and published copies
against its retained-byte budget; selection costs one unit of logical work per
result element. All live operands are evaluated and finite-checked, so an invalid
or nonfinite unselected branch still raises, even under a constant condition.
Scientific index domains, representations and symmetry annotations are not
admitted by this selection primitive. TensorIR runtime reference JVP/VJP/dot
tests support external fixed Boolean masks in finite generic real graphs, and
comparison selectors with the explicit `allow_comparison_where=True` option;
see the [TensorIR derivative contract](../developer/tensor_ir.md#derivative-programs-151).
`xp.grad` and `xp.value_and_grad` accept `allow_external_mask_where=True`
for first-order fixed external-mask gradients on generic finite FP64 arrays
with `backend="reference"` or the default native CPU backend. Select only real
arguments with `argnums`; pass the mask as a separate non-differentiated argument.
For example:

```python
def selected_energy(x, y, mask):
    return xp.sum(xp.where(mask, x * x, y * y))

derivative = xp.value_and_grad(
    selected_energy, argnums=(0, 1), allow_external_mask_where=True
)
value, (dx, dy) = derivative(x, y, mask)
derivative.close()
```

All live branches and forward derivative domains remain checked, including
unselected branches and disconnected or empty requested gradients. Prepared
calls read the current mask; the function does not freeze its contents. The
combined value/gradient call uses one native endpoint and retains inherited
byte/work/node budgets and independent executor lifetime. With this external-mask-only
option, computed predicates,
Boolean differentiation, scientific annotations, FP32/mixed/packed policies,
higher-order transforms and CUDA remain unsupported. Default generated derivatives still reject `where`. The low-level `linearize` and `transpose_program`
functions accept `allow_external_mask_where=True` to generate first-order
JVP/VJP programs for fixed external masks on generic finite FP64 arrays,
replayable by the reference interpreter and materializing native CPU owner.
Only real branch tangents/cotangents are generated. Comparison selectors require
the separate `allow_comparison_where=True` option described above; comparisons
and masks never receive derivatives. See the derivative contract linked above for its eager
validation and higher-order boundaries.
The materializing native CPU forward path admits only
float64 branches with the same generic-domain restrictions and explicit broadcasts.
Its byte ABI stores each Boolean in one byte and aligns FP64 fields. Every live
input and branch is materialized and validated before selection; unselected
invalid branches fail without publishing outputs. Byte, work and node budgets
include the selector and both branches. Native CUDA/portable/scalar/bundle paths
and float32/mixed CPU branches reject before compiler discovery or device execution.
This captured contract does not change
the broader standard eager namespace or promise full IEEE conformance.

`einsum` and `slice` remain bounded GenerativeQC extensions, not standard API
functions. They preserve explicit-output/no-ellipsis contraction notation and
exact finite scientific validation. Use `xp.capabilities()` to distinguish
`eager_functions`, `eager_dtypes` and `eager_device` from `compiled_functions` and
the compiler-owned operation matrix. Symbolic arrays never advertise standardized
discovery, even though concrete arrays do.
