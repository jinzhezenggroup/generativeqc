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
`sqrt` and exact `pow`. Other operations fail closed; full eager conformance does
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

The current compiled frontend supports ordinary shape broadcasting for generic arrays,
`@`, `.T`, `.mT`, `matrix_transpose`, reshape with one inferred `-1`
dimension, and static indexing with integers, slices (including negative
strides), `None`/newaxis, and ellipsis. Common elementwise conveniences
`square` and `reciprocal` lower to existing TensorIR multiply/divide nodes.
`sum` and `mean` accept static axes, and generic arrays may use
`keepdims=True` to retain reduced singleton axes. Shape utilities also include
`broadcast_shapes`, `broadcast_arrays`, `expand_dims`, `squeeze`,
`moveaxis` and `flip`. `expand_dims` accepts the 2025.12 multi-axis tuple
form, and `permute_dims` accepts negative axes; all reuse existing TensorIR
reshape, transpose, broadcast and gather semantics. Six elementwise comparisons
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

## Compose complete RHF molecular energies

`RHF` is a GenerativeQC extension, not a standard Array API function. It binds
an immutable all-electron, closed-shell molecular method to the existing native
integral/J/K/SCF/analytic-force endpoint. It can be used inside ordinary captured
Array API expressions:

```python
rhf = xp.RHF(("O", "H", "H"), basis="sto-3g", backend="cpu")

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

Configuration requires a nonempty tuple of at most 256 nuclei, a positive even
all-electron population, a named admitted all-electron basis, float64 coordinates,
finite positive tolerances and bounded iteration counts. Scientifically annotated
coordinates must use complete, spin-independent `atom` and `cartesian` index
spaces; matching AO/orbital extents do not authorize a nuclear-coordinate binding.
These domains survive source snapshots and AD. ECP, density fitting,
unrestricted spin and nested molecular dependencies in geometry are not supported
by this source. Capture admits at most 16 distinct molecular sources. Native
nonconvergence, basis/derivative/resource errors propagate without publication
or fallback. `resource_budget` is the existing `ResourceBudget` for native method
preflight; the compiler's `max_bytes`/`max_work`/`max_nodes` bound **tensor algebra**,
not the independent SCF/J/K endpoint's work or resident memory.

`backend="cuda"` must be selected on both `RHF` and `compile`/AD, with the same
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
Shape/axis values must be static. Captured `flip` retains
its 65,536-index budget per reversed axis. Scientific domains retain explicit
metadata requirements. Compiled reductions do not introduce implicit dtype
conversions, compiled empty means remain unsupported, and reference execution
requires finite inputs/results with positive `log`/`pow` and nonnegative `sqrt`
domains. Exact rational constants cannot represent negative zero. Integer/complex
compilation, dynamic indexing/shapes and arbitrary Python control flow are not
implied by eager support. Unsupported capture functions, including linalg/fft,
raise instead of evaluating a NumPy fallback during tracing.

`einsum` and `slice` remain bounded GenerativeQC extensions, not standard API
functions. They preserve explicit-output/no-ellipsis contraction notation and
exact finite scientific validation. Use `xp.capabilities()` to distinguish
`eager_functions`, `eager_dtypes` and `eager_device` from `compiled_functions` and
the compiler-owned operation matrix. Symbolic arrays never advertise standardized
discovery, even though concrete arrays do.
