# Experimental symbolic Array API

GenerativeQC exposes a bounded Array-API-shaped frontend at
`generativeqc.experimental.array_api`. The ordinary user path is deliberately
shape/dtype based: users write array expressions and do not need to construct
TensorIR `IndexSpace`, `Index`, or `TensorSpec` objects.

This is an **experimental public preview**, not a Python Array API conformance
claim. The surface may change between releases and symbolic arrays deliberately
do not implement `__array_namespace__`.

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

The namespace also works eagerly on its NumPy-backed reference arrays. The same
function body can therefore be evaluated directly and then compiled without
rewriting its array expressions. The first compiled call captures a
specialization from the concrete argument shapes and dtypes, lowers the
expression to canonical TensorIR, caches that program, and executes it through
the independent reference interpreter. Use
`observable.lower(C, occupation, O)` when the TensorIR `Program` itself is
needed for inspection, optimization, or a separate native compilation step.

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

The preview also supports finite, uniform array creation through `zeros`, `ones`,
`full`, and `zeros_like`, `ones_like`, `full_like`. Inside `@xp.compile`,
these constructors lower **statically known** shapes to a single exact TensorIR
scalar constant and broadcast (no size-proportional literal payload):

```python
@xp.compile
def shifted(x):
    return x + xp.ones_like(x) * xp.full(x.shape, 0.25, dtype=x.dtype)
```

Outside capture, creation executes eagerly on NumPy/CPU. Only the existing
`float32` and `float64` dtypes and `device=None` are admitted: an explicit
device request never triggers a hidden host transfer. `full(shape, integer)`
without an explicit floating dtype is rejected rather than silently claiming
the standard's unsupported default integer dtype. Scientific AO/occupied/etc.
arrays cannot be passed to `*_like` without their explicit TensorIR metadata.

The current preview supports ordinary shape broadcasting for generic arrays,
`@`, `.T`, `.mT`, `matrix_transpose`, reshape with one inferred `-1`
dimension, and static indexing with integers, slices (including negative
strides), `None`/newaxis, and ellipsis. Common elementwise conveniences
`square` and `reciprocal` lower to existing TensorIR multiply/divide nodes.
`sum` and `mean` accept static axes, and generic arrays may use
`keepdims=True` to retain reduced singleton axes. Generic arrays accept finite Python
float literals as ordinary scalar values, so expressions such as `x + 0.5`
have eager/compiled parity. The compiler records the exact binary value of that
Python float. Negative-zero float literals are rejected because exact rational
constants cannot preserve their sign. Explicitly scientific arrays retain the
stricter exact-scalar spelling rules.

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

## Current limits

The eager namespace and reference compiled-call path currently accept CPU/NumPy
`float32` and `float64` arrays. Static creation is capture-aware but does not
supply the standard's full dtype defaults, devices or constructor set. There is no implicit dtype promotion, dynamic Python control
flow, or implicit external-device transfer. `xp.asarray` refuses to silently
copy a foreign DLPack array to the host; use `import_dlpack` for the explicit
same-device handoff. Every eager namespace operation checks this host boundary
before NumPy dispatch, including nested host containers and foreign DLPack/CUDA
array protocols. Exact scalar literals (`int`, `Fraction`, or rational
strings) and finite Python floats are converted to the array operand dtype;
array operands of different dtypes are rejected.

Eager functions retain the bounded compiled contract: `sum` does not yet
accept a `dtype` conversion; `mean` rejects reductions over zero elements
rather than producing a nonfinite value; `take` requires a static tuple of
nonnegative in-bounds indices; `slice` requires one nonnegative in-bounds
half-open range per axis. Normal `x[...]` indexing remains a separate path
supporting negative indices and strides. For scientifically annotated arrays,
reduction `keepdims=True` remains unsupported without explicit TensorIR index
metadata, rather than silently recreating a discarded AO/occupied/etc. domain.
Real-valued operations require finite inputs and results, strictly positive
inputs for `log`/`pow`, and nonnegative inputs for `sqrt`. `square` does
not inherit the positive-base restriction of `pow`.

General `einsum` remains a GenerativeQC extension for high-rank scientific
contractions, using explicit-output equations without ellipses or implicit
singleton-label broadcasting. The eager path validates this subset through the
canonical frontend. The goal is that common expressions use normal array syntax, while
specialized quantum-chemistry algebra can still use `einsum` when it is the
clearest representation.

Use `xp.capabilities()` for the exact current subset. It continues to report
`array_api_version=None` and `array_namespace_protocol=False` until a declared
standard version has a dedicated conformance matrix.
