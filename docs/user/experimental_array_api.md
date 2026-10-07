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

The first call captures a specialization from the concrete argument shapes and
dtypes, lowers the expression to the canonical TensorIR, caches that program,
and executes it through the independent reference interpreter. Use
`observable.lower(C, occupation, O)` when the TensorIR `Program` itself is
needed for inspection, AD, optimization, or a separate native compilation step.

The current preview supports ordinary shape broadcasting for generic arrays,
`@`, `.T`, `.mT`, `matrix_transpose`, reshape with one inferred `-1`
dimension, and static indexing with integers, slices (including negative
strides), `None`/newaxis, and ellipsis. Elementwise exact scalar expressions
such as `x + 1` and `x - Fraction(1, 2)` are also supported.

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

The reference compiled-call path currently accepts CPU/NumPy `float32` and
`float64` arrays. There is no implicit dtype promotion, dynamic Python control
flow, or implicit external-device transfer. `xp.asarray` refuses to silently
copy a foreign DLPack array to the host; use `import_dlpack` for the explicit
same-device handoff.

General `einsum` remains a GenerativeQC extension for high-rank scientific
contractions. The goal is that common expressions use normal array syntax, while
specialized quantum-chemistry algebra can still use `einsum` when it is the
clearest representation.

Use `xp.capabilities()` for the exact current subset. It continues to report
`array_api_version=None` and `array_namespace_protocol=False` until a declared
standard version has a dedicated conformance matrix.
