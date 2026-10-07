# Experimental symbolic Array API

GenerativeQC exposes a bounded symbolic array frontend at
`generativeqc.experimental.array_api`. It is intended for advanced users who
want to write tensor expressions with ordinary array-like syntax and lower them
to the same TensorIR used by built-in compiler paths.

This is an **experimental public preview**, not a claim of Python Array API
standard conformance. The surface may change between releases, unsupported
semantics fail closed, and symbolic arrays deliberately do not implement
`__array_namespace__`.

## Basic use

```python
from fractions import Fraction

from generativeqc.experimental import array_api as xp
from generativeqc.extensions import tensor

ao = xp.IndexSpace("ao", "ao", 3)
p = xp.Index("p", ao)
spec = xp.TensorSpec((p,), role="input")

program = xp.trace(
    lambda x: {"norm2": xp.sum(Fraction(1, 2) * x * x)},
    {"x": spec},
)

result = tensor.execute(program, {"x": [1.0, 2.0, 3.0]})
print(result.outputs["norm2"])
```

`trace` returns the canonical TensorIR `Program`; no frontend-only runtime node
survives capture. The resulting program can therefore be inspected,
differentiated, interpreted, or explicitly compiled through
`generativeqc.extensions.tensor` under that API's existing capability rules.

## Supported preview surface

The current symbolic operations include elementwise `+`, `-`, `*`, `/`, unary
negation, `pow`, `exp`, `log`, `sqrt`, reductions with `sum`,
`permute_dims`, explicit `reshape` and `broadcast_to`, static `slice` and
`take`, rank-2 `matmul`, and the GenerativeQC `einsum` extension.

Scientific domains remain explicit. Equal integer extents do not make AO,
occupied, virtual, auxiliary, spin, or batch axes interchangeable. Shape-changing
operations therefore require explicit TensorIR `Index` metadata whenever the
new shape cannot preserve those semantics automatically.

Exact scalar coefficients accept integers, `Fraction`, or rational strings.
Floating-point spellings such as `0.5` are rejected where they would weaken
canonical scientific identity.

Use `xp.capabilities()` to inspect the exact supported subset at runtime. The
report identifies the surface as experimental and reports
`array_api_version=None` and `array_namespace_protocol=False` until a dedicated
versioned conformance matrix justifies advertising the standard protocol.

## DLPack interoperability

The preview also exposes the qualified DLPack handoff helpers
`dlpack_device` and `import_dlpack`. They require a verified same-device
zero-copy handoff and fail rather than silently moving data across devices.
This interop boundary does not imply external-framework autograd through SCF,
CC, or other iterative quantum-chemistry solvers.
