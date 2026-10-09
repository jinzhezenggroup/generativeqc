# Array API frontend for TensorIR

Issue #633 introduced a bounded symbolic array frontend owned by the compiler.
Normal array expressions are captured once and lowered to the existing TensorIR;
TensorIR remains the scientific IR and keeps the stronger quantum-chemistry type
system. The compiler owner remains `generativeqc_compiler.array_api`, while
advanced users access the curated experimental facade at
`generativeqc.experimental.array_api`.

The frontend does **not** currently claim Array API conformance. The public
facade is an Array-API-shaped experimental preview and fails closed outside the
declared subset. In particular, `VibeArray` deliberately does not implement
`__array_namespace__` yet: the Array API standard uses that method as a
compliance discovery signal and requires the returned namespace to provide the
standard's top-level API. Advertising the protocol for this bounded subset
would therefore misidentify a preview object as conforming.

## Layering

```text
generativeqc.experimental.array_api
        |
        v
compiler-owned VibeArray / namespace
        |
        v
existing TensorIR nodes
        |
   +----+----+
   |         |
   AD      CPU/CUDA lowering
```

MethodIR, IntegralIR, ProgramIR, stationary/implicit solves and integral
providers retain their existing ownership. The frontend does not turn ERI,
J/K, XC, SCF iteration or eigensolvers into generic array primitives.

## Preserved scientific semantics

A `VibeArray` wraps an ordinary TensorIR node. Therefore capture preserves:

- AO / occupied / virtual / auxiliary / batch / spin index-space identity;
- selected index ranges and ordered gathers;
- orbital representation and declared symmetry;
- exact rational compile-time coefficients;
- input/parameter role and differentiability;
- TensorIR logical identity, serialization, optimization and JVP/VJP behavior.

The public facade has two deliberately different scientific modes and two
execution modes. Generic arrays use ordinary shape semantics, including
broadcasting, whether evaluated eagerly through the NumPy reference namespace or
captured by `compile`. Scientifically annotated arrays keep TensorIR domain identity:
equal numerical shapes do not make AO/occupied/virtual/auxiliary domains
compatible.

## Initial capability subset

| Surface | Initial contract |
| --- | --- |
| `+ - * /` | Generic arrays use shape broadcasting; scientific arrays require compatible TensorIR domains |
| unary `-` | Exact coefficient lowering |
| `pow/exp/log/sqrt` | Existing TensorIR real-valued contracts |
| `sum` | Explicit reduction, `keepdims=False`, no implicit dtype conversion |
| `permute_dims`, `.T`, `.mT` | Array-style axis and matrix transpose, including 2025.12 signed axis positions |
| `reshape` | Shape-only for generic arrays, including one `-1`; scientific arrays require explicit target metadata |
| `expand_dims/squeeze` | Static singleton axis insertion/removal (including signed and multiple axes); scientific dimensions need explicit typed TensorIR metadata |
| `moveaxis/flip` | Signed-axis reorder through transpose; generic flip through static gather up to 65,536 indices per reversed axis |
| `broadcast_shapes/broadcast_arrays` | Static integer-shape calculus / individual explicit generic broadcasts; each input retains its dtype |
| `broadcast_to` | Shape-only for generic arrays; scientific arrays require explicit indices/axis map |
| indexing | Generic integer/slice/newaxis/ellipsis; scientific mode retains strict rank-preserving slices |
| `take` | Static integer gather along one axis |
| `matmul`, `@` | Vector/matrix/batched generic arrays; scientific annotated path remains strict |
| `asarray` | CPU/NumPy float32/float64 eager arrays; no silent external-device transfer |
| `astype` | Explicit float32/float64 conversion through an existing TensorIR `cast` (or NumPy eager); native physical copy/alias guarantees are not independently claimed |
| `can_cast/result_type/isdtype/finfo` | Limited real-floating subset; no integer/bool/complex families or their dtype rules |
| `zeros/ones/full` | Static shapes, finite float32/float64 values; symbolic creation uses one exact scalar + TensorIR broadcast; integer/bool default `full` dtypes remain unsupported |
| `zeros_like/ones_like/full_like` | Generic arrays only, preserve shape/dtype by default; scientifically annotated axes require an explicit typed TensorIR construction |
| `compile` | Shape/dtype-specialized public TensorIR capture with reference execution; differentiable inputs are explicit |
| `einsum` | GenerativeQC extension lowered to existing TensorIR einsum |
| dtype promotion | Generic float32/float64 operands promote to float64, using explicit TensorIR casts before ordinary binary arithmetic, matmul and einsum; scientific typed arrays remain strict |
| dynamic Python control flow | Not supported |

Scientifically annotated arrays keep the exact scalar spelling contract:
`int`, `Fraction`, or a rational string. Generic public arrays additionally
accept finite Python float literals; capture converts each literal to the exact
binary rational represented by that Python float so eager/compiled array syntax
does not require special coefficient spelling. Negative-zero float literals
fail closed because the exact rational constant representation cannot preserve
their sign.

The eager namespace admits operands through the same CPU/NumPy float32/float64
boundary as `asarray`, before calling NumPy or a foreign array hook. This guard
also checks nested containers and advertised DLPack/CUDA array protocols. Exact scalar
literals are materialized in the array operand dtype. For generic arrays,
float32/float64 mixed operands promote via an explicit TensorIR cast, while
scientific annotated TensorIR domains continue rejecting implicit precision
changes. A pure `astype` may deliberately insert a precision conversion in
either mode. The existing backend/reference interpreter is still responsible
for execution and floating accuracy checks. Eager reductions and static selection retain the same bounded
controls as capture: no reduction dtype/keepdims extension, nonnegative
in-bounds `take` indices, and one valid half-open `slice` range per axis.
Real-valued eager execution rejects nonfinite inputs/results and enforces the
TensorIR `log`/`pow` positive-input and `sqrt` nonnegative-input domains. Eager
`einsum` validates its equation and label extents through the canonical frontend,
retaining explicit outputs, no ellipses, and no implicit label broadcasting.

Generic public inputs are assigned compiler-owned anonymous array dimensions
whose identity is intentionally shape-based. This lets ordinary broadcasting,
reshape, indexing, transpose and matmul lower to explicit TensorIR view/
contraction nodes. Scientific inputs are different: integer extents alone never
define AO/occupied/virtual/auxiliary meaning. Their reshape/broadcast operations
continue to require explicit metadata, and TensorIR still rejects equal-sized
but scientifically distinct populations.

Internal compiler consumers continue to import
`generativeqc_compiler.array_api.namespace` explicitly. The public facade
re-exports the same canonical operations rather than constructing a second IR
or mathematical identity. `__array_namespace__` and a versioned Array API
declaration will only be added after a dedicated conformance matrix proves that
the advertised namespace meets the corresponding standard version.

## Public experimental example

```python
from generativeqc.experimental import array_api as xp

@xp.compile
def observable(C, occupation, O):
    density = (C * occupation) @ C.T
    return xp.sum(density * O)

value = observable(C, occupation, O)
program = observable.lower(C, occupation, O)
```

No TensorIR type declarations are needed on this ordinary path. The same
namespace functions execute eagerly when given NumPy-backed arrays and lower to
symbolic TensorIR when `compile` supplies `VibeArray` inputs. The concrete
shape/dtype signature constructs a cached generic TensorIR specialization.
Creation inside the trace uses context-local capture admission (reset even on a
failed trace); creation outside a trace stays eager. Uniform symbolic arrays are
broadcasts of one exact scalar, not size-proportional constant payloads.
`lower` exposes the ordinary compiler-owned `Program`; there is still no
frontend-only runtime node or second mathematical IR. Inferred inputs are
non-differentiable by default; `@xp.compile(differentiable=("x", ...))` promotes
only the named inputs to differentiable TensorIR parameters for JVP/VJP use. The
first compiled-call backend is the independent NumPy TensorIR reference
interpreter. Native CPU/CUDA execution remains a separate explicit
lowering/qualification step.

## Native SCF adoption

The CPU SCF density and energy-weighted-density production helpers are generated
at build time from the validated `array_api.scf -> TensorIR` topology. The
dynamic native lowering preserves the existing AO/orbital storage contract and
the legacy FP64 product/accumulation order, including the exact density witness
used by occupied density-fitting exchange. Python tracing is therefore absent
from the SCF iteration hot path.

This cutover covers CPU density construction and its compact occupied-factor
consumer. The same generated header also consumes the canonical TensorIR
`diis_gram_program` and `diis_extrapolation_program` equations for CPU
production DIIS. Only the pure residual Gram and Fock-history contraction move
to generated scientific ownership: chronological history mutation, optional
metric normalization, the augmented pivoted solve, dependent-history retirement,
and fallback semantics remain native solver policy. The specialization preserves
the historical FP64 reduction order.

Resident CUDA SCF density and DIIS kernels retain their existing device ownership
for now; moving those kernels requires separate stream/layout and performance
qualification. Exposing the frontend as an experimental public facade does not
change production execution ownership or add an Array API conformance claim.

## Ownership

`generativeqc_compiler.array_api` remains the compiler owner above
`generativeqc_compiler.tensor`. Its dependency direction is deliberately one-way:
the frontend may import TensorIR, while TensorIR cannot import the frontend.
`generativeqc.experimental.array_api` is only a curated public facade over that
owner. This keeps TensorIR usable by hand-built/generated equations, avoids a
second array implementation, and keeps array syntax out of mathematical IR
identity.

The design rationale, rejected alternatives and invariants are retained in the
[Array API frontend architecture note](../../.agents/notes/implemented/architecture/2026-09-20-array-api-tensorir-frontend.md).
The public-preview boundary is recorded in the
[experimental Array API note](../../.agents/notes/implemented/architecture/2026-10-07-experimental-array-api-public-preview.md).

The native cutover is a checked specialization, not a general C++ graph emitter.
Its topology, FP64 order and source-identity contract are recorded in the
[native specialization note](../../.agents/notes/implemented/architecture/2026-09-20-native-array-scf-specialization.md).
