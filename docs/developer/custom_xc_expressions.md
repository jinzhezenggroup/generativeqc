---
orphan: true
---

# Custom XC scalar expression contract

`generativeqc.extensions.xc.expression(source, *, name="user-expression")`
captures a restricted string declaration once as an immutable `UserExpression`.
This is schema `generativeqc.xc-expression.v1`, a prerequisite for issue #2330.
It accepts neither Python functions nor Array API callbacks, decorators, foreign
`Expr` objects, or caller-owned graphs. The parser does not execute user code and
is not a general security sandbox for arbitrary Python.

```python
from generativeqc.extensions import xc

formula = xc.expression(
    "(rho_a**2 + rho_b**2)/3 - 0.25*rho_a*rho_b",
    name="illustrative polynomial",
)
energy_density = formula.evaluate(rho_a=0.2, rho_b=0.03)
report = xc.inspect(formula)
restored = xc.UserExpression.loads(formula.dumps())
assert restored.identity == formula.identity
```

This polynomial demonstrates the contract; it is not a physically calibrated XC
functional. `evaluate` is a diagnostic CPU point evaluation through the existing
compiler scalar Graph, with no native library or device involved.

(xc-expression-values)=
## Ingredients, units and domain

The only admitted inputs are collinear spin densities `rho_a` and `rho_b`, each
in bohr^-3, both finite and nonnegative. They are independent spin channels;
unpolarized inputs must explicitly supply half the total density in each channel.
Zero total density and zero minority spin are included. There is no clipping,
tail regularization, unit conversion, or implicit spin averaging.

The vocabulary reserved for later slices is `sigma_aa = grad(rho_a).grad(rho_a)`,
`sigma_ab = grad(rho_a).grad(rho_b)`, `sigma_bb = grad(rho_b).grad(rho_b)`
(bohr^-8), and `tau_a`, `tau_b = (1/2) sum_i occupation_i |grad(orbital_i)|^2`
per spin (bohr^-5). All five are rejected by v1. There is no GGA or meta-GGA
admission. Total `rho`, vector fields, arrays and undeclared inputs are rejected.

Output is energy **per volume**, hartree/bohr^3, not energy per particle.
Coefficients carry the corresponding atomic units; dimensional analysis is the
declarant's responsibility. V1 has no per-particle constructor or conversion at
zero density. It does not define integrated energy, XC potential, or forces.

Point inputs are Python `float` (binary64) or integers converted to binary64;
Boolean and array inputs are rejected. Overflow and nonfinite results fail.
Underflow follows binary64 arithmetic. Polynomial intermediate rounding and
canonical reassociation follow the canonical Graph, without a numerical accuracy
promise across all finite densities. Individual formulas need independent
numerical and scientific qualification; a successful point call grants none.

(xc-expression-errors)=
## Language and bounds

Allowed syntax is numeric integer/float literals, `rho_a`, `rho_b`, parentheses,
unary `+`/`-`, addition, subtraction, multiplication, division by a nonzero
constant expression, and powers with a constant integer exponent in 0..4.
The conservative declared total polynomial degree must be at most 8, including
subexpressions later eliminated by simplification. Division by densities,
fractional/negative powers, functions, comparisons, Boolean operations, conditionals,
attributes, indexing, assignment, comprehensions and loops are rejected.
Thus common physical formulas with fractional density powers are unsupported.
No global names or mutable captured state are available, and nothing is called
inside SCF.

Limits are 4096 source characters, 256 AST nodes (including operator/context
nodes), depth 32 with the expression root at depth zero, 512 pre-canonical Graph
nodes, and 256 bits for each exact constant numerator and denominator. These
limits are fixed by the schema. Replay envelopes are limited to 65536 characters.
Generated constants are checked before parent operations can erase or convert
them, including input-dependent syntax that folds to constants.
Nonfinite literals are rejected even in eliminated branches such as `0*1e309`.
Invalid syntax, ingredients, domains, budgets and mismatched replay envelopes
raise `ValueError`; non-string declarations and unsupported point input types
raise `TypeError`. Invalid JSON raises `ValueError` through `JSONDecodeError`.
Integer and rational coefficients remain exact. Float literals first follow
Python binary64 literal parsing, then the scalar Graph's exact shortest-decimal
coefficient convention; this is not arbitrary-precision decimal source parsing.

(xc-expression-ownership)=
## Identity and audit

The compiler-owned adapter reuses `integral.expr.Graph` operations, associative
canonicalization, evaluation, and symbolic differentiation. It introduces no
alternate scientific algebra or derivative engine. `scalar_graph()` returns a
fresh detached graph for compiler research; mutating it cannot change the sealed
formula. It does not confer permission to use that graph in a native method.
Exact coefficient capture does not qualify arbitrary exported Graph derivatives:
existing research AD simplification limits remain outside this contract.

The semantic identity hashes the versioned contract and reachable canonical
nodes with remapped indices, following existing XC graph identity conventions.
It includes spin, units, energy convention and finite-domain policy. It excludes
the descriptive name and original source spelling, paths, timestamps and bytecode.
Associatively reordered equal declarations may share identity; v1 is not a
complete polynomial-equivalence solver. Name `PBE` confers no built-in identity
or qualification. Mathematical changes surviving canonicalization change identity.

`to_payload()` and `xc.inspect()` return detached audit copies. Serialization
retains original source, name, canonical payload and semantic identity; `loads`
rebuilds and verifies their relation rather than trusting supplied graph nodes.
Copying and process replay preserve identity. Audit source provenance is separate
from semantic identity, so equal mathematics may retain different declarations.

(xc-expression-backends)=
## Capability boundary

| Stage | CPU | CUDA |
| --- | --- | --- |
| Representation | Bounded scalar contract | Same target-neutral contract |
| Pointwise validation of a user formula | UNQUALIFIED | UNQUALIFIED |
| Native compilation | UNQUALIFIED | UNQUALIFIED |
| Self-consistent SCF | UNQUALIFIED | UNQUALIFIED |
| Analytic nuclear forces | UNQUALIFIED | UNQUALIFIED |

`xc.inspect(formula)["capability"]` exposes these separate stages.
`UserExpression` is deliberately distinct from `FunctionalSpec`;
`method.compose(..., xc=formula)` rejects it. Native Calculator admission is
unchanged. There is no Python SCF callback or host/device/reference fallback.
Issue #2330's derivative pipeline integration, native LDA SCF, GGA/meta-GGA,
forces, CUDA and end-to-end usability remain separate work. This contract does
not complete #2330, #558 or #1118.
