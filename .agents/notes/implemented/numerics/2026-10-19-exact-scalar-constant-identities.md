# Decision: preserve exact scalar coefficient identity before FP64 lowering

Status: implemented
Date: 2026-10-19

## Problem

At baseline `f7e03398a5f1510cbe44c16c74b4e8b8b0798a40`, canonical
`integral.expr.Graph.is_constant` converted both a coefficient payload and its
query to float. For `q = Fraction(2**53 + 1, 2**53)`, creating the constant
before variable `x`, then differentiating `multiply_many((x, q))`, returned
the exact constant one instead of q. Creating x first returned q. Binary
multiply discarded q in either operand order. Fractions immediately below
one had the same failure; `Fraction(1, 10**400)` was mistaken for exact zero.
Large finite rationals could raise OverflowError during an identity check.
These are exact-payload/symbolic errors, before native FP64 rounding.

## Decision and compatibility

Only the shared constant predicate changes. Fraction payloads compare with
the query normalized through the existing source-literal coefficient rule;
this preserves `is_constant(constant(0.1), 0.1)`. Approximate float payloads
compare directly against the query, so an approximate 0.1 is equal to the
float 0.1 but not to exact Fraction(1, 10). No tolerance is used. Binary and
n-ary arithmetic, interning, derivatives, rebuilds, hash algorithms and
emitters keep their existing owners and mechanisms.

The complete call-site audit found add's two zero checks, multiply's zero/one
checks, and the exact-one branch-boundary test in `xc.libxc_bulk`.
Correct zero/one inputs retain their behavior. The bulk branch rewrite must
only recognize an actual one, not a rational merely rounding to one.
Existing zero-annihilator/domain/IEEE policies are retained, not redesigned.
Mixed approximate arithmetic continues using the existing float-fold policy.

## Rejected alternatives

- Reorder two-constant folding: would leave variable products, sums and shared
  predicate users exposed to the same rounding error.
- Patch only UserExpression or introduce another AD/algebra owner: would hide
  the canonical integral/XC defect and leave construction-order dependence.
- Change hashes or round stored Fractions: would erase distinct mathematics.
- Qualify generic AD or native endpoints from this result: a bounded polynomial
  regression cannot establish those scientific contracts.

## Evidence

- Baseline direct reproduction retained both the failing constant-first case
  and passing variable-first control. The initial independent regression run
  produced 17 failures before the production fix (including both sides of
  unity, underflow/overflow, exact predicates and polynomial derivatives).
- Exact Fraction hand derivatives for `q*x*x + (q-1)*x` are checked at three
  rational points: `2*q*x+(q-1)`, `2*q`, then zero. Tests also cover operand
  and allocation orders, exported/rebuilt graphs, repeated differentiation,
  constant folding/cancellation, distinct equal-double rationals, true float
  controls, exact zero/one and the custom Boys leaf rule.
- A bounded one-off audit replayed the baseline predicate extracted from the
  committed source in an isolated process. All 14 legacy fixtures retained
  reachable mathematics and scalar-emitter source bytes: three PSSS axes,
  three one-electron derivative families, and polarized/unpolarized LDA_X,
  LDA_XC_PW, PBE and R2SCAN through second order. XC expression hashes were
  unchanged. This is a sample, not an exhaustive functional census.
- The audit artifacts and baseline RED are retained under ignored
  `.artifacts/exact-rational/`; no permanent source/body fingerprint test is
  introduced. Generator-source provenance naturally changes with expr.py;
  artifact identities that include generator source hashes must invalidate.
  This audit compares scalar emission and mathematical identity, not an
  assertion that all build artifact identities remain unchanged.
- Local validation: 46 Graph tests, 294 shared XC/Maple/UserExpression/
  derivative-IR/structure tests, 51 weighted-ERI independent finite-difference
  tests, and seven selected one-electron derivative tests passed. The remaining
  structure import test initially failed because Microsoft Store Python's
  isolated mode could not import user-installed NumPy; it passed in an isolated
  dependency environment. Compiler structure checked 568 modules with zero
  dependency errors. Ruff 0.17.0 and CI-pinned ty 0.0.82 (NumPy/Tomli, Linux
  target) passed. Initial unpinned/type-tool environment failures were retained
  separately and do not represent scientific failures.
- The unmodified public-contract checker fails on Windows path separators.
  A diagnostic that changes only in-memory declaration keys to POSIX spelling
  reports zero errors; this does not count as the unchanged checker passing.
  Required Linux CI remains the authoritative public-contract gate. No slice A
  schema, manifest or implementation file was changed to suppress the failure.

## Consequences and revisit conditions

Erroneously simplified near-one and underflow graphs deliberately change;
otherwise-correct sampled graphs retain source/mathematical identity.
This is one prerequisite of issue #2330 B, not completion of B or the parent.
The delivered slice A schema-v2 contract, capability fields and historical
research-only generic scalar-AD limitations remain unchanged. Native CPU,
CUDA, SCF, force and complete-endpoint qualification remain UNQUALIFIED for
arbitrary user expressions. No device allocation or native build is needed
to establish this exact symbolic bug.

Revisit if callers require approximate/tolerance predicates, broader mixed
exact/approximate folding policy, or scientific admission of further scalar
primitives. Such work requires a separate contract and independent oracles.

## References

- Issue #2330 (B prerequisite; remaining B/C/D/E/F/G scope remains open).
- `tests/python/test_expr_rational_identity.py`.
- [Current compiler ownership](../../../../docs/developer/compiler_architecture.md).
