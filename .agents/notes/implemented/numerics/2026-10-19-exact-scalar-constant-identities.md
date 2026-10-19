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

The shared constant predicate, normalization cache and constant interning
keys change. Fraction
payloads compare with the query normalized through the existing source-literal
coefficient rule;
this preserves `is_constant(constant(0.1), 0.1)`. Approximate float payloads
compare directly against the query, so an approximate 0.1 is equal to the
float 0.1 but not to exact Fraction(1, 10). No tolerance is used. Binary and
n-ary arithmetic, derivative rules, rebuilds, artifact hash algorithms and
emitters keep their existing owners and mechanisms.

Normalization uses a type-aware unbounded cache. A float 0.1 and
`Fraction.from_float(0.1)` compare equal in Python, but the former is a source
decimal literal and the latter is an explicitly supplied binary rational.
The previous untyped cache could return the first interpretation for either
input. An automatic review of the first fix identified that predicate queries
also seeded this cache: `is_constant(constant(0), 0.1)` changed a subsequent
binary Fraction constant to decimal 1/10, even in another Graph. This was
independently reproduced and corrected; the same alias was also reachable
through constant construction before this PR. Typed keys preserve each
input's existing interpretation.

A second automatic review identified a separate provenance collision:
constructing `constant(Fraction.from_float(0.1))` before
`approximate_constant(0.1)` reused the Fraction node, while reverse order
reused a float node. The documented predicate therefore still depended on
construction order. This was independently reproduced before changing both
constant interning layers: the local lookup uses `(type, value)` and the
scalar value-number key includes constant payload type. Node's public shape
and the shared value-numbering implementation are unchanged. The added type
is a semantic distinction, not an attempt to preserve erroneous identities;
previously conflated exact/approximate graphs deliberately change identity.

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
- Six fresh-process regression cases independently failed on the first fix
  for predicate/float/rational cache-seeding orders at 0.1 and 0.3. They check
  both the original and a second Graph against direct Fraction constructors;
  no test clears or inspects the production cache.
- Eight provenance cases failed on the second head, covering both creation
  orders at 0.1, 0.5, 1.0 and 0.0 across independent Graphs, cloning and
  associative rebuild. Exact and approximate payload types remain distinct
  even when their numeric values coincide; existing zero/one simplifications
  retain their behavior.
- A bounded one-off audit replayed baseline predicate, normalization and both
  interning methods extracted from the
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
- Local validation: 60 Graph tests, 295 shared XC/Maple/UserExpression/
  derivative-IR/structure tests, 51 weighted-ERI independent finite-difference
  tests, and eight selected one-electron controls passed. The shared
  suite initially had 294 passes and one structure import failure because
  Microsoft Store Python's isolated mode could not import user-installed
  NumPy; the complete 295-test suite passed in an isolated dependency
  environment. Compiler structure checked 568 modules with zero
  dependency errors. Ruff 0.17.0 and CI-pinned ty 0.0.82 (NumPy/Tomli, Linux
  target) passed. Initial unpinned/type-tool environment failures were retained
  separately and do not represent scientific failures.
- The unmodified public-contract checker fails on Windows path separators.
  A diagnostic that changes only in-memory declaration keys to POSIX spelling
  reports zero errors; this does not count as the unchanged checker passing.
  Required Linux CI remains the authoritative public-contract gate. No slice A
  schema, manifest or implementation file was changed to suppress the failure.
- The c9 author-head CI retained one genuine failure:
  `test_catalog_and_human_report_reproduce_offline`. Its 442 independent bulk
  Libxc E/vxc/fxc cases all passed. Offline regeneration changes only six
  unpolarized `graph_nodes` counts by one: LDA_C_ML1, LDA_C_ML2,
  MGGA_C_REVTPSS, MGGA_C_TPSS, MGGA_C_TPSSLOC and MGGA_C_TPSS_GAUSSIAN.
  Their exact/approximate constants now occupy distinct nodes. Cohort totals, source
  inputs, statuses, admission and the generated human report do not change.
  The necessary generated catalog refresh and its single product-output
  checksum are included; no hash algorithm, source input, oracle or gate is
  changed to suppress the failure.
- An additional before/after audit covers those six registrations in both
  spins through order two. All twelve original Libxc fixture comparisons pass
  at unchanged rtol=2e-9/atol=2e-10, and all numerical output bytes are equal
  before/after. Six polarized scalar-emitter sources are byte-identical; each
  unpolarized source swaps only one commutative multiply's operand order
  (`0.5*v` to `v*0.5`, or `2*v` to `v*2`) after interning preserves type.
  This explicitly extends, rather than overstates, the earlier fourteen-case
  unchanged-source sample.
- The #2276/#2609 author confirmed the bounded generated-product seam in PR
  #2617 comment 6266798274. Their held/unmerged changes are not imported.
  Preserve every other parent registry entry. Bind the catalog checksum to
  actual committed LF bytes: Windows CRLF output has a different checksum
  despite identical JSON data. Generation and product registration were
  independently reproduced in an explicit committed-LF snapshot.

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
