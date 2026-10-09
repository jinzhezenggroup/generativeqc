# Decision: bind CPU pullback evidence to the executed common transform

Status: implemented (validation only; retention decisions remain open)
Date: 2026-10-09

## Problem and scope

Issue #1626 asks for evidence for six retained materializations. Weight lookup
parity or a small force finite difference does not establish the allocation
lifetimes and executed work of the common shell pullback. This note records a
bounded CPU harness, not a performance result or issue closure.

The harness compiles the **actual complete**
`src/posthf/mp2_derivative_common.cpp`, with the actual `basis.cpp`,
`cpu_linalg.cpp` (scalar provider), and `mp2_gradient.cpp`. Function-section
linking discards unrelated consumers; no substitute transform, reference type,
rank-two linalg routine, factor-validation helper, or resource plan is supplied.
Production code and generated CUDA code are unchanged.

The common source's SHA-256 is
`3618e9fca87435c4e24fd732f5b3cde751256a783d546fde3891e702d0f4d3fc`.
The pytest harness binds this **entire** source hash and requires exactly one
match for each instrumentation anchor. Source drift must trigger a review of
observation coverage before updating the binding.

## Independent numerical acceptance

`tests/python/test_mp2_derivative_pullback_probe.py` builds an unmodified-source
executable and a separately instrumented copy. The C++ driver is
`tests/native/mp2_derivative_pullback_probe.cpp`.

The independent oracle uses a direct eight-loop, long-double MO-to-AO contraction,
then applies an independently enumerated ERI permutation group on the AO result
for dense local-weight expectations. It does not copy the staged transform.
Factorized weights are expanded by contribution scatters, without calling the
production factorized lookup. Independent direct rank-two contractions verify
coefficient orientation as well.

Every callback weight and final atom-gradient component must agree within
`3e-11 * (1 + abs(expected))`, with nonfinite values explicitly rejected.
Every factorized fixture is also executed using its identical expanded dense
weights: canonical dense and ordered factorized final gradients must agree.
Arbitrary nonsymmetric dense weights remain covered separately. Both executables
must produce bit-identical weight, gradient, work and allocation receipts, apart
from the explicitly added counters.

Synthetic center derivatives come from the pair-symmetric polynomial
`u*v + 0.17*u*u*v*v`, with `u=a_mu+a_nu`, `v=a_kappa+a_lambda` and distinct AO
center derivatives. The oracle contracts all ordered AO quartets before atom
scatter. This tests eightfold symmetry without making center slots identical.
It is an algebraic derivative-consumer fixture, **not** a real ERI/force endpoint.

Eight shell arrangements cover two s shells; one p shell; s/p and reordered p/s;
s/p/s with two shells on the same atom; Cartesian d/s; spherical d/s; and one
spherical f shell. They include repeated shell quartets, unequal shell widths,
nontrivial atom indices, a spectator atom, and occupied/virtual edge populations.
Each runs with nonsymmetric full coefficients and with exact zeros in
coefficients/Fock weights. The 32 parametrized tests execute 48 scientific
configurations through both executables (96 successful invocations).

## Executed work and measured allocation scope

Instrumented counters increment at the actual source accumulation statements:
dense first, factorized Coulomb, exchange and correlation first contributions,
second, third and local. Additional counters count projected dense weight reads
and orbit multiplications. These are **executed accumulation-site iterations**,
not static work estimates, total scalar FLOPs, hardware FMA counts, or endpoint
work. For example, a Coulomb update contains two scalar multiplies; projection
contains eight weighted terms. Rank-two GEMM work, zero-gate checks, callback
arithmetic, and other bookkeeping are outside these counters.

Independent expected counts enumerate admitted canonical shell orbits and their
prefixes; factorized counts account for exact nonzero gates. All three downstream
factorized stages execute N^5 accumulation iterations. Dense first executes N^5
iterations and eight projected reads each. Canonical dense downstream counts and
callback multiplicities are checked exactly; orbit scaling is counted separately.

A global C++ allocation tracker records allocation payloads, frees, total bytes,
call count and high-water bytes. The one-electron callback's gradient is
preallocated before tracing, ownership-transferred from the callback, and reported
as **72 retained output bytes**, separately from observed allocations. Oracle
vectors, inputs and callback scratch are excluded. Callback execution pauses
observation; frees still follow their allocation epoch. This is C++ allocation
payload evidence, not allocator metadata/rounding, RSS, CUDA memory or complete
endpoint peak evidence.

At every callback, the observed live allocation payload is checked against the
two retained AO rank-two outputs, offset metadata and the simultaneous four shell
buffers:

`8 * (di*N^3 + di*dj*N^2 + di*dj*dk*N + di*dj*dk*dl)`.

Dense and factorized first are mutually exclusive branches of the **same**
allocation. All tracked allocations must be freed by invocation return. The
actual conventional MP2 plan must cover the measured numeric peak plus the
separately retained gradient, with offset metadata excluded consistently from
that numeric-buffer comparison. This check does not claim a complete MP2/CC
endpoint measurement or a CC plan execution.

Representative observed receipts (the reproduction command regenerates them):

| Fixture | Mode | Callback calls | Second / third / local iterations | Shell peak bytes | Common peak bytes |
| --- | --- | ---: | --- | ---: | ---: |
| two s, N=2, full | dense | 6 | 24 / 20 / 12 | 120 | 208 |
| two s, N=2, full | factorized | 16 | 32 / 32 / 32 | 120 | 208 |
| s/p/s, N=5, zero gates | dense | 21 | 2250 / 1850 / 1065 | 6528 | 6960 |
| s/p/s, N=5, zero gates | factorized | 81 | 3125 / 3125 / 3125 | 6528 | 6960 |
| Cartesian d/s, N=7, full | dense | 6 | 14749 / 12985 / 11137 | 53040 | 53848 |
| Cartesian d/s, N=7, full | factorized | 16 | 16807 / 16807 / 16807 | 53040 | 53848 |
| spherical d/s, N=6, zero gates | dense | 6 | 6696 / 5796 / 4836 | 26840 | 27440 |
| spherical d/s, N=6, zero gates | factorized | 16 | 7776 / 7776 / 7776 | 26840 | 27440 |

The common peak includes two AO rank-two outputs and offsets, but excludes the
72-byte retained gradient. It is not a complete endpoint peak. For N=5 zero-gate
factorized input the first-stage counters are 116 Coulomb, 116 exchange and 126
correlation updates, rather than the full-input formula.

## Reproduction and remaining decisions

Run with a verified compiler cache on PATH (or `CCACHE` set):

```sh
ccache --version
ccache --show-stats
python -m pytest -q tests/python/test_mp2_derivative_pullback_probe.py
ccache --show-stats
ruff check tests/python/test_mp2_derivative_pullback_probe.py
ruff format --check tests/python/test_mp2_derivative_pullback_probe.py
```

The required cached-native fixture fails rather than silently skipping absent
compiler/cache prerequisites. Runtime progress tracing is disabled explicitly;
OpenBLAS is disabled in these probe builds. On the local qualification run,
ccache 4.14.1 was verified and the existing cache was reused. The final focused
run passed all 32 tests; Ruff checks passed. These are focused harness results,
not a full repository test or full-native integration pass.

All six #1626 retention decisions remain **pending evidence**:

1. Dense first: algebra/lifetime/work covered here; profitability still open
2. Factorized first: algebra/lifetime/zero gates/work covered; profitability open
3. Second: stage allocation and executed work covered; retention choice open
4. Third: stage allocation and executed work covered; retention choice open
5. Local: cotangent/consumer/scatter covered; retention choice open
6. Dense initial orbital-RHS oracle: not executed or changed by this harness

Actual callers remain factorized conventional MP2 and dense conventional RCCSD
and RCCSD(T). RI-MP2 bypasses this common shell pullback. No backend-wide speed,
profitability, full method qualification, CUDA qualification, or closure claim
follows from these bounded synthetic-input CPU tests. Revisit retention only with
complete endpoint timing/work and a justified alternative at the actual caller.
