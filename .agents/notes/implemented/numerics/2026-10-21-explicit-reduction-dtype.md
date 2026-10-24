# Decision: explicit finite reduction dtype at the generic frontend

Status: implemented
Date: 2026-10-21

## Decision

Refs #2111. Generic finite sum/prod accept explicit float32/float64 targets by
composing the existing astype/cast before the existing reduction. None and
same-dtype requests retain graph identity. Public dtype objects use the existing
adapter normalization; no new type system, IR, math owner or binder is introduced.

Scientific metadata is checked before conversion, including same-dtype requests:
anonymous matrix dimensions alone do not establish a generic dense value when
spin, symmetries or representation annotations remain. The existing scientific
sum with dtype=None and the explicit astype index-preserving contract are intact.

Cross-dtype preflight counts the full target buffer. Sum also conservatively
counts reduction/output buffers; prod includes its existing gathers, canonical
reshapes, odd carries and final reshape. The 256 MiB frontend bound supplements
backend limits. No-op dtype requests do not invent a new conversion allocation.

## Rejected alternatives and qualification boundaries

Reduce-then-cast loses the requested accumulator precision; changing astype or
core promotion would enlarge scientific policy. Existing canonical cast AD is
used for first-order tangents/cotangents, not a classical derivative of IEEE
rounding. Zero-safe product differentiation uses multiplication without division.

Public explicit-dtype sum on CUDA uses the existing scoped capture context to
reject unqualified requests. Prod retains its public CUDA rejection. No backend
marker enters Program identity, and raw primitive admission stays unchanged.
Opt-in CPU remains restricted to FP64 inputs, outputs and arithmetic. A float32
input converted to float64 is still outside that native contract.

## Evidence and revisit conditions

`tests/python/test_array_api_reduction_dtype.py` defines NumPy and pinned 2025.12
array-api-strict value/shape/dtype oracles, cast-order counterexamples, replay,
serialization/optimization, first-order cast conventions, zero factors, finite
guards, metadata, symbolic storage bounds and native FP64 no-fallback checks.
Test definitions are not execution receipts. Native FP32, CUDA, molecular
endpoints and performance remain unqualified; #2111 stays broader than this slice.
Revisit only with independent evidence for a separately authorized backend or
dtype expansion.
