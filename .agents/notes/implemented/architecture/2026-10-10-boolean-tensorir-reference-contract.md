# Decision: typed Boolean TensorIR data before native lowering

Status: implemented
Date: 2026-10-10

## Problem

The experimental Array API frontend could compare real values only by leaving
TensorIR. A floating 0/1 mask would give truth values the wrong dtype and allow
them into scientific arithmetic or differentiation. The existing `int64` values
are runtime index controls, with a separate role and storage contract.

## Decision

`TensorSpec(dtype="bool")` is a one-byte logical data value with input, constant,
and intermediate roles. It is general, symmetry-free, and non-differentiable.
Boolean constants contain JSON Boolean literals, distinct from rational real
constants. The six comparisons accept same-domain, same-representation real
operands and produce Boolean intermediates. Generic frontend arrays may first
broadcast and promote float32/float64 with explicit casts; scientific domains
must match exactly. Boolean transpose, reshape, slice, gather, and broadcast
remain typed views. Numeric arithmetic, reductions, einsum, casts, and AD on
Boolean graphs reject explicitly.

The NumPy TensorIR interpreter is the bounded reference executor. It checks
input dtype/shape, finite real inputs, output dtype/shape, detached results, and
retained bytes using `itemsize=1` for Boolean values. Production preparation
rejects every live Boolean or comparison node before CPU/CUDA/portable/scalar
lowering, allocation, transfer, or JIT. Native buffer layout and ABI have not
been qualified by this reference contract.

## Rejected alternatives

- Floating masks or `int64` index maps as Boolean arrays: both lose type and
  permit incorrect arithmetic or storage assumptions.
- Silent native lowering through existing float scalar code: its storage,
  emission, and device behavior are not qualified for Boolean values.
- Claiming full Array API 2025.12 comparison conformance: the current reference
  executor rejects NaN/Inf at the input boundary, while the standard defines
  explicit nonfinite comparison results.

## Invariants

- Boolean literals, input roles, and comparison operators participate in
  deterministic serialization, logical hashes, and exact CSE.
- Scientific equal extents never establish equal index domains.
- Boolean values never supply tangents, cotangents, or numeric coefficients.
- The frontend does not expose `__array_namespace__` or claim full conformance.

## Evidence and remaining scope

`tests/python/test_array_api_boolean_comparisons.py` checks all six operators
against NumPy across scalar, empty, broadcast, mixed dtype, and signed-zero
cases, plus an `array-api-strict` finite reference. It also checks serialization,
hashes, bounded bytes, optimizer identity, AD/backend rejection, and scientific
domain guards. The larger #2137 still owns `where`, real-branch AD, native
CPU/CUDA qualification, nonfinite comparison support, and conformance testing.

## Revisit when

One backend has a reviewed Boolean ABI/storage layout, explicit comparison
lowering, and source-matched execution evidence; or the reference executor has
a separately accepted IEEE nonfinite policy.

## References

- https://github.com/jinzhezenggroup/generativeqc/issues/2137
- https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.equal.html
- https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.where.html
