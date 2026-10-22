# Decision: explicit empty contraction identities in the reference evaluator

Status: implemented
Date: 2026-10-19

## Problem

An empty vecdot regression intermittently failed in the NumPy 2.1.3 CI lane.
The same behavior reproduces deterministically with NumPy 2.3.5: one empty
operand has zero strides, another has a contiguous reduction stride, and the
zero-stride operand has NaN in backing storage outside its logical contents.
The direct einsum kernel reads the backing value despite an empty reduction,
then multiplies it by zero. Both logical arrays contain no nonfinite elements.

## Decision

For an einsum with any zero-size operand, the reference interpreter returns the
correctly shaped and typed zero result before applying its existing coefficient.
Reference JVP and VJP use the same empty contraction identity. Live input,
tangent and cotangent validation, coefficient representability checks and
logical resource budgets remain in their existing owners. The ordinary
nonempty einsum path and explicit mixed-precision reference path are unchanged.

## Rejected alternatives

- Retrying CI or relaxing finite checks hides a numerical correctness defect
- Changing the pinned NumPy lane is insufficient: multiple versions reproduce
- Repairing forward execution alone leaves the reference JVP rule exposed
- Eager namespace, native lowering and runtime engines do not need a new policy

## Evidence and invariants

`tests/python/test_tensor_empty_contractions.py` uses inaccessible NaN backing
storage to make the semantic failure deterministic without asserting that NumPy
must remain broken. It covers public and reference forward/JVP/VJP, repeated
labels, zero output dimensions, mixed empty/nonempty operands, coefficient sign
and representability, dtype/shape preservation, invalid live data and budgets.
All regression expectations follow empty-sum identities rather than NumPy's
affected kernel or internal call ordering.

## References

- GenerativeQC PR #2618 queue run 39251884887, job 118057436719
- NumPy v2.1.3 `numpy/_core/src/multiarray/einsum_sumprod.c.src`
- NumPy v2.1.3 `numpy/_core/src/multiarray/einsum.c.src`
