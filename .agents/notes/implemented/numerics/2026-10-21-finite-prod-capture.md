# Decision: bounded finite product capture

Status: implemented
Date: 2026-10-21

## Decision

Refs #2111. Preserve the existing eager prod and compose generic finite f32/f64
capture from canonical multiply, gather and reshape. Reduce axes from last to
first. Each round multiplies adjacent pairs in the even prefix and saves an
odd last element separately. Multiply saved tails together and combine with
the final pair result. Graph depth grows with the sum of logarithmic axis
extents. No new IR, interpreter, AD engine, emitter or runtime ABI is needed.

Validate metadata, dtype, axes and keepdims before identities or empty outputs.
A genuinely empty reduction returns scalar one with the required broadcast;
an empty retained dimension returns an empty reshape. Same-dtype explicit
requests are admitted, cross-dtype conversion remains unqualified.

## Bounds

Preflight at most 65,536 gather positions, 64 pair rounds, 512 new nodes and
256 MiB of newly retained tensor storage before constructing any index tuples.
Each pair round budgets two gathers, both possible canonical reshapes and
one multiply. Each odd tail budgets its gather, possible canonical reshape
and any carry multiply. Axis-final carry multiplication and output reshape
are included, as are empty-product scalar/broadcast and identity output.
Existing input graphs, returned copies and full derivative programs retain
the established backend resource gates. These conservative frontend bounds
are not an endpoint performance claim.

## Rejected alternatives

Duplicating an odd last element (valid for idempotent extrema) squares a factor.
Total-product/input division gives wrong or undefined derivatives at zero.
Scalar unrolling produces a deep graph; a new reduction primitive or concat
would enlarge ownership and backend scope. Per-value Python/NumPy execution
would violate prepared native execution.

## Qualification

Independent NumPy/array-api-strict forward checks and scalar monomial derivative
oracles cover zero, one-zero and multiple-zero factors. Multiplication is smooth
at zero; no selector opt-in is required. Tests exercise serialization,
optimization, replay, noncontiguous input, signed zero, scalar/empty/rank64,
metadata and resource rejection. Linux native FP64 tests forbid reference
execution/AD. Tests are acceptance definitions, not execution receipts.
Native f32, CUDA, higher derivatives, molecular endpoints and performance remain
unqualified. The public capture backend context rejects prod on CUDA while
preserving generic TensorIR primitive admission and serialization identities.
The internal compiler retains array_api_version=None.

## Revisit when

A separately reviewed native reduction or expanded dtype/device contract has
independent evidence. Parent #2111 remains broader than this finite slice.
