# Decision: generic float32/float64 Array API promotion via explicit TensorIR cast

Status: implemented
Date: 2026-10-09

## Problem

The experimental Array API preview rejected all mixed float32/float64 array operands, contrary to the Python Array API 2025.12 real-valued promotion lattice. The existing TensorIR is intentionally strict: scientific AO/occupied/virtual/auxiliary/spin tensors use exact index domains and explicit precision schedules. Relaxing TensorIR primitive dtype checks globally would undermine that contract.

## Decision

- **Only anonymous generic Array API inputs** automatically promote float32 + float64 to float64, independently of operand order and array rank. The tracer inserts the already supported first-class TensorIR `cast` node before binary add/subtract/multiply/divide, matrix multiplication, or generic extension `einsum`.
- Scalar Python arithmetic retains the existing weak-scalar policy: the scalar converts to the array's dtype. Explicit scientific arrays of different dtypes remain incompatible unless the programmer calls `astype` intentionally.
- Add a public real-valued subset of the standard `astype`, `can_cast`, `finfo`, `isdtype`, and `result_type` functions. `astype` lowers to TensorIR `cast` for symbolic arrays. The NumPy eager path honors same-dtype `copy=False` identity and default `copy=True` detached storage. A symbolic copy is an immutable SSA value, not a promise of native physical copy allocation.
- Keep `float32` and `float64` as the **only** admitted public numerical dtypes. Reject integer, bool, complex and external devices; do not invent their promotion/dtype defaults or weaken finite-result rules. Native CPU/CUDA execution is separately qualified.
- Update `capabilities()["dtype_promotion"]` to `generic-float32-float64-explicit-cast`, with `scientific_dtype_promotion=False`. Retain `array_api_version=None` and no `__array_namespace__`.

## Rejected alternatives

1. Relax core TensorIR `_common` dtype constraints for all scientific methods: would silently rewrite precision/accuracy contracts.
2. Automatically cast operands only in the eager NumPy reference path: compiled TensorIR capture would still fail and numerical identity would diverge.
3. Choose a dtype based on the first operand's precision: violates type promotion commutativity and loses accuracy when float64 is second.
4. Model non-supported bool/integer/complex data with real-valued surrogate arrays to increase a function-name count: incorrect semantics and misleading standard conformance.
5. Claim physical copy/zero-copy semantics for any native artifact based on a Python `astype` expression alone: this is a separate buffer-ownership/runtime guarantee.

## Invariants

- Numeric promotion occurs at the generic frontend boundary; ordinary TensorIR math continues to require matching dtypes and typed index populations.
- Every promoted operand emits an explicit inspectable cast node. No uncontrolled FP32/FP64, CPU/GPU or scientific representation conversion occurs.
- Generic graph identity, JVP/VJP and native code planning reuse existing TensorIR rules; compiled arithmetic does not call NumPy per node in native production paths.
- Unavailable complex, integer, boolean, dynamic-device and NaN/Inf semantics remain fail closed and advertised as unsupported.
- `astype(copy=True)` eager arrays allocate fresh host storage, while the symbolic SSA path carries no undocumented native physical alias guarantee.

## Evidence

`tests/python/test_array_api_float_dtype.py` provides a multi-operation float32/float64 promotion matrix, scalar/array dtype behavior, NumPy eager and captured numerical parity, explicit cast graph assertions, scientific-domain negative cases, JVP, dtype inspection and input/device guards. Existing frontend and broadcast tests are updated to accept this **generic only** promotion.

## Consequences

The preview becomes more useful for ordinary mixed real-float programs but is still not a conformant general-purpose Array API: no bool, integer, complex, IEEE nonfinite special values, standardized namespace discovery or native backend guarantee. Continue broader type support through #2111 only with proper bool/int TensorIR representation and differential/dispatch policies.

## References

- #2111 (completion tracker)
- https://data-apis.org/array-api/2025.12/API_specification/type_promotion.html
- https://data-apis.org/array-api/2025.12/API_specification/data_type_functions.html

Agent: ChatGPT
Model: GPT-6
