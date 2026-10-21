# Decision: finite centered variance capture

Status: implemented
Date: 2026-10-21

## Decision

Refs #2111. Compose generic finite float32/float64 variance with existing mean,
sum, subtraction, multiplication, rational coefficients and shape operations.
Do not introduce another IR operation, AD rule, backend or runtime. Validate
scientific metadata before scalar, empty and empty-axis cases. Same output
dtype, signed unique axes and keepdims follow the existing frontend contracts.

Use centered squares rather than E[x*x]-E[x]**2, which loses small variance at
large common offsets. Python int/float correction becomes its exact rational
value for static M-correction. Require M>0 and M-correction>0; reject nonfinite,
dynamic/bool correction and an unrepresentable reciprocal. The finite execution
contract validates actual intermediate values. Eager IEEE semantics are intact.

## Resource and backend boundaries

Preflight 8 input-sized and 10 reduced-sized tensor slots, including potential
canonical reshapes, mean broadcasting/singleton removal, both sums and coefficient
scaling. This conservative 18-node bound also accounts for real zero-byte nodes.
Reject retained storage above 256 MiB before constructing any reduction node.
Existing input graph, backend work/byte/node and derivative limits still apply.
No new shape-rank or extent whitelist is introduced.

Public CUDA uses the existing scoped capture backend context to reject variance
at all public primal and derivative entries. Program metadata and raw manual IR
backend admission remain unchanged. Native CPU must satisfy existing FP64 input,
output and every-live-operation admission before invoking the compiler.

## AD and qualification

Existing first-order composition is checked against the independent analytic
derivative 2*(x-mean)/(M-correction), axis-broadcast cotangents, finite differences
and adjoint identities. Constant inputs have zero derivative. This follows the
existing real-arithmetic AD convention, not classical derivatives of IEEE rounding.
NumPy and pinned array-api-strict 2.6.1 (2025.12 mode) independently check shape,
dtype and values. Native FP64 checks forbid reference execution and AD fallback.
Tests define gates; they are not standalone execution receipts.

Native FP32, public CUDA, std, higher derivatives, molecular endpoints,
performance and full conformance remain unqualified. Parent #2111 stays broader
than this slice. Revisit only with independent evidence and separate backend or
domain authorization.
