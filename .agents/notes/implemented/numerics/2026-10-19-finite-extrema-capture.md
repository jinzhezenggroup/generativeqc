# Decision: bounded finite-real minimum/maximum capture

Status: implemented
Date: 2026-10-19

## Problem

The public eager 2025.12 namespace already exposes minimum and maximum. Its
private capture adapter and compiler namespace lacked a lowering. Issue #2111
tracks a much larger surface and backend qualification; this slice does not
complete that parent issue.

## Decision

Use the existing typed comparison, generic promotion/broadcast and where nodes.
At least one operand must be a generic real array. Validate scientific metadata
before promotion, and finite scalar representability before building a constant,
even for an empty result. Pre-broadcast operands once so predicates and selected
branches share logical retained storage under existing interpreter budgets.

Ties select x1, retaining its array zero sign. Array API 2025.12 explicitly leaves
signed-zero ordering implementation-defined. The eager backend may select another
zero sign. Static negative-zero literals remain outside the exact rational scalar
contract. Public eager IEEE/NaN behavior is unchanged; captured live operands must
all be finite. Selected-branch first-order AD uses the existing explicit comparison
selector opt-in. Interior JVP/VJP/dot qualification does not establish a derivative
at a tie, Hessian support, full Array API conformance, or CUDA qualification.

## Rejected alternatives

- A new TensorIR mathematical primitive or separate AD rule duplicates existing
  comparison/selection semantics and adds unnecessary engine scope.
- Expanding minimum/maximum arithmetically can overflow at finite extremes and
  loses branch/zero semantics.
- Repeated broadcasts for predicate and branches consume logical storage before
  CSE; raising budgets would hide the unnecessary work.
- Relaxing scientific metadata or admitting nonfinite capture would widen the
  existing interpreter/backend contracts without appropriate independent gates.

## Evidence and invariants

`tests/python/test_array_api_extrema.py` uses NumPy and the installed pinned
array-api-strict 2.6.1 oracle, tests direct callable compilation, weak scalars on
both sides, promotion, broadcasting, 0D/empty arrays, finite extremes, zero ties,
serialization/determinism, rejection domains, unchanged resource gates and
first-order AD with unequal directions and weighted broadcast adjoints. The
native FP64 test forbids reference interpreter execution and checks forward,
JVP and VJP independently. A test definition is not an execution receipt;
native capability flags remain unset until separate backend qualification.

No changes to scientific defaults, runtime/compiler IR primitives, GPU paths,
public discovery, shared API manifests or the already merged clip contract.

## Revisit when

The shared exact scalar representation supports signed zero, or an explicitly
reviewed IEEE/extended-dtype capture contract and backend qualification exists.

## References

- Issue #2111
- `docs/user/experimental_array_api.md`
- https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.minimum.html
- https://data-apis.org/array-api/2025.12/API_specification/generated/array_api.maximum.html
