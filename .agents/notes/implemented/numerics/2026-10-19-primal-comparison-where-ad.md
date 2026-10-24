# Decision: first-order real-branch AD with primal comparison selectors

Status: implemented
Date: 2026-10-19

## Problem

Issue #2137 requires differentiating the selected real branches of `where`
without differentiating its Boolean condition. Fixed external selectors were
qualified, but a real comparison selector caused the whole derivative request
to reject. Merely admitting comparisons in the context proof would still
incorrectly differentiate predicate-only arithmetic, including the singular
slope of `sqrt(0)` used only in a valid primal comparison.

## Decision

Add `allow_comparison_where=True`, separate from the existing external-only
option. Reuse the canonical context proof, branch JVP/VJP rules, real view
adjoints, eager validation graph and native materializing CPU owner. Stop
real derivative ancestry at Boolean edges. Predicate-only real computations
remain live, finite/domain-checked primals but have no requested derivative
work. Values reused by real branches retain the existing derivative checks.
Reference admission is generic finite FP32/FP64; generated and public transforms
are first-order generic finite FP64 on reference/materializing CPU only.

A selector is fixed at its call's primal, including at equality. The selected
branch gives the reported directional/adjoint convention at switches; this
makes no mathematical differentiability claim there. No finite difference at a
switch qualifies that convention. VJP snapshots include predicate inputs.

Generated execution inputs remain non-differentiable, and requested JVP
seeds are retained by validation edges even if their only primal use is a
predicate. Serialization, CSE/DCE and removed descriptive provenance cannot
open higher-order AD or drop these seed checks. Existing external-mask and
pure-real generated equations preserve their prior admission and identities.

## Rejected alternatives

- Promoting comparison primitives to differentiable algebra would create
  Boolean tangents/cotangents and give predicates the wrong mathematical role.
- Silently widening the external-mask option would break its declared ownership
  and existing fail-closed callers.
- Differentiating all real predicate ancestors would raise on valid primal-only
  comparisons such as `sqrt(p)>0` at `p=0`.
- Pruning unselected real branches by mask value would remove existing eager
  primal/forward-domain checks and change error semantics.
- A parallel derivative algebra or production reference fallback is unnecessary;
  the existing real branch/view rules and native owner provide the behavior.

## Evidence and acceptance

`tests/python/test_array_api_comparison_where_ad.py` checks all six comparisons,
real FP32/FP64 reference derivatives, FP64 public reference/native CPU
transforms, broadcast and repeated-gather adjoints, snapshot mutation, empty
and scalar results, predicate-only singular slopes, retained unselected branch
errors, serialized seed validation, first-order rejection and executor cleanup.
Native cases forbid reference execution/AD. Independent NumPy analytic and dot
gates use rtol=atol=2e-13 in FP64 (2e-6 in FP32). Central differences away from
switches use step 1e-5 with rtol=atol=2e-8 in FP64, and step 2e-3 with
rtol=2e-3/atol=2e-4 in FP32. Equality tests check the explicit selected-branch
convention rather than differentiability.

No CUDA, high-order AD, scientific selection, mixed precision, performance or
new full-standard conformance claim follows. CPU/CUDA capability and resource
error gates remain required.

## References

- Refs #2137 (requested P0) and #2111.
- Supersedes only the computed-selector exclusion in the
  [external-mask reference decision](2026-10-13-external-mask-reference-where-ad.md).
- [Current contract](../../../../docs/developer/tensor_ir.md#derivative-programs-151).
