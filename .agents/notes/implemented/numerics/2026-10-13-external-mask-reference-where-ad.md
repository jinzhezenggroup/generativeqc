# Decision: external-mask where derivatives in the CPU reference interpreter

Status: implemented
Date: 2026-10-13

## Problem

`where` already selects finite generic real branches using genuine Boolean data,
but the runtime reference derivative endpoints reject all live Boolean values.
Giving `where` unrestricted AD capability would incorrectly admit computed
conditions and unqualified generated/public compiled/native derivative paths.

## Decision

The existing `tensor/autodiff.py` owner validates a narrow execution context:
all live values are generic float32/float64 or external Boolean selectors;
selectors originate from inputs/constants through the existing type-preserving
transpose/reshape/slice/static-gather/broadcast views. Every Boolean value must
serve a selector view or `where` condition. Boolean outputs, computed comparisons,
scientific annotations, integer controls and mask differentiation reject.

The forward interpreter skips Boolean tangent storage and selects branch
tangents with the actual mask. Reverse mode allocates only real/control bars,
routes real bars to the two branches with Boolean selection, and reuses existing
broadcast/scatter adjoints. Boolean bars are never materialized. This extends
only runtime reference JVP/VJP/dot-test, with an explicit versioned context in
the existing capability owner. The unrestricted primitive rule registry and
generated derivative admission are unchanged.

## Rejected alternatives

- Float 0/1 predicates and Boolean/integer-control conflation lose typed semantics.
- A global `real_ad`/`AD_PRIMITIVES` flag would promote unqualified consumers.
- Lazy evaluation or derivative error suppression would change the existing
  all-live primal contract and derivative-domain checks.
- A separate IR, derivative algebra or capability registry duplicates ownership.

## Invariants

All live primals are evaluated and finite/domain checked, including unselected
branches. JVP still evaluates real derivative nodes; unselected singular or
overflowing derivatives can fail. VJP retains its existing requested-input
reachability pruning, without introducing mask-dependent graph pruning.
Inputs remain unchanged and published derivative arrays are detached. Existing
retained-byte and logical work accounting remains authoritative; masks have
primal storage but no derivative storage. Pure-real identities/rule version
remain unchanged. Generated/public compiled AD, HVP and every native `where`
backend remain unsupported.

## Evidence

`tests/python/test_tensor_where_reference_ad.py` executes the complete reference
primal, JVP, VJP and dot endpoints. Independent NumPy selection and hand-derived
polynomial/exp/log slopes and broadcast/scatter bars supplement fixed-mask
central differences (FP32 step 2e-3, FP64 step 1e-5 with dtype-specific gates).
Coverage includes both dtypes, constant/input/view masks, scalar/empty/strided
data, sparse/dense/all-true/all-false masks, nested selection, changed-mask
replay, casts/promotion, aliases, repeated inputs, partial requests, budgets,
error recovery and unchanged generated/public/native rejection.

The host qualification runs retain raw logs and JUnit records under ignored
`.artifacts/where-ad/`; their skips do not qualify native/device execution.
The source baseline is upstream `3ea6f7988ace5b0d711c243cdc3aadff64defe75`.
Reproduce the new tests with `PYTHONPATH=python python -m pytest
tests/python/test_tensor_where_reference_ad.py`, and retain Python/NumPy/source
identities with the run. No GPU, QC endpoint performance, Array API full-version
conformance or broader conditional/scientific AD acceptance follows.

## Revisit when

Computed conditions or generated/public/native selected-branch AD acquire their
own explicit mathematical domain, independent oracles, execution qualification
and error/identity/resource contracts.

## References

- Refs #2137 and #2111; both parent scopes remain open.
- [Current derivative contract](../../../../docs/developer/tensor_ir.md#derivative-programs)

## Follow-up: explicit primal comparison selectors

The [2026-10-19 decision](2026-10-19-primal-comparison-where-ad.md) separately
qualifies comparison selectors through an explicit option, preserving the
external-only default and Boolean non-differentiability recorded here.
