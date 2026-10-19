# Decision: expose bounded fixed-mask scalar gradients

Status: implemented
Date: 2026-10-18

## Problem

Issue #2137 requests real branch derivatives for where. Generated fixed-mask
FP64 VJP programs existed after #2595, but public grad/value_and_grad could not
select that qualified path.

## Decision

Expose allow_external_mask_where=False explicitly on those two scalar transforms.
Pass the opt-in to the canonical transpose_program owner; do not implement another
selection adjoint or change native lowering. The generated fixed-mask graph
freezes real execution feeds as non-differentiable. Combining it with the
original differentiable primal causes a same-name input type conflict. Rebuild
the primal with the compiler's canonical graph rebuilder against the generated
inputs; never restore their differentiability flags to make combination succeed. Require reference or materializing
CPU execution. argnums selects real inputs while Boolean selectors stay external.
The default, public jvp/vjp, computed predicates and broader scientific/precision
contracts retain their current boundaries.

## Rejected alternatives

Automatically enabling selection would weaken an explicit compatibility boundary.
Differentiating every positional input would treat Boolean selectors as numerical
variables. Extending jvp/vjp here would also require a separate argument/seed
selection contract; it is deferred rather than inferred from dtype.

## Invariants

Retain all live primal/forward domains, byte/work/node gates, detached results,
current-mask replay and independent close ownership. Use the generated VJP guards
for disconnected/empty gradients, even when grad omits the primal output.

## Evidence

The public regression suite tests NumPy analytic gradients and displaced-energy
finite differences, reference/native parity, repeated gather/broadcast adjoints,
mask replay, unselected singular branches, empty/disconnected inputs, budgets,
scientific/dtype/backend rejections and owner lifetime. Native tests forbid the
reference interpreter and reference AD evaluators. Acceptance uses rtol/atol
2e-14 for the selected analytic controls and 2e-9 for central differences with
step 1e-5. The broader Array API/where regression passed 1612 tests with 14 skips;
the final public fixed-mask plus capability suite passed 158 tests (overlapping
the broader run). The independent type, evidence, codegen ownership and compiler
structure checks passed. A cache-miss native execution observed actual sccache
prefixes on both compile and link; its foreground server reported one C++ cache
miss and an uncacheable link, not a cache-hit performance claim.

## Revisit when

A separately qualified public selected-argument JVP/VJP contract, broader dtype,
scientific domain or backend admission is available. Parent issues #2137/#2111
remain broader than this scalar-transform slice.
