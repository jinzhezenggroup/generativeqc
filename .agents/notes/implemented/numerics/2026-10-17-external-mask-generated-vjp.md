# Decision: opt-in generated VJP for fixed external Boolean selectors

Status: implemented
Date: 2026-10-17

## Problem

P0 #2137 requires real branch JVP/VJP for `where` without numerical masks or
Boolean adjoints. The fixed external-mask reference domain and explicit generated
FP64 JVP already exist; reverse DAG generation still rejects every Boolean
ancestor. Public AD, computed predicates and higher-order qualification are
separate work.

## Decision

Add `transpose_program(..., allow_external_mask_where=True)` for the same generic,
finite FP64 domain as explicit JVP generation. Reuse the shared external-mask
admission proof and the existing derivative/view rules. A `where` sends a real
cotangent to its selected real branch and a literal real zero to the other;
the Boolean condition receives no adjoint. Broadcast and gather adjoints keep
their existing reduction/scatter ownership. Repeated-label einsum projections
retain their existing element-budget gates.

A demanded adjoint alone does not preserve eager scientific domains: an
unselected sqrt(0), an unreturned invalid branch, or an unused cotangent feed
could disappear. Build a zero-direction forward validation DAG from the existing
JVP rules, not a second derivative algebra. Shared guards zero every primal,
forward-domain value and requested cotangent before full reduction, then form
an exact +1 multiplier on each published gradient. This avoids overflowing a
validation sum and preserves finite selected cotangent bits, including -0.
Guards remain output-reachable through optimization/serialization, even for an
empty or disconnected requested input. Native byte/work/node budgets charge
this additional work; no endpoint speedup is claimed.

As with fixed-mask JVP, generated inputs become non-differentiable execution
data. The typed boundary survives reconstruction without provenance. Explicit
provenance additionally identifies first-order materializing-CPU qualification. Disconnected requested inputs
must satisfy the same dtype and generic-domain admission because their zero
adjoints become published live values.
Default rejection, computed comparisons, Boolean outputs/AD, scientific domains,
FP32/mixed precision, packed policies, CUDA/portable/scalar and public compiled
transforms remain outside this slice. Ordinary non-selection graphs retain
legacy generation version and mathematical derivative identities.

## Rejected alternatives

- Float-valued masks or Boolean adjoints violate the typed selector contract.
- Retained definitions alone do not force execution after DCE.
- Primal-only guards omit singular forward derivative domains.
- Adding zero guards to gradients destroys selected negative-zero signs.
- Calling the reference VJP/interpreter from native execution is not native AD.

## Evidence and revisit conditions

Independent NumPy finite differences, analytic branch derivatives, indexed
broadcast/gather pullbacks and JVP/VJP dot identities qualify reference and
real native CPU execution. Failure injection covers eager domains, missing and
invalid unused cotangents, resource limits and failed replay followed by recovery.
Existing JVP regressions remain required. Broaden the domain only with separate
predicate, precision, device and higher-order scientific qualification.

Refs #2137; previous fixed-mask JVP contract and #2518 signed-zero guards.
