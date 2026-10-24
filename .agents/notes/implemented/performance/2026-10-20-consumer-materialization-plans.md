# Decision: prove consumer substitution before removing dense intermediates

Status: implemented
Date: 2026-10-20

## Problem

Shape-only complexity and exact-zero diagnostics cannot prove that a dense value
can disappear while preserving all its consumers. #1580 still needs reusable
consumer-aware planning beyond its existing symbolic contraction-tree search.
The paused d2291fb4 draft was unqualified; it is not production capability.

## Decision

Expose an explicit bounded TensorIR plan API. Only a real-FP64 einsum with one
reader edge into an einsum can substitute its visible factors. Private reduction
indices are renamed, rational factors multiply exactly, and the existing tree
search supplies a candidate. Accept only strict total-materialization savings
with nonincreasing work degree, concrete work and simultaneous-live logical peak.

Caller input bytes are charged through return; output lifetimes are retained.
Use the shared lifetime analyzer, without claiming actual allocator/view/provider
workspace accounting. Recompute value/reader contexts after accepted substitutions:
replacement rebuilds descendants, so a stale original-node list can silently
omit newly rebuilt high-rank consumers. A previously rejected producer must
be reconsidered when its reader changes; node identity alone is not a decision
cache key.

## Rejected alternatives

- Blindly reuse the paused draft: input bytes were separate from its claimed
  peak, and its fixed candidate list could miss rebuilt consumers.
- Enable automatically: reassociation changes floating-point reduction and
  coefficient order; production admission and endpoint profitability are absent.
- Infer factorization from MP2/CC four-index shapes: no separability proof exists.
  Report the required factor/consumer/alias/derivative/resource contract instead.
- Duplicate shared symbolic tree search or storage analysis: these are already
  authoritative compiler owners.

## Invariants

Preserve requested dense outputs, immutable inputs, input roles, differentiability,
exact symbolic coefficients and output specification/order. Repeated edges,
multiple consumers, views/nonlinear boundaries, explicit arithmetic schedules,
unsupported dtypes/diagonals/rules and exhausted resources retain the original
program with an explanation. Physical input aliasing grants no extra savings.
No default compiler preparation, production route or CUDA lowering is promoted.

## Evidence

The production-derived acceptance is an explicit auxiliary slice of
`cc/df_spectator_pairs.py`'s `ac,ijcd,bd->ijab` ladder, expressed via
`ac,bd->abcd` followed by the tau contraction. Compare with independently
materialized NumPy einsums, analytical NumPy JVP/VJP and two-step independent
arithmetic finite differences. Generated derivative programs use the same gates.
Numerical parity tolerances are 2e-13 for the ladder, 3e-13 for derivative
comparisons, and 3e-8 for directional differences at 1e-5 and 3e-6.

Behavioral coverage includes deterministic serialization/planning, rational
coefficients/private dummy indices, visible/shared/view boundaries, precision,
provider availability, exact/below-limit resources, work/storage regression,
finite search and dead-definition/input/output lifetimes. See
`tests/python/test_tensor_materialization.py`.

## Consequences and revisit conditions

This is a partial #1580 compiler delivery. Opaque #1574/#1575 sources, broader
provider/factorization rules and production selection remain open. Add a rule only
when its mathematical, derivative, consumer, alias and resource proof is explicit;
promote execution only with the corresponding real-backend endpoint evidence.

Agent: Codex
Model: GPT-6
