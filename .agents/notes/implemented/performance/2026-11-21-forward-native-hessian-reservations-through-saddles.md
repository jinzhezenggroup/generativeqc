# Decision: Forward native Hessian reservations through molecular saddles

Status: implemented
Date: 2026-11-21

## Problem

The Python molecular saddle evaluator owns each prepared energy/force frame
and consumes it with the same CPU Hessian bridge as `Calculator.hessian`.
It exposed the integral, grid and work caps, but hid the aggregate and
response byte caps. Enlarging only the exposed caps cannot admit a resident
coordinate-source panel that exceeds the response reservation; a bounded
column schedule can require substantially more work.

## Decision

Expose the existing Hessian byte controls in `optimize_saddle`. Resolve them
through the shared cap validator before preparation, keep their 512 MiB /
128 MiB defaults and nesting, and forward the resolved pair for each retained
Hessian, including final verification. Do not add a second optimizer storage
policy or silently pick larger limits. Integral, grid, output, solver, and
semantic work caps retain their own checks. Fallback still requires the
native owner's full admission, under the same physical model and source
identity. These limits do not bound primal SCF or process RSS.

## Evidence and boundary

On the fixed 35-atom Ni/I₂ seed, the exact integration wheel's shape planner
reports a 309,251,728-byte optional first-source panel, larger than the
previously inexpressible 128 MiB response reservation, and 349,439,992 bytes
for moving-grid/XC work. Shape support is not actual allocation, bounded
runtime, or saddle convergence. The new adapter tests exercise malformed
capacity refusal before SCF, actual aggregate and response refusals after a
valid SCF on H₂, and successful Hessian recovery on the same calculator.
They do not qualify the Ni/ECP endpoint or justify changing defaults.

Related: #2848 and the standalone Hessian/HVP controls for #2844. See
`docs/user/geometry_optimization.md` for the public contract.
