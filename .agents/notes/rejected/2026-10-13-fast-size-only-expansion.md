# Decision: do not expand Fast admission based only on larger AO counts

Status: rejected
Date: 2026-10-13

## Problem

The current budget-selected XC warmup gives small complete-endpoint improvements.
The user reasonably asked whether larger systems expose a larger benefit. A
size-only timing comparison would be misleading: public admission ends at 256
AOs, while diagnostic mixed work outside that domain can immediately fail and
retry strictly. Neither type of fallback is sustained FP32 execution.

## Decision

Keep the production domain unchanged. Measure ordinary FP64, diagnostic mixed
XC and the public audit fallback as different routes. Preserve all first calls,
repeats, rejected mixed work, independent-reference failures and actual solve /
force / Fock counts. Do not promote a small timing ratio from a mostly-FP64
fallback to evidence for expanding mixed-precision admission.

## Evidence

On RTX 5090, Hcore-seeded PBE0/def2-SVP complete energy-plus-force endpoints give
1.0381x at 192 AOs and 1.0221x at 384 AOs in the diagnostic mode. The former uses
10 mixed + 9 strict Focks; the latter uses only 1 mixed + 20 strict Focks with an
execution retry in every sample. Ordinary FP64 uses 17 and 22 Focks respectively.
The larger geometry is a synthetic translated octamer array, not qualification
on chemically diverse large molecules. All listed endpoints pass independent
PySCF moving-grid energy/force gates; strict final accuracy does not imply the
rejected low-precision work was acceptable.

A public 384-AO Fast diagnostic executes two FP64 endpoints, taking 53.265 s,
with 45 total Focks and two force evaluations. Returned-attempt precision counts
alone do not describe this audit's total cost. Keep this single first call
separate from repeated steady-process comparisons.

Bounded 384-AO ablations isolate the retry to adding mixed contractions:
point-only and point-plus-AO modes both run 12 mixed + 9 strict Focks without
retry, whereas points-plus-AO-plus-contractions retries after one mixed Fock.
The precise device failure cause is not yet exported/identified. Metric electron
counts are correct; a negative-density explanation remains a hypothesis, not
a diagnosis. Do not patch this by clipping densities or suppressing validity
guards. A future fix needs the exact failing operator/inputs and independent
local plus full-endpoint checks.

The 384-AO strict profile attributes about 35.9% of kernel-duration sums to
canonical/resident J/K and 3.3% to XC point kernels. These are not endpoint
fractions or a comprehensive gradient profile. J/K and all analytic forces
remain FP64, so increasing system size alone does not create an all-FP32 route.

At 768 AOs both the diagnostic mixed route and ordinary FP64 fail in generated
analytic forces with out-of-memory errors on RTX 5090. No complete timing or
force-error gate is accepted. The ordinary independent PySCF reference also
fails to converge within 180 cycles; a bounded second-order retry is cancelled
after the native force failure. Preserve the incomplete reference checkpoint
and cancellation separately rather than relabeling it as converged or failed
numerically. This is an endpoint/resource limitation of this campaign, not
evidence that all large-system FP32 algorithms lack value.

## Revisit when

Identify and qualify the contraction failure with an explicit bounded fallback,
then demonstrate sustained mixed work and complete-endpoint improvements across
representative geometries, AO sizes and initial guesses. Widening derivative or
J/K precision needs its own independent gates, force/energy consistency checks
and application-budget semantics. Do not loosen final convergence, change the
grid/model, feed oracle densities into native work, or omit required audits to
manufacture a crossover.

## References

- [Full scaling evidence](../../../benchmarks/results/pbe0-fast-scaling-20261013/README.md)
- [Implemented XC warmup](../implemented/numerics/2026-10-13-budget-selected-fp32-xc.md)
- [Current application contract](../../../docs/user/application_precision.md)
