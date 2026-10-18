# Decision: bound the pure-J Direct default below its failed endpoint

Status: implemented
Date: 2026-10-16

The temporary exact-direct upper bound is superseded by
[stationary-density energy confirmation](../numerics/2026-10-16-stationary-pure-j-energy-confirmation.md)
and its new source-bound complete-endpoint qualification. The containment below
did not repair the SCF tail; its original negative observation remains rejected.

## Problem

PR #2532 promotes resident final validation for exact-direct and density-fitted
pure-J KS at 544 or more AOs. Its frozen Direct/96 moved endpoint instead takes
327.888863 to 380.302499 seconds, a 15.985% slowdown, with 17 versus 21 SCF
iterations/builds. Changed work prevents a causal timing attribution to final
validation; it does not establish complete-endpoint nonregression. Repairing
the evidence verifier alone therefore left the production default overbroad.

## Decision

Supersede the exact-direct scope of the
[original pure-J promotion](2026-10-15-pure-j-resident-final-validation.md):

- Exact-direct pure-J defaults to resident validation only for 544 <= AOs < 1088.
  At 1088 AOs and above, unset uses the existing host proof.
- Density-fitted pure-J retains its separately qualified default at 544 or more
  AOs. The live producer's fitted-Coulomb flag distinguishes this route; size
  alone must not disable its successful 1088-AO population.
- Exact-direct global hybrids retain the existing 384-AO lower bound and no new
  upper bound. The negative observation concerns pure J, not global hybrids.
- Explicit selector `0` still forces the host proof. Explicit `1` remains an
  experimental request and cannot bypass numerical, resource or lifetime guards.

The upper bound is a conservative default-admission hold, not an inferred
hardware crossover or a claim that every smaller shape has been measured.
No new SCF, force, matrix-product, precision, basis or convergence algorithm is
introduced. This contains exposure to the failed candidate; it does not fix or
explain that candidate's intrinsic performance.

## Rejected alternatives

- Waiving the moved failure because iterations differ, or normalizing time by
  iterations, would hide a real complete-endpoint negative.
- Raising the lower bound to 1088 would disable Direct/48's positive benefit
  while retaining the failed Direct/96 route. Direct/48 moved-warm's 1.868%
  benefit remains enabled; the descriptive 2% label is not an enablement gate.
- Disabling all pure-J or all large resident validation would discard the
  independently successful smaller Direct, DF and existing hybrid populations.
- Restoring unrestricted defaults without a source-bound, reviewed endpoint
  basis would only move the unresolved failure out of the verifier.

## Invariants

The held route uses the same strict host proof as explicit `0`; existing
token, generation, matrix-handle, scratch-packet and published W/total-D lease
guards remain unchanged for resident requests. All scientific gates, complete
host exports, D4/gCP and canonical spherical def2-mTZVPP remain unchanged.

The original observations, summaries, receipts, source patch and plots remain
byte-identical. The historical campaign must still fail verification at
`none/96/moved`; narrowing the current default must not reclassify that failure
or relabel old candidate timings as measurements of the revised default.

## Evidence

The retained campaign contains 192 complete comparison calls with independent
energy/force gates. It remains evidence for its frozen source, not this revised
checkout. All other retained comparisons satisfy the complete-endpoint verifier;
the Direct/96 moved failure is preserved explicitly.

Host tests execute the production C++ policy at both AO boundaries, for Direct,
DF and hybrid producers, spin/precision exclusions, unset/0/1/error selectors
and very large sizes. A focused compiled regression test fails on the previous
policy and passes with the upper bound. The evidence regression test replays
the real archived samples and still rejects the held comparison. No new GPU
campaign or performance qualification is claimed.

## Revisit when

Removing or raising the Direct-only upper bound requires reviewed,
source-bound complete cold/warm/moved/moved-warm endpoints with independent
scientific acceptance, all semantic work counts and the original negative
retained. A repaired verifier or warm-only benefit is not sufficient.

## References

- PR #2532 and its complete-endpoint nonregression review.
- `benchmarks/results/r2scan3c-final-validation-20261015/`.
- `src/dft/cuda_ks_final_validation_policy.hpp`.
- `tests/python/test_ks_device_final_validation_policy.py`.
- `tests/python/test_r2scan3c_final_validation_evidence.py`.
