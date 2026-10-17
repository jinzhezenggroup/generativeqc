# Decision: guarded shared FP32 XC matrix products

Status: implemented
Date: 2026-10-17

## Problem

Fast SCF retained double-precision XC matrix products even when the physical
precision phase admitted mixed arithmetic. Simply narrowing AO columns loses
tiny density and kinetic-energy-density tails, sometimes violating the point
domain. The change must be shared by numerical capability, not functional name,
and must not introduce another optimization of the FP64 path.

## Decision

Use the existing scalar-parametric density and potential contractions with
FP32 inner products and inner sums. Normalize each density input column with a
binary exponent before narrowing, then restore its scale in FP64 storage.
Potential contractions retain FP64 accumulation across point tiles. Open the
existing diagonal schedule to float; retain bounded strict fallback when mixed
range checks reject the input. Strict point algebra, AO generation, refinement,
and analytic forces retain their existing precision and algorithms.

The shared physical precision phase owns admission and precision reporting.
Independent strict consumers retain their strict default. No functional-name
dispatch, new FP64 algorithm, cache, or scheduling optimization is included.

## Invariants

- Fast must perform genuine mixed work, one solve and zero numerical retries.
- Preserve tails rather than flooring/clipping densities or relaxing domains.
- Preserve the independently qualified energy and analytic-force gates.
- Do not attribute prior Fast improvements or later point experiments to this
  isolated matrix change.

## Evidence

The isolated PR parent is `6788c3aa3dbaf5c68b47d40108f702244a59a0e4`.
Its canonical production inventory exactly matches the tested candidate
`c757d8f6b65017d1e56f432c2167f1ed2737723f68d1f9a67f1bc6f2587314fc`.
The matched baseline is
`face2e30c5f7479a3e2f94ae4851f6780bc9fd5e247eb5b2c5a031daabb2d565`.

All 27 predeclared complete 96-atom/768-AO endpoints pass science/work checks.
Each method has three alternating old-Fast/new-Fast/strict triplets on clean,
exclusive Slurm RTX 5090 allocations; no observation or first-use cost is
removed. Median paired previous-Fast/new-Fast ratios are 1.0954287634 for PBE,
1.0949698750 for PBE0, and 1.0331274092 for WB97M-V. All three pairs improve
for each method and all pass the predeclared 1.02 incremental gate. Median
paired strict/new-Fast ratios are 1.3338933505, 1.3641372139, and 1.6573183665;
these latter numbers describe cumulative Fast, not matrix-only benefit.

Native qualification covers 108 mixed XC cases, 84 Boys cases and the existing
materialized J/K case, with three zero-error memchecks. The full numerical
population includes eight families at two small sizes in strict/Fast modes,
six 96-atom returns and four physical-tail checks. Fast E/maxF/RMS gates remain
`1e-5`/`1e-4`/`3e-5`; strict controls retain `1e-7`/`1e-6`/`3e-6` in atomic
units. Independent GPU4PySCF references qualify full energy/force returns.
Node2's NVML mismatch limits its evidence to numerics, never timing.

The compact auditable record is
`benchmarks/results/fast-xc-matrix-fp32-20261017/README.md` and its
`summary.json.xz`. It retains every paired observation, source/runtime hashes,
and independent audit results without publishing raw arrays, binaries or traces.

## Alternatives and consequences

Unscaled float dots were rejected after physical-tail failures; global strict
recovery, density floors and functional whitelists were not accepted fixes.
Binary scaling and FP64 cross-tile accumulation intentionally bound the scope
of the precision change. More FP32 work requires its own execution evidence,
independent numerical gates and matched complete-endpoint qualification.

## References

Historical trials remain in the proposed shared-FP32-XC-matrix, diagonal-XC,
and current-master-integration notes dated 2026-10-17. Their identities and
partial/pilot results are historical evidence, not qualification of this PR.
