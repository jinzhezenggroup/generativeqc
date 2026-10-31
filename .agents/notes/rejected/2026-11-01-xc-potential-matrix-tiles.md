# Decision: retain the 16x16 CUDA XC matrix schedule after the unroll fix

Status: rejected (8x8 and 32x32 global defaults)
Date: 2026-11-01

## Problem

Bounding potential-loop unrolling removes the register-resource launch failure
of the existing explicit 32x32 schedule. That establishes feasibility, not a
reason to select 32x32 as the global default. Both schedules must be measured
with the same generated arithmetic and complete requested outputs.

## Decision and evidence

Retain the production 16x16 matrix schedule. In the finite eight-CPU n1 Slurm
allocation 15031, the four complete cold endpoint changes for tile32 are
PBE +3.84%, r2SCAN +4.51%, r2SCAN-3c +3.02%, and M06-2X +2.67%.
Warm changes are +1.03% to +1.46%. All eight populations independently pass
the unchanged README energy/full-force gates, and paired semantic work is
unchanged. The sampled regression rejects a global promotion; this pilot does
not prove tile32 is slower for every system or GPU.

Both builds use production source identity
`ce5f498a8d173d214bc46d686256caba37836541445f6a089601a550066d6ad0`.
CMake's explicit matrix-tile override changes the generated code without
changing that source identity. The evidence therefore binds the generated
source, CMake configuration and actual library hashes independently:

- tile16: `3e86bc042c0480da6716588ff9cde38e4987c217446c5cf1dbcab3c70dc29c9a`;
- tile32: `9478341106a4572a60cd1560bdf8b432dc4726f3a249a4f0be48f3f2830f3132`.

The native point/ragged suites pass in the tile32 binary. The final GPU4PySCF
command in job 15031 was given a native-only argument and failed before timing;
its original exit 2 and log remain visible. That failure does not erase the
eight completed native populations or establish a reference ranking.

## Reference-cache control

The separate fresh-reference job 15032 exposed a first GPU4PySCF cold endpoint
of 30.9079 seconds, followed by 23.2696 seconds. Retain that entire initial
block as exploratory cache-conditioning evidence. Do not average it into a
claim that tile16 has a large cold advantage.

Job 15033 explicitly conditions both engines with one complete cold/one-warm
population, then measures two fresh-owner cold/three-warm ABBA pairs. Tile16
has median cold 23.437873 seconds versus GPU4PySCF 23.404893 seconds (+0.14%);
warm medians are 6.806990 and 9.883326 seconds (-31.13%). This narrow cold margin
is not an established win. All ten reference-control populations across the
two jobs independently qualify, and all original samples remain retained.
The same explicit conditioning protocol is applied to the subsequent n4/n2
reference controls; they are not pooled across devices or allocations.

## Additional 8x8 experiment

Slurm job 15034 measures the other existing supported tile against the same
16x16 unroll4 binary on n1, with one cold and three frozen warm endpoints per
population. All eight populations independently qualify and paired semantic
work remains equal. Cold changes are PBE +0.16%, r2SCAN +2.25%, r2SCAN-3c +3.05%
and M06-2X +1.19%; warm changes range from +0.07% to +1.99%. The native point and
ragged suites pass and the job exits successfully. This pilot also gives no
basis for replacing the global 16x16 default. The archive preserves its
independent binary/configuration hashes under `tile8-followup/` and the reader
recomputes its eight gates and paired timing changes.

## Revisit when

Reconsider a shape-specific tile only after complete endpoint measurements
identify a useful regime and resource admission remains portable. Do not
infer optimal tile size from register counts or from GPU names. Numerical
controls must continue to cover 8/16/32 schedules, both FMA compiler modes,
ragged/indexed maps and finite-cancellation fallbacks.

## References

The [unroll evidence archive](../../../benchmarks/results/xc-potential-unroll4-20261101/README.md)
retains this experiment under `tile32-followup/`; its reader recomputes all 18
independent qualifications, configuration hashes and timing comparisons.
