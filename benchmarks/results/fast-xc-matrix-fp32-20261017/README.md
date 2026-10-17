# Shared FP32 XC matrix products, 96-atom qualification

This source-matched summary qualifies FP32 density/potential inner products
and opens the existing diagonal matrix schedule to float. It does not enable
FP32 AO production, point algebra or forces, and introduces no new FP64
algorithm, cache or scheduling optimization. Selection is by shared execution
capability and numerical shape, not by functional name.

## Tested composition

- Candidate production identity: `c757d8f6b65017d1e56f432c2167f1ed2737723f68d1f9a67f1bc6f2587314fc`.
- Baseline production identity: `face2e30c5f7479a3e2f94ae4851f6780bc9fd5e247eb5b2c5a031daabb2d565`.
- Isolated PR parent: `6788c3aa3dbaf5c68b47d40108f702244a59a0e4`, including the merged guarded Boys change.
- The isolated PR and tested candidate have exactly equal canonical production
  inventories. Later master changes are not silently included in this claim.
- Density dot products and their inner sums are FP32; binary column scaling
  preserves tails in FP64 storage. Potential dots use FP32 inner sums and
  FP64 accumulation across point tiles. Strict refinement remains FP64.

## Complete endpoint performance

Three alternating triplets per method, all 96 atoms/768 AOs, measured through
finite exclusive Slurm allocations on matched RTX 5090 CUDA/NVML devices.
Each fresh-process endpoint includes construction, complete energy and analytic
forces, export and teardown. Existing caches remain enabled; there is no warmup
subtraction, first-use deletion, reference timing denominator or replacement
of a failed/slow sample. The strict arm retains the optimized FP64 path.

All 27 endpoints pass the independent science/work checks. Both Fast arms
execute one solve, zero numerical retries and equal per-pair Fock counts.
The predeclared incremental speed gate is a median paired old/new ratio at
least 1.02 and at least two positive pairs per method.

| Method | Previous Fast median (s) | FP32-matrix Fast median (s) | FP64 median (s) | Median paired previous/new | Median paired FP64/new Fast |
| --- | ---: | ---: | ---: | ---: | ---: |
| pbe-rks | 60.150401 | 54.917076 | 73.259431 | 1.095429x | 1.333893x |
| pbe0-rks | 60.845092 | 55.605171 | 75.838212 | 1.094970x | 1.364137x |
| wb97mv-rks | 649.646045 | 628.743855 | 1042.109213 | 1.033127x | 1.657318x |

Medians of paired ratios, rather than ratios of displayed medians, are the
qualification statistics. GPU groups, UUIDs, every retained sample hash,
source/runtime hashes and semantic-work differences are in `summary.json.xz`.
Raw endpoint arrays, build logs, native binaries and profiler traces are not
published by this compact evidence record.

## Independent numerical and safety gates

- Native: 108 mixed XC matrix cases, 84 Boys cases, one materialized J/K case,
  and three memchecks reporting zero errors.
- Complete numerical population: eight families at two small sizes in strict
  and Fast modes, six 96-atom strict/Fast returns, and four physical-tail checks.
- Fast E/maxF/RMS gates: `1e-5` Eh / `1e-4` Eh/bohr / `3e-5` Eh/bohr.
  Strict controls retain `1e-7` / `1e-6` / `3e-6` in the same units.
- Complete energy/force accuracy uses authenticated independent GPU4PySCF
  references. Node2's known NVML mismatch limits its evidence to numerical
  qualification; no node2 timing enters the performance population.
- Isolated PR checks: 40 host tests pass, 108 opt-in native cases are explicitly
  skipped locally and covered by the frozen real-GPU qualification, 565 compiler
  modules have zero dependency errors, 21 documentation tests and 61 subtests
  pass, and the Sphinx warnings-as-errors build passes.

These empirical gates are not a universal FP32 error bound. Unsupported
domains retain explicit bounded fallbacks; no strict recovery is relabeled
as a successful mixed result. Reproduce native device checks only through
finite scheduler allocations, preserving scheduler-assigned device visibility.
