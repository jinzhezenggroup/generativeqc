# H100: Direct RHF per-iteration reductions on a full CTA

Every Direct RHF SCF iteration runs two single-block reductions over
`nbf x nbf` matrices:

- **Energy contraction:** `compute_energy_kernel` streams D, H and F with one
  32-thread warp per system.
- **Convergence walk:** `update_convergence_kernel` computes the residual
  maximum, density RMS and conditional copy.

At 768 AOs (water-96, def2-SVP) both are latency-bound. A node-traced nsys of a
cold README Direct HF96 call puts them at 89 ms (12 launches) and 179 ms
(11 launches).

- **Energy:** `compute_energy_kernel` now folds warp sums in fixed warp order,
  and RHF launches it with 512 threads. A single warp keeps the incumbent
  arithmetic exactly.
- **Convergence:** the existing one-block 256-thread walk
  (`GENERATIVEQC_HF_BLOCK_CONVERGENCE`, #3042) becomes the default in its
  qualified domain: restricted, nonmixed, nonincremental quartet Direct without
  physical-reference export. `0` restores the 32-thread walk.

## Measurements

The runs used fresh consecutive processes on one physical H100 80GB in qz Job
`gqc-h100-opt-1118-qb`, in the order base, reduced, reduced, base.
- **Base:** the parent snapshot, with a 32-thread energy and the walk unset
  (off at that time).
- **Reduced:** the same tree plus the 512-thread energy, run with
  `GENERATIVEQC_HF_BLOCK_CONVERGENCE=1`, the new default.
- **Shared changes:** both trees also contained the since-merged
  #3115/#3116/#3118/#3119/#3120/#3121 changes.

| Case (Direct) | base cold (s) | reduced cold | base warm | reduced warm |
| --- | --- | --- | --- | --- |
| HF 96 | 6.651, 6.664 | 6.435, 6.417 | 1.400, 1.401 | 1.372, 1.378 |
| HF 48 | 4.286, 4.229 | 4.204, 4.182 | 0.370, 0.369 | 0.364, 0.363 |

Warm values are medians of three calls.
- **Counts and qualification:** iteration and Fock counts are identical, and
  every arm qualifies.
- **Forces:** same-phase forces of the two arms agree within 5.7e-13 Eh/Bohr.
- **Single-change A/Bs:**
  - Walk alone: same-binary, environment-controlled. HF96 cold 6.60-6.63 →
    6.45-6.48 s.
  - Energy alone: separate build pair. HF96 cold 6.64 → 6.56-6.58 s, warm
    1.398 → 1.384 s.
- **Files:** `endpoint.json` holds every arm, sample and qualification error;
  `results.json.xz` adds all force vectors.

## Gates

- `native-tests.txt`: `generativeqc_cuda_force_convergence_tests
  --rhf-block-convergence` passes its 32/256 matrix-walk and energy-refinement
  controls. `generativeqc_uhf_tests --rhf-block-convergence-cache` passes its
  bidirectional cached-policy refusal. Both ran on the measured build.
- `memcheck.txt`: compute-sanitizer 12.9.79 memcheck over one water-12 Direct
  RHF energy+force call with the block walk reports 0 errors.

These measurements do not qualify any other GPU, and the selection has no
architecture switch.
