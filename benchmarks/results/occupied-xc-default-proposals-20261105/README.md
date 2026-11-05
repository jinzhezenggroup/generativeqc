# Automatic occupied-XC admission qualification

This capsule measures **actual unset controls**, not the historical explicit-1
route relabeled as the default. The native owner automatically admits bounded
occupied-feature initializers when eligible; explicit `0` and malformed values
retain matrix-only work. Original-D strict refinement, publication, forces,
precision and scientific gates are unchanged.

## Immutable source and experiment

- Scientific/runtime source: `ddadce1e29453e0f5cabc4c12948c7c798e8d66c`.
- Native source identity:
  `739538b11aec9053c6a8e2bd526f441f907f0d93ffc3d8824b1c4292b2583d99`.
- Library SHA-256:
  `be126dbe2567668cfb7e747177d71fd811f8f1c7d832e608f7aed5f0ca9adb92`.
- Release/C++20/O3/sm_120, CUDA 12.9.1, strict XC `--fmad=false`, verified
  sccache 0.16.0; existing shared caches retained. The incremental build compiles
  only the changed C++ selector and native identity, not CUDA producers.
- Slurm node1/main, RTX 5090, driver 580.95.05, November 5, 2026 local date.
  Every GPU command has a finite `srun` allocation and preserves its device
  visibility. Scheduler IDs, package versions and source/binary identities are
  retained in the receipts. No production CPU scientific oracle is introduced.

The only native/runtime source difference from the previously qualified
`07ec901f833c584518a6cc8481efef5649ba90ab` is the unset selector and its comment
in `src/dft/cuda_ks.cpp`. The source patch is retained. Earlier emitted-kernel
normal/memcheck/synccheck/racecheck results remain explicitly attributed to that
earlier source, not newly claimed for this selector commit.

Delivery also preserves the PR branch's independent integration of upstream
`0a214362fa9d762cf0a3a6be821d7cb5d185f257` and raw-output retention guards.
Those later integrations do not relabel the measurements as a merged-target
native image qualification. Current catalog source identities are regenerated
without upgrading any historical Libxc qualification claim.

A later source-review correction reserves two strict target evaluations inside
small caller iteration budgets and avoids provisional epochs when every AO
domain falls back to matrix products. Compiled host regressions cover those
bounds. The measurements above remain pinned to `ddadce1e2`; they are reused
for the unchanged numerical kernels and measured workloads, not relabeled as a
new GPU/performance run of that bounded admission correction. Receipt replay
also rejects duplicate oracle keys and force vectors jointly truncated from
the fixed three-atom small workload.

## Complete original 96-atom workload

Three alternating fresh-process matrix/default pairs use the original Direct
Standard r2SCAN-3c workload, unchanged grid and solver, complete native D4/gCP,
analytic forces and host export. Cold includes constructor, preparation, SCF
and E+F; imports and CUDA context creation precede that timer. Frozen warm
reuses the held cold state. Clean timings have no tracing/profiling enabled.

| Complete E+F | Matrix (`0`) median | Default (unset) median | Iterations/Focks, both |
| --- | ---: | ---: | ---: |
| Cold | 30.439342 s | 28.893960 s | 10/10 |
| Frozen warm | 11.216157 s | 11.179427 s | 1/1 |

Cold takes **5.077% less time (1.0535x)**, above the unchanged 2% floor from
`max(2%, 2 * sum(relative MADs))`. Warm differs by 0.33%, below that floor;
there is **no warm speedup claim** and no credible warm regression. This is not
an all-functional, molecule-size or trajectory-wide performance guarantee.

All 12 large E+F observations retain the existing audited GPU4PySCF grid7/8 and
CPU/GPU-spot reference gates: `1e-5 Eh/atom`, `1e-4` maximum and `3e-5` RMS
`Eh/Bohr`. The paired matrix/default gates remain `2e-9 Eh` and
`1e-7 Eh/Bohr`. Every full force vector and semantic work record is retained;
historical references, package contracts and scientific inputs are unchanged.

## Small complete endpoints and native lifecycle

`endpoints.json.xz` contains 80 complete E+F observations: default versus `0`,
five alternating repetitions, RKS/UKS, cold/warm/moved/moved-warm. Its v2 schema
records both cohort labels and actual environment values (`null` means absent).
Independent and paired gates are still `2e-9 Eh` / `1e-7 Eh/Bohr`; independent
maxima are `6.679e-13 Eh` / `3.155e-10 Eh/Bohr`. SCF/Fock work matches between
routes in all populations. No general small-system timing claim is made.

Native tests exercise default, explicit 1, 0 and malformed controls, canonical
frame lifetime, geometry changes, strict matrix tails, StrictAudit exclusion,
and independent E+F with production CPU derivatives forbidden. The capsule
retains normal and memcheck logs alongside host policy/lifecycle checks; earlier
236-case emitted-kernel sanitizer receipts are linked, not duplicated.

## Offline and device reproduction

```bash
PYTHONPATH=.:python python benchmarks/results/occupied-xc-default-proposals-20261105/replay.py
PYTHONPATH=.:python python -O benchmarks/results/occupied-xc-default-proposals-20261105/replay.py
```

The offline replay rechecks actual controls, source/binary identity,
independent and paired vectors, completeness, grid/solver identity, work and
the predeclared cost floor. Corrupted control, source, energy, force and missing
observations must fail even under optimized Python. The generic small-endpoint
reader preserves historical v1 explicit-1 receipts without changing their labels.
The large-reference qualifier requires NumPy, but not native/CUDA execution.

`qualification.json.xz` retains the complete Slurm script. To rerun, check out
the scientific source, build the identified Release CUDA configuration with a
verified compiler cache, select that fresh native image, and use fresh output
paths. Small endpoints use independently converged default-grid PySCF vectors;
large records use the existing audited dense-grid reference reuse contract.
The historical recovery and rejected implementations remain in the
[earlier capsule](../occupied-xc-audited-proposals-20261104/README.md).
