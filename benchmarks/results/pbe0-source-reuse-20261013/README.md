# Small-hybrid exact FP64 source reuse

These are retained measurements of the native policy submitted in this change,
not a mixed-precision Fast qualification. The unrelated experimental facade is
not submitted. Its historical `fast` rows remain in the raw files for integrity;
use the ordinary `fp64` rows when evaluating this optimization.

## Complete endpoint evidence

Node n1, RTX 5090, finite Slurm job 7980. PBE0/RKS, spherical def2-SVP, no DF,
Hcore guess on every call; grid (48,16,32), unit radii, three Becke passes,
no pruning; energy/density/screening controls 1e-10/1e-8/1e-12.
Five alternating-order measured repetitions follow retained first calls.
Timing includes construction, SCF, all analytic forces, host export, release
and CUDA synchronization. Process caches are warm, densities are not.

| System | AOs | Explicit no-reuse control (s) | Ordinary FP64 default (s) | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Water | 24 | 2.114368 | 1.758106 | 1.203 |
| Water tetramer | 96 | 6.120576 | 3.401016 | 1.800 |

The no-reuse control reproduces the former small-system schedule; it is **not
the current FP64 default**. One SCF and force evaluation run in each endpoint.
Water/tetramer have 13/14 Fock builds; zero mixed builds. The retained sources
construct 52,975/9,471,311 canonical ERI values once. These are source-value
counts, not primitive FLOPs; a complete no-reuse ERI work counter is absent.

Independent PySCF 2.11.0 SCF and full moving-grid gradients supply the reference.
All samples pass 1e-7 Eh energy and 1e-6 Eh/bohr maximum force-component gates.
The oracle is separate from production and never supplies initial density.
These discrete-model gates do not bound basis/grid error or qualify NVE MD.

## Source identity and reconstruction

Measurements used master 4f4c1aeb0bd32baf4bbdda276947b7ce911c453d plus the
uncompressed `runtime.patch.xz`. The submitted native change is identical;
subsequent master integration is not relabeled as a new measurement.

- Native source: b72ce4fd6ac8046b6b35a8eeb4641ab44f65113b71320025ee9bc5012b7731a2
- Library SHA256: c6cd6e18f6a00a0669fd51e4d74b435b22c99717299530b1c0ed3f2b90e2f7df
- Uncompressed patch SHA256: 4924ef22f75cd9f1eb802a06dcc3bc68c7af5b717ffc48a5a879e0798accad5f
- Uncompressed reference SHA256: f764b2350565182729fab3a58dac3b928baf6f781ecd7187f6bc1b9a0b833bcd

The patch archives both benchmark drivers and the historical facade along with
native changes. To reproduce the **historical experiment**, apply it in a
separate checkout of that base, not on top of this PR. Build with the retained
CMake settings (Release, CUDA 12.9.1, SM120, fast-compile OFF, native stationary
and PBE0 stationary AOT ON). Decompress reference.json.xz and run:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:15:00 \
  python -m benchmarks.pbe0_fast --repeats 5 \
  --reference reference.json --output native.json
```

Use an allowed node n1/n2/n4/n5 and preserve Slurm-assigned visibility. Set
OMP_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=4 and select the matching library.
The retained CUDA host-link override only repairs the original relocated
toolchain; a fresh local build does not require it. Repeated compilation used
ccache. Imports, building and probes are not part of endpoint medians.

## Validation receipts

- Four small-source GPU tests pass on the final binary (Slurm 7982): independent
  signed RKS/UKS J/K, partial/full residency, PBE0/B3LYP states, warm/frozen/
  external seeds, FP64/AUTO refinement, chunk widths 1/2 and capacity guards.
- A complete resident PBE0 gradient test passes (Slurm 7980), including its
  no-production-CPU-derivative guard and independent reference.
- Final compute-sanitizer memcheck (Slurm 7982): zero errors, zero bytes leaked.
  Instrumented timings are not used for speed claims.

Raw JSON, logs and the CMake cache are preserved byte-for-byte. Historical
`strict_fp64` facade rows demonstrate no independent Fast advantage and must
not be reinterpreted as a successful mixed-precision implementation.
