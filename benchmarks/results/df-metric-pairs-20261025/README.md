# Auxiliary metric derivative pair reuse

Complete contiguous metric panels now evaluate each unordered auxiliary pair
derivative once and scatter both original weights. This preserves arbitrary
nonsymmetric adjoints and the generated scientific recurrence. Partial, offset,
strided, serial and distributed-gradient panels retain the bounded dense route;
`GENERATIVEQC_DF_METRIC_DERIVATIVE_PAIRS=0` explicitly rolls back.

## Complete endpoint observations

Three alternating fresh-process rounds retain **36 populations / 144 complete
energy-and-force calls** for Standard/Fast PBE48/96 and r2SCAN-3c96. Cold includes
construction, preparation, SCF, forces and host export. Warm retains three calls
from each frozen cold density. [All medians and ranges](timings.md) remain visible.

| Method | Atoms | Profile | Control cold (s) | Candidate cold (s) | Control warm (s) | Candidate warm (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| PBE | 48 | Standard | 6.137 | 6.093 | 1.880 | 1.828 |
| PBE | 48 | Fast | 5.515 | 5.468 | 1.619 | 1.570 |
| PBE | 96 | Standard | 16.337 | 16.120 | 6.070 | 5.877 |
| PBE | 96 | Fast | 15.311 | 15.074 | 5.662 | 5.453 |
| r2SCAN-3c | 96 | Standard | 34.654 | 34.065 | 11.066 | 10.554 |
| r2SCAN-3c | 96 | Fast | 32.225 | 31.721 | 9.956 | 9.434 |

The small PBE cold differences are scoped observations. The r2SCAN-3c96
gain is approximately 0.5 s in the complete force endpoint. This does not
establish parity with GPU4 or replace the complete README figure cohort.

## Work and correctness

Diagnostic traces reduce PBE48 metric derivative tasks from 3,268,864 to
1,635,336; PBE96 from 13,075,456 to 6,539,536; r2SCAN-3c96 from 46,457,856
to 23,232,336. Dense weights/storage and every original contribution remain.
Warm metric contraction time changes 104.3 -> 52.8 ms, 371.9 -> 186.7 ms,
and 1016.1 -> 507.9 ms respectively. These intrusive region timings explain
the work reduction and are excluded from the endpoint table.

All paired samples preserve SCF iterations, Fock counts and scientific settings.
The maximum paired force change is 3.80e-8 Eh/Bohr, within the existing 3e-7
Fast gate; Standard retains a 1e-8 gate. Energy change is bounded by 1e-8 Eh.
All **72 candidate sample gates** pass against independent GPU4 references.
All **30 original/moved/property checks**, including twelve same-owner rollback
comparisons, pass common gates of 1e-5 Eh/atom, 1e-4 maximum and 3e-5 RMS force
in Eh/Bohr. These unchanged observable gates do not reclassify earlier README
default-grid accuracy failures.

Final related tests: **166 pass** (123 broad response cases, 30 full g-auxiliary
endpoints, 12 native g/range-separated checks and one invalid-control recovery
case). They cover independent arbitrary/skew/one-sided/diagonal weights,
Cartesian/spherical/mixed bases, both moving auxiliary centers, full/split
panels, finite differences, resource limits and original fallback routes.
Host resource-identity checks independently cover the new rollback control.

The original broad run passed 119 and failed four assertions. All four failures
reproduce on the unchanged control: current RHF can fit in 4 MiB, and the current
manifest qualifies auxiliary class 003 for Rys. The revised tests retain the
1-MiB rejection, independently qualify the 4-MiB endpoint with finite differences,
and retain per-class manifest and full-force comparisons. No tolerance is relaxed.
The original broad run's 30 g-endpoint skips were caused by its absent opt-in
gate; the separately enabled run passes all 30. Original failures and reruns remain.

## Provenance and reproduction

Control: `fba34e9192edf0ea0d5baf6e3cd1f0b9ab14dc5d`.
Candidate: `a55214c7c458694aaa93a5b533e71b842f3a73d6`.
Both contain the preceding HF and response-capacity changes and current master
at measurement time. Each 12,796-file measurement tree matches its Git archive;
the candidate also matches every actual incremental-build input. Core/AOT hashes,
verified sccache 0.16.0, compiler commands and cache statistics are retained.
CUDA 12.9.1, GCC 13.3, Release sm120, node1 RTX 5090 and finite Slurm jobs with
assigned device visibility, four CPU cores and one BLAS/OMP thread are recorded.
Later edits add tests, documentation and control provenance only.

```bash
PYTHONPATH=python:. python benchmarks/results/df-metric-pairs-20261025/reproduce.py \
  --output .artifacts/df-metric-pairs-replay
```

This replays every paired and independent gate and regenerates the table from
lossless receipts. GPU reexecution uses `jobs.json.xz` after adapting installation
paths and preserving finite `srun`. Fixed auxiliary tables remain in the preceding
[README campaign](../df-dft48-work-20261025/README.md). Source archives, libraries
and full traces remain in ignored node1 artifacts; no release assets are needed.

The separate `tile-trials.json.xz` retains all nine exploratory force-grid tile
attempts. PBE improves at 512 points, while r2SCAN-3c96 exceeds the additional
host workspace budget. Tile 128 regresses all three. These diagnostic wrapper
results do not change the production tile default or enter the endpoint table.
