# FP32-only pair-axis qualification, 2026-10-16

Extend cooperative Hermite-axis preparation only to the shared mixed Direct
J/K specialization. FP64 keeps its existing specialization; no functional-name
dispatch, force-grid scheduling change or numerical gate change is introduced.

Complete 96-atom/768-AO WB97M-V energy/force endpoints, spherical def2-SVP,
`GridSpec(48,16,32)`, fresh processes/calculators/hcore densities. Creation, SCF,
forces, export, teardown and actual runtime preparation remain timed. Only
separately recorded build attestation is excluded. All calls are retained.

| Same-device pair | Previous Fast (s) | FP32-axis Fast (s) | Old/new |
| --- | ---: | ---: | ---: |
| n4, job 1070 | 788.762789 | 753.150144 | 1.047285 |
| n5, job 1659 | 795.201434 | 758.461470 | 1.048440 |
| n4, job 1072 | 749.660461 | 697.617563 | 1.074601 |

Median paired ratio: **1.048440**, three positive pairs. Each row uses one
physical GPU; rows are not pooled into a cross-device timing denominator.
All six Fast returns pass independent GPU4PySCF energy/max-force/RMS gates
`1e-5 / 1e-4 / 3e-5` in Eh and Eh/bohr. Each executes the same 21 Focks
(17 mixed, four strict), identical operator work and zero execution retries.

The source-matched n4 strict control takes 1063.483326 seconds with 34 strict
Focks and passes its independent `1e-7 / 1e-6 / 3e-6` gates. It is one control,
not a median FP64/Fast ratio or equal-work arithmetic-throughput measurement.
Separate 96-atom PBE/PBE0 repeats are neutral; no gain is claimed for them.

`summary.json.xz` retains compact timings, observable errors, work counts, actual
devices/drivers and source/core/receipt digests. The retained accuracy oracle's
historical timing is never used. The third population was explicitly amended
before observing it after the generic v7 trial failed its predeclared gate;
all v7 evidence remains separate outside Git. No failed or slower Fast sample
is removed from this axis population.

The summary is losslessly compressed; `xz -dc summary.json.xz` reproduces the
original JSON bytes (Git blob `51352dd114b1aae7540c4f2b8201d117349161ad`).
This storage change does not alter qualification or numerical evidence.

Qualification base: `072ba064643a21f470cd889c965475e2d1f8475c`, not current-master
integration qualification. Raw matrices, full receipts, failures and scripts
stay outside Git in the existing local evidence tree. No Release or raw bundle
is published. The separate Boys-table FP32 trial is not measured by this report.

## PR-head integration

`integration.json` independently qualifies the exact PR source at `a226b3cc0`,
based on `8d7246d25`. This is not a qualification of later upstream commits.
The rebuilt core and all six official force manifests authenticate the same
source identity. Node2 job 3270 passes all-order mixed J/K, exchange-only,
strict/derivative/response coverage and memcheck with zero errors. Its retained
NVML mismatch limits that receipt to functional validation; two initial compiler
environment failures are recorded separately rather than counted as passes.

On the clean n5 RTX 5090, job 1668 independently qualifies WB97M-V96 Fast and
FP64 complete endpoints (757.880654 and 1071.574263 seconds), plus both modes
for eight method families at three and twelve atoms: 34 complete returns.
All Fast returns execute genuine FP32 work with FP64 accumulation, one qualified
budget solve and zero retries. The WB96 pair is an integration control, not a
repeat-median performance cohort. Only the historical three-pair table supports
the axis speedup claim. Full receipts and the recomputing audit stay outside Git.
