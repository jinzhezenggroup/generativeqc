# Fast hybrid proposal-source qualification

This compares the new source-screening default with its explicit opt-out on
master `ececb48cd56a00a56fbb6223dc53bc61dfe6e11d`, including #2692 and #2696.
Fast's bounded mixed stage uses `1e-8` integral-source screening; the primary
provider, fresh FP64 closing pair and analytic forces retain the requested
screening. The [decision note](../../../.agents/notes/implemented/performance/2026-10-23-fast-provisional-source-screening.md)
records the model, size, resource and lifetime boundaries.

Five fresh alternating off/default process pairs per size each include
construction, preparation, cold energy/analytic forces and two frozen warm
energy/force calls. Warm results first take each process's median, then the
median across processes. No sample is excluded or selected for low latency.

| Atoms | Phase | Off median (s) | Default median (s) | Time reduction |
| ---: | --- | ---: | ---: | ---: |
| 12 | cold | 2.398919 | 2.405251 | -0.264% |
| 12 | warm | 0.382575 | 0.380022 | 0.667% |
| 48 | cold | 10.196071 | 9.546306 | **6.373%** |
| 48 | warm | 2.307475 | 2.314476 | -0.303% |
| 96 | cold | 27.642817 | 25.128536 | **9.096%** |
| 96 | warm | 6.527128 | 6.534383 | -0.111% |

The predeclared gain/nonregression floor is max(2%, twice the sum of relative
median absolute deviations). Only 48/96 cold gains exceed that floor. The
12-atom and warm differences do not establish a gain or regression. Cold Fock
counts are 10/9/9 at 12/48/96 atoms; frozen warm calls use one. All 25 paired
controls have identical phase, iteration and physical Fock counts.

## Work, scientific gates and fallbacks

At 96 atoms, cached integral values decline from 1,476,473,708 to 1,035,084,412
(29.90%), avoiding 1,765,557,184 bytes of FP32 value storage. Seven mixed Focks
and two strict closing Focks remain. Separate component profiles confirm less
source/consumer work and unchanged strict work. Their inclusive intervals are
not summed into endpoint timings or pooled with clean samples.

The 50 retained processes contain **180 complete energy/force endpoints**:

- 30 clean processes / 90 calls;
- 10 lifecycle processes / 60 calls, including PBE0/B3LYP/M06-2X at 12 atoms
  and PBE0 at 48/96, original/moved/frozen-moved/reset states;
- six excluded-owner processes / 18 Standard Direct and Standard/Fast DF calls;
- two tiny-source-capacity processes / six calls; and
- two diagnostic processes / six calls, excluded from clean timing.

Every call passes independent tight full-density GPU4PySCF references. Native
owners initialize and reconverge their own densities. The maximum errors are
`1.84083e-10 Eh/atom`, `1.13717e-5 Eh/Bohr` maximum force and
`3.38396e-6 Eh/Bohr` RMS force, below the unchanged gates of
`2.5e-6`, `2.5e-5` and `7.5e-6`, respectively. This qualifies solver behavior
on stock grids and sampled geometries; it is not a grid-convergence or universal
functional-performance ranking.

The native incremental, DF cold-factor and Fast anchor suites and eight public
GPU cases pass on the final library. Public regressions compare automatic/forced/disabled residency
and off/on source screening with independent PySCF forces, including frozen
and moved replay. Additional tests cover two-owner batches, reset, a one-byte
source cap, a public resource ledger and invalid selector handling.

## Provenance and reproduction

Final core SHA256:
`378e026ff3ea23ff48987147df8aa1795d881728e3fd1657c52389a00bb3fb90`.
All 2,429 production/build/driver inputs match the retained source manifest,
whose digest is
`eff667bbc389dc0bc723751a195d15cf07f253abc7e43d19a7c0c66691c765f1`.
The build uses GNU 13.3.0, CUDA 12.9.86, Release `120-real`, PCH disabled and
verified sccache 0.16.0 launchers. Compiler commands and before/after cache
statistics are retained. PBE0 force AOT reuse passed contract and binary-hash
checks; matching B3LYP/M06-2X force AOT was rebuilt. Their identities are in
the receipt.

The qualification ran in finite Slurm job 13580 on n1/node1 RTX 5090,
preserving assigned `CUDA_VISIBLE_DEVICES=1` and one numerical host thread.
Off means `GENERATIVEQC_KS_MIXED_SOURCE_COARSE_SCREENING=0`; default leaves it
unset. Every raw endpoint records that control and the loaded binary identity.

```bash
python benchmarks/results/fast-source-screening-20261023/reproduce.py
```

This recomputes all 180 observable checks, required populations, per-process
statistics, the gain/nonregression gates and paired work comparisons from
`qualification.json.xz`. The archive also preserves the independent references,
source/AOT identities, exploratory samples and compact diagnostic summaries.
Full build/test logs, profiler streams, binaries and finite-Slurm runners remain
in ignored local `.artifacts/` and
`n1:/data/jzzeng/fast-df-next-20261023/`.

The first exploratory rebuild reused stale runtime objects because copied
source timestamps predated the new baseline objects. Its selector/counter was
absent from the library; those 21 calls remain explicitly marked invalid
mechanism evidence. They are not candidate timing evidence. After recompiling
the changed runtime, the separate two-pair pilot showed a 9.55% cold reduction;
those 21 calls and the three-call untouched baseline smoke remain separate from
the final 180-call qualification. Initial AOT identity refusals and the build
correction are retained, without weakening any provenance or numerical gate.

## Storage-only compaction

The existing aggregate evidence limit remains 64 MiB. This change losslessly
recompresses two historical JSON records from gzip to XZ, saving 302,954 bytes:
`cpu-eri-coulomb-20261001/pilot-campaign.json.xz` and
`cuda-ownership/one-electron/samples.json.xz`. Their decoded bytes match the
original files at `ececb48cd56a00a56fbb6223dc53bc61dfe6e11d` exactly. Publication
and retention hashes follow the new storage; all samples, failures and scientific
identities remain intact. Existing readers already support both formats.
