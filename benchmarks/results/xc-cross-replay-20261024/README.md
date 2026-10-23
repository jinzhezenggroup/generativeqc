# Direct DFT cold: XC cross tile replay

RTX 5090, n1/node1, 2026-10-24. Baseline:
`c1243a9d6ca693d7cb4058605ce1e191e2d8ce22` (ordered DIIS, PR #2714).
Candidate checks FP64 cross-potential range at point-tile boundaries,
replaying nonfinite tiles with the old per-term fallback. Precision, local
FMA order, grids and solver policy are unchanged. See the
[decision note](../../../.agents/notes/implemented/performance/2026-10-24-xc-cross-tile-replay.md).

## Complete endpoints

Three fresh-process rounds rotate baseline/candidate/stock GPU4PySCF order.
Timers include constructor, preparation, SCF, analytic forces and host export,
then three warm calls from each owner's frozen cold density. Imports/AOT
compilation are excluded; no reference density enters production. All samples
remain, including candidate PBE's first cold at 20.4954 s. This default-grid
comparison is not a globally fastest-configuration ranking.

| Direct / 96 atoms | Baseline cold (s) | Candidate cold (s) | Reduction | GPU4PySCF cold (s) | Remaining gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| PBE | 19.9984 | 18.3808 | 8.09% | 16.1515 | 13.80% |
| PBE0 | 23.5646 | 22.1812 | 5.87% | 20.1248 | 10.22% |
| r2SCAN-3c | 44.3808 | 42.8280 | 3.50% | 32.4612 | 31.94% |

Medians exceed the predeclared reduction floor
`max(2%, 2 * (baseline MAD / median + candidate MAD / median))`, 2% here.
All timings, MADs, ranges and per-process warm medians are retained. Native
cold iterations/reported Fock builds remain 11/11, 9/9 and 10/10; warm is 1/1.
Reference cycles are 11/9/11. Reference Fock counts and detailed native
quartet/J/K counts remain unavailable. Native grids contain 1,078,336 points;
reference grids, actual controls and work receipts remain in raw storage.

All-nine-sample warm medians (baseline/candidate/reference) are
5.4605/5.3471/5.5814, 6.9955/6.8215/6.7364 and 15.8413/15.6636/14.8543 s.
3c's 1.12% warm reduction is below the floor. OMP/OpenBLAS/MKL each use one
thread. Other jobs used other GPUs on this shared host. Slurm jobs: clean
13629, qualification 13637, profile 13638, stability follow-up 13641.

## Mechanism

A separate 3c cold-plus-warm profile reduces
`batch_local_diagonal_potentials<double>` from 5.7735 to 4.0725 GPU seconds
over the same 2,904 launches (29.46%). Exclusive SCF wall time falls from
22.6152 to 21.1000 s cold and 2.0809 to 1.9269 s warm. Density contraction
remains about 4.3665 GPU seconds; the two leading integral-force kernels
still total 12.5877 GPU seconds across both calls. Cold preparation costs
4.5625 wall seconds. Profiles are excluded from clean timing; these costs
are not additive estimates of the gap to a differently structured engine.

The isolated old/new CUDA probe covers 64/128/256/384 active AOs, one/four
jets, 16 batches, 256 points, four alternating rounds and twelve launches
per sample. Kernel reductions are 27–65%, with bitwise equal outputs.
Whole-endpoint bitwise equivalence is not claimed.

## Qualification and limitations

All 166 planned E/F calls pass: 108 clean, 40 holdout (Standard 3/12/48,
Fast/96, DF/96), 18 cold/warm/moved. Static checks reuse audited dense
references in [readme-final-20261022](../readme-final-20261022/README.md),
with 1e-5 Eh/atom, 1e-4 maximum and 3e-5 RMS Eh/Bohr gates. Moved tests
shift atom 1 by 0.02 Bohr in z, with fresh tighter full-density GPU4PySCF
references after native timing and 2.5e-6/2.5e-5/7.5e-6 gates. These are
matched-default-grid regressions, not new grid-convergence evidence.
Ten diagnostic and four profile calls qualify separately.

One moved-cohort 3c candidate cold call, with coarse FP32 screening disabled,
takes 13 iterations versus ten: DIIS creates a large density excursion at
iteration ten, then recovers. Independent E/F gates pass, but its paired
force difference 3.988e-6 fails the optional 1e-8 diagnostic. Fixed two-round
follow-up with both binaries returns ten cold iterations in all four runs;
all twelve calls pass the same geometry-bound references. The original
outlier remains. Fast/3c also fails the optional diagnostic at 2.279e-8.
Neither cohort is pooled into default timings; no claim of deterministic
SCF follows from preserving local FMA order.

144 CUDA cases pass, including 36 new cancellation cases across tile
boundaries, mapped AO tails, both spins and both implicit-FMA settings.
Memcheck/synccheck each pass 64 selected cases with zero errors. Four native
E/V/batching/spatial-grid/matrix-schedule suites pass. Focused host checks:
148 passed, ten CUDA-only skips. Compiler structure and pre-commit pass.
Two-spin kernel coverage does not establish UKS molecular performance.

## Provenance and rechecking

CUDA 12.9.1, GCC 13.3, Release/sm120, tile16, `--fmad=false`, FastCompile
OFF; verified sccache 0.16.0, reused cache and launcher-bound commands.
Cache statistics are shared-node totals. The build verified 6,000 inputs.
Two final tests differ only by Python formatting and a C++ comment; exact
versions/diffs are retained. All other 5,998 inputs, including production,
match the snapshot. Receipts name the common base; the manifest/patch binds
the candidate. Binary hashes and source identities are checked.

Tracked `summary.json.xz` retains timings, work, qualification maxima and
identities. Exact-text `records.json.xz` and `sources.json.xz` remain at
`n1:/data/jzzeng/direct-dft-next-20261024/`, with checksums in the summary,
alongside binaries, commands, scripts, logs and profiler reports. This
respects the aggregate evidence cap; no sample was removed or replaced.

Copy both bundles to `.artifacts/remote/` (or pass `--raw-directory`), then:

```bash
PYTHONPATH=python:. python benchmarks/results/xc-cross-replay-20261024/recheck.py
```

This host-only reader requalifies every call, checks reference/receipt
identities and recomputes timing statistics, also under `python -O`. It
validates historical evidence without pinning current production source.
