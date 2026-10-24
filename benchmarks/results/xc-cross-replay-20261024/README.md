# Historical Direct DFT cold: XC cross tile replay

RTX 5090, n1/node1, 2026-10-24. Baseline:
`c1243a9d6ca693d7cb4058605ce1e191e2d8ce22` (ordered DIIS, PR #2714).
Candidate checks FP64 cross-potential range per point tile; nonfinite tiles
replay the old per-term fallback. Precision, local FMA order, grids and solver
policy are unchanged. See the
[decision note](../../../.agents/notes/implemented/performance/2026-10-24-xc-cross-tile-replay.md).

## Complete endpoints

Three fresh-process rounds rotate baseline/candidate/stock GPU4PySCF. Timing
covers construction, preparation, SCF, forces and host export; three warm calls
reuse frozen cold density. Imports/AOT compilation are excluded; production uses
no oracle density. All samples remain, including first candidate PBE cold
(20.4954 s). Default-grid results are not a global ranking.

| Direct / 96 atoms | Baseline cold (s) | Candidate cold (s) | Reduction | GPU4PySCF cold (s) | Remaining gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| PBE | 19.9984 | 18.3808 | 8.09% | 16.1515 | 13.80% |
| PBE0 | 23.5646 | 22.1812 | 5.87% | 20.1248 | 10.22% |
| r2SCAN-3c | 44.3808 | 42.8280 | 3.50% | 32.4612 | 31.94% |

Medians exceed the predeclared reduction floor
`max(2%, 2 * (baseline MAD / median + candidate MAD / median))`, 2% here.
Timings, MADs, ranges and warm process medians are retained. Native
cold iterations/Fock builds are 11/11, 9/9 and 10/10; warm is 1/1.
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
over 2,904 launches (29.46%). Exclusive SCF wall time falls from
22.6152 to 21.1000 s cold and 2.0809 to 1.9269 s warm. Density contraction:
4.3665 GPU seconds; two leading integral-force kernels: 12.5877 GPU seconds
across both calls; cold preparation: 4.5625 wall seconds. Profiles are excluded
from clean timing; costs cannot be added to estimate inter-engine gaps.

The old/new probe uses 64/128/256/384 active AOs, one/four jets, 16 batches,
256 points, four alternating rounds and twelve launches/sample. Kernel gains
are 27–65% with bitwise-equal outputs; endpoints need not be bitwise equal.

## Qualification and limitations

All 166 planned E/F calls pass: 108 clean, 40 holdout (Standard 3/12/48,
Fast/96, DF/96), 18 cold/warm/moved. Static checks reuse audited dense
references in [readme-final-20261022](../readme-final-20261022/README.md),
with 1e-5 Eh/atom, 1e-4 maximum and 3e-5 RMS Eh/Bohr gates. Moved tests
shift atom 1 by 0.02 Bohr in z, with fresh tighter full-density GPU4PySCF
references after native timing and 2.5e-6/2.5e-5/7.5e-6 gates. These are
matched-default-grid regressions, not new grid-convergence evidence.
Ten diagnostic and four profile calls qualify separately.

One moved-cohort 3c candidate cold call with coarse FP32 screening disabled
takes 13 iterations versus ten after a DIIS density excursion at iteration ten.
Independent E/F gates pass; paired force difference 3.988e-6 fails the optional
1e-8 diagnostic. Two-round follow-up with both binaries gives ten cold iterations
in all four runs; all twelve calls pass the same geometry-bound references.
The outlier remains. Fast/3c also fails the optional diagnostic at 2.279e-8.
Neither cohort enters default timings; local FMA order does not ensure
deterministic SCF.

144 CUDA cases pass, including 36 new cancellation cases across tile
boundaries, mapped AO tails, both spins and both implicit-FMA settings.
Memcheck/synccheck each pass 64 selected cases with zero errors. Four native
E/V/batching/spatial-grid/matrix-schedule suites pass. Focused host checks:
148 passed, ten CUDA-only skips. Compiler structure and pre-commit pass.
Two-spin kernel coverage does not establish UKS molecular performance.

## Provenance and rechecking

CUDA 12.9.1, GCC 13.3, Release/sm120, tile16, `--fmad=false`, FastCompile OFF.
Verified sccache 0.16.0 wraps compilation; cache stats are shared-node totals.
Of 6,000 verified inputs, two tests later changed formatting/a C++ comment
(diffs retained); the other 5,998, including production, match. Receipts name
the base; manifests/patches bind candidate, binary and source.

`summary.json.xz` stores timing/work/qualification/identity summaries and bundle
checksums. Exact `records.json.xz` and `sources.json.xz`, binaries, commands,
scripts, logs and profiles remain at `n1:/data/jzzeng/direct-dft-next-20261024/`.
The evidence cap is unchanged; no sample is removed or replaced.

Copy both bundles to `.artifacts/remote/` (or pass `--raw-directory`), then:

```bash
PYTHONPATH=python:. python benchmarks/results/xc-cross-replay-20261024/recheck.py
```

This host reader checks every call, reference/receipt identity and statistic,
including under `python -O`, without pinning current production source.

[Subsequent DIIS repair and fresh endpoint requalification](../../../.agents/notes/implemented/numerics/2026-10-24-normalized-diis-affine-guard.md)
retain this campaign and its limitations as historical evidence.
