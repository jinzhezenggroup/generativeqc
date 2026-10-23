# Conservative Direct block-screening qualification

FP32 coarse screening preserves the tested FP64 task set and observables, but
shows no meaningful molecular speedup. Keep
`GENERATIVEQC_DIRECT_FP32_BLOCK_SCREENING` default-off. INT8/WMMA remain
standalone experiments.

## Provenance and protocol

Base `a524db91d` plus this change; native source identity
`67542bfded3279389c27a336e5f65fac4633724a2e7371f779513c9910d8521a`;
loaded core SHA256
`631a5cb7e89d67b5698671e0bcb74f929d2537bbcef1f20dbb64104320302194`.
All endpoints use this image. Driver/test and AOT hashes are retained.

Node n1, RTX 5090, CUDA 12.9.1, Release sm_120, ccache 4.5.1. All device work
used finite Slurm `main` / `gpu:5090:1` allocations with scheduler visibility.
PBE0/def2-SVP and r2SCAN-3c (including D4/gCP) use 96-atom water clusters.
C7 is pinned QM7 C7H16 Direct RHF/def2-SVP, **not** full CCSD(T)/CBS.

Three rounds × two arms × three systems: 18 fresh processes, each with cold
and three frozen warm E/F calls. Round one also displaces one atom by 0.02 Bohr;
round two reverses arm order. Total: 78 complete Standard FP64 Direct E/F calls.
Cold includes constructor/preparation, after imports/CUDA context initialization.
Warm freezes each owner's own cold density. Clean timers exclude profiling.

Tighter GPU4PySCF references run after first-round candidate timing, with D4/gCP
for r2SCAN-3c. Every sample of both arms was compared to its original/moved
reference before compaction. DFT grids match defaults; this is not a grid study.

## Endpoint results

Median seconds, three cold and nine warm samples per cell:

| System | FP64 cold | FP32 cold | FP64 warm | FP32 warm |
| --- | ---: | ---: | ---: | ---: |
| PBE0-96 | 29.370681 | 29.391279 | 7.925767 | 7.932827 |
| r2SCAN-3c-96 | 51.243392 | 51.298567 | 17.848709 | 17.871032 |
| C7 RHF | 3.084085 | 3.077435 | 0.285214 | 0.285204 |

Both arms have cold SCF/Fock counts 9/9, 10/10 and 10/12 respectively; warm is
1/1 throughout. Moved geometry has one sample per arm: correctness/invalidation
coverage, insufficient for a speed claim. The reader reports ranges and moved
timings. Worst cross-arm differences are 1.819e-12 Eh and 1.619e-11 Eh/Bohr.

Worst independent-reference errors across all samples:

| System | Energy/atom (Eh) | Force max (Eh/Bohr) | Force RMS (Eh/Bohr) |
| --- | ---: | ---: | ---: |
| PBE0-96 | 7.011e-13 | 9.348e-7 | 2.053e-7 |
| r2SCAN-3c-96 | 5.244e-12 | 2.005e-6 | 7.112e-7 |
| C7 RHF | 6.006e-13 | 4.515e-8 | 1.299e-8 |
| Gate | 2.5e-6 | 2.5e-5 | 7.5e-6 |

Separate diagnostic cold/warm DFT force censuses agree in every work field:
PBE0 admits 92,223,676 quartets and rejects zero block pages; r2SCAN-3c admits
243,123,642 quartets and rejects 30,112 pages. These are force-owner counts,
not SCF totals. C7 uses another owner: its quartet census is unavailable, not
zero. Its kernel launch counts and SCF/Fock counts are retained.

Selected diagnostic GPU milliseconds per call:

| Kernel / system | FP64 cold | FP32 cold | FP64 warm | FP32 warm |
| --- | ---: | ---: | ---: | ---: |
| Spatial low force / PBE0 | 1374.721 | 1376.320 | 1374.538 | 1381.059 |
| Spatial low force / r2SCAN | 3859.749 | 3865.864 | 3859.518 | 3865.910 |
| Generated task compaction / C7 | 0.319 | 0.360 | 0.318 | 0.360 |
| Exact low force page / C7 | 33.945 | 33.954 | 33.960 | 33.955 |

Full selected kernel names/counts are in `profiles`. Screening is fused with
compaction or integral/contraction work, so these are not isolated screening
timings. Do not sum them into clean wall time. Schwarz remains FP64. Profile
scopes separate prepare/cold/warm; NVTX cold covers execution while clean cold
also includes prepare.

## Integer experiment

All six modes match host FP64 task identities. Tile-local power-of-two scales
and upward INT8 codes in [0,64] enclose bounds; survivors receive FP64 refinement.
Actual WMMA (`IMMA.16816.S8.S8` observed in SASS) pads an outer product with
15 zero K lanes. Tiled scalar INT8 has identical warp/refinement scheduling.

Three process repeats × 30 timed iterations, 256/2048/8192 bounds, Fock and force
gates, sparse and all-survive thresholds. GPU timers include packing, mask writes
and refinement; allocation, transfer and host checks are outside. The dense
square mask is synthetic, not a production bounded queue.

Median force-screening milliseconds at 8192 bounds / 67,108,864 candidates:

| Mode | Sparse (tau=1e-12) | All survive (tau=0) |
| --- | ---: | ---: |
| FP64 | 0.549117 | 0.561439 |
| FP32 per visit | 0.600374 | 1.468697 |
| Prepared FP32 | 0.341929 | 1.015654 |
| Flat INT8 | 0.202737 | 1.008693 |
| Tiled INT8 | 0.153578 | 0.541374 |
| INT8 Tensor Core | 0.131090 | 0.542804 |

Sparse survivors: 2,673,008 in every mode. WMMA is about 15% faster than tiled
INT8 here, with no benefit in the all-survive control. Integer modes lose at
256 bounds. Prepared metadata merits experiments; these results establish
neither molecular speedup nor a qualified 12-byte density ABI.

## Validation and retained evidence

Both FTZ modes pass 3,061,731 cases against sequential host FP64 and an
extended-precision rejection oracle: zero task mismatches, including thresholds,
subnormals, overflow, infinity and NaN. Independent UHF force regression passes.
Compute Sanitizer reports zero errors for arithmetic and INT8/WMMA probes.

`manifest.json` binds `receipts.json.xz`: every endpoint timing/energy/scalar
error, selected kernel rows, full DFT census, synthetic observations, validation
counts, identities, cache statistics and reproduction scripts. Full force arrays,
logs, traces and excluded trials stay in local artifacts. Force errors and
cross-arm maxima were computed before compaction; the reader rechecks scalar
gates but cannot reconstruct those arrays. Raw-record hashes bind the samples.

Pilots are excluded. Inconsistent scalar/WMMA densities failed task checks and
were fixed; malformed printf metadata was discarded before rebuilding with
`-Werror=format`. Neither trial enters accepted timings.

## Reproduction

Recompute summaries and scalar gates without a GPU:

```bash
python3 benchmarks/results/direct-screening-20261023/summarize.py
```

The receipt's `reproduction` object has the exact build/environment, campaign,
profile and GPU-test scripts. Adjust paths, verify ccache, configure CXX/CUDA
launchers and build core plus PBE0/r2SCAN stationary AOT manifests. Run scripts
with `srun --partition=main --nodelist=n1 --gres=gpu:5090:1 --nodes=1 --ntasks=1`
and a finite `--time`; preserve scheduler visibility. Use 60 minutes for the
campaign, 20 for profiles and 10 for GPU tests. C7 needs the pinned QM7 dataset.
Compile the probe with ccache + NVCC C++20/O3/sm_120 and `-Werror=format`; run
all sizes with tolerances `1e-12` and `0` through five-minute allocations.
See the [decision note](../../../.agents/notes/implemented/performance/2026-10-23-conservative-block-screening.md).
