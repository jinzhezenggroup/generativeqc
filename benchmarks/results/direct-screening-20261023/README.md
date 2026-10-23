# Prepared conservative Direct block screening

Prepared FP32 intervals remove repeated conversion and FP64 refinement for
proven survivors. Complete molecular timings remain close; the selector
`GENERATIVEQC_DIRECT_FP32_BLOCK_SCREENING` stays default-off. INT8/WMMA remain
standalone experiments. This replaces the initial rejection-only report,
recoverable at commit `2fe65ffff7c92ef6b7c98057e21dd95f5b715315`.

## Source and protocol

Base `1e77b239e551856741abedc0b45a6e8461f261b9` plus this change; native source identity
`cfb0d305f8537387a109eeab1ff5be4ab70d63f5da3c1061277bb8bde1564783`; loaded core SHA256
`e2c0149921399175ecc88e85f7876a1fbe609a15b027cd707dc47007f7e697d9`. All accepted endpoints use this image.
Driver/test, AOT and probe hashes are retained in the receipts.

Node n1, RTX 5090, CUDA 12.9.1, Release sm_120, verified ccache 4.5.1.
All device work uses finite Slurm `main` / `gpu:5090:1` allocations and preserves
scheduler visibility. One allocation holds the accepted campaign, with clean
timers followed by separate profiles and synthetic measurements.

Three rounds × two arms × three systems: 18 fresh processes and 78 complete E/F
calls. Each process has cold plus three frozen warm samples; round one adds a
0.02-Bohr displacement and candidate independent references; round two reverses
arm order. Cold includes constructor/preparation after imports/context setup.
Warm retains each owner's own cold density. PBE0/def2-SVP and r2SCAN-3c with
D4/gCP use 96-atom water. C7 is pinned QM7 C7H16 Direct RHF/def2-SVP, **not**
CCSD(T)/CBS. Grids match defaults; this is not a grid-convergence study.

The flag prepares four bytes per block and system within existing reductions,
with charged storage and geometry/density invalidation. Lower/upper FP32
intervals certify admission/rejection; only uncertain candidates read original
FP64 metadata. The 24-byte shell-density ABI, exact shell gates, Schwarz, ERI,
J/K, energy and force arithmetic retain their existing precision.

## Complete endpoint results

Median seconds, three cold and nine warm samples per cell:

| System | FP64 cold | FP32 cold | FP64 warm | FP32 warm |
| --- | ---: | ---: | ---: | ---: |
| PBE0-96 | 28.309551 | 28.401147 | 6.905804 | 6.950624 |
| r2SCAN-3c-96 | 49.100527 | 48.991587 | 15.723778 | 15.713471 |
| C7 RHF | 3.108950 | 3.113103 | 0.284292 | 0.284762 |

Both arms have the same SCF/Fock counts:
PBE0-96: cold 9/9, warm 1/1; r2SCAN-3c-96: cold 10/10, warm 1/1; C7 RHF: cold 10/12, warm 1/1. Changed geometry has one sample per arm/system: invalidation
coverage, insufficient for a speed claim. The reader reports ranges and moved
timings. Earlier binaries are historical context, not matched timing controls.

Every raw sample of both arms was compared to a tighter independent GPU4PySCF
reference, including r2SCAN-3c D4/gCP, before compaction. Worst errors:

| System | Energy/atom (Eh) | Force max (Eh/Bohr) | Force RMS (Eh/Bohr) |
| --- | ---: | ---: | ---: |
| PBE0-96 | 6.916e-13 | 9.348e-07 | 2.053e-07 |
| r2SCAN-3c-96 | 5.215e-12 | 2.005e-06 | 7.112e-07 |
| C7 RHF | 6.055e-13 | 4.515e-08 | 1.299e-08 |
| Gate | 2.5e-6 | 2.5e-5 | 7.5e-6 |

All cross-arm pairs at the same geometry differ by at most
2.274e-12 Eh and 1.190e-11 Eh/Bohr, below the separate
1e-9 Eh / 1e-8 Eh/Bohr gates.

## Work and profiles

Diagnostic cold/warm DFT force censuses match in every work field:

- pbe0-96: 92,223,676 admitted quartets and 0 rejected block pages per call.
- r2scan-3c-96: 243,123,642 admitted quartets and 30,112 rejected block pages per call.

These are force-owner counts, not SCF totals.
C7 uses another owner: its quartet census is unavailable, not zero. Kernel
launch and SCF/Fock counts are retained.

Selected diagnostic GPU milliseconds per execution window:

| Kernel / system | FP64 cold | FP32 cold | FP64 warm | FP32 warm |
| --- | ---: | ---: | ---: | ---: |
| Fused low force / PBE0 | 1362.643 | 1360.676 | 1362.158 | 1360.740 |
| Fused low force / r2SCAN | 3810.106 | 3807.950 | 3810.189 | 3805.795 |
| Task compaction / C7 | 0.316 | 0.320 | 0.318 | 0.318 |
| Exact force page / C7 | 33.555 | 33.394 | 33.555 | 33.400 |

Screening is fused with compaction or integral/contraction work, so these are
not isolated screening timings. Do not sum them into clean wall time. The
receipts retain exact kernel names/counts and separate prepare/cold/warm rows,
including Schwarz kernels. NVTX cold covers execution; clean cold also includes
preparation. Loose block maxima and the system-wide density bound limit page
rejection; eliminating repeated screening leaves the integral task set intact.

## FP32, INT8 and Tensor Core controls

All seven modes match every host-FP64 task identity. Three process repeats ×
30 timed iterations cover 256/2048/8192 bounds, Fock/force and sparse/all-survive
thresholds. GPU timers include packing, mask writes and refinement; allocation,
transfers and host checks are outside. The dense square mask is synthetic, not
a production bounded queue.

Median force-screening milliseconds at 8192 bounds / 67,108,864 candidates:

| Mode | Sparse (tau=1e-12) | All survive (tau=0) |
| --- | ---: | ---: |
| FP64 | 0.553318 | 0.565885 |
| FP32 per visit | 0.605855 | 1.474377 |
| Prepared FP32 rejection | 0.344154 | 1.021492 |
| Prepared FP32 interval | 0.618688 | 0.472905 |
| Flat INT8 | 0.202871 | 1.014289 |
| Tiled INT8 | 0.155022 | 0.546429 |
| INT8 Tensor Core | 0.131872 | 0.544722 |

The interval arm certifies both outcomes and avoids the old survivor recheck.
Checking lower bounds first favors all-survive workloads but adds work to the
sparse control. This is not a universal screening speedup. INT8 uses upward
tile-local codes with power-of-two scales; WMMA pads an outer product with
15 zero K lanes. Tiled scalar INT8 has identical warp/refinement scheduling.
These experiments establish neither molecular Tensor Core benefit nor a
qualified 12-byte shell-density ABI.

## Validation and reproduction

Both FTZ modes pass 3,061,731 cases with zero task mismatches against sequential
host FP64 and an extended-precision rejection oracle, including threshold,
subnormal, overflow, infinity and NaN cases. Thirty-five Python tests cover
resource identity/budget/replay, independent UHF forces, signed raw J/K, and
full/SR/LR shared-owner matrices with changed geometry. Compute Sanitizer reports
zero errors for arithmetic, all seven probe arms, and complete C7 cold/warm/moved
execution with the cache enabled. Shared J/K caches retain spin-density ownership.
The diagnostic stage recompiles the probe; its timed image is separately hashed
and memchecked after timing. Its SASS contains `IMMA.16816.S8.S8` instructions.

Recompute numerical errors and check archive integrity, work counts and summaries
without a GPU:

```bash
python3 benchmarks/results/direct-screening-20261023/summarize.py
```

`manifest.json` binds `receipts.json.xz`, which includes all accepted endpoint
timings, force/reference arrays, geometries, selected profiles, DFT census,
synthetic observations, identities and exact reproduction scripts. The reader
recomputes independent errors and all cross-arm maxima from the retained arrays.
Full raw records/logs/profiler traces remain in local artifacts; raw-record hashes
bind the observations. Shared ccache snapshots are not exclusive per-build
accounting. The original repository evidence budget is unchanged.

For reproduction, adjust paths in the retained scripts, run `build.sh`, then
run:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 --time=01:00:00 bash qualify.sh
```

Keep the scheduler's device visibility. Verify ccache and build
the core, PBE0/r2SCAN AOT manifests and both FTZ tests; C7 needs the pinned QM7
dataset. `qualify.sh` runs validation, clean campaign and diagnostics in order.
Run `probe-final-memcheck.sh` in a three-minute allocation afterward to verify
the actual timed probe image; its sanitized timings are excluded.
NVCC include discovery must use the actual CUDA installation, not a cache-wrapper
directory. Preliminary, sanitized and interrupted pre-integration timings are
excluded from accepted performance comparisons.

See the [decision note](../../../.agents/notes/implemented/performance/2026-10-23-prepared-block-screening.md).
