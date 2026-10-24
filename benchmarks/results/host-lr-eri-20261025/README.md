# Opt-in host LR ERI cache screen

This records the original opt-in implementation. The subsequent automatic
default policy has a separate [source-bound qualification](DEFAULT.md).

This experiment retains omitted exact long-range FP64 ERIs in pageable RAM for
WB97M-V/def2-SVP spherical density-fitted E+F. All three arms retain the same fitted
primary provider. **Direct** below disables only LR value residency; it does not
switch the calculation from DF to direct J/K.

The implementation remains opt-in. These are single screening observations at
two static geometries, not a statistical speedup gate or a default-policy
promotion. Preparation, SCF, full analytic forces and host export are included
in cold timing. Warm uses the engine's frozen cold density and the same owner.

## Source and execution

- Base: PR #2744 head `bd2fed8ee4709192a580361c97ac14ce40a4930f`.
- Measured Git tree: `3507e6cb6741a329bd26308a9ceb0639025af004` (an actual tree
  object, not a fabricated commit). The packet retains its reconstructing patch
  and changed-file manifest. Later evidence/prose and ownership-audit additions do not change
  the measured executable implementation.
- Library SHA-256:
  `393fe70191d7a885f76330d686535f076a71808058b9cce8ef8e0273b99ec0d9`.
- Release CUDA 12.9.1, `sm_120`, verified ccache 4.5.1 for C++/CUDA and sccache
  0.16.0 for runtime JIT. All 550 generated compilation commands use ccache.
  Cache statistics are retained; cache activity is not assumed to mean a hit.
- This build uses the scalar CPU linear algebra fallback. It is shared by every
  new arm, but differs from the historical OpenBLAS build. Historical #2744 wall
  times must not be pooled with this cohort.
- Formal timing: one RTX 5090 on n5, Slurm job 1843, 6 allocated CPUs, 96 GiB
  RAM, one BLAS/OpenMP thread, finite 02:30:00 allocation. Each arm has a fresh
  process; separate 96/108 primers warm generated compilation artifacts.
- Diagnostic CUDA-event profiles and sanitizer: n4 job 1334. Job 1336 records
  Nsight and the 96-atom cache controls. All GPU work uses
  Slurm-assigned visibility. Profiled endpoints and primers are excluded from
  the formal comparison.
- n5 job 1844 separately checks host-enabled 96-atom admission after the formal
  cohort; its profiled latency is also excluded.
- GPU value ceiling 16 GiB; host ceiling 32 GiB; requested tile ceiling
  256 MiB. Actual tiles shrink without evicting GPU buckets. Both component
  trace variables are unset for the formal timing loop.
- Stock method quadratures, strict FP64 and the existing driver defaults;
  `GENERATIVEQC_SCF_VALIDATION_GEMM=1` enables the declared force-response
  solver route. The complete settings, grid work, convergence and forces are
  retained, including the driver's force-response-override declaration.

## Complete endpoints

| Atoms | LR policy | Cold E+F (s) | Frozen-density warm E+F (s) |
| --- | --- | ---: | ---: |
| 96 | Direct | 617.876154 | 39.680324 |
| 96 | GPU | 310.452769 | 39.681283 |
| 96 | GPU+RAM | 310.445138 | 39.622892 |
| 108 | Direct | 768.181020 | 56.414080 |
| 108 | GPU | 686.400475 | 56.345633 |
| 108 | GPU+RAM | 354.802842 | 56.385833 |

Each size runs Direct, GPU and GPU+RAM in that order; order effects and run-to-run
variability are not estimated by this screen. At 108 atoms, host backing lowers
cold latency by 331.598 seconds (48.31%) versus GPU partial residency. At 96
atoms the observed 0.008-second difference is not evidence of a speedup.

The existing Direct arm performs 28 cold Fock builds; cached arms perform 29
(28 SCF actions plus one post-SCF action). All have 28 SCF iterations. The
GPU-versus-GPU+RAM comparison therefore preserves the Fock count. Frozen-density
warm performs one strict Direct Fock; it is not a repeated cached-SCF workload.

## Work and transfer accounting

The 108-atom screened source has 2,156,803,082 canonical values: 17,254,424,656
FP64 bytes (16.0694 GiB), smaller than a naive unscreened fourth-power estimate.
The retained GPU buckets cover 1,169,998,706 values (54.2469%). Host storage adds
986,804,376 values (45.7531%), or 7,894,435,008 bytes (7.3523 GiB).

CUDA-event traces retain every measured LR action and each bucket's recurrence,
GPU replay or host replay region. These identify actual canonical visits and
recurrence evaluations; `direct_candidate_upper_bound` remains an upper bound.
Host replay regions include CPU staging, waits and contraction, so their
durations are not isolated DMA bandwidth measurements. Nsight separately
measures H2D and kernel interval overlap.

The n4 108-atom diagnostic medians over the 28 cold SCF actions are:

| LR policy | Recurrences/action | GPU values/action | Host values/action | Direct region (s) | Host replay region (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 2,156,803,082 | 0 | 0 | 16.9827 | 0 |
| GPU | 986,804,376 | 1,169,998,706 | 0 | 12.7754 | 0 |
| GPU+RAM | 0 | 1,169,998,706 | 986,804,376 | 0 | 0.7036 |

The host arm's empty Direct bucket regions contribute only about 5 microseconds
of tracing overhead. GPU replay medians are 0.2691 and 0.2688 seconds in the GPU
and GPU+RAM arms. Their admitted GPU populations are identical.

On n4 at 96 atoms, GPU coverage is 1,706,098,336 of 1,873,221,716 values; the host
tier fills the remaining 167,123,380 values (1,336,987,040 H2D bytes/action).
Those diagnostic runs are a different node/prewarming cohort and do not supply
the formal n5 latency comparison. Automatic admission depends on available
device capacity. The additional n5 96-atom host-enabled profile admits
1,727,360,068 GPU values and **zero host values**, retaining 145,861,648 Direct
recurrences per SCF action (median 2.299 seconds). Host H2D and device staging are
both zero. Thus the unchanged formal 96-atom latency is consistent with the
bounded admission fallback; it does not measure a transfer/recompute crossover.
The refusal's physical-VRAM versus total-budget cause was not separately
instrumented. A supplementary Slurm memory-query step could not start while
the allocation was busy, so no free-memory snapshot is used as evidence.
Reserving staging before greedy GPU selection is a possible follow-up with its
own A/B qualification; this implementation preserves the incumbent GPU buckets.

The separate n4 first-action Nsight capture transfers exactly 7,894,435,008 bytes
in 147 H2D copies. DMA active time is 321.091 ms (24.586 GB/s while active);
resident contraction overlaps 8.795 ms, or 2.739% of DMA time. Kernel active time
is 296.652 ms and the combined device activity span is 750.215 ms. Copy and
contraction use distinct streams. This establishes real but modest overlap;
avoiding recurrences is the dominant benefit. DMA busy bandwidth excludes CPU
pageable-to-pinned copies and idle gaps, and is not an endpoint throughput claim.
The capture disables Nsight's CUDA event completion tracing to avoid its
documented cross-stream false-dependency hazard. An initial capture using that
default was deliberately stopped (job 1335) and remains separate from accepted
evidence.

The cached cold calculation includes 28 cached SCF LR actions, one uncached
post-SCF action, source construction and the force path. Warm has one uncached
LR action. These consumers are separated in the offline report instead of
treating every Fock as a cache hit or multiplying one kernel time by an assumed
iteration count.

## Numerical and lifetime qualification

Every complete E+F receipt is checked against the retained independent
PySCF/GPU4PySCF dense reference, including dense-grid stability and the CPU/GPU
small-system spot check. Original producer identities remain unchanged through
the existing audited oracle-reuse contract. Common gates are energy/atom
`1e-5 Eh`, force max `1e-4 Eh/bohr`, and force RMS `3e-5 Eh/bohr`. Cache parity
against the new Direct arm additionally requires total energy `1e-8 Eh` and
force max `1e-7 Eh/bohr`.

All 15 retained complete receipts (30 E+F calls, including diagnostics and
primers) qualify. The six formal A/B/C receipts account for 12 of those calls.
The worst independent errors are `3.188e-6 Eh/atom`, `1.995e-5 Eh/bohr` force
max, and `9.392e-6 Eh/bohr` force RMS. Formal cache parity is at most
`3.638e-12 Eh` in total energy and `3.180e-13 Eh/bohr` in force max.

The final library passed 51 real-GPU tests: four independent host matrix cases,
seven host E+F/lifetime/policy cases and 40 existing range/mixed/replay cases.
The matrix cases also exercise original/displaced geometries, signed/zero
densities, restricted/unrestricted contraction, screened row prefixes and
4 KiB tiles. Bounded staging tests admit only 8192 total device bytes while
preserving GPU values and independent LR K, and verify zero-budget refusal
followed by a successful retry. Compute Sanitizer reports zero errors and zero
leaked bytes. The final ownership registration also passes 192 CPU structure
and dependency-boundary tests.

Two early setup failures are retained in node-local artifacts: a probe launched
before library transfer finished, and an older packaged cuSOLVER loaded ahead
of CUDA 12.9. They are not numerical or timing results. Final jobs preload the
matching toolkit libraries and use the verified transferred binary.

## Reproduction and storage

From the repository root:

```bash
PYTHONPATH=python:. python benchmarks/results/host-lr-eri-20261025/reproduce.py
```

`measurements.json.xz` retains original complete endpoint JSON text, every LR
CUDA-event record from the declared profiles, source/build/cache/Slurm/test
receipts and the Nsight reduction. Hashes bind each retained member. Full
component traces and Nsight databases stay in ignored `.artifacts/host-eri/`
and their node-local evidence directories; their hashes and sizes are recorded.
The Nsight reducer is retained for rechecking the original database. No external
archive, GitHub Release or published release tag is created.

The packet's `benchmark.sh`, `profile.sh`, and `nsys.sh` preserve executed node
paths and settings. The reusable entry point is
`python benchmarks/host_eri_cache.py {direct,gpu,host} --method wb97m-v --mode df
--atoms {96,108} --warm-repeats 1 --inputs <prepared-inputs> --output <receipt>`
inside a finite `srun` allocation, using the same source/binary/input identity
for all arms. See the [current cache contract](../../../docs/developer/xc_native_cuda.md#experimental-pageable-lr-eri-backing)
and [decision note](../../../.agents/notes/implemented/performance/2026-10-24-host-lr-eri-experiment.md).
