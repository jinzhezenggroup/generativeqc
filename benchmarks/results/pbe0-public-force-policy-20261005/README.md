# Public force AO policy: complete endpoints and kernel attribution

These are two complete, source-pinned 48/96-atom exact-direct PBE0/def2-SVP
campaigns and separate intrusive warm profiles. No production default changes
in this publication. The current force-map candidate improves both measured
warm sizes, but remains above #1895's 1.25x matched-reference target. Earlier
cold regressions and all SCF trajectories remain visible.

## Current composition: n1 RTX 5090 Slurm 5823

Both arms use the public default SCF local-AO, phased Becke and indexed force
schedules. The candidate enables the shared public force-map registry profile;
the control replaces that registry with an empty tuple. The numerical cutoff
is 1e-16 and optional map cache allowance is 16 MiB. Neither arm uses the old
low-level atom-only `auto` policy from #1833.

Seconds below are complete E+F observations, including the harness's recorded
preparation charge. Warm rows show the median of all five fixed native/reference
density replays. Cold/moved rows have one observation per arm in this campaign.

| Atoms | Phase | Dense force control | Force-map candidate | Candidate/control |
|---|---|---:|---:|---:|
| 48 | cold | 125.6164 | 124.7586 | 0.9932 |
| 48 | warm | 13.1681 | 10.7415 | 0.8157 |
| 48 | moved | 58.8560 | 58.3532 | 0.9915 |
| 48 | moved-warm | 13.1585 | 10.7474 | 0.8168 |
| 96 | cold | 303.7963 | 287.0286 | 0.9448 |
| 96 | warm | 46.8146 | 32.4317 | 0.6928 |
| 96 | moved | 138.7730 | 129.4576 | 0.9329 |
| 96 | moved-warm | 46.9672 | 32.4393 | 0.6907 |

The 48-atom cold/moved SCF builds match at 25/13. At 96 atoms, cold builds
differ: control 31 versus candidate 30; moved builds match at 12. Every native
warm and moved-warm call reports one actual public Fock build and no fallback.
Do not normalize timing by iterations or attribute the entire cold difference
to force maps. The 48-atom process order is control then candidate; 96 reverses
that order. These are ordered observations, not randomized causal estimates.

Matched reference warm medians are 5.9937/13.7208 s for 48/96 atoms, giving
candidate/reference ratios 1.7921/2.3637. Reference iterations vary and are
retained for every sample, including moved-warm. These reference trajectories
are not interchangeable with #1912's older 10.2 s 96-atom warm reference.
The reference uses full-density rebuilds, CUDA LibXC and moving-grid forces;
DF/COSX and a CPU XC fallback are excluded.

## Earlier composition and negative observations

Slurm 5817 used the earlier narrowed SCF admission composition. Its warm
control/candidate medians were 13.1848/11.0296 s at 48 atoms and
46.9207/32.5421 s at 96 atoms. The 96-atom cold endpoint regressed from
260.0548 to 281.8816 s (8.393%), with builds increasing from 25 to 29.
The 48-atom moved endpoint was 1.052% slower. No row is discarded because a
later source or SCF trajectory looks better. The full data and reconstruction
patch live under `narrow/`; they do not qualify the broader current SCF default.

The shared-profile default decision remains separate. Supplemental cold
repetition and any later promotion must identify their own source, process
ordering, cache conditions and complete trajectories. This publication itself
does not resolve the scientific/representativeness limits of interpolation
through a workload profile or prove profitability for every eligible molecule.

## Actual semantic work

Current original-warm force observations:

| Atoms | Grid points | 256-point tiles | Active AO-square sum | Dense AO-square sum | Retained map bytes |
|---|---:|---:|---:|---:|---:|
| 48 | 1,179,648 | 4,608 | 26,954,388,992 | 173,946,175,488 | 5,145,520 |
| 96 | 2,359,296 | 9,216 | 81,057,099,776 | 1,391,569,403,904 | 12,129,408 |

Every original-warm map lookup hits the retained cache, with no discovery or
budget fallback. Cold and moved map work, actual SCF local-AO work, preparation,
component timers and all per-call Fock counts remain in the raw records.
Observed Becke pair-state productions are 1,330,642,944 / 10,758,389,760.
The separate grid plan's pair-visit estimates are not substituted for executed
pair productions. `native_primitive_records` counts prepared records, not
executed four-center derivative primitive products. Those molecular derivative
class/primitive counts are not measured by these profiles.

## Intrusive warm attribution

Slurm 5838 (48 atoms) and 5830 (96 atoms) capture only the first unchanged warm
`PreparedBatch.execute` using CUDA profiler start/stop and one NVTX E+F range.
All twelve calls still finish and pass the same independent numerical gates.
Profiler timing is separate from the clean endpoint campaign above. Nsight
reports CUDA-event tracing overhead and possible false displayed dependencies;
summed kernel durations are not endpoint wall time.

| Observed GPU kernel family | 48 atoms, s | 96 atoms, s |
|---|---:|---:|
| Bounded Direct derivative | 3.8102 | 13.4196 |
| Generated Direct J value kernels | 1.1772 | 2.4724 |
| Generated Direct K value kernels | 1.0631 | 1.9853 |
| Becke partition derivative phases | 0.7651 | 4.0954 |
| AO/grid geometry response | 1.1443 | 2.7062 |
| XC density, point and potential kernels | 0.9009 | 1.9896 |
| AO values/jets | 0.4977 | 1.1339 |
| One-electron derivative | 0.1610 | 1.0916 |

J/K attribution uses the **pinned source's submission order**, not separate
NVTX ranges: `src/scf/cuda/direct_jk.cpp::enqueue_cuda_direct_jk_device_impl`
submits generated Coulomb then generated exchange. The capture contains one
reported Fock build and two identical ordered 20-class launch inventories on
one stream. `ordered-fock-kernels.csv` retains every duration and launch
resource field; their sum is checked against Nsight's combined kernel report.
These columns exclude setup, other kernels, scatter and synchronization, and
are not complete J/K phase timers. The complete aggregation retains unmatched
kernels and a GEMM group whose caller is not classified.

The coarse force geometry timer must not be labelled "Becke-only": the actual
Becke phases and AO/grid response are distinct kernel families. The 96-atom
bounded derivative is 43.96% of summed GPU duration, providing motivation for
further derivative work reduction without presuming a per-class bottleneck.

## Reproduction and provenance

Run offline from the repository root:

```bash
PYTHONPATH=python:. python benchmarks/results/pbe0-public-force-policy-20261005/verify.py
```

The verifier authenticates stored bytes, checks every record and route, and
recomputes 864 endpoint E/F pairings (576 native/reference and 288
candidate/control), plus 144 profile/reference pairings. Energy and force gates
remain 1e-8 Eh and 1e-7 Eh/Bohr. Original JSON bytes are losslessly gzip-compressed;
no missing historical counts are manufactured. The summary is regenerated from
the records, including failures of performance comparisons.

`identities.json` pins the two commits, library hashes, source identities and
permanent reconstruction base `9d0f4fc0ddbe9c1dc4539cd4f6ad8a6e019ba990`.
Each cohort's `source.patch.gz` reconstructs all canonical source-identity
inputs and the measured harness/native tests from that base. Both source
identities were independently reproduced; see `reconstruction.json`.
Unrelated historical notes and Python regression tests are outside this patch
scope. The original executable hashes are receipts, not a promise that a future
toolchain reproduces identical binary bytes.

Original runner, verifier, scheduler, source and binary receipts are retained.
`profile/` holds 96-atom data and `profile48/` holds 48-atom data. Raw `.nsys-rep`
and SQLite exports remain in the ignored local artifact directories recorded
by their capture receipts; only the lossless numerical records, exported CSVs,
logs, scripts and report checksums are published. The original 48-atom driver's
preflight also hashes the common 96-atom shell driver; the actual executed
48-atom script is separately retained and authenticated here. No external
backup, release asset or release tag is created.
