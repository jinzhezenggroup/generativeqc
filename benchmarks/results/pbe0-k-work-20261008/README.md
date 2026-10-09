# Primitive-work-aware K: PBE0 cold qualification

At qualification on 2026-10-08, `work` was experimental and opt-in, and `fill`
was the default. The frozen `summary.json` preserves that qualification state.
On 2026-10-09, the user accepted `work` as the default with explicit `fill`
rollback; see the [default decision](../../../.agents/notes/implemented/performance/2026-10-09-direct-k-work-default.md).
The equations, density screening, precision, independent J ownership and force
operators are unchanged. See the [work-bucket decision](../../../.agents/notes/implemented/performance/2026-10-08-direct-k-work-buckets.md).

## Conditions

RTX 5090 on n1, CUDA 12.9.1, Release SM120, strict FP64, def2-SVP spherical,
48x16x32 grid, screening 1e-12, energy tolerance 1e-12 and density tolerance
1e-10. All real-GPU work runs through finite Slurm allocations and preserves
assigned visibility. Clean comparisons use eight allocated CPUs and eight
BLAS/OpenMP threads. The baseline concentration trace uses one allocated CPU;
its intrusive device sums are not pooled with clean wall samples.

Cold means fresh prepared owner/density and preparation plus the first
synchronized energy/host-force endpoint. Compiler/runtime caches are reused;
imports, CUDA context initialization and Calculator construction are excluded.
Each clean comparison uses the same newly built candidate library, with
`fill, work, work, fill` in separate processes. The library/source identities,
SCF/Fock counts, independent numerical gates and sample timings are retained in
[summary.json](summary.json).

## Work and timing

| Public AOs | Fixed initial-density K: fill / work | K wall reduction | Cold E+F: fill / work | Observed cold reduction |
| ---: | ---: | ---: | ---: | ---: |
| 384 | 948.52 / 732.08 ms | 22.82% | 103.37 / 91.66 s | 11.33% |
| 768 | 1669.46 / 1379.59 ms | 17.36% | 203.49 / 198.74 s | 2.33% |

**The cold columns are observations, not a stable speedup or promotion claim.**
There are only two clean samples per mode. At 384 AOs, fill uses 26/23 Fock
builds and work uses 23/23; at 768 AOs, fill uses 24/28 and work uses 28/25.
The SCF trajectories are not matched. These cold observations alone do not
justify default promotion; the later user-accepted default decision retains this
limitation rather than treating the observations as a stable cold speedup.

The diagnostic fixed-density test uses the same independent PySCF MINAO initial
density and full-scale Libcint raw-K oracle for both schedules. It collects six
clean samples per mode, excluding preparation, via the existing qualification
SCF device-K seam: density upload, transformation, K, projection, host export
and synchronization. It does not time the independent host Fock/one-electron
assembly or replace a complete PBE0 cold measurement. The seam is compiled
against the candidate's headers; the private handle layout is checked against
`src/api/c_api_fock.cpp`. Events and borrowed task counters are disabled during
clean samples. A separate census observes exactly 23,480,495 / 81,907,624
generated streaming admissions at 384 / 768 AOs, with identical per-class arrays
for fill and work. Native `dddd`/fallback tasks are unobserved by this census.

## 768-AO concentration

[classes-fill-cold-96.json](classes-fill-cold-96.json) validates 27 paired
J-then-K passes over 21 classes, including native `dddd`. K value kernels sum
to 44.28994 s. Classes containing d contribute 60.30%; psss/psps together
contribute 14.40%. The largest classes are ddds 9.31%, ppps 8.71% and dpps
8.20%; the top three sum to 26.21%. High-angular work matters, but a few classes
do not monopolize K time. These percentages are of K device time, not of the
complete endpoint.

[classes-work-cold-96.json](classes-work-cold-96.json) independently validates
25 paired passes and confirms 19 separately compiled work kernels execute in
K, never J. Native `dddd` and whole-CTA `ddpp` retain their original workers.
Its cumulative K device time is 33.12914 s. Different trajectories, densities,
Fock counts and trace CPU allocations prevent a direct device-time speedup claim
between these two captures. Neither trace supplies primitive-operation counts.

## Numerical and synchronization gates

- Sixteen restricted/unrestricted, symmetric/nonsymmetric, full/scaled/zero/
  restored raw-K comparisons against independent Libcint: max error 4.6863e-13.
- Full-size frozen-density K: max error 1.8902e-11, retaining the 1e-8 gate.
- Every complete cold sample converges and passes the independent energy/force
  gates. The native CUDA provider suite passes through-f values and its
  Cartesian order-two / CPU finite-difference derivative checks.
- Work-mode memcheck and synccheck each report zero errors.
- Concurrent emitted-worker tests check exactly-once admission, bucket-homogeneous
  consumption, precision tags, saturation, canonical legality, empty/inactive
  rows and per-bin tail flushing. Ordinary emitted streaming bodies remain
  byte-identical across 22 compiler selections.

Raw endpoints, oracle matrices, compiler/cache records, sanitizer logs,
qualification scripts and Nsight captures remain in ignored local/n1 artifact
directories. Compact records include checksums. Re-reduce a full-range RKS trace
with `benchmarks/pbe0_k_work_profile.py --trace TRACE.sqlite --endpoint COLD.json --output CLASSES.json`;
the reducer rejects incomplete/misaligned class passes rather than attributing
all J/K or force time to K.
