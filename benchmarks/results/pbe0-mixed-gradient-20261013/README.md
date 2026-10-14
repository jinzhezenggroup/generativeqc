# Exact-PBE0 mixed weighted gradient qualification

**Private candidate, not public Fast admission.** This implements the upstream
study's useful distinction between strict SCF and mixed gradient arithmetic,
without replacing exact four-center J/K by density fitting. It does not
reproduce upstream's DF-r2SCAN experiment or its reported 6.8–7.1x gradient gain.

## Measured endpoints

RTX 5090, CUDA 12.9.1, Release/sm_120, fast-compile disabled; exact PBE0/RKS,
spherical def2-SVP, unpruned moving grid `(48,16,32)`, Hcore initial density,
SCF energy/density tolerances `1e-10/1e-8`, screening `1e-12`, maximum 180 cycles.
All GPU work uses finite Slurm allocations and preserves their device visibility.

Both comparators use **the same complete native stationary endpoint**. This is
not the public Python stationary endpoint used in the earlier size-scaling
campaign. Each clean sample includes fresh construction, strictly FP64 SCF,
all gradient sources, export, teardown and CUDA synchronization. No converged
density is reused between timed endpoints. One warmup pair and an instrumented
four-gradient same-state audit are retained but excluded from timing medians;
three subsequent pairs alternate strict/mixed order.

| AOs | Endpoint FP64 / mixed (s) | Endpoint ratio | Gradient FP64 / mixed (s) | Gradient ratio |
| ---: | ---: | ---: | ---: | ---: |
| 192 | 11.435413 / 11.291817 | 1.0127x | 5.604270 / 5.472606 | 1.0241x |
| 384 | 56.355400 / 55.882824 | 1.0085x | 36.509198 / 36.038969 | 1.0130x |

The 192-AO case is the WATER27 S4 octamer. The 384-AO case is two translated
copies, not a chemically diverse or optimized larger molecule. Both modes use
17 SCF iterations / 18 Fock builds at 192 AOs, and 22 / 23 at 384 AOs.
Three repeats describe these observations, not statistical confidence bounds.
**This is not a useful multi-fold Fast result; increasing size did not help.**

## Arithmetic and error gates

- FP32: explicitly narrowed leaves, weighted derivative DAG arithmetic,
  component-partition sums and primitive/source sums for angular orders 0–5.
- FP64: geometry and Boys moments, density/normalization weights, translation
  recovery, global force accumulation, higher-order derivatives, SCF, XC,
  one-electron and Pulay work. Double output arrays do not imply double inner
  arithmetic. This is not an all-FP32 gradient.
- A nonfinite mixed task retries once in FP64 **before any scatter**. This does
  not detect every finite rounding/underflow error and is not an error bound.

Every endpoint passes the independent moving-grid PySCF oracle: energy
`1e-7 Eh`, mixed maximum force component `1e-5 Eh/bohr`, mixed RMS component
`3e-6 Eh/bohr`. The strict comparator also passes its tighter `1e-6 Eh/bohr`
maximum-component gate. Maximum observed mixed errors are `7.7383e-7` and
`1.1726e-6 Eh/bohr`; the latter would fail a `1e-6` target. Same-state maximum
differences are `7.7387e-7` and `1.1726e-6`; RMS differences are `2.6779e-7`
and `3.4535e-7 Eh/bohr`. Strict replay changes by less than `2.6e-13`.

Separate instrumented receipts count attempted mixed shell tasks,
primitive/source evaluations and nonfinite strict retries:

| AOs | Mixed tasks | Primitive/source evaluations | Strict retries |
| ---: | ---: | ---: | ---: |
| 192 | 5,961,644 | 91,654,664 | 0 |
| 384 | 23,141,636 | 352,656,912 | 0 |

The instrumented replay has identical counts. Clean timings disable these
atomic counters and their explicit host-copy fence. Strict high-order work is
not counted as FP32. An adversarial overflow test independently exercises the
retry and verifies agreement with a finite FP64 answer.

## Why the gain is small

Separate Nsight Systems captures cover a single complete 192-AO gradient,
excluding SCF. In the strict capture, the native `geometry_kernel` accounts for
4.551 s, **79.9% of summed GPU kernel durations**; the mixed capture spends
4.541 s there. It still executes FP64 AO/XC/moving-grid work. These are kernel
shares, not endpoint wall-time percentages.

The affected bounded derivative kernel falls from 0.5154 to 0.3930 s, about
1.31x, while order-seven and other strict kernels remain. A local improvement
cannot deliver a large endpoint gain with this work distribution. The captures
are diagnostics, not clean timing samples. They do not prove the same bottleneck
share for the separate public Python force endpoint.

The next experiment should qualify shared FP32 AO/XC gradient contractions
and an efficient complete consumer, retaining sensitive partition/root work
and FP64 final reductions where justified. A better exact FP64 schedule must
also benefit Strict; it cannot be presented as precision-specific Fast gain.
No Hessian, TS, IRC, diffuse/cancellation-wide or NVE qualification is claimed.

## Validation and provenance

Native source identity: `49b0d2310434ae12e5f253c945ad9c43cf7638c57b61d18a482096957fb54a9e`.
Library SHA-256: `9918bc4480ca7a2265eb3bf681b39b6f47d3e4b82abc91ccb980116e5b5cd02a`.
The independent references exactly match the earlier campaign's complete
scientific protocol; their file hashes, raw sample hashes, work counts, ranges,
Slurm jobs and error summaries are retained in `summary.json.xz`.

- Water (24 AO), tetramer (96 AO) and adversarial retry: **3 real-GPU tests pass**.
- Compute Sanitizer memcheck: water and overflow retry, **zero errors**.
- Typed arithmetic/resource/owner/lifetime/benchmark checks: **692 host tests pass**.
- Existing low-order/order-four/order-five compiled source and weighted-ERI
  numerical checks pass. The updated host fixture retains default transfer
  counts and tests the optional receipt fence and copy-failure recovery.
- Compiler structure: 537 modules, zero dependency errors.

Initial missing-lease failures, a sanitizer stack-allocation OOM, host-shim
compile failures and incomplete public-control attempts are retained, not
treated as timings. The new force-cursor counters require an additional
24 bytes in the shape-only capacity authority; that omission is corrected.
Subsequent 192/384 endpoints and sanitizer runs pass. The initial sanitizer OOM
occurred at `cudaDeviceSetLimit`; it is not established to have the same cause
as the envelope omission.

Raw JSON, logs, Nsight reports and source/build receipts remain under ignored
`.artifacts/pbe0-gradient-fp32-20261013/` and on node1 under
`/data/jzzeng/pbe0-gradient-fp32-20261013/`. No release or external backup is made.
The benchmark's post-run gates were strengthened after timing; the retained
raw arrays also pass the new strict/replay/work gates without altering samples.

A separate public FP64 regression control, using each library's matching Python
sources and complete packaged stationary artifacts, gives medians of 8.610820 s
before and 8.568433 s after at 192 AOs (three samples each, 17 iterations,
independent gates pass). The overlapping sample ranges show no observed
default-off slowdown; the small difference is not a claimed optimization.
This public endpoint remains separate from the native mixed comparison above.
The initial control mixed source/library identities, then omitted the candidate
build's Python s/p/d artifact. Both fail closed. Building the existing
`generativeqc_stationary_pbe0_rks_spd_manifest` target fixes the qualification
setup without changing the native library hash or bypassing artifact checks.

Reproduce with a source-matched library and test bridge, plus the retained
independent reference and source `python/` on `PYTHONPATH`:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:15:00 \
  python -m benchmarks.pbe0_mixed_gradient \
  --case water-octamer-s4-def2-svp-spherical \
  --probe /path/to/mixed_force_endpoint_probe.so \
  --reference /path/to/reference-192.json --repeats 3 \
  --output .artifacts/pbe0-gradient-reproduction.json
```
