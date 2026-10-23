# Proposal: preserve native cold factors before extending delta controllers

Status: proposed
Date: 2026-10-23

## Objective and measured source

Improve complete Fast/DF energy-and-force endpoints after #2696. This is an
investigation, not an implemented optimization or a new default promotion.
The checkout is based on `bfb1584b3e6aed39c8e7f6276f553d4f5ee7d8ec`. The measured
production inputs are the #2696 qualification source plus its native cold-seed
fix, with library SHA256
`117feee7922ac187fb781ece52be59e345be04aceb98d6bf0fb2d8ff93d14e9b`.
The later checkout commit only changes raw-output retention in the benchmark.

Slurm job 13482 on n1/node1 used one RTX 5090 with a finite ten-minute allocation,
preserving `CUDA_VISIBLE_DEVICES=2`. Seven fresh processes produced 21 complete
E/F endpoints. All passed the earlier independent full-density GPU4PySCF gates:
2.5e-6 Eh/atom, 2.5e-5 maximum and 7.5e-6 RMS force in Eh/Bohr. Inputs were
PBE0/def2-SVP spherical water-96 with native default initialization. References
never supplied production densities. These are stock-grid solver checks.

The original build, matching force AOT and independent references remain at
`n1:/data/jzzeng/pr2696-fast-df-20261023/`. New raw records, CUDA/host component
traces and allocation receipt are at `n1:/data/jzzeng/fast-df-perf-20261023/`,
mirrored in this worktree's ignored `.artifacts/diagnosis/`. The runner is
`.artifacts/diagnose-fast-df.sh`, also retained on n1 as
`/data/jzzeng/diagnose-fast-df-20261023.sh`.

## Negative result: disabling Fast residency

Two clean alternating-order rounds compared default Fast Direct against
`GENERATIVEQC_KS_RESIDENT_VALUES=0`. The latter retains the existing streamed
mixed provider, rather than selecting Standard's strict generated provider.

| Selection | Cold seconds, both samples | Frozen-warm median seconds |
| --- | --- | ---: |
| Default | 28.858758, 28.779074 | 7.461960 |
| Residency disabled | 59.808739, 59.896363 | 7.476309 |

Both selections use nine cold Focks and one per warm endpoint. Disabling this
cache is strongly negative for the measured case. Separate diagnostic runs
agree with that direction; they are excluded from clean timing populations.
The default trace attributes about 4.72 seconds to the cold begin/source setup,
but streaming its mixed work costs much more. Do not remove the cache merely
because its setup is visible.

Earlier #2696 samples show a size crossover: Fast cold at 12/48/96 atoms was
2.481/10.412/30.543 seconds versus Standard/Direct incremental medians
3.091/11.237/26.224 seconds. Those Fast values are single samples and the
campaigns are separate; they motivate a repeated plan-selection experiment,
not a universal cutoff or a new performance claim. Fast has looser admitted
controls than Standard and passes the same independent observable gates.

## First implementation candidate: DF native cold factor handoff

`src/scf/initial_guess/huckel.cpp` already constructs occupied AO coefficients
and forms `D = 2 C C^T`. `HuckelDensityResult` returns only D and diagnostics,
discarding C. The DFT batch owner then passes the native cold density to CUDA
without a factor certificate. The first fitted K therefore uses the dense
provider, while subsequent orbital-generated densities use occupied factors.

The separate Standard DF trace contains one dense K at 3919.466 ms and ten
occupied K builds at approximately 538.5 ms each (eight remaining cold builds
and two frozen warm calls). Its complete cold endpoint is 36.799 seconds.
The approximately 3.38-second operator-cost difference is a useful opportunity
estimate, not a measured endpoint saving: the inputs differ by iteration and
the handoff is not implemented yet.

Preserve the already-computed Hückel factor and carry its exact occupation,
normalization, geometry/source, shape and lifetime through native seed admission.
Use the existing occupied K provider only when that certificate is valid.
Preserve dense fallback for arbitrary external/moved inputs, unsupported
initial guesses and missing capacity. Check any seed normalization against the
factor; do not silently substitute final canonical orbitals or approximate D
with a new low-rank decomposition. Charge the retained factor to the complete
host/device resource bounds. No additional CPU/reference calculation is needed.

Acceptance must compare the first K and full Fock at the same density against
the dense provider, independent complete E/F endpoints, cold/reset/moved/warm
lifecycle and resource/error fallbacks. Qualify 12/48/96 atoms and additional
hybrids with repeated complete timing; do not infer speedup from the trace alone.

## Subsequent candidates

- Fast: compare its current resident mixed execution against the already
  qualified strict incremental execution. Preserve the application budget and
  truthful precision receipts. Select using measured workload/resource costs;
  avoid assuming mixed arithmetic is always the fastest or adding molecule-name
  switches. A future mixed-delta composition needs explicit anchor/rebase rules.
- DF preparation: the diagnostic resident three-center materialization costs
  about 6.49 seconds, including about 4.54 seconds of metric eigen-projection.
  Investigate contraction layout and reuse under the existing complete resource
  bounds, without changing auxiliary/metric scientific identities.
- DF repeated endpoints: force response is about 4.43 seconds per E/F call in
  this diagnostic. Source/derivative reuse can benefit warm calls, whereas the
  proposed first-K handoff only targets cold work. Inclusive nested trace totals
  must not be added together or treated as clean endpoint measurements.

## Follow-up

The native cold-factor handoff is implemented in the
[decision note](../implemented/performance/2026-10-23-ks-df-native-cold-factor.md).
The diagnostic timings above remain historical opportunity evidence; they are
not substituted for the implementation’s separate complete-endpoint qualification.
