# Decision: accelerate the actual r2SCAN-3c cold density owner

Status: implemented
Date: 2026-11-07

## Problem

Issue #2976 prioritizes complete r2SCAN-3c cold, not another small warm/kernel
improvement. Actual A100 LSF228017 attributed about 3.45 s to SCF matrix density
projection: three original/strict calls near 490 ms and seven occupied proposals
still carrying about 283 ms of explicit matrix work each. Occupied projection
itself was only about 462 ms total. AO evaluation was about 580 ms and point
algebra about 45 ms. Stationary force remained about 5.63 s, including nested
Direct 3.65 s and order-six 1.79 s; these nested intervals must not be added.
Native CUDA-event owner intervals include stream gaps, not kernel-busy time.
The intrusive 17.737768 s observation is not the clean performance baseline.

The density portfolio already had a mapped strict-FP64 library recipe but lacked
method-owned production qualification. Indexed potential already used the same
hardware/work gate. Rewriting point expressions, repeating a rejected AO cache,
shortening the occupied proposal stage, or tuning DIIS would miss this owner.

## Decision

Admit the existing bounded indexed density Dgemm recipe through an explicit
method-owned preparation argument. Reuse the established architecture/throughput
predicate, large local meta-GGA work gate, optional capacity and finite provider
refusals. Neither product names nor an environment override grant qualification.
Typed FP64 Dgemm uses default cuBLAS math on this handle only, admitting FP64
matrix units without FP32/TF32 inputs. Other handles remain pedantic.

The optional device allowance is at most 128 MiB, including 96 MiB opaque provider
allowance plus the exact spin-matrix cache (9,469,952 bytes for RKS/1088 AOs).
Host metadata remains charged. Public ledgers and explicit reference budgets keep
the incumbent; the unpromised scope does not wrap their existing preparation.
Resource refusal, empty maps and unsupported capability remain bounded generated
paths. The new rollback is `GENERATIVEQC_CUDA_XC_DENSITY_GEMM=0`; unset/1 cannot
bypass the gates. This is not an AO/jet cache and does not move work to CPU/PySCF.

## Invariants

- Preserve the original half-before-add symmetric-D graph, mapped AO domain,
  all density-product jets, finite publication and point/force mathematics.
- Preserve occupied witness/refusal semantics, strict physical handoff, final
  density proof, original grid and 1e-9/1e-5/1e-12 SCF/screen settings.
- Retain first constructor+prepare+SCF+analytic-force+host-export cold, then three
  same-owner warm calls seeded only by that engine's frozen cold density.
- Keep paired native gates 1e-8 energy, 1e-7 max force and 3e-8 RMS force. The
  independent static reference gates remain 1e-5 energy/atom, 1e-4 max force and
  3e-5 RMS force; they are not the tighter paired-native gates.
- Require both A100 cold ratios below 0.98 and every warm ratio at most 1.03;
  protect 5090 cold and every warm by 1.03. Do not reverse a negative first pair.

## Evidence

The final image includes actual master6572be190 (#3012), source
`7fa6df22bf7bbf8912375aeab23b1e2e1149ea711373d336e833d1ca2dbf4931`,
ELF `3f902abe8e5082c8644e1e42d43a5ce540e3610875d77aa58d2ef89269c08847`.
The earlier e8 image was compiled before the actual master merge; its receipt's
moving-ref base label was misleading. It has no GPU qualification and was rebuilt,
not relabelled. Complete native observations use only the final source/image.

| Device / allocation | Order | Rollback cold / s | Default cold / s | Rollback warm median / s | Default warm median / s |
| --- | --- | ---: | ---: | ---: | ---: |
| A100 / LSF228050 | forward | 17.412127 | 14.509533 | 6.982215 | 6.561433 |
| A100 / LSF228050 | reverse | 17.421643 | 14.499977 | 6.967975 | 6.558612 |
| RTX5090 / Slurm16704 | forward | 28.500869 | 28.426446 | 10.517211 | 10.544306 |
| RTX5090 / Slurm16704 | reverse | 28.454144 | 28.434399 | 10.501042 | 10.510058 |

A100 saves about 2.9 s (16.7%) in each complete cold pair, with all warm ratios
0.936–0.947. 5090's maximum warm ratio is1.004643; its tiny cold differences are
noise-level non-regression evidence, not a new 5090 speedup. Both retain ten cold
Focks and unchanged semantic J/K/grid work. Native paired force max is about
1.03e-11 on A100. These are two single-cold order checks, not a confidence interval
or universal accuracy-qualified production ranking.

The A100 native arms and independent full-AO bilinear fixture completed, but its
same-job GPU4PySCF reference failed before a physical endpoint: NVRTC could not
find `cuda_fp16.h`. Preserve LSF228050's actual exit1, failed reference with zero
samples and all four successful native populations. The environment-only repair
sets the existing CUDA header root and runs only a fresh reference and separate
intrusive attribution; it does not repeat or relabel the clean native calls.
Reference repair, independent E/F gates and attribution are retained in the
source-matched capsule linked below. Do not describe the A100 reference as a
same-allocation latency race.

Environment-only LSF228063 completes with exit0. Its fresh reference cold is
14.595587 s, numerically qualifying all original native exports at the unchanged
common gates: maximum energy error/atom3.91e-12, force max1.35e-6 and RMS4.50e-7.
Native14.50 s is close to that separate-allocation reference observation; it is
not evidence of a statistically established lead. Its separate intrusive traces
reduce the cold matrix-projection owner from3456.553 to610.226 ms with unchanged
explicit matrix FMA counts, AO visits and tile domains. Candidate density storage
is exactly9,469,952 bytes plus100,663,296 provider allowance; potential allowance
is unchanged. Direct force stays about3653 ms and nested order-six1786 ms.
Do not add those nested force intervals or call event elapsed time kernel-busy.

The linked-native fixture uses no test-only offer and independently checks
PBE/r2SCAN, RKS/UKS, full/noncontiguous/empty maps, two densities and one-byte
resource fallback. 5090 memcheck reports zero errors; racecheck reports zero
hazards/errors/warnings. 100 CPU XC, 20 host owner/policy and 40 offline evidence
replay/corruption cases (including Python `-O`) pass. Missing
retained-fixture and missing-native-library setup failures remain recorded rather
than becoming passes. Builds use private foreground sccache0.16.0, reuse the
existing cache and restore the borrowed source/runtime checksum.

The fixed checkout evidence cap had only about34 KiB remaining. Reuse the existing
exact-FP64 byte-plane codec for complete endpoint records; retain named-owner
attribution sums/counters and full raw stream hashes, not the verbose per-tile
timing dump in Git. Full dumps remain ignored artifacts. Four historical recovery
inventories use the existing authenticated compressed-manifest envelope, with
all original bytes and449 historical recovery entries checked unchanged. No
numerical sample or failure is removed, decoder contract changed, new external
archive published, or cap raised. The capsule records each affected metadata path
and both compressed/decoded hashes. This storage-only work does not requalify old
performance populations.

## Rejected alternatives and limits

Four Nsight deployment attempts failed before any production endpoint; no licensed
tool binaries are published. Native optional events supplied sufficient attribution.
No repeated tuning sweep, looser gate, cache, CPU production oracle or marketing
GPU-name rule is introduced. Algebra is generic, but complete profitability was
sampled only on the recorded r2SCAN-3c geometry and two devices. Unqualified
grids/displacements, other workloads and other FP64-matrix GPUs remain follow-up
qualification, not implied results. Stationary force remains the next large owner.

## References

- Tracking external memory: issue #2976.
- [Current contract](../../../../docs/developer/xc_native_cuda.md).
- [Raw source-matched evidence and offline replay](../../../../benchmarks/results/r2scan-density-gemm-20261107/README.md).
