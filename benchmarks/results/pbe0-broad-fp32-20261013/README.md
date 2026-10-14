# Broad FP32 stationary-gradient candidate: exact PBE0

This implements FP32 arithmetic, not an FP64 optimization renamed Fast. It is
still a **private native-controller candidate**, not public
`PrecisionCalculator(policy="fast")`, and not literal all-FP32 computation.

## Complete endpoints

Final source-matched measurement: Slurm **8154**, node1 RTX 5090, assigned
visibility `3`, CUDA 12.9.1, Release/sm_120, fast-compile disabled. Exact non-DF
PBE0/RKS, spherical def2-SVP, Hcore initial guess, moving grid `(48,16,32)`;
energy/density/screening tolerances `1e-10 / 1e-8 / 1e-12`, maximum 180 iterations.
The 384-AO fixture is two translated S4 water octamers, not an optimized cluster.

Each sample includes fresh construction, unchanged FP64 SCF, the **complete
native gradient**, export, teardown and synchronization. One warmup pair is
retained but excluded; medians use three alternating-order pairs. Separate
instrumented same-state replays never supply clean timings.

| AOs | Complete FP64 / mixed (s) | Complete speedup | Gradient FP64 / mixed (s) | Gradient speedup |
| ---: | ---: | ---: | ---: | ---: |
| 192 | 11.249630 / 8.039422 | 1.3993x | 5.523108 / 2.310956 | 2.3900x |
| 384 | 55.390198 / 32.428234 | 1.7081x | 35.859010 / 12.918232 | 2.7758x |

Strict/mixed SCF work is identical: 17 iterations / 18 Focks at 192 AOs and
22 / 23 at 384 AOs. Geometry/grid semantic work also matches; all samples pass
the independent gates, with zero ordinary task or complete-gradient retries.
Three repeats give descriptive medians, not statistical confidence bounds.

**The comparator is this same native executor in FP64, not the public Python
force executor. These ratios do not establish a public Fast speedup or superiority
over the current public Calculator.** Do not combine them with older public
endpoint timings, or upstream gpu4pyscf-mixedprec's DF-r2SCAN gradient-only 7x.
Native scheduling/public integration remain separate work.

## Actual arithmetic and error budget

The shared compiler IR now lowers these regions to FP32 operands and arithmetic:
weighted ERI derivatives through order five; AO spatial jets; density projection
products and feature sums; interior PBE exchange **and** correlation; AO
pullbacks, Becke reverse traversal and point-local force accumulation. Projection
and local force sums really accumulate in FP32; they are not FP64 computations
followed by an output cast. No TF32 substitution is used.

SCF/J/K, one-electron/Pulay derivatives, higher-order ERI derivatives, Boys
moments, physical XC normalization, center preparation and global source/force
sums remain FP64. Double transport buffers also remain. Typed geometry-point
counts describe attempted launches, **not** a proof that every XC tail point
executed FP32. Representability boundaries use the unchanged strict equation.

Independent PySCF SCF and complete moving-grid forces gate every endpoint:

| AOs | Energy error (Eh) | Maximum force-component error (Eh/bohr) | RMS force error (Eh/bohr) |
| ---: | ---: | ---: | ---: |
| 192 | 3.30e-12 | 8.28e-7 | 2.94e-7 |
| 384 | 1.14e-11 | 1.24e-6 | 3.85e-7 |

Mixed acceptance is `1e-7 Eh`, `1e-5 Eh/bohr` maximum and `3e-6 Eh/bohr` RMS;
strict force acceptance is `1e-6 Eh/bohr`. The larger case **does not meet a
1e-6 mixed force target**. Neither empirical gates nor nonfinite checks bound
finite error for an arbitrary molecule or certify Hessians, TS/IRC or NVE MD.

An FP32 density-tail underflow discovered during testing is repaired by
recomputing the affected bilinears in FP64, not by clipping density or removing
grid points. Unresolved numerical failure retries the entire gradient once in
FP64 at the same authenticated state. Reset fences borrowed streams while
discarding failed numerical status; CUDA/token/resource errors are not numerical
retries. Optional float storage stays within the native resource allowance.

## Evidence and reproduction

`summary.json.xz` records source/library/probe and raw-file hashes, every compact
timing sample, work receipts, independent errors, replay gates and validation.
Failed/intermediate experiments and source archives remain in ignored
`.artifacts/pbe0-broad-fp32-20261013/`, with the corresponding campaign on node1.
No Release or external backup is published.

Validation: 83 targeted host tests (24 skips), 109 host contract tests, and
537 compiler modules checked without dependency errors. Slurm 8162 passes
22 public/native CUDA endpoint tests; Slurm 8160 passes six selected tests under
memcheck, with zero errors. Coverage includes spherical/Cartesian water,
tetramer, ammonia, independent Libxc/Decimal/long-double ingredient oracles,
strict same-state replay, nonfinite task retry and pending-stream reset.

Build a Release CUDA library with the native `pbe0_rks` stationary profile and
link `tests/native/mixed_force_endpoint_probe.cpp` against that library. Omit
`--reference` to generate the independent PySCF oracle before timing, or supply
a retained JSON containing the matching `protocol` and `reference` records.
With source `python/` and the repository on `PYTHONPATH`, run:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=00:15:00 \
  python -m benchmarks.pbe0_mixed_gradient \
  --case water-octamer-s4-def2-svp-spherical --probe PROBE.so \
  --reference REFERENCE.json --output .artifacts/broad-fp32.json \
  --repeats 3 --precision-mask 7
```

Mask bits select weighted derivatives (1), geometry (2), and grid production
(4); they are qualification controls, not public user-facing precision knobs.
See the [decision note](../../../.agents/notes/implemented/numerics/2026-10-13-broad-fp32-stationary-candidate.md)
for shared algebra, scratch lifetimes, rejected fixes and promotion boundaries.
