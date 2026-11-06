# Guarded full-Direct default qualification

`GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY` unset now selects the same guarded
dense-Direct replay policy as explicit `1`; explicit `0` restores eager
preparation. Strict FP64 RKS/device-fused XC, immutable replay-source policy and
lifetime, and actual following-force memory headroom remain required. There is
no GPU-model dispatch, CPU oracle in production, tolerance relaxation or fitting
substitution. See the [default decision](../../../.agents/notes/implemented/performance/2026-11-05-default-direct-range-warm-anchor.md).

The [original matched trial](../direct-range-warm-anchor-20261105/README.md)
reports 9.222% A100 and 7.349% RTX 5090 cold reductions, with 6.152% and 5.700%
cold/three-warm lifecycle reductions. Its opt-in source/binary identities and
timings are preserved, not relabeled as new-default measurements.

## New-default observations

One unset-default cold/three-warm population per device, full-Direct
WB97M-V/def2-SVP energy and analytic forces, water96 (96 atoms/768 spherical
AOs). Cold includes construction, preparation, SCF, forces and host export;
warm calls freeze only their own owner's accepted density. The native runner
rejects PySCF/GPU4PySCF imports. No successful matched population is repeated.

| Device/allocation | Cold (s) | Warm 1 (s) | Warm 2 (s) | Warm 3 (s) |
| --- | ---: | ---: | ---: | ---: |
| A100-PCIE-40GB, LSF 225544 | 118.503707 | 21.692416 | 21.598975 | 21.600666 |
| RTX 5090, Slurm 15959 | 203.074699 | 19.287336 | 19.304196 | 19.325214 |

Both cold endpoints use 18 iterations/18 Focks/18 J/36 K applications; all warm
calls use one iteration/one Fock. VV10 work agrees with the retained optimized
trial. No incumbent rebase or deferred preparation Fock appears in any call.
Warm ratios to the earlier eager observations are 0.998309/1.000232/1.001367
(A100) and 1.011313/1.011175/1.010297 (5090), passing the predeclared per-call
1.05 timing guard. These are separate single populations, not statistical proof
of zero warm regression or another matched timing claim.

Two independent same-grid Direct capacity/lifetime cases and two public
default/strict/moved-geometry E+F cases pass on each device. The independent
large stock-default GPU4PySCF reference from LSF 225523 is **reused**, not rerun.
All 16 new native/reference pairs pass the common E/atom, force-max and force-RMS
gates (1e-5 Ha, 1e-4 and 3e-5 Ha/Bohr). Strict observable parity against the
original optimized image also passes (1e-9 Ha/atom, 1e-7 and 3e-8 Ha/Bohr).
The native/reference grid counts differ; this is not a tight/grid-extrapolated
oracle, and native A100 cold performance does not reach GPU4PySCF parity.

## Replay and identity

Run `python benchmarks/results/direct-range-default-20261105/replay.py`.
This CPU-only reader checks the compact new observations and the hash-verified
original publication, which owns the shared molecular inputs and independent
reference. Every new large sample's full-precision E/F/work payload is retained
in `publication.json.xz`; raw compiler/test/trace logs remain under ignored
`.artifacts/cold-direct-default-20261105/` with publication receipt hashes.

Production commit: `00ec18f032a1574ed495d0cdc9615657ec204687`.
Native source: `c69a04e5c7bc79220e4e52516c79ab86d76ed42b42b446de657e66a8264f7f32`.
Core ELF: `3cf435f6fb2ae36cc411d27a44a88498544ec896bc6892ecf793e9a898c04dcb`.
The same SM80/SM120 Release bytes execute on both actual allocated devices,
fast-compile off. A single-owner incremental build uses verified sccache 0.16.0
(two CXX misses/compilations, no failures); no full CUDA rebuild or cache clearing.
LSF is terminal DONE; Slurm runner and client return zero. Accounting storage is
disabled on Slurm, so no `sacct` terminal receipt is claimed. Finite GPU limits
are 15 minutes on `gpu_chem/gpu22` and 12 minutes on `main/node1`; assigned
device visibility remains unchanged. Engineering memory: #2976; PR #3006.
