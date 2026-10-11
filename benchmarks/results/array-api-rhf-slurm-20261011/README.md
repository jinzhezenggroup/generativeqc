# Array API RHF: real-device Slurm qualification, 2026-10-11

Measurement source: `aecab1e3ad9f5ce6bc12de884b87eed35b1628d7` (tree
`b73dacbca0c542f9ee40cf96239be39424ecfe37`), based on master
`140160231f23d61cea6ce103ec5a46a0da915b31`. The same source archive and rebuilt
CUDA library run on both machines. `provenance.json` records their hashes,
finite scheduler commands, build flags, toolchain and raw-file checksums.

## Correctness

- n1 / node1, Slurm **7586**, assigned `CUDA_VISIBLE_DEVICES=1`: **67 passed**,
  zero skipped, in 335.34 seconds. RTX 5090, driver 580.95.05.
- n4 / node4, Slurm **767**, assigned `CUDA_VISIBLE_DEVICES=0`: **67 passed**,
  zero skipped, in 433.48 seconds. RTX 5090, driver 580.178.04.
- These suites include **26 actual CUDA cases** per machine; the other cases
  exercise host/compiler/orchestration contracts, not GPU execution.
- Complete FP64 nuclear gradients: H2/STO-3G, HeH+/def2-SVP and water/STO-3G.
  Check against the independent central energy-difference oracle (absolute
  directional tolerance `5e-7` Hartree/Bohr), translation invariance and native
  endpoint parity. H2 also checks the retained independent energy oracle.
- Compiled nonlinear geometry/energy composition and reusable mutation-safe
  VJP run on CUDA. Scalar float32/float64 exp/log/sqrt/power primal/JVP/VJP run
  ordinary, fused and recomputed schedules, including resident execution,
  domain/overflow recovery and stale-output rejection.

The first n4 gate (**764**) exposes the dormant FP32 transcendental planner
rejection: one failure, three passes. Removing only the stale FP64 admission
restriction reuses the existing typed emitter, AD and error guards. The bounded
follow-up (**765**) passes all 24 scalar/frontend cases; both final source-matched
67-case suites pass after the fix. This does not relax molecular FP64 admission
or enable fast math. Retain the failed gate rather than treating a CUDA planner
check as real-device success.

n2 device-discovery job **2893** fails NVML initialization with a driver/library
mismatch. Do not claim PRO 6000 qualification. CPU-only Slurm jobs **2894/2895**
build/rebuild the matching CUDA library there; actual execution uses n1/n4.
No n3/node3 allocation or device-visibility override occurs.

## Complete energy-plus-nuclear-gradient endpoint

Host Bohr coordinates enter; converged total energy and complete nuclear
gradients return to host. Both paths use the same native FP64 exact-J/K RHF
driver, STO-3G basis, fresh Hcore start and convergence/screening settings.
Alternate native/generated order over ten warm repetitions. Keep cold JIT and
all raw samples in the endpoint JSON; these medians do not include compilation.

| Machine | Molecule | Native median (ms) | Array API median (ms) | Overhead |
| --- | --- | ---: | ---: | ---: |
| n1, 7586 | H2 | 4.164925 | 4.813250 | 15.5663% |
| n1, 7586 | water | 101.301279 | 102.508631 | 1.1918% |
| n4, 767 | H2 | 4.193418 | 4.948853 | 18.0148% |
| n4, 767 | water | 99.437864 | 100.245991 | 0.8127% |

Water meets the observed 5% timing target on both devices. **H2 does not**;
retain this tiny-endpoint overhead rather than cherry-picking water. The added
absolute cost is about 0.65-0.76 ms for H2 and 0.81-1.21 ms for water; these
measurements alone do not identify its host/device breakdown.

Every warm path takes two SCF iterations for H2 and twelve for water. Each
generated endpoint performs eleven real native force calls (one cold plus ten
warm), totaling 22/132 SCF iterations respectively. Reported cold parity errors
are zero for H2 and at most `1.4211e-14` Hartree / `1.7903e-15` Hartree/Bohr for
water. Every warm replay separately satisfies `1e-10` energy / `1e-8` gradient
acceptance against the original native result.

## Remaining qualification boundaries

This is a fixed native RHF method primitive with compiled surrounding algebra
and its first-order chain rule, not an arbitrary electronic-functional compiler.
Inputs/results are host-staged, not resident public GPU arrays. It does not
qualify DFT, higher derivatives or arbitrary algorithms.

Native strict-Fock counters are valid and retained, but detailed precision /
exact-J/K operator inventories remain explicitly incomplete. Peak host/device
memory is not measured. Therefore every JSON keeps
`gpu_performance_qualified=false`, including water where the observed timing
target passes. Numerical and timing observations are not a universal <=5%
performance or memory/work qualification.

## Reproduction and retained evidence

Check out the measurement source on an allowed remote machine. Build Release
with CUDA 12.9, `120-real`, the verified cache launcher and the exact flags in
`provenance.json`. This qualification enables native shell AOT and disables
stationary CPU/CUDA force AOT and automatic XC CUDA AOT; those options are not
silently replaced by another prebuilt library. Both execution jobs have finite
20-minute limits and leave Slurm visibility unchanged.

`run.sh` and `build.sh` retain the exact environment/commands used. `n1/` and
`n4/` contain unmodified benchmark JSON, JUnit, test logs, device/toolchain/
source/library receipts and cache statistics. `build/` retains initial/final
build/cache receipts. Full compiler commands/build logs and the 498 MiB library
remain ignored locally under `.artifacts/array-api-rhf-slurm-20261011/`, and on
the remote isolated checkout under
`/home/jzzeng/codes/qc-array-api-rhf-d15383b49-20261011/.artifacts/qualification/`.
The directory's older initial-SHA label is not its final source provenance;
always use the recorded final archive and source identity.
